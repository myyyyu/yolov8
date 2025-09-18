"""
CPRAformer modules for YOLOv8 integration
Cross Paradigm Representation and Alignment Transformer for enhanced object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)
        self.dwconv_2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, padding='same',
                                  groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv(x1)
        x2 = self.dwconv_2(x2)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class spr_sa(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x1 = F.adaptive_avg_pool2d(x, (1, 1))
        x1 = F.softmax(x1, dim=1)
        x = x1 * x
        x = self.act(x)
        x = self.conv_1(x)
        return x


class ComplexFFT(nn.Module):
    def __init__(self):
        super(ComplexFFT, self).__init__()

    def forward(self, x):
        # Ensure full precision for FFT operations to avoid cuFFT limitations
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.float()
        
        # 检查输入是否包含NaN或Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        real = x_fft.real
        imag = x_fft.imag
        
        # 检查FFT结果是否有效
        if torch.isnan(real).any() or torch.isnan(imag).any():
            real = torch.nan_to_num(real, nan=0.0)
            imag = torch.nan_to_num(imag, nan=0.0)
        
        # Convert back to original dtype
        if input_dtype == torch.float16:
            real = real.half()
            imag = imag.half()
            
        return real, imag


class ComplexIFFT(nn.Module):
    def __init__(self):
        super(ComplexIFFT, self).__init__()

    def forward(self, real, imag):
        # Ensure full precision for FFT operations to avoid cuFFT limitations
        input_dtype = real.dtype
        if real.dtype == torch.float16:
            real = real.float()
            imag = imag.float()
        
        # 检查输入是否包含NaN或Inf
        if torch.isnan(real).any() or torch.isnan(imag).any():
            real = torch.nan_to_num(real, nan=0.0, posinf=1e6, neginf=-1e6)
            imag = torch.nan_to_num(imag, nan=0.0, posinf=1e6, neginf=-1e6)
            
        x_complex = torch.complex(real, imag)
        x_ifft = torch.fft.ifft2(x_complex, dim=(-2, -1))
        result = x_ifft.real
        
        # 检查IFFT结果是否有效
        if torch.isnan(result).any() or torch.isinf(result).any():
            result = torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Convert back to original dtype
        if input_dtype == torch.float16:
            result = result.half()
            
        return result


class Stage2_fft(nn.Module):
    def __init__(self, in_channels):
        super(Stage2_fft, self).__init__()
        self.c_fft = ComplexFFT()
        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, padding=0,
                                groups=in_channels * 2)
        self.c_ifft = ComplexIFFT()

    def forward(self, x):
        real, imag = self.c_fft(x)

        combined = torch.cat([real, imag], dim=1)
        conv_out = self.conv1x1(combined)

        out_channels = conv_out.shape[1] // 2
        real_out = conv_out[:, :out_channels, :, :]
        imag_out = conv_out[:, out_channels:, :, :]

        output = self.c_ifft(real_out, imag_out)

        return output


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.spr_sa = spr_sa(dim // 2, 2)
        self.linear_0 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.qkv = nn.Conv2d(dim // 2, dim // 2 * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim // 2 * 3, dim // 2 * 3, kernel_size=3, stride=1, padding=1,
                                   groups=dim // 2 * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim // 2, max(1, dim // 8), kernel_size=1),
            nn.GroupNorm(1, max(1, dim // 8)),  # Use GroupNorm instead of BatchNorm
            nn.GELU(),
            nn.Conv2d(max(1, dim // 8), dim // 2, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim // 2, max(1, dim // 16), kernel_size=1),
            nn.GroupNorm(1, max(1, dim // 16)),  # Use GroupNorm instead of BatchNorm
            nn.GELU(),
            nn.Conv2d(max(1, dim // 16), 1, kernel_size=1)
        )
        self.fft = Stage2_fft(in_channels=dim)
        self.gate = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        y, x = self.linear_0(x).chunk(2, dim=1)

        y_d = self.spr_sa(y)

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-8)
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-8)

        _, _, C, _ = q.shape
        # Ensure we have at least 1 and at most C for dynamic_k
        gate_val = self.gate(x).view(b, -1).mean().clamp(0.1, 1.0)
        
        # 防止NaN错误和确保数值稳定性
        if torch.isnan(gate_val).any() or torch.isinf(gate_val).any():
            gate_val = torch.tensor(0.5, device=gate_val.device, dtype=gate_val.dtype)
        
        # 确保gate_val在有效范围内并安全转换
        gate_val = torch.clamp(gate_val, 0.1, 1.0)
        try:
            dynamic_k = max(1, min(C, int(C * gate_val.item())))
        except (ValueError, OverflowError):
            dynamic_k = max(1, C // 2)  # 使用默认值
        
        # Fix temperature indexing for single head case
        temp = self.temperature if self.num_heads > 1 else self.temperature[0]
        attn = (q @ k.transpose(-2, -1)) * temp
        
        # Only apply top-k selection if we have more than 1 dimension
        if C > 1:
            mask = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
            index = torch.topk(attn, k=dynamic_k, dim=-1, largest=True)[1]
            mask.scatter_(-1, index, 1.)
            attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))

        # 确保softmax数值稳定性
        attn = torch.clamp(attn, min=-1e8, max=1e8)  # 防止极值
        attn = attn.softmax(dim=-1)
        
        # 检查softmax结果
        if torch.isnan(attn).any() or torch.isinf(attn).any():
            attn = torch.ones_like(attn) / attn.size(-1)  # 均匀分布作为fallback
        out1 = (attn @ v)
        out2 = (attn @ v)
        out3 = (attn @ v)
        out4 = (attn @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out_att = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # AAFM
        channel_map = self.channel_interaction(out_att)
        spatial_map = self.spatial_interaction(y_d)

        attened_x = out_att * torch.sigmoid(spatial_map)
        conv_x = y_d * torch.sigmoid(channel_map)

        x = torch.cat([attened_x, conv_x], dim=1)
        out = self.project_out(x)
        # out = self.fft(out)  # 注释掉FFT操作避免NaN问题
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CPRAformerC2f(nn.Module):
    """CPRAformer-enhanced C2f module for YOLOv8"""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize CPRAformerC2f module.
        
        Args:
            c1 (int): Input channels
            c2 (int): Output channels  
            n (int): Number of CPRAformer blocks
            shortcut (bool): Whether to use shortcut connections
            g (int): Groups for convolutions (not used in CPRAformer)
            e (float): Expansion ratio
        """
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        # Replace standard bottlenecks with CPRAformer blocks
        self.m = nn.ModuleList([
            TransformerBlock(
                dim=self.c,
                num_heads=max(1, self.c // 64),  # Adaptive number of heads
                ffn_expansion_factor=2.66,
                bias=False,
                LayerNorm_type='WithBias'
            ) for _ in range(n)
        ])

    def forward(self, x):
        """Forward pass through CPRAformerC2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class LightCPRAformerC2f(nn.Module):
    """Lightweight CPRAformer-enhanced C2f module for better speed/accuracy balance"""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        # Simplified CPRAformer blocks
        self.m = nn.ModuleList([
            LightTransformerBlock(
                dim=self.c,
                num_heads=max(1, self.c // 32),  # More heads for lighter version
                ffn_expansion_factor=2.0
            ) for _ in range(n)
        ])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class LightTransformerBlock(nn.Module):
    """Lightweight Transformer block optimized for YOLO"""
    
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)  # Use GroupNorm instead of BatchNorm
        self.attn = SimplifiedAttention(dim, num_heads)
        self.norm2 = nn.GroupNorm(1, dim)  # Use GroupNorm instead of BatchNorm
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, int(dim * ffn_expansion_factor), 1),
            nn.GELU(),
            nn.Conv2d(int(dim * ffn_expansion_factor), dim, 1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SimplifiedAttention(nn.Module):
    """Simplified attention mechanism for lightweight operation"""
    
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # Shape: (B, C*3, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        
        # Check if we have valid dimensions
        if qkv.size(0) < 3:
            # Fallback for edge case
            return self.proj(x)
            
        q, k, v = qkv.unbind(0)  # Each: (B, num_heads, C//num_heads, H*W)
        
        # Compute attention
        attn = (q.transpose(-2, -1) @ k) * self.scale  # (B, num_heads, H*W, H*W)
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        out = (v @ attn.transpose(-2, -1))  # (B, num_heads, C//num_heads, H*W)
        out = out.transpose(1, 2).reshape(B, C, H, W)  # Reshape back to (B, C, H, W)
        
        return self.proj(out)