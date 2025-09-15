# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
ContMix module adapted for Ultralytics YOLO
Based on OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels
Paper: https://arxiv.org/abs/2502.20087
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from timm.models.layers import DropPath

try:
    from natten.functional import na2d_av
    has_natten = True
except ImportError:
    has_natten = False
    warnings.warn("natten not installed. Using fallback implementation for ContMix.")

from .conv import Conv

__all__ = ['ContMixBlock', 'ContMixC2f']


class LayerNorm2d(nn.LayerNorm):
    """2D Layer Normalization"""
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)
    
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x.contiguous()


class GRN(nn.Module):
    """Global Response Normalization from ConvNeXt V2"""
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


class SEModule(nn.Module):
    """Squeeze-and-Excitation module"""
    def __init__(self, dim, red=8):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        return x * self.proj(x)


class ResDWConv(nn.Conv2d):
    """Depthwise convolution with residual connection"""
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
    
    def forward(self, x):
        return x + super().forward(x)


class ContMixBlock(nn.Module):
    """
    ContMix block adapted for YOLO
    Simplified version without dilated reparam for easier integration
    """
    def __init__(self, 
                 c1, 
                 c2=None,
                 kernel_size=7,
                 smk_size=5,
                 num_heads=2,
                 mlp_ratio=2,
                 drop_path=0,
                 **kwargs):
        super().__init__()
        
        c2 = c2 or c1  # output channels same as input if not specified
        mlp_dim = int(c1 * mlp_ratio)
        self.kernel_size = kernel_size
        self.smk_size = smk_size
        self.num_heads = num_heads * 2
        head_dim = c1 // self.num_heads
        self.scale = head_dim ** -0.5
        
        # Pre-processing
        self.dwconv1 = ResDWConv(c1, kernel_size=3)
        self.norm1 = LayerNorm2d(c1)
        
        # Weight generation for dynamic convolution
        self.weight_query = nn.Sequential(
            nn.Conv2d(c1, c1//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1//2),
        )
        self.weight_key = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(c1, c1//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1//2),
        )
        self.weight_value = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
        )
        
        # Dynamic kernel weight projection
        self.weight_proj = nn.Conv2d(49, kernel_size**2 + smk_size**2, kernel_size=1)
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
        )
        
        # Large kernel convolution (simplified without dilation)
        self.lepe = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=kernel_size, padding=kernel_size//2, groups=c1),
            nn.BatchNorm2d(c1),
        )
        
        self.se_layer = SEModule(c1)
        self.gate = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
        )
        
        # Output projection
        self.proj = nn.Sequential(
            nn.BatchNorm2d(c1),
            nn.Conv2d(c1, c2, kernel_size=1),
        )
        
        # FFN
        self.dwconv2 = ResDWConv(c2, kernel_size=3)
        self.norm2 = LayerNorm2d(c2)
        self.mlp = nn.Sequential(
            nn.Conv2d(c2, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, c2, kernel_size=1),
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        # Relative position bias
        self.get_rpb()

    def get_rpb(self):
        """Initialize relative position bias"""
        self.rpb_size1 = 2 * self.smk_size - 1
        self.rpb1 = nn.Parameter(torch.zeros(self.num_heads, self.rpb_size1, self.rpb_size1))
        self.rpb_size2 = 2 * self.kernel_size - 1
        self.rpb2 = nn.Parameter(torch.zeros(self.num_heads, self.rpb_size2, self.rpb_size2))
        nn.init.trunc_normal_(self.rpb1, std=0.02)
        nn.init.trunc_normal_(self.rpb2, std=0.02)

    @torch.no_grad()
    def generate_idx(self, kernel_size):
        rpb_size = 2 * kernel_size - 1
        idx_h = torch.arange(0, kernel_size)
        idx_w = torch.arange(0, kernel_size)
        idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
        return (idx_h, idx_w, idx_k)

    def apply_rpb(self, attn, rpb, height, width, kernel_size, idx_h, idx_w, idx_k):
        """Apply relative position bias"""
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_h[kernel_size//2] = height - (kernel_size-1)
        num_repeat_w[kernel_size//2] = width - (kernel_size-1)
        bias_hw = (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*kernel_size-1)) + idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + idx_k
        bias_idx = bias_idx.reshape(-1, int(kernel_size**2))
        bias_idx = torch.flip(bias_idx, [0])
        rpb = torch.flatten(rpb, 1, 2)[:, bias_idx]
        rpb = rpb.reshape(1, int(self.num_heads), int(height), int(width), int(kernel_size**2))
        return attn + rpb

    def forward(self, x):
        input_resolution = x.shape[2:]
        B, C, H, W = x.shape
        
        x = self.dwconv1(x)
        identity = x
        x = self.norm1(x)
        gate = self.gate(x)
        lepe = self.lepe(x)
        
        # Handle small feature maps
        is_pad = False
        if min(H, W) < self.kernel_size:
            is_pad = True
            if H < W:
                size = (self.kernel_size, int(self.kernel_size / H * W))
            else:
                size = (int(self.kernel_size / W * H), self.kernel_size)
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            H, W = size
        
        # Generate dynamic weights
        query = self.weight_query(x) * self.scale
        key = self.weight_key(x)
        value = self.weight_value(x)
        
        query = rearrange(query, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        key = rearrange(key, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        weight = einsum(query, key, 'b g c n, b g c l -> b g n l')
        weight = rearrange(weight, 'b g n l -> b l g n').contiguous()
        weight = self.weight_proj(weight)
        weight = rearrange(weight, 'b l g (h w) -> b g h w l', h=H, w=W)
        
        # Split into two branches
        attn1, attn2 = torch.split(weight, split_size_or_sections=[self.smk_size**2, self.kernel_size**2], dim=-1)
        rpb1_idx = self.generate_idx(self.smk_size)
        rpb2_idx = self.generate_idx(self.kernel_size)
        attn1 = self.apply_rpb(attn1, self.rpb1, H, W, self.smk_size, *rpb1_idx)
        attn2 = self.apply_rpb(attn2, self.rpb2, H, W, self.kernel_size, *rpb2_idx)
        attn1 = torch.softmax(attn1, dim=-1)
        attn2 = torch.softmax(attn2, dim=-1)
        
        value = rearrange(value, 'b (m g c) h w -> m b g h w c', m=2, g=self.num_heads)
        
        # Apply dynamic convolution
        if has_natten:
            x1 = na2d_av(attn1, value[0], kernel_size=self.smk_size)
            x2 = na2d_av(attn2, value[1], kernel_size=self.kernel_size)
        else:
            # Fallback implementation
            pad1 = self.smk_size // 2
            pad2 = self.kernel_size // 2
            
            v1 = rearrange(value[0], 'b g h w c -> b (g c) h w')
            v2 = rearrange(value[1], 'b g h w c -> b (g c) h w')
            
            v1 = F.unfold(v1, kernel_size=self.smk_size).reshape(B, -1, H-2*pad1, W-2*pad1)
            v2 = F.unfold(v2, kernel_size=self.kernel_size).reshape(B, -1, H-2*pad2, W-2*pad2)
            
            v1 = F.pad(v1, (pad1, pad1, pad1, pad1), mode='replicate')
            v2 = F.pad(v2, (pad2, pad2, pad2, pad2), mode='replicate')
            
            v1 = rearrange(v1, 'b (g c k) h w -> b g c h w k', g=self.num_heads, k=self.smk_size**2, h=H, w=W)
            v2 = rearrange(v2, 'b (g c k) h w -> b g c h w k', g=self.num_heads, k=self.kernel_size**2, h=H, w=W)
            
            x1 = einsum(attn1, v1, 'b g h w k, b g c h w k -> b g h w c')
            x2 = einsum(attn2, v2, 'b g h w k, b g c h w k -> b g h w c')
        
        x = torch.cat([x1, x2], dim=1)
        x = rearrange(x, 'b g h w c -> b (g c) h w', h=H, w=W)
        
        if is_pad:
            x = F.adaptive_avg_pool2d(x, input_resolution)
        
        x = self.fusion_proj(x)
        x = x + lepe
        x = self.se_layer(x)
        x = gate * x
        x = self.proj(x)
        
        # Residual connection
        x = identity + self.drop_path(x)
        x = self.dwconv2(x)
        
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class ContMixC2f(nn.Module):
    """
    ContMix-enhanced C2f block for YOLO
    Replaces some Bottleneck blocks with ContMix blocks
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, contmix_ratio=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        
        # Mix of regular bottlenecks and ContMix blocks
        n_contmix = max(1, int(n * contmix_ratio))
        n_regular = n - n_contmix
        
        self.m = nn.ModuleList()
        # Add regular bottlenecks first
        for _ in range(n_regular):
            self.m.append(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0))
        # Add ContMix blocks
        for _ in range(n_contmix):
            self.m.append(ContMixBlock(self.c, self.c, kernel_size=7, smk_size=5, num_heads=2, mlp_ratio=2))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# Import Bottleneck from block module for ContMixC2f
from .block import Bottleneck