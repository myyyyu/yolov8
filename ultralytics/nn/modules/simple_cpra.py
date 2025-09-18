#!/usr/bin/env python3
"""
Simplified CPRAformer for better YOLO integration
只保留核心的跨域注意力机制，去掉复杂的FFT和过度设计的部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCrossAttention(nn.Module):
    """简化的跨域注意力机制"""
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.norm = nn.BatchNorm2d(dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 生成Q,K,V
        qkv = self.qkv(x)  # (B, C*3, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        
        # 确保维度正确
        if qkv.size(1) != 3:
            # 如果维度不匹配，直接返回归一化的输入
            return self.norm(x)
            
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # 每个: (B, num_heads, C//num_heads, H*W)
        
        # 计算注意力
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 应用注意力
        out = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        out = self.proj(out)
        
        return self.norm(out)

class SimpleTransformerBlock(nn.Module):
    """简化的Transformer块"""
    
    def __init__(self, dim, num_heads=8, ffn_ratio=4.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SimpleCrossAttention(dim, num_heads)
        self.norm2 = nn.BatchNorm2d(dim)
        
        # 简化的FFN
        hidden_dim = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )
        
    def forward(self, x):
        # 残差连接
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class SimpleCPRAformerC2f(nn.Module):
    """简化的CPRAformer C2f模块"""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        # 简化的Transformer块
        self.m = nn.ModuleList([
            SimpleTransformerBlock(
                dim=self.c,
                num_heads=max(4, self.c // 32),  # 适中的头数
                ffn_ratio=2.0  # 减小FFN比例
            ) for _ in range(n)
        ])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))