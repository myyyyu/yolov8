#!/usr/bin/env python3
"""
超轻量级CPRAformer - 减少内存占用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightCrossAttention(nn.Module):
    """超轻量级跨域注意力机制 - 减少内存占用"""
    
    def __init__(self, dim, reduction=4):
        super().__init__()
        # 大幅减少中间层维度
        self.reduced_dim = max(16, dim // reduction)
        
        # 使用更小的中间层
        self.compress = nn.Conv2d(dim, self.reduced_dim, 1)
        self.expand = nn.Conv2d(self.reduced_dim, dim, 1)
        
        # 简单的通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )
        
        # 简单的空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 通道注意力
        channel_weight = self.channel_attn(x)
        x_channel = x * channel_weight
        
        # 空间注意力
        spatial_weight = self.spatial_attn(x)
        x_spatial = x_channel * spatial_weight
        
        # 压缩和扩展
        compressed = self.compress(x_spatial)
        expanded = self.expand(compressed)
        
        return expanded

class UltraLightTransformerBlock(nn.Module):
    """超轻量级Transformer块"""
    
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = LightweightCrossAttention(dim, reduction=8)  # 大幅减少
        self.norm2 = nn.BatchNorm2d(dim)
        
        # 极简的FFN
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )
        
    def forward(self, x):
        # 残差连接
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class UltraLightCPRAformerC2f(nn.Module):
    """超轻量级CPRAformer C2f模块"""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        # 超轻量级Transformer块
        self.m = nn.ModuleList([
            UltraLightTransformerBlock(self.c) for _ in range(n)
        ])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))