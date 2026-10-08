"""3D Pyramid Vision Transformer backbone (PVT-Tiny) for F2Mix.

Adapted from "Pyramid Vision Transformer: A Versatile Backbone for Dense
Prediction without Convolutions" (https://arxiv.org/abs/2102.12122) with the
patch embedding extended to 3D volumes. ``forward`` returns the last-stage
feature map (B, embed_dims[-1], d, h, w).
"""

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.drop(self.fc1(x)))))


class Attention(nn.Module):
    """Spatial-reduction attention over 3D tokens."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W, D):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        # NOTE: drop_path is kept as an attribute for API compatibility; the
        # plain block applies no stochastic depth.
        self.drop_path = drop_path
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, H, W, D):
        x = x + self.attn(self.norm1(x), H, W, D)
        return x + self.mlp(self.norm2(x))


class PatchEmbed3D(nn.Module):
    """Volume to patch embedding."""

    def __init__(self, patch_size=4, in_chans=1, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, C, d, h, w)
        _, C, D, H, W = x.shape
        return x.flatten(2).transpose(1, 2), (D, H, W)


class PVTBackbone(nn.Module):
    """PVT encoder with configurable stages.

    Input (B, 1, D, H, W) -> last-stage feature map (B, embed_dims[-1], d, h, w).
    """

    def __init__(self, patch_size=4, in_chans=1, embed_dims=(64, 128, 320),
                 num_heads=(1, 2, 5), mlp_ratios=(8, 8, 4), qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=(2, 2, 2), sr_ratios=(8, 4, 2)):
        super().__init__()
        self.pool_size = (1, 1, 1)
        self.depths = depths
        self.num_stages = len(depths)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stages):
            patch_embed = PatchEmbed3D(
                patch_size=patch_size if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i])
            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'pos_drop{i + 1}', nn.Dropout(p=drop_rate))
            setattr(self, f'block{i + 1}', nn.ModuleList([
                Block(dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                      qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                      drop_path=dpr[cur + j], norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                for j in range(depths[i])]))
            cur += depths[i]

        self.norm = norm_layer(embed_dims[-1])
        self.feat_channels = embed_dims[-1]
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            pos_drop = getattr(self, f'pos_drop{i + 1}')
            block = getattr(self, f'block{i + 1}')

            x, (H, W, D) = patch_embed(x)
            x = pos_drop(x)
            for blk in block:
                x = blk(x, H, W, D)
            if i != self.num_stages - 1:
                x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        x = self.norm(x)
        return x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()


def pvt_tiny(**kwargs) -> PVTBackbone:
    return PVTBackbone(patch_size=4, embed_dims=[64, 128, 320], num_heads=[1, 2, 5],
                       mlp_ratios=[8, 8, 4], qkv_bias=True, depths=[2, 2, 2],
                       sr_ratios=[8, 4, 2], **kwargs)


def pvt_small(**kwargs) -> PVTBackbone:
    return PVTBackbone(patch_size=4, embed_dims=[64, 128, 320], num_heads=[1, 2, 5],
                       mlp_ratios=[8, 8, 4], qkv_bias=True, depths=[3, 4, 6],
                       sr_ratios=[8, 4, 2], **kwargs)
