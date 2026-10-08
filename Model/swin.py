"""3D Swin Transformer backbone for F2Mix.

A compact re-implementation of "Swin Transformer: Hierarchical Vision
Transformer using Shifted Windows" (https://arxiv.org/abs/2103.14030) extended
to 3D volumes, following Video Swin Transformer. ``forward`` returns the
last-stage feature map (B, embed_dim * 2**(num_layers-1), d, h, w).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


def get_window_size(x_size, window_size, shift_size):
    """Clamp window/shift sizes to the input size."""
    use_window_size = list(window_size)
    use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            use_shift_size[i] = 0
    return tuple(use_window_size), tuple(use_shift_size)


def compute_mask(D, H, W, window_size, shift_size, device):
    """Attention mask for the shifted (SW-MSA) blocks."""
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0
    for d in (slice(0, -window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)):
        for h in (slice(0, -window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)):
            for w in (slice(0, -window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size[0] * window_size[1] * window_size[2])
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))


def window_partition(x, window_size):
    """(B, D, H, W, C) -> (num_windows*B, wd, wh, ww, C)."""
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1],
               window_size[1], W // window_size[2], window_size[2], C)
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, *window_size, C)


def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2],
                     window_size[0], window_size[1], window_size[2], -1)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.drop(self.fc1(x)))))


class WindowAttention3D(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))
        trunc_normal_(self.relative_position_bias_table, std=.02)

        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # The relative position bias is precomputed for the nominal window
        # size; skip it when the window was clamped to a smaller feature map.
        nominal_tokens = (self.window_size[0] * self.window_size[1]
                          * self.window_size[2])
        if N == nominal_tokens:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                nominal_tokens, nominal_tokens, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.attn_drop(attn.softmax(dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock3D(nn.Module):
    """3D Swin block: (W-MSA / SW-MSA) + MLP with residual connections."""

    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        D, H, W = self.D, self.H, self.W
        assert L == D * H * W, 'input feature has wrong size'

        # Clamp window/shift sizes to the actual feature size (keeps the
        # pre-computed mask consistent with the block's internal padding).
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # Pad to multiples of the window size.
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        # Cyclic shift.
        if any(shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1],
                                              -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows and flatten each window to a token sequence.
        x_windows = window_partition(shifted_x, window_size)
        x_windows = x_windows.view(-1, window_size[0] * window_size[1]
                                   * window_size[2], C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *window_size, C)
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)

        # Reverse cyclic shift and remove padding.
        if any(shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1],
                                              shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        if pad_d1 or pad_r or pad_b:
            x = x[:, :D, :H, :W, :].contiguous()

        # Attention output with residual connection, then the MLP branch.
        x = shortcut + x.view(B, D * H * W, C)
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    """Downsample by concatenating 2x2x2 neighbouring patches then projecting."""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        B, D, H, W, C = x.shape
        pad_d = (2 - D % 2) % 2
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        D, H, W = x.shape[1:4]

        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 1::2, :]
        x6 = x[:, 1::2, 1::2, 0::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)

        return self.reduction(self.norm(x))


class PatchEmbed3D(nn.Module):
    """Volume to patch embedding with optional normalization."""

    def __init__(self, patch_size=(4, 4, 4), in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # (B, C, d, h, w)
        if self.norm is not None:
            D, H, W = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.proj.out_channels, D, H, W)
        return x


class StemConv(nn.Module):
    """Light convolutional stem before patch embedding (as in the paper)."""

    def __init__(self, in_chans=1, embed_dim=96):
        super().__init__()
        self.ext = nn.ModuleList([
            nn.Conv3d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(embed_dim),
        ])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for i, layer in enumerate(self.ext):
            x = layer(x)
            if i % 2 == 0:
                x = self.relu(x)
        return x


class SwinTransformer3DBackbone(nn.Module):
    """3D Swin Transformer encoder.

    Input (B, 1, D, H, W) -> last-stage feature map
    (B, embed_dim * 2**(num_layers-1), d, h, w).
    """

    def __init__(self, patch_size=(2, 2, 2), in_chans=1, embed_dim=32,
                 depths=(2, 2, 2, 2), num_heads=(2, 4, 8, 16),
                 window_size=(2, 7, 7), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, patch_norm=False):
        super().__init__()
        self.pool_size = (1, 1, 1)
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size

        self.stem = StemConv(in_chans, embed_dim)
        self.patch_embed = PatchEmbed3D(patch_size, embed_dim, embed_dim,
                                        norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleDict({
                'blocks': nn.ModuleList([
                    SwinTransformerBlock3D(
                        dim=int(embed_dim * 2 ** i_layer),
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        shift_size=(0, 0, 0) if (j % 2 == 0) else tuple(w // 2 for w in window_size),
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i_layer]) + j], norm_layer=norm_layer)
                    for j in range(depths[i_layer])]),
                'downsample': PatchMerging(int(embed_dim * 2 ** i_layer), norm_layer)
                if i_layer < self.num_layers - 1 else None,
            })
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.feat_channels = self.num_features
        self.norm = norm_layer(self.num_features)
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
        x = self.stem(x)
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            B, C, D, H, W = x.shape
            window_size, shift_size = get_window_size((D, H, W), self.window_size,
                                                      tuple(w // 2 for w in self.window_size))
            # Blocks operate on flattened token sequences (B, L, C).
            x_tok = x.flatten(2).transpose(1, 2)

            # The mask is shared by the shifted blocks of this layer and is
            # computed with the same clamped window/shift sizes they use.
            mask = None
            if any(shift_size):
                Dp = int(np.ceil(D / window_size[0])) * window_size[0]
                Hp = int(np.ceil(H / window_size[1])) * window_size[1]
                Wp = int(np.ceil(W / window_size[2])) * window_size[2]
                mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

            for blk in layer['blocks']:
                blk.H, blk.W, blk.D = H, W, D
                x_tok = blk(x_tok, mask)

            if layer['downsample'] is not None:
                # PatchMerging expects (B, D, H, W, C) and returns (B, D', H', W', C').
                x = layer['downsample'](x_tok.view(B, D, H, W, -1))
                x = x.permute(0, 4, 1, 2, 3).contiguous()
            else:
                x = x_tok.transpose(1, 2).view(B, C, D, H, W)

        x = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C) for the final norm
        x = self.norm(x)
        return x.permute(0, 4, 1, 2, 3).contiguous()
