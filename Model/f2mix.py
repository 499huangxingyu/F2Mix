"""Shared F2Mix framework.

This module implements the model-agnostic parts of F2Mix:

- ``FeatureExpander``  : the FExp decoder that upsamples encoder feature maps
  back to the input resolution (used by the FCPC module).
- ``FeatureRefinement``: the FR module. It builds the primary feature ``f_p``
  from the original and augmented streams, removes the redundant component of
  the copy-paste features via orthogonal projection, and fuses everything for
  classification.
- ``F2MixNet``         : a plug-and-play wrapper that combines any backbone
  exposing ``forward(x) -> (B, C, d, h, w)`` with the modules above.

Reference: "F2Mix: Fine-grained augmentation and feature copy-paste
consistency for heterogeneous CT image classification".
"""

from math import prod
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleBlock(nn.Module):
    """Trilinear upsampling followed by conv-bn-relu (paper: FExp block)."""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: Tuple[int, int, int]):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(self.upsample(x))))


class FeatureExpander(nn.Module):
    """Decode encoder feature maps back to full input resolution.

    The number of upsampling stages and their scale factors are derived
    automatically from the encoder feature-map size and the input size, so the
    expander adapts to any backbone.
    """

    def __init__(self, feat_size: Sequence[int], input_size: Sequence[int], in_channels: int):
        super().__init__()
        feat_size, input_size = list(feat_size), list(input_size)
        if len(feat_size) != 3 or len(input_size) != 3:
            raise ValueError('feat_size and input_size must be 3-tuples (D, H, W)')

        stages = nn.ModuleList()
        cur_size, cur_ch = feat_size, in_channels
        # Keep doubling each spatial dim until it reaches the input size.
        # A dim that is already at full resolution keeps a scale of 1.
        while any(c * 2 <= t for c, t in zip(cur_size, input_size)):
            scale = tuple(2 if c * 2 <= t else 1 for c, t in zip(cur_size, input_size))
            cur_size = [min(c * s, t) for c, s, t in zip(cur_size, scale, input_size)]
            out_ch = max(cur_ch // 2, 16)
            stages.append(UpsampleBlock(cur_ch, out_ch, scale))
            cur_ch = out_ch

        # Final projection to a single-channel full-resolution feature map.
        stages.append(nn.Sequential(
            nn.Conv3d(cur_ch, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
        ))
        self.stages = stages
        self.input_size = tuple(input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        if x.shape[2:] != self.input_size:  # exact-resolution safeguard
            x = F.interpolate(x, size=self.input_size, mode='trilinear', align_corners=True)
        return x


class FeatureRefinement(nn.Module):
    """FR module: primary feature + orthogonal projection redundancy filter.

    Given the pooled features of the four views (A, B, A', B'):
      1. ``f_p = Linear(concat(f_A, f_B))``          (primary feature)
      2. ``f_x^orth = f_x - (f_x . f_p / |f_p|^2) f_p``  for x in {A', B'}
      3. ``f_cat = conv/linear fusion(f_p, f_A^orth, f_B^orth)``
    """

    def __init__(self, feat_dim: int, num_classes: int, eps: float = 1e-8):
        super().__init__()
        # Linear projection of the concatenated (A, B) features.
        self.embed = nn.Linear(feat_dim * 2, feat_dim)
        # Fusion of the primary feature and the two refined features.
        self.fusion = nn.Linear(feat_dim * 3, feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.eps = eps

    def orthogonal_projection(self, f: torch.Tensor, f_p: torch.Tensor) -> torch.Tensor:
        """Remove the component of ``f`` that is parallel to ``f_p``."""
        proj = (f * f_p).sum(dim=1, keepdim=True) / (f_p.pow(2).sum(dim=1, keepdim=True) + self.eps)
        return f - proj * f_p

    def forward(self, f_a: torch.Tensor, f_b: torch.Tensor,
                f_a_cp: torch.Tensor, f_b_cp: torch.Tensor) -> torch.Tensor:
        # Primary feature from the two clean views.
        f_p = self.embed(torch.cat([f_a, f_b], dim=1))
        # Redundancy filtering of the copy-paste views.
        f_a_orth = self.orthogonal_projection(f_a_cp, f_p)
        f_b_orth = self.orthogonal_projection(f_b_cp, f_p)
        fused = self.fusion(torch.cat([f_p, f_a_orth, f_b_orth], dim=1))
        return self.classifier(fused)


class F2MixNet(nn.Module):
    """Plug-and-play F2Mix wrapper around an arbitrary 3D backbone.

    Args:
        backbone: module with ``forward(x) -> (B, C, d, h, w)`` and a
            ``pool_size`` attribute (output size of the adaptive average pool).
        num_classes: number of output classes.
        input_size: spatial size of the input volume (D, H, W).
    """

    def __init__(self, backbone: nn.Module, num_classes: int = 2,
                 input_size: Tuple[int, int, int] = (48, 256, 256)):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool3d(backbone.pool_size)

        # Infer the feature geometry with a dry forward pass so that the
        # expander and head dims adapt to any backbone automatically.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_size)
            feat_map = backbone(dummy)
        _, channels, *feat_size = feat_map.shape

        self.expander = FeatureExpander(feat_size, input_size, channels)
        self.refinement = FeatureRefinement(channels * prod(backbone.pool_size), num_classes)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (pooled flat feature, pre-pool feature map)."""
        feat_map = self.backbone(x)
        pooled = self.pool(feat_map).flatten(1)
        return pooled, feat_map

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor,
                x_a_cp: torch.Tensor, x_b_cp: torch.Tensor):
        """Forward pass over the four views (A, B, A', B').

        Returns:
            (expanded F_A, expanded F_B, expanded F_{A'}, expanded F_{B'}, logits)
        """
        f_a, map_a = self.encode(x_a)
        f_b, map_b = self.encode(x_b)
        f_a_cp, map_a_cp = self.encode(x_a_cp)
        f_b_cp, map_b_cp = self.encode(x_b_cp)

        logits = self.refinement(f_a, f_b, f_a_cp, f_b_cp)

        expanded = [self.expander(m) for m in (map_a, map_b, map_a_cp, map_b_cp)]
        return expanded[0], expanded[1], expanded[2], expanded[3], logits
