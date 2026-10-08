"""Copy-paste operators and consistency losses for F2Mix.

- ``image_copy_paste``      : the ICP operation (image level).
- ``feature_copy_paste``    : the FCP operation (feature level), reusing the
  ICP patch position so that image-level and feature-level copy-paste are
  strictly aligned.
- ``MutualInformation``    : Parzen-window MI loss used as the feature
  copy-paste consistency loss (paper Sec. Loss functions).
- ``NCC``                  : local normalized cross-correlation (ablation).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def sample_patch_position(size_a, size_b, patch_size, generator=None):
    """Randomly sample the top-left corner of an in-plane patch.

    The copy-paste region spans the full depth axis, preserving the spatial
    structural integrity of 3D volumes.

    Args:
        size_a: spatial size of A (D, H, W).
        size_b: spatial size of B (D, H, W).
        patch_size: in-plane patch size (ph, pw).
    Returns:
        (start_h, start_w, ph, pw)
    """
    ph, pw = patch_size
    _, H, W = size_a
    start_h = np.random.randint(0, max(H - ph, 1))
    start_w = np.random.randint(0, max(W - pw, 1))
    return int(start_h), int(start_w), int(ph), int(pw)


def _exchange(a: torch.Tensor, b: torch.Tensor, position) -> tuple:
    """Exchange the patch at ``position`` between tensors a and b.

    ``position`` indexes the in-plane dims (dims 2 and 3); the depth dim is
    copied in full. Works for both images (B, 1, H, W, D) and full-resolution
    feature maps.
    """
    sh, sw, ph, pw = position
    out_a, out_b = a.clone(), b.clone()
    patch_a = a[:, :, sh:sh + ph, sw:sw + pw, :].clone()
    patch_b = b[:, :, sh:sh + ph, sw:sw + pw, :].clone()
    out_a[:, :, sh:sh + ph, sw:sw + pw, :] = patch_b
    out_b[:, :, sh:sh + ph, sw:sw + pw, :] = patch_a
    return out_a, out_b


def image_copy_paste(a: torch.Tensor, b: torch.Tensor, patch_size):
    """ICP: exchange an in-plane patch between the original (A) and the
    FGA-augmented (B) volumes.

    Returns (A', B', patch_position).
    """
    position = sample_patch_position(a.shape[-3:], b.shape[-3:], patch_size)
    cp_a, cp_b = _exchange(a, b, position)
    return cp_a, cp_b, position


def feature_copy_paste(f_a_cp: torch.Tensor, f_b_cp: torch.Tensor, position):
    """FCP: repeat the copy-paste at feature level with the same spatial
    position used by the ICP operation.

    ``f_a_cp``/``f_b_cp`` are the full-resolution expanded feature maps of
    A' and B'. Returns (F'_{A'}, F'_{B'}).
    """
    return _exchange(f_a_cp, f_b_cp, position)


class MutualInformation(nn.Module):
    """Differentiable mutual information via Parzen windowing.

    Feature values are clamped to [0, 1] and softly assigned to ``num_bins``
    evenly spaced bins with a Gaussian kernel (bandwidth 1/31 for 32 bins).
    MI is computed per sample over the flattened spatial and channel
    dimensions and averaged across the batch. ``forward`` returns the
    negative MI (a minimizable loss).
    """

    def __init__(self, sigma_ratio: float = 1.0, minval: float = 0.0,
                 maxval: float = 1.0, num_bins: int = 32):
        super().__init__()
        bin_centers = np.linspace(minval, maxval, num=num_bins)
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio  # 1/31 by default
        self.preterm = 1 / (2 * sigma ** 2)
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = Variable(
            torch.linspace(minval, maxval, num_bins), requires_grad=False)

    def mi(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_true = torch.clamp(y_true, 0., self.max_clip)
        y_pred = torch.clamp(y_pred, 0., self.max_clip)

        y_true = y_true.reshape(y_true.shape[0], -1).unsqueeze(2)
        y_pred = y_pred.reshape(y_pred.shape[0], -1).unsqueeze(2)
        nb_voxels = y_pred.shape[1]

        # Reshape bin centers: (1, 1, num_bins)
        vbc = self.vol_bin_centers.reshape(1, 1, -1).to(y_true.device)

        # Soft bin assignment via a Gaussian kernel.
        i_a = torch.exp(-self.preterm * (y_true - vbc) ** 2)
        i_a = i_a / torch.sum(i_a, dim=-1, keepdim=True)
        i_b = torch.exp(-self.preterm * (y_pred - vbc) ** 2)
        i_b = i_b / torch.sum(i_b, dim=-1, keepdim=True)

        # Joint and marginal distributions.
        pab = torch.bmm(i_a.permute(0, 2, 1), i_b) / nb_voxels
        pa = torch.mean(i_a, dim=1, keepdim=True)
        pb = torch.mean(i_b, dim=1, keepdim=True)
        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6

        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across the batch

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return -self.mi(y_true, y_pred)


class NCC(nn.Module):
    """Local (over window) normalized cross-correlation loss (ablation)."""

    def __init__(self, win: int = 5, eps: float = 1e-8):
        super().__init__()
        self.win = win
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            x, y = x.unsqueeze(1), y.unsqueeze(1)
        weight = torch.ones((1, 1, self.win, self.win, self.win),
                            device=x.device, requires_grad=False)
        pad = self.win // 2
        conv = lambda t: F.conv3d(t, weight, padding=pad)

        x_sum, y_sum = conv(x), conv(y)
        x2_sum, y2_sum, xy_sum = conv(x * x), conv(y * y), conv(x * y)
        win_size = self.win ** 3
        u_x, u_y = x_sum / win_size, y_sum / win_size

        cross = xy_sum - u_y * x_sum - u_x * y_sum + u_x * u_y * win_size
        x_var = x2_sum - 2 * u_x * x_sum + u_x * u_x * win_size
        y_var = y2_sum - 2 * u_y * y_sum + u_y * u_y * win_size
        cc = cross * cross / (x_var * y_var + self.eps)
        return -1.0 * torch.mean(cc)


def build_consistency_loss(name: str) -> nn.Module:
    """Factory for the feature copy-paste consistency loss."""
    losses = {'mi': MutualInformation, 'mse': nn.MSELoss, 'ncc': NCC}
    if name not in losses:
        raise ValueError(f'Unknown consistency loss "{name}". Available: {sorted(losses)}')
    return losses[name]()
