"""Dataset with the FGA (Fine-Grained Augmentation) pipeline.

Each HDF5 split file must contain ``<split>/path`` (NIfTI file paths) and
``<split>/label`` (integer labels), with ``<split>`` in {train, val, test}.

FGA pipeline (applied to generate the augmented view B from the original A):
  1. Resample the volume to ``input_size`` with trilinear interpolation.
  2. 3x3x3 median-filter denoising.
  3. Unsharp masking (Gaussian sigma=1.0, alpha=10).
  4. Min-max normalization to [0, 1].
  5. Contrast scaling by a factor of 4 (augmented view only).
Weak transformations (random flip / rotation) are applied independently to A
and B with probability 0.1 each, after FGA.
"""

import random

import h5py
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, median_filter
from torch.utils.data import Dataset


def load_nifti(path: str) -> torch.Tensor:
    """Load a NIfTI volume as a float32 tensor."""
    return torch.tensor(nib.load(path).get_fdata(), dtype=torch.float32)


class FGAugmenter:
    """Fine-Grained Augmentation operator (paper Sec. FGA module)."""

    def __init__(self, denoise: bool = True, sharpen: bool = True,
                 contrast_factor: float = 4.0, sharpen_sigma: float = 1.0,
                 sharpen_alpha: float = 10.0, filter_size: int = 3):
        self.denoise = denoise
        self.sharpen = sharpen
        self.contrast_factor = contrast_factor
        self.sharpen_sigma = sharpen_sigma
        self.sharpen_alpha = sharpen_alpha
        self.filter_size = filter_size

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """Apply FGA to a raw-intensity (Hounsfield unit) volume."""
        if self.denoise:
            volume = median_filter(volume, size=self.filter_size)
        if self.sharpen:
            blurred = gaussian_filter(volume, self.sharpen_sigma)
            volume = volume + self.sharpen_alpha * (volume - blurred)
        # Min-max normalization then contrast scaling (augmented view only).
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        return volume * self.contrast_factor


def weak_transform(volume: torch.Tensor, prob: float = 0.1) -> torch.Tensor:
    """Light auxiliary augmentation: random flip and rotation.

    Applied independently to each view with a low probability (paper: 10%).
    ``volume`` has shape (D, H, W); flips/rotations act in-plane (H, W).
    """
    if random.random() < prob:
        volume = torch.flip(volume, dims=[-1])          # horizontal flip
    if random.random() < prob:
        volume = torch.flip(volume, dims=[-2])          # vertical flip
    if random.random() < prob:
        volume = torch.rot90(volume, k=random.randint(1, 3), dims=(-2, -1))
    return volume


class HDF5VolumeDataset(Dataset):
    """Dataset of (original, FGA-augmented, label) CT volume triplets.

    Volumes are resampled to ``input_size`` (D, H, W) and returned as tensors
    of shape (1, H, W, D) with the depth axis last, matching the network's
    in-plane copy-paste convention.

    Args:
        hdf5_path: path to the HDF5 split file.
        split: group name, one of {train, val, test}.
        input_size: resampled volume size (D, H, W).
        use_fga: generate the FGA-augmented view (training only).
        weak_prob: probability of each weak transformation.
    """

    def __init__(self, hdf5_path: str, split: str = 'train',
                 input_size=(48, 256, 256), use_fga: bool = True,
                 weak_prob: float = 0.1):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be train/val/test, got '{split}'")
        self.hdf5_path = hdf5_path
        self.split = split
        self.input_size = input_size
        self.use_fga = use_fga
        self.weak_prob = weak_prob
        self.fga = FGAugmenter()

        with h5py.File(hdf5_path, 'r') as f:
            self.paths = [p.decode('utf-8') if isinstance(p, bytes) else str(p)
                          for p in f[f'{split}/path'][:]]
            self.labels = f[f'{split}/label'][:]

    def __len__(self):
        return len(self.paths)

    def _load_and_resample(self, path: str) -> torch.Tensor:
        volume = load_nifti(path)
        volume = torch.nn.functional.interpolate(
            volume[None, None], size=self.input_size, mode='trilinear',
            align_corners=False).squeeze(0).squeeze(0)
        return volume  # (D, H, W)

    def __getitem__(self, idx: int):
        volume = self._load_and_resample(self.paths[idx])
        label = int(self.labels[idx])

        if self.use_fga:
            # FGA modifies voxel intensities only; the spatial structure of
            # the original view A is preserved.
            augmented = torch.tensor(
                self.fga(volume.numpy()), dtype=torch.float32)
            augmented = weak_transform(augmented, self.weak_prob)
        else:
            augmented = volume

        original = weak_transform(volume, self.weak_prob)
        original = (original - original.min()) / (original.max() - original.min() + 1e-8)

        # Rearrange (D, H, W) -> (1, H, W, D): depth last for in-plane CP.
        original = original.permute(1, 2, 0).unsqueeze(0).contiguous()
        augmented = augmented.permute(1, 2, 0).unsqueeze(0).contiguous()

        return original, augmented, label
