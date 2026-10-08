"""Grad-CAM style heatmap visualization for a trained F2Mix model.

The expanded feature map of the original view (F_A) is upsampled to the input
resolution and overlaid on selected slices.

Usage:
    python Plot/Hotmap_test.py --data_path your/path/to/dataset.hdf5 \
        --test_model results/your_dataset/densenet/models/best_model.pth
"""

import argparse
import os
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model import build_model                       # noqa: E402
from Utils.common import load_weights               # noqa: E402
from Utils.datasets import HDF5VolumeDataset        # noqa: E402


def get_sample(dataset: HDF5VolumeDataset):
    idx = random.randrange(len(dataset))
    original, _augmented, label = dataset[idx]
    return original.unsqueeze(0), label


def main():
    parser = argparse.ArgumentParser(description='F2Mix heatmap visualization')
    parser.add_argument('--data_path', type=str, default='your/path/to/dataset.hdf5')
    parser.add_argument('--backbone', type=str, default='densenet')
    parser.add_argument('--test_model', type=str,
                        default='results/your_dataset/densenet/models/best_model.pth')
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--save_path', type=str, default='heatmap.png')
    parser.add_argument('--slices', type=int, nargs='+', default=[6, 12, 24, 36],
                        help='depth indices of the slices to visualize')
    args = parser.parse_args()

    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')

    model = build_model(args.backbone).float().to(device)
    load_weights(args.test_model, model, device=str(device))
    model.eval()

    dataset = HDF5VolumeDataset(args.data_path, 'test', use_fga=False)
    volume, label = get_sample(dataset)
    volume = volume.to(device)
    print(f'volume shape {tuple(volume.shape)}, label {label}')

    with torch.no_grad():
        # At inference the same volume feeds all four views; the first
        # expanded feature map (F_A) serves as the class activation map.
        f_a, *_ = model(volume, volume, volume, volume)
        cam = F.interpolate(f_a, size=volume.shape[2:], mode='trilinear',
                            align_corners=False).squeeze(0).squeeze(0)

    n = len(args.slices)
    plt.figure(figsize=(8, 4 * n))
    for i, s in enumerate(args.slices):
        plt.subplot(n, 2, 2 * i + 1)
        plt.imshow(volume[0, 0, :, :, s].cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.subplot(n, 2, 2 * i + 2)
        plt.imshow(cam[:, :, s].cpu().numpy())
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(args.save_path, bbox_inches='tight')
    print(f'Heatmap saved to {args.save_path}')


if __name__ == '__main__':
    main()
