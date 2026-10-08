"""3D VGG backbone for F2Mix.

VGG-16 style 3D CNN. ``forward`` returns the last convolutional feature map
(B, 512, d, h, w).
"""

from typing import Callable, Union

import torch
import torch.nn as nn

CFG_16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def make_layers(cfg, in_channels: int = 1,
                norm_layer: Callable[..., nn.Module] = nn.BatchNorm3d) -> nn.Sequential:
    layers = []
    channels = in_channels
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv3d(channels, v, kernel_size=3, padding=1))
            if norm_layer is not None:
                layers.append(norm_layer(v))
            layers.append(nn.ReLU(inplace=True))
            channels = v
    return nn.Sequential(*layers)


class VGGBackbone(nn.Module):
    """VGG-16 encoder without the classification head.

    Input (B, 1, D, H, W) -> feature map (B, 512, d, h, w).
    """

    def __init__(self, cfg=None, in_channels: int = 1, norm_layer: Union[Callable[..., nn.Module], None] = nn.BatchNorm3d):
        super().__init__()
        self.pool_size = (1, 1, 1)
        self.features = make_layers(cfg or CFG_16, in_channels, norm_layer)
        self.feat_channels = 512
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def vgg16(**kwargs) -> VGGBackbone:
    return VGGBackbone(CFG_16, **kwargs)
