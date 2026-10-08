"""3D Res2Net backbone (R2N.) for F2Mix.

Res2Net groups the 3x3x3 convolution of a bottleneck into scale sub-groups to
capture multi-scale features. Following the paper's configuration, three
stages are used. ``forward`` returns the layer3 feature map.
"""

from typing import Optional

import torch
import torch.nn as nn


class SEModule(nn.Module):
    """Squeeze-and-excitation block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module_input = x
        x = self.avg_pool(x)
        x = self.sigmoid(self.fc2(self.relu(self.fc1(x))))
        return module_input * x


class Res2NetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, downsample: Optional[nn.Module] = None,
                 stride: int = 1, scales: int = 4, groups: int = 1, se: bool = False,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (64 / 64.0))
        # Downsampling is done by the 1x1 expansion conv so that all scale
        # sub-groups share the same resolution and can be accumulated.
        self.conv1 = nn.Conv3d(inplanes, width * scales, kernel_size=1,
                               stride=stride, bias=False)
        self.bn1 = norm_layer(width * scales)

        if scales == 1:
            self.conv2 = nn.ModuleList([nn.Conv3d(width, width, kernel_size=3,
                                                  stride=1, padding=1, groups=groups, bias=False)])
        else:
            self.conv2 = nn.ModuleList([
                nn.Conv3d(width, width, kernel_size=3, stride=1,
                          padding=1, groups=groups, bias=False)
                for _ in range(scales - 1)
            ])
        self.bn2 = nn.ModuleList([norm_layer(width) for _ in range(scales - 1)])
        self.conv3 = nn.Conv3d(width * scales, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(planes * self.expansion) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        xs = torch.chunk(out, self.scales, 1)
        ys = [xs[0]]
        for s in range(1, self.scales):
            if s == 1:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)
        out = self.bn3(self.conv3(out))
        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Res2NetBackbone(nn.Module):
    """Res2Net-50 style encoder with three stages (as used in the paper).

    Input (B, 1, D, H, W) -> feature map (B, 512, d, h, w).
    """

    def __init__(self, layers=(3, 4, 6), in_channels: int = 1, width: int = 8,
                 scales: int = 4, groups: int = 1, se: bool = False,
                 zero_init_residual: bool = False, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self.pool_size = (1, 1, 1)
        planes = [int(width * scales * 2 ** i) for i in range(4)]

        self.inplanes = planes[0]
        self.conv1 = nn.Conv3d(in_channels, planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(planes[0], layers[0], norm_layer, scales, groups, se)
        self.layer2 = self._make_layer(planes[1], layers[1], norm_layer, scales, groups, se, stride=2)
        self.layer3 = self._make_layer(planes[2], layers[2], norm_layer, scales, groups, se, stride=2)
        self.feat_channels = planes[2] * Res2NetBottleneck.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Res2NetBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, planes, blocks, norm_layer, scales, groups, se, stride=1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Res2NetBottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * Res2NetBottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * Res2NetBottleneck.expansion),
            )
        layers = [Res2NetBottleneck(self.inplanes, planes, downsample, stride=stride,
                                    scales=scales, groups=groups, se=se, norm_layer=norm_layer)]
        self.inplanes = planes * Res2NetBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Res2NetBottleneck(self.inplanes, planes, scales=scales,
                                            groups=groups, se=se, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)
