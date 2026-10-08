"""Modified 3D DenseNet backbone (DN.) for F2Mix.

A compact densely connected 3D CNN. ``forward`` returns the final feature map
(B, 96, d, h, w) which is consumed by the shared F2Mix wrapper.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features: int, growth_rate: int, drop_rate: float):
        super().__init__(
            nn.BatchNorm3d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_input_features, growth_rate, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv3d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_input_features: int, growth_rates, drop_rate: float):
        super().__init__()
        in_features = num_input_features
        for i, growth in enumerate(growth_rates):
            layer = _DenseLayer(in_features, growth, drop_rate)
            in_features += growth
            self.add_module(f'denselayer{i + 1}', layer)
        self.out_features = in_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__(
            nn.BatchNorm3d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False),
        )


class DenseNetBackbone(nn.Module):
    """Modified DenseNet encoder used in the paper (DN.).

    Input (B, 1, D, H, W) -> feature map (B, 96, d, h, w).
    ``pool_size = (2, 2, 2)`` gives a 768-d per-stream pooled feature.
    """

    def __init__(self, in_channels: int = 1, base_filters: int = 16, drop_rate: float = 0.0):
        super().__init__()
        self.pool_size = (2, 2, 2)

        blocks = OrderedDict()
        blocks['conv0'] = nn.Conv3d(in_channels, base_filters, kernel_size=3,
                                    stride=(1, 2, 2), padding=(1, 1, 1), bias=False)

        block_specs = [
            ([24, 24], 24),   # (growth rates, transition output channels)
            ([32, 32], 32),
            ([64, 64], 64),
            ([96, 96], 96),
        ]
        in_features = base_filters
        for i, (growth_rates, trans_out) in enumerate(block_specs):
            block = _DenseBlock(in_features, growth_rates, drop_rate)
            blocks[f'denseblock{i + 1}'] = block
            in_features = block.out_features
            blocks[f'transition{i + 1}'] = _Transition(in_features, trans_out)
            in_features = trans_out
            if i < len(block_specs) - 1:
                blocks[f'maxpool{i + 1}'] = nn.MaxPool3d(kernel_size=2, stride=2)

        blocks['bn5'] = nn.BatchNorm3d(in_features)
        blocks['relu5'] = nn.ReLU(inplace=True)
        self.features = nn.Sequential(blocks)
        self.feat_channels = in_features

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
