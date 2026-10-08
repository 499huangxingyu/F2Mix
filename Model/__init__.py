"""Model registry for F2Mix.

Any backbone that exposes ``forward(x) -> (B, C, d, h, w)`` and a
``pool_size`` attribute can be plugged into the shared :class:`F2MixNet`
wrapper.
"""

from Model.densenet import DenseNetBackbone
from Model.f2mix import F2MixNet, FeatureExpander, FeatureRefinement
from Model.pvt import PVTBackbone, pvt_tiny
from Model.pvt_v2 import PVTv2Backbone, pvt_v2_b1
from Model.res2net import Res2NetBackbone
from Model.resnet import ResNetBackbone, resnet50
from Model.swin import SwinTransformer3DBackbone
from Model.vgg import VGGBackbone, vgg16

# Registry of supported backbones (name -> zero-arg factory).
BACKBONES = {
    'densenet': DenseNetBackbone,
    'resnet': resnet50,
    'res2net': Res2NetBackbone,
    'vgg': vgg16,
    'pvt': pvt_tiny,
    'pvt_v2': pvt_v2_b1,
    'swin': SwinTransformer3DBackbone,
}


def build_backbone(name: str):
    """Build a backbone by name."""
    if name not in BACKBONES:
        raise ValueError(f'Unknown backbone "{name}". Available: {sorted(BACKBONES)}')
    return BACKBONES[name]()


def build_model(backbone: str, num_classes: int = 2,
                input_size=(48, 256, 256)) -> F2MixNet:
    """Build the full F2Mix model for a given backbone."""
    return F2MixNet(build_backbone(backbone), num_classes=num_classes, input_size=input_size)


__all__ = [
    'BACKBONES', 'build_backbone', 'build_model', 'F2MixNet',
    'FeatureExpander', 'FeatureRefinement',
    'DenseNetBackbone', 'ResNetBackbone', 'Res2NetBackbone', 'VGGBackbone',
    'PVTBackbone', 'PVTv2Backbone', 'SwinTransformer3DBackbone',
]
