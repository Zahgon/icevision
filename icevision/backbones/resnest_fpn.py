__all__ = [
    "resnest50_fpn",
    "resnest101_fpn",
    "resnest200_fpn",
    "resnest269_fpn",
]

from icevision.imports import *
from icevision.utils import *
from icevision.backbones.resnet_fpn_utils import patch_param_groups
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection.backbone_utils import (
    resnet_fpn_backbone,
    BackboneWithFPN,
)

from icevision.soft_dependencies import SoftDependencies

if SoftDependencies.resnest:
    import resnest.torch


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


def resnest_fpn_backbone(
    backbone_fn,
    pretrained,
    trainable_layers=3,
    returned_layers=None,
    extra_blocks=None,
):
    pass


def _resnest_fpn(name, pretrained: bool = True, **kwargs):
    pass


def resnest50_fpn(pretrained: bool = True, **kwargs):
    pass


def resnest101_fpn(pretrained: bool = True, **kwargs):
    pass


def resnest200_fpn(pretrained: bool = True, **kwargs):
    pass


def resnest269_fpn(pretrained: bool = True, **kwargs):
    pass
