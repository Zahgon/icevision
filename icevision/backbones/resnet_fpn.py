__all__ = [
    "resnet18_fpn",
    "resnet34_fpn",
    "resnet50_fpn",
    "resnet101_fpn",
    "resnet152_fpn",
    "resnext50_32x4d_fpn",
    "resnext101_32x8d_fpn",
    "wide_resnet50_2_fpn",
    "wide_resnet101_2_fpn",
]

from icevision.imports import *
from icevision.utils import *
from icevision.backbones.resnet_fpn_utils import *
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def _resnet_fpn(name: str, pretrained: bool = True, **kwargs):
    pass


def resnet18_fpn(pretrained: bool = True, **kwargs):
    pass


def resnet34_fpn(pretrained: bool = True, **kwargs):
    pass


def resnet50_fpn(pretrained: bool = True, **kwargs):
    pass


def resnet101_fpn(pretrained: bool = True, **kwargs):
    pass


def resnet152_fpn(pretrained: bool = True, **kwargs):
    pass


def resnext50_32x4d_fpn(pretrained: bool = True, **kwargs):
    pass


def resnext101_32x8d_fpn(pretrained: bool = True, **kwargs):
    pass


def wide_resnet50_2_fpn(pretrained: bool = False, **kwargs):
    pass


def wide_resnet101_2_fpn(pretrained: bool = False, **kwargs):
    pass
