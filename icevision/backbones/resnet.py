__all__ = [
    "resnet_param_groups",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext101_32x8d",
]

from icevision.imports import *
from icevision.utils import *


def _resnet_features(model: nn.Module, out_channels: int):
    pass


def resnet_param_groups(model: nn.Module) -> List[nn.Parameter]:
    pass


def resnet18(pretrained: bool = True):
    pass


def resnet34(pretrained: bool = True):
    pass


def resnet50(pretrained: bool = True):
    pass


def resnet101(pretrained: bool = True):
    pass


def resnet152(pretrained: bool = True):
    pass


def resnext101_32x8d(pretrained: bool = True):
    pass
