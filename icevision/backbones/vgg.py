__all__ = ["vgg_param_groups", "vgg11", "vgg13", "vgg16", "vgg19"]

from icevision.imports import *
from icevision.utils import *


def _vgg_features(model: nn.Module):
    pass


def vgg_param_groups(model: nn.Module) -> List[List[nn.Parameter]]:
    pass


def vgg11(pretrained: bool = True):
    pass


def vgg13(pretrained: bool = True):
    pass


def vgg16(pretrained: bool = True):
    pass


def vgg19(pretrained: bool = True):
    pass
