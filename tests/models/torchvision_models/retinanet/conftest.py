import pytest
from torch import nn

from icevision.models.torchvision import retinanet


@pytest.fixture
def fridge_retinanet_model() -> nn.Module:
    backbone = retinanet.backbones.resnet18_fpn(pretrained=False)
    return retinanet.model(num_classes=5, backbone=backbone)
