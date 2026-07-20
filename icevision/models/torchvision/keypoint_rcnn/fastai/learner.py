__all__ = ["learner"]

from icevision.imports import *
from icevision.engines.fastai import *
from icevision.models.torchvision.fastai.learner import rcnn_learner
from icevision.models.torchvision.keypoint_rcnn.fastai.callbacks import *


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    **learner_kwargs
):
    pass
