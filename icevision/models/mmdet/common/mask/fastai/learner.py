__all__ = ["learner"]

from icevision.imports import *
from icevision.models.mmdet.fastai.learner import mmdetection_learner
from icevision.models.mmdet.common.mask.fastai.callbacks import MaskMMDetectionCallback


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    **learner_kwargs,
):
    pass
