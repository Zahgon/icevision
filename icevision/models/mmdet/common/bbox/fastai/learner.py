__all__ = ["learner"]

from icevision.imports import *
from icevision.models.mmdet.fastai.learner import mmdetection_learner
from icevision.models.mmdet.common.bbox.fastai.callbacks import BBoxMMDetectionCallback


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    **learner_kwargs,
):
    pass
