__all__ = ["learner"]

from icevision.imports import *
from icevision.engines.fastai import *
from icevision.models.ultralytics.yolov5.fastai.callbacks import Yolov5Callback
from yolov5.utils.loss import ComputeLoss


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    **learner_kwargs,
):
    pass
