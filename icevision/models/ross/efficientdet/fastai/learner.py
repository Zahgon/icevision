__all__ = ["learner"]

from icevision.imports import *
from icevision.engines.fastai import *
from icevision.models.ross.efficientdet.loss_fn import loss_fn
from icevision.models.ross.efficientdet.fastai.callbacks import EfficientDetCallback


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    **learner_kwargs,
):
    pass
