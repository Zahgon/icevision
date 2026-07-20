__all__ = ["learner"]


from icevision.imports import *
from icevision.engines.fastai import *
from icevision.models.fastai.unet.fastai.callbacks import *


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    loss_func=fastai.CrossEntropyLossFlat(axis=1),
    **kwargs,
):
    pass
