from icevision.imports import *
from icevision.engines.fastai import *
from icevision.models.torchvision.fastai.learner import rcnn_learner
from icevision.models.torchvision.faster_rcnn.fastai.callbacks import *


def learner(
    dls: Sequence[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs: Optional[Sequence[fastai.Callback]] = None,
    **learner_kwargs
) -> fastai.Learner:
    pass
