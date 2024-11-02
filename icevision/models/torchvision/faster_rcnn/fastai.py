__all__ = ["FasterRCNNCallback", "learner"]

from typing import Sequence, Union, Optional
from torch.utils.data import DataLoader
from torch import nn

from fastai.data import core as fastai_core
from icevision.engines import fastai
from fastcore.utils import L

from icevision.models.torchvision.fastai_learner import rcnn_learner
from icevision.models.torchvision.fastai_callbacks import RCNNCallback
from icevision.models.torchvision.faster_rcnn import convert_raw_predictions


class FasterRCNNCallback(RCNNCallback):
    def convert_raw_predictions(self, batch, raw_preds):
        return convert_raw_predictions(
            batch=batch,
            raw_preds=raw_preds,
            records=self.learn.records,
            detection_threshold=0.0,
        )


def learner(
    dls: Sequence[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs: Optional[Sequence[fastai.Callback]] = None,
    **learner_kwargs
) -> fastai.Learner:
    """Fastai `Learner` adapted for Faster RCNN.

    # Arguments
        dls: `Sequence` of `DataLoaders` passed to the `Learner`.
        The first one will be used for training and the second for validation.
        model: The model to train.
        cbs: Optional `Sequence` of callbacks.
        **learner_kwargs: Keyword arguments that will be internally passed to `Learner`.

    # Returns
        A fastai `Learner`.
    """
    cbs = [FasterRCNNCallback()] + L(cbs)
    return rcnn_learner(dls=dls, model=model, cbs=cbs, **learner_kwargs)
