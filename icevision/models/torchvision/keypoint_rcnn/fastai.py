__all__ = ["learner", "KeypointRCNNCallback"]

from typing import Union, List

from fastcore.foundation import L
from torch import nn
from torch.utils.data import DataLoader

from icevision.engines import fastai
from icevision.models.torchvision.fastai_learner import rcnn_learner
from icevision.models.torchvision.fastai_callbacks import RCNNCallback
from icevision.models.torchvision.keypoint_rcnn import convert_raw_predictions


class KeypointRCNNCallback(RCNNCallback):
    def convert_raw_predictions(self, batch, raw_preds):
        return convert_raw_predictions(
            batch=batch,
            raw_preds=raw_preds,
            records=self.learn.records,
            detection_threshold=0.0,
        )


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    **learner_kwargs
):
    """Fastai `Learner` adapted for RCNN.

    # Arguments
        dls: `Sequence` of `DataLoaders` passed to the `Learner`.
        The first one will be used for training and the second for validation.
        model: The model to train.
        cbs: Optional `Sequence` of callbacks.
        **learner_kwargs: Keyword arguments that will be internally passed to `Learner`.

    # Returns
        A fastai `Learner`.
    """
    cbs = [KeypointRCNNCallback()] + L(cbs)
    return rcnn_learner(dls=dls, model=model, cbs=cbs, **learner_kwargs)
