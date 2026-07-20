__all__ = ["MMDetectionCallback"]

from icevision.imports import *
from icevision.engines.fastai import *
from icevision.models.mmdet.utils import *

from icevision.models.mmdet.common.bbox.prediction import convert_raw_predictions


class _ModelWrap(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, xb):
        pass

    def forward_test(self, xb):
        pass


class MMDetectionCallback(fastai.Callback):
    def after_create(self):
        pass

    def before_batch(self):
        pass

    @abstractmethod
    def convert_raw_predictions(self, batch, raw_preds, records):
        """Convert raw predictions from the model to library standard."""

    def after_loss(self):
        pass
