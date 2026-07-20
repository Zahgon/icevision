__all__ = ["RCNNCallback"]

from icevision.imports import *
from icevision.engines.fastai import *
from icevision.models.torchvision import faster_rcnn


class RCNNCallback(fastai.Callback, ABC):
    @abstractmethod
    def convert_raw_predictions(self, raw_preds):
        """Convert raw predictions from the model to library standard."""

    def before_batch(self):
        pass

    def after_pred(self):
        pass

    def before_validate(self):
        pass

    def after_loss(self):
        pass
