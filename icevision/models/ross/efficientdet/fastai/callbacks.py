__all__ = ["EfficientDetCallback"]

from icevision.models.ross import efficientdet
from icevision.engines.fastai import *


class EfficientDetCallback(fastai.Callback):
    def before_batch(self):
        pass

    def after_pred(self):
        pass
