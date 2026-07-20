__all__ = ["ModelAdapter"]

from icevision.imports import *
from icevision.metrics import *
from icevision.engines.lightning.lightning_model_adapter import LightningModelAdapter
from icevision.models.ross import efficientdet


class ModelAdapter(LightningModelAdapter, ABC):

    def __init__(self, model: nn.Module, metrics: List[Metric] = None):
        super().__init__(metrics=metrics)
        self.model = model

    def forward(self, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outs):
        pass
