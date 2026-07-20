__all__ = ["RCNNModelAdapter"]

from icevision.imports import *
from icevision.utils import *
from icevision.metrics import *
from icevision.engines.lightning.lightning_model_adapter import LightningModelAdapter
from icevision.models.torchvision.loss_fn import loss_fn
from icevision.core.record_components import (
    InstanceMasksRecordComponent,
    BBoxesRecordComponent,
)
from icevision.core.mask import MaskArray
from icevision.core.bbox import BBox


class RCNNModelAdapter(LightningModelAdapter, ABC):
    def __init__(self, model: nn.Module, metrics: Sequence[Metric] = None):
        super().__init__(metrics=metrics)
        self.model = model

    @abstractmethod
    def convert_raw_predictions(self, batch, raw_preds, records):
        """Convert raw predictions from the model to library standard."""

    def forward(self, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outs):
        pass
