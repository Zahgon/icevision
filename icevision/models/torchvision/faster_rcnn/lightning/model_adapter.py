__all__ = ["ModelAdapter"]

from icevision.models.torchvision.lightning.model_adapter import *
from icevision.models.torchvision.faster_rcnn.prediction import *


class ModelAdapter(RCNNModelAdapter):

    def convert_raw_predictions(self, batch, raw_preds, records):
        return convert_raw_predictions(
            batch=batch, raw_preds=raw_preds, records=records, detection_threshold=0.0
        )
