__all__ = ["ModelAdapter"]

from abc import ABC
from typing import List

from torch import nn

from icevision.engines.lightning.lightning_model_adapter import LightningModelAdapter
from icevision.metrics import Metric
from icevision.models.ultralytics import yolov5


class ModelAdapter(LightningModelAdapter, ABC):
    """Lightning module specialized for EfficientDet, with metrics support.

    The methods `forward`, `training_step`, `validation_step`, `on_validation_epoch_end`
    are already overriden.

    # Arguments
        model: The pytorch model to use.
        metrics: `Sequence` of metrics to use.

    # Returns
        A `LightningModule`.
    """

    def __init__(self, model: nn.Module, metrics: List[Metric] = None):
        super().__init__(metrics=metrics)
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        (xb, yb), _ = batch
        preds = self(xb)

        loss = self.compute_loss(preds, yb)

        self.log("train_loss", loss)

        return loss

    def compute_loss(self, preds, yb):
        return yolov5.loss_fn(preds, yb, self.model)

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, loss_log_key="val")

    def _shared_eval(self, batch, loss_log_key):
        (xb, yb), records = batch

        inference_out, training_out = self(xb)
        preds = self.convert_raw_predictions(
            batch=xb,
            raw_preds=inference_out,
            records=records,
            detection_threshold=0.001,
            nms_iou_threshold=0.6,
        )
        loss = self.compute_loss(training_out, yb)

        self.accumulate_metrics(preds)

        self.log(f"{loss_log_key}_loss", loss)

    def convert_raw_predictions(
        self, batch, raw_preds, records, detection_threshold, nms_iou_threshold
    ):
        return yolov5.convert_raw_predictions(
            batch=batch,
            raw_preds=raw_preds,
            records=records,
            detection_threshold=detection_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )

    def on_validation_epoch_end(self):
        self.finalize_metrics()

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch=batch, loss_log_key="test")

    def on_test_epoch_end(self):
        self.finalize_metrics()
