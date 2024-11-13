from abc import ABC
from typing import List

from torch import nn

from icevision.engines.lightning.lightning_model_adapter import LightningModelAdapter
from icevision.metrics import Metric
from icevision.metrics.keypoints.keypoint_metrics import KeypointMetrics
from icevision.models.custom import keypoints


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

    def __init__(self, model: nn.Module):
        super().__init__(metrics=[KeypointMetrics()])
        self.model = model
        self.loss_fn = keypoints.KeypointLoss()
        # self.loss_fn = keypoints.JointsMSELoss()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        (xb, yb), records = batch
        raw_preds = self(xb)
        loss = self.compute_loss(raw_preds, yb)

        self.log(f"train_loss", loss)

        return loss

    def compute_loss(self, preds, yb):
        return self.loss_fn(preds, yb)

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, loss_log_key="val")

    def _shared_eval(self, batch, loss_log_key):
        (xb, yb), records = batch

        raw_preds = self(xb)

        preds = self.convert_raw_predictions(xb, yb, raw_preds, records)
        self.accumulate_metrics(preds)

        loss = self.compute_loss(raw_preds, yb)

        self.log(f"{loss_log_key}_loss", loss, prog_bar=True)

    def convert_raw_predictions(self, xb, yb, raw_preds, records):
        # Note: raw_preds["detections"] key is available only during Pytorch Lightning validation/test step
        # Calling the method manually (instead of letting the Trainer call it) will raise an exception.
        return keypoints.convert_raw_predictions(
            batch=(xb, yb),
            raw_preds=raw_preds,
            records=records,
            detection_threshold=0.0,
        )

    def on_validation_epoch_end(self):
        self.finalize_metrics()

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch=batch, loss_log_key="test")

    def on_test_epoch_end(self):
        self.finalize_metrics()
