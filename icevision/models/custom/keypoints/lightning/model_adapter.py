from abc import ABC
from typing import List, Dict

from lightning.pytorch.trainer.states import TrainerFn
from torch import nn, compile, optim, _dynamo

from icevision.engines.lightning.lightning_model_adapter import LightningModelAdapter
from icevision.metrics import Metric
from icevision.metrics.keypoints.keypoint_metrics import KeypointMetrics
from icevision.models.custom import keypoints
from icevision.utils.schedulers import WarmupCosineScheduler


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

    def __init__(self, model: nn.Module, learning_rate: float = 1e-4, torch_compile: bool = True):
        super().__init__(metrics=[KeypointMetrics()])
        self.model = model
        # self.loss_fn = keypoints.KeypointLoss()
        # self.loss_fn = keypoints.JointsMSELoss()
        self.loss_fn = keypoints.KeypointHeatmapLoss()
        self.save_hyperparameters("learning_rate", "torch_compile")

    def configure_optimizers(self):
        warmup_epochs = max(int(0.05 * self.hparams.max_epochs), 2)
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = WarmupCosineScheduler(optimizer=optimizer, warmup_epochs=warmup_epochs, max_epochs=self.hparams.max_epochs)
        return ({"optimizer": optimizer, "lr_scheduler": scheduler},)

    def configure_model(self) -> None:
        already_compiled = isinstance(self.model, _dynamo.eval_frame.OptimizedModule)
        if self.hparams.torch_compile and not already_compiled:
            self.model = compile(self.model)

    def setup(self, stage: str):
        if stage == TrainerFn.FITTING:
            self.hparams["max_epochs"] = self.trainer.max_epochs

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
