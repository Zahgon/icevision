__all__ = ["LightningModelAdapter"]

import pytorch_lightning as pl
from icevision.imports import *
from icevision.metrics import *


class LightningModelAdapter(pl.LightningModule, ABC):
    def __init__(
        self,
        metrics: List[Metric] = None,
        metrics_keys_to_log_to_prog_bar: List[tuple] = None,
    ):
        """
        To show a metric in the progressbar a list of tupels can be provided for metrics_keys_to_log_to_prog_bar, the first
        entry has to be the name of the metric to log and the second entry the display name in the progressbar. By default the
        mAP is logged to the progressbar.
        """
        super().__init__()
        self.metrics = metrics or []
        self.metrics_keys_to_log_to_prog_bar = metrics_keys_to_log_to_prog_bar or [
            ("AP (IoU=0.50:0.95) area=all", "COCOMetric")
        ]

    def accumulate_metrics(self, preds):
        pass

    def finalize_metrics(self) -> None:
        pass
