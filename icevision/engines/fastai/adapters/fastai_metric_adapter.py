__all__ = ["FastaiMetricAdapter"]

from typing import Dict

from fastai import metrics as fastai_metrics
from icevision.metrics import Metric


class FastaiMetricAdapter(fastai_metrics.Metric):
    def __init__(self, metric: Metric):
        self.metric = metric

    def reset(self):
        pass

    def accumulate(self, learn: fastai_metrics.Learner):
        self.metric.accumulate(preds=learn.converted_preds)

    @property
    def value(self) -> Dict[str, float]:
        # return self.metric.finalize()
        # HACK: Return single item from dict
        logs = self.metric.finalize()
        return next(iter(logs.values()))

    @property
    def name(self) -> str:
        return self.metric.name
