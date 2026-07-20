__all__ = ["FastaiMetricAdapter"]

from icevision.imports import *
from icevision.metrics import Metric
from icevision.engines.fastai.imports import *


class FastaiMetricAdapter(fastai.Metric):
    def __init__(self, metric: Metric):
        self.metric = metric

    def reset(self):
        pass

    def accumulate(self, learn: fastai.Learner):
        pass

    @property
    def value(self) -> Dict[str, float]:
        pass

    @property
    def name(self) -> str:
        pass
