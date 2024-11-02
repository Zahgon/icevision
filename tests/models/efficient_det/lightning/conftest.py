import pytest
from torch.optim import SGD

from icevision import models


@pytest.fixture
def light_model_cls():
    class LightModel(models.ross.efficientdet.lightning.ModelAdapter):
        def __init__(self, model, metrics):
            super(LightModel, self).__init__(model, metrics)
            self.was_finalize_metrics_called = False
            self.logs = {}

        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-3)

        def finalize_metrics(self):
            self.was_finalize_metrics_called = True

        def log(self, key, value, **args):
            super(LightModel, self).log(key, value, **args)
            self.logs[key] = value

    return LightModel
