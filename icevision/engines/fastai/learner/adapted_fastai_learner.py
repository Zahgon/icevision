__all__ = ["adapted_fastai_learner"]

from typing import List, Union
from torch.utils.data import DataLoader
from torch import nn
from fastai.data import core as fastai_core
from fastai.learner import Learner


from icevision.engines.fastai import convert_dataloaders_to_fastai, FastaiMetricAdapter
from icevision.metrics import Metric


# TODO: param_groups fix for efficientdet
def adapted_fastai_learner(
    dls: List[Union[DataLoader, fastai_core.DataLoader]],
    model: nn.Module,
    metrics=None,
    device=None,
    splitter=None,
    **learner_kwargs,
) -> Learner:
    # convert dataloaders to fastai
    fastai_dls = convert_dataloaders_to_fastai(dls=dls, device=device)

    # convert metrics to fastai
    metrics = metrics or []
    fastai_metrics = [
        FastaiMetricAdapter(metric) if isinstance(metric, Metric) else metric
        for metric in metrics
    ]

    if splitter == None:
        if hasattr(model, "param_groups"):

            def splitter(model):
                return model.param_groups()

        else:
            raise ValueError(
                "If the parameter `splitter` is not specified, "
                "the model should define a method called `param_groups`"
            )

    learn = Learner(
        dls=fastai_dls,
        model=model,
        metrics=fastai_metrics,
        splitter=splitter,
        **learner_kwargs,
    )
    learn.freeze()
    return learn
