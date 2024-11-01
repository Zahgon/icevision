__all__ = ["base_show_results"]

import random
from typing import Optional

from torch import nn

from icevision.data.dataset import Dataset
from icevision.utils.utils import denormalize_imagenet
from icevision.visualize.show_data import show_preds


def base_show_results(
    predict_fn: callable,
    model: nn.Module,
    dataset: Dataset,
    num_samples: int = 6,
    ncols: int = 3,
    denormalize_fn: Optional[callable] = denormalize_imagenet,
    show: bool = True,
    **predict_kwargs,
) -> None:
    records = random.choices(dataset, k=num_samples)
    preds = predict_fn(model, records, **predict_kwargs)

    show_preds(
        preds,
        denormalize_fn=denormalize_fn,
        ncols=ncols,
        show=show,
    )
