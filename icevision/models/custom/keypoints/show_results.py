__all__ = ["show_results", "interp"]

from typing import Optional

import torch
from torch import nn

from icevision.data.dataset import Dataset
from icevision.models.base_show_results import base_show_results
from icevision.models.interpretation import Interpretation
from icevision.utils.utils import denormalize_imagenet

from icevision.models.custom.keypoints.dataloaders import (
    valid_dl,
    # infer_dl,
)
from icevision.models.custom.keypoints.prediction import (
    predict,
    # predict_from_dl,
)


def show_results(
    model: nn.Module,
    dataset: Dataset,
    detection_threshold: float = 0.5,
    num_samples: int = 6,
    ncols: int = 3,
    denormalize_fn: Optional[callable] = denormalize_imagenet,
    show: bool = True,
    device: Optional[torch.device] = None,
) -> None:
    return base_show_results(
        predict_fn=predict,
        model=model,
        dataset=dataset,
        num_samples=num_samples,
        ncols=ncols,
        denormalize_fn=denormalize_fn,
        show=show,
        detection_threshold=detection_threshold,
        device=device,
    )


def _rename_losses_effdet(loss):
    loss["effdet_total_loss"] = loss["loss"]
    _ = loss.pop("loss", None)
    return loss


def _sum_losses_effdet(loss):
    _loss = loss.copy()
    _ = _loss.pop("effdet_total_loss", None)
    loss["loss_total"] = sum(_loss.values())
    return loss


_LOSSES_DICT = {
    "effdet_total_loss": [],
    "class_loss": [],
    "box_loss": [],
    "loss_total": [],
}
#
# interp = Interpretation(
#     losses_dict=_LOSSES_DICT,
#     valid_dl=valid_dl,
#     infer_dl=infer_dl,
#     predict_from_dl=predict_from_dl,
# )
#
# interp._rename_losses = _rename_losses_effdet
# interp._sum_losses = _sum_losses_effdet
