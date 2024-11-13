__all__ = ["show_results", "interp"]

from typing import Optional

import torch
from torch import nn

from icevision.core.record_components import LossesRecordComponent
from icevision.data.dataset import Dataset
from icevision.models.base_show_results import base_show_results
from icevision.models.interpretation import Interpretation, _move_to_device
from icevision.utils.torch_utils import tensor_to_image
from icevision.utils.utils import denormalize_imagenet, pbar

from icevision.models.custom.keypoints.dataloaders import (
    valid_dl,
    # infer_dl,
)
from icevision.models.custom.keypoints.prediction import (
    predict,
    predict_from_dl,
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


def _rename_losses_custom(loss):
    raise NotImplementedError
    loss["effdet_total_loss"] = loss["loss"]
    _ = loss.pop("loss", None)
    return loss


def _sum_losses_custom(loss):
    raise NotImplementedError
    _loss = loss.copy()
    _ = _loss.pop("effdet_total_loss", None)
    loss["loss_total"] = sum(_loss.values())
    return loss


_LOSSES_DICT = {
    "loss_total": [],
}

interp = Interpretation(
    losses_dict=_LOSSES_DICT,
    valid_dl=valid_dl,
    infer_dl=valid_dl,
    predict_from_dl=predict_from_dl,
)


def loop_custom(dl, model, losses_stats, device):
    samples_plus_losses = []

    with torch.no_grad():
        for (x, y), sample in pbar(dl):
            torch.manual_seed(0)
            x, y = _move_to_device(x, y, device)
            loss = model.training_step(((x, y), None), 0)
            loss = loss.detach().cpu().numpy().item()
            losses_stats["loss_total"].append(loss)

            loss_comp = LossesRecordComponent()
            loss_comp.set_losses({"loss_total": loss})
            sample[0].add_component(loss_comp)
            sample[0].set_img(tensor_to_image(x[0]))
            samples_plus_losses.append(sample[0])
    return samples_plus_losses, losses_stats


interp._loop = loop_custom