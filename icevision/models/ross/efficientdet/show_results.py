__all__ = ["show_results", "interp"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.base_show_results import base_show_results
from icevision.models.ross.efficientdet.dataloaders import (
    build_infer_batch,
    valid_dl,
    infer_dl,
)
from icevision.models.ross.efficientdet.prediction import (
    predict,
    predict_from_dl,
)
from icevision.models.interpretation import Interpretation


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
    pass


def _rename_losses_effdet(loss):
    pass


def _sum_losses_effdet(loss):
    pass


_LOSSES_DICT = {
    "effdet_total_loss": [],
    "class_loss": [],
    "box_loss": [],
    "loss_total": [],
}

interp = Interpretation(
    losses_dict=_LOSSES_DICT,
    valid_dl=valid_dl,
    infer_dl=infer_dl,
    predict_from_dl=predict_from_dl,
)

interp._rename_losses = _rename_losses_effdet
interp._sum_losses = _sum_losses_effdet
