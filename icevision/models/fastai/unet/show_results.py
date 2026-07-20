__all__ = ["show_results", "interp"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.fastai.unet.prediction import *
from icevision.models.base_show_results import *
from icevision.models.fastai.unet.dataloaders import (
    valid_dl,
    infer_dl,
)
from icevision.models.interpretation import Interpretation

from icevision.models.interpretation import _move_to_device
from icevision.core.record_components import LossesRecordComponent


def show_results(
    model: nn.Module,
    dataset: Dataset,
    num_samples: int = 6,
    ncols: int = 3,
    denormalize_fn: Optional[callable] = denormalize_imagenet,
    show: bool = True,
    device: Optional[torch.device] = None,
) -> None:
    pass


def loop_unet(dl, model, losses_stats, device):
    pass


_LOSSES_DICT = {
    "loss_unet": [],
    "loss_total": [],
}

interp = Interpretation(
    losses_dict=_LOSSES_DICT,
    valid_dl=valid_dl,
    infer_dl=infer_dl,
    predict_from_dl=predict_from_dl,
)

interp._loop = loop_unet
