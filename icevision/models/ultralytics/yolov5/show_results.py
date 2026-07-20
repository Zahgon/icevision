__all__ = ["show_results", "interp"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.base_show_results import base_show_results
from icevision.models.ultralytics.yolov5.dataloaders import (
    build_infer_batch,
    valid_dl,
    infer_dl,
)
from icevision.models.ultralytics.yolov5.prediction import (
    predict,
    predict_from_dl,
)
from icevision.models.interpretation import Interpretation

from icevision.models.interpretation import _move_to_device
from icevision.core.record_components import LossesRecordComponent
from yolov5.utils.loss import ComputeLoss


def show_results(
    model: nn.Module,
    dataset: Dataset,
    detection_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
    num_samples: int = 6,
    ncols: int = 3,
    denormalize_fn: Optional[callable] = denormalize_imagenet,
    show: bool = True,
    device: Optional[torch.device] = None,
) -> None:
    pass


def loop_yolo(dl, model, losses_stats, device):
    pass


_LOSSES_DICT = {
    "loss_yolo": [],
    "loss_total": [],
}

interp = Interpretation(
    losses_dict=_LOSSES_DICT,
    valid_dl=valid_dl,
    infer_dl=infer_dl,
    predict_from_dl=predict_from_dl,
)

interp._loop = loop_yolo
