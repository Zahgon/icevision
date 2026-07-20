__all__ = ["show_results"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.base_show_results import base_show_results
from icevision.models.mmdet.common.bbox.dataloaders import *
from icevision.models.mmdet.common.bbox.prediction import *


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
