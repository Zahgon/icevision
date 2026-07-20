__all__ = [
    "create_coco_api",
    "convert_records_to_coco_style",
    "convert_preds_to_coco_style",
    "convert_record_to_coco_annotations",
    "coco_api_from_records",
    "coco_api_from_preds",
    "create_coco_eval",
]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def create_coco_api(coco_records) -> COCO:
    pass


def coco_api_from_preds(preds, show_pbar: bool = False) -> COCO:
    pass


def coco_api_from_records(records, show_pbar: bool = False) -> COCO:
    pass


def create_coco_eval(
    records,
    preds,
    metric_type: str,
    iou_thresholds: Optional[Sequence[float]] = None,
    show_pbar: bool = False,
) -> COCOeval:
    pass


def convert_record_to_coco_image(record) -> dict:
    pass


def convert_record_to_coco_annotations(record):
    pass


def convert_preds_to_coco_style(preds, show_pbar: bool = False):
    pass


def convert_records_to_coco_style(
    records,
    images: bool = True,
    annotations: bool = True,
    categories: bool = True,
    show_pbar: bool = True,
):
    pass
