from icevision.imports import *
from icevision import BBox, BaseRecord


def get_best_score_item(prediction_items: Collection[Dict]):
    pass


def pairwise_iou_record_record(target: BaseRecord, prediction: BaseRecord):
    pass


def match_records(
    target: BaseRecord, prediction: BaseRecord, iou_threshold: float = 0.5
) -> Collection:
    pass
