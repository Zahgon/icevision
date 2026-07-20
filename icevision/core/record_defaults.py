__all__ = [
    "ObjectDetectionRecord",
    "InstanceSegmentationRecord",
    "SemanticSegmentationRecord",
    "KeypointsRecord",
    "GrayScaleObjectDetectionRecord",
    "GrayScaleInstanceSegmentationRecord",
    "GrayScaleKeypointsRecord",
]

from icevision.core.record import *
from icevision.core.record_components import *
from icevision.core import tasks


def ObjectDetectionRecord():
    return BaseRecord(
        (
            FilepathRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
        )
    )


def InstanceSegmentationRecord():
    pass


def SemanticSegmentationRecord(gray=False):
    pass


def KeypointsRecord():
    pass


def GrayScaleObjectDetectionRecord():
    pass


def GrayScaleInstanceSegmentationRecord():
    pass


def GrayScaleKeypointsRecord():
    pass
