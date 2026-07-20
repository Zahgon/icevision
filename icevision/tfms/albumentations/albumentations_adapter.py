__all__ = [
    "Adapter",
    "AlbumentationsAdapterComponent",
    "AlbumentationsImgComponent",
    "AlbumentationsSizeComponent",
    "AlbumentationsInstancesLabelsComponent",
    "AlbumentationsBBoxesComponent",
    "AlbumentationsMasksComponent",
    "AlbumentationsKeypointsComponent",
    "AlbumentationsIsCrowdsComponent",
]

import albumentations as A
from itertools import chain

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.tfms.transform import *
from icevision.tfms.albumentations.albumentations_helpers import (
    get_size_without_padding,
    get_transform,
)


@dataclass
class CollectOp:
    fn: Callable
    order: float = 0.5


class AlbumentationsAdapterComponent(Component):
    @property
    def adapter(self):
        pass

    def setup(self):
        pass

    def prepare(self, record):
        pass

    def collect(self, record):
        pass


class AlbumentationsImgComponent(AlbumentationsAdapterComponent):
    def setup_img(self, record):
        pass

    def collect(self, record):
        pass


class AlbumentationsSizeComponent(AlbumentationsAdapterComponent):
    order = 0.2

    def setup_size(self, record):
        pass

    def collect(self, record) -> ImgSize:
        pass


class AlbumentationsInstancesLabelsComponent(AlbumentationsAdapterComponent):
    order = 0.1

    def set_labels(self, record, labels):
        pass

    def setup_instances_labels(self, record_component):
        pass

    def collect_labels(self, record):
        pass


class AlbumentationsBBoxesComponent(AlbumentationsAdapterComponent):
    def setup_bboxes(self, record_component):
        pass

    def collect(self, record) -> List[BBox]:
        pass

    @staticmethod
    def _clip_bboxes(xyxy, h, w):
        pass


class AlbumentationsMasksComponent(AlbumentationsAdapterComponent):
    def setup_masks(self, record_component):
        pass

    def collect(self, record):
        pass


class AlbumentationsKeypointsComponent(AlbumentationsAdapterComponent):
    def setup_keypoints(self, record_component):
        pass

    def collect(self, record):
        pass

    @classmethod
    def _remove_albu_outside_keypoints(cls, tfms_kpts, kpts_visible, size_no_padding):
        pass

    @staticmethod
    def _check_kps_coords(p, size_no_padding):
        pass


class AlbumentationsIsCrowdsComponent(AlbumentationsAdapterComponent):
    def setup_iscrowds(self, record_component):
        pass

    def collect(self, record):
        pass


class AlbumentationsAreasComponent(AlbumentationsAdapterComponent):
    def setup_areas(self, record_component):
        pass

    def collect(self, record):
        pass


class Adapter(Transform, Composite):
    base_components = {
        AlbumentationsImgComponent,
        AlbumentationsSizeComponent,
        AlbumentationsInstancesLabelsComponent,
        AlbumentationsBBoxesComponent,
        AlbumentationsMasksComponent,
        AlbumentationsIsCrowdsComponent,
        AlbumentationsAreasComponent,
        AlbumentationsKeypointsComponent,
    }

    def __init__(self, tfms):
        super().__init__()
        self.tfms_list = tfms

    def create_tfms(self):
        pass

    def apply(self, record):
        pass






    def _filter_attribute(self, v: list):
        pass


def _flatten_tfms(t):
    pass


def _is_iter(o):
    pass
