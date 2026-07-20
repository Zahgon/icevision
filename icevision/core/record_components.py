__all__ = [
    "RecordComponent",
    "ClassMapRecordComponent",
    "RecordIDRecordComponent",
    "ImageRecordComponent",
    "FilepathRecordComponent",
    "SizeRecordComponent",
    "BaseLabelsRecordComponent",
    "InstancesLabelsRecordComponent",
    "ClassificationLabelsRecordComponent",
    "GrayScaleRecordComponent",
    "BBoxesRecordComponent",
    "BaseMasksRecordComponent",
    "InstanceMasksRecordComponent",
    "SemanticMaskRecordComponent",
    "AreasRecordComponent",
    "IsCrowdsRecordComponent",
    "KeyPointsRecordComponent",
    "ScoresRecordComponent",
    "LossesRecordComponent",
]

from icevision.utils.imageio import open_gray_scale_image
from icevision.imports import *
from icevision.utils import *
from icevision.core.components import *
from icevision.core.bbox import *
from icevision.core.mask import *
from icevision.core.exceptions import *
from icevision.core.keypoints import *
from icevision.core.class_map import *
from icevision.core import tasks


class RecordComponent(TaskComponent):
    @property
    def record(self):
        pass

    def as_dict(self) -> dict:
        pass

    def _load(self) -> None:
        pass

    def _unload(self) -> None:
        pass

    def _num_annotations(self) -> Dict[str, int]:
        pass

    def _autofix(self) -> Dict[str, bool]:
        pass

    def _aggregate_objects(self) -> Dict[str, List[dict]]:
        pass

    def _repr(self) -> List[str]:
        pass

    def builder_template(self) -> List[str]:
        pass

    def _builder_template(self) -> List[str]:
        pass

    def _format_builder_template(self, lines):
        pass

    def setup_transform(self, tfm) -> None:
        pass


class ClassMapRecordComponent(RecordComponent):
    def __init__(self, task):
        super().__init__(task=task)
        self.class_map = None

    def set_class_map(self, class_map: ClassMap):
        self.class_map = class_map

    def _repr(self) -> List[str]:
        pass

    def as_dict(self) -> dict:
        pass

    def _builder_template(self) -> List[str]:
        pass


class RecordIDRecordComponent(RecordComponent):
    def __init__(self, task=tasks.common):
        super().__init__(task=task)
        self.record_id = None

    def set_record_id(self, record_id: int):
        self.record_id = record_id

    def _repr(self) -> List[str]:
        pass

    def as_dict(self) -> dict:
        pass


class ImageRecordComponent(RecordComponent):
    def __init__(self, task=tasks.common):
        super().__init__(task=task)
        self.img = None

    def set_img(self, img: Union[PIL.Image.Image, np.ndarray]):
        assert isinstance(img, (PIL.Image.Image, np.ndarray))
        self.img = img
        if isinstance(img, PIL.Image.Image):
            width, height = img.size
        elif isinstance(img, np.ndarray):
            height, width, _ = self.img.shape
        self.composite.set_img_size(ImgSize(width=width, height=height), original=True)

    def _repr(self) -> List[str]:
        pass

    def _unload(self):
        pass

    def as_dict(self) -> dict:
        pass

    def setup_transform(self, tfm) -> None:
        pass


class FilepathRecordComponent(ImageRecordComponent):
    def __init__(self, task=tasks.common, gray=False):
        super().__init__(task=task)
        self.gray = gray
        self.filepath = None

    def set_filepath(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)

    def _load(self):
        pass

    def _autofix(self) -> Dict[str, bool]:
        pass

    def _repr(self) -> List[str]:
        pass

    def as_dict(self) -> dict:
        pass

    def _builder_template(self) -> List[str]:
        pass


class SizeRecordComponent(RecordComponent):
    def __init__(self, task=tasks.common):
        super().__init__(task=task)
        self.img_size = None

    def set_image_size(self, width: int, height: int):
        pass

    def set_img_size(self, size: ImgSize, original: bool = False):
        self.img_size = size
        self.width, self.height = size

        if original:
            self.original_img_size = size

    def setup_transform(self, tfm) -> None:
        pass

    def _repr(self) -> List[str]:
        pass

    def as_dict(self) -> dict:
        pass

    def _aggregate_objects(self) -> Dict[str, List[dict]]:
        pass

    def _builder_template(self) -> List[str]:
        pass


class GrayScaleRecordComponent(FilepathRecordComponent):

    def _load(self):
        pass


class BaseLabelsRecordComponent(ClassMapRecordComponent):
    def __init__(self, task=tasks.common):
        super().__init__(task=task)
        self.label_ids: List[int] = []
        self.labels: List[Hashable] = []

    def set_labels_by_id(self, labels: Sequence[int]):
        self.label_ids = list(labels)
        if self.class_map is not None:
            self.labels = self._labels_ids_to_names(labels)

    def add_labels_by_id(self, labels: Sequence[int]):
        self.label_ids.extend(labels)
        if self.class_map is not None:
            self.labels.extend(self._labels_ids_to_names(labels))

    def set_labels(self, labels_names: Sequence[Hashable]):
        pass

    def add_labels(self, labels_names: Sequence[Hashable]):
        pass

    def is_valid(self) -> List[bool]:
        pass

    def _labels_ids_to_names(self, labels_ids):
        return [self.class_map.get_by_id(id) for id in labels_ids]

    def _labels_names_to_ids(self, labels_names):
        pass

    def _num_annotations(self) -> Dict[str, int]:
        pass

    def _autofix(self) -> Dict[str, bool]:
        pass

    def _remove_annotation(self, i):
        pass

    def _aggregate_objects(self) -> Dict[str, List[dict]]:
        pass

    def _repr(self) -> List[str]:
        pass

    def as_dict(self) -> dict:
        pass

    def _builder_template(self) -> List[str]:
        pass


class InstancesLabelsRecordComponent(BaseLabelsRecordComponent):
    def __init__(self, task=tasks.detection):
        super().__init__(task=task)

    def setup_transform(self, tfm) -> None:
        pass


class ClassificationLabelsRecordComponent(BaseLabelsRecordComponent):
    def __init__(self, task=tasks.classification, is_multilabel: bool = False):
        super().__init__(task=task)
        self.is_multilabel = is_multilabel

    def _autofix(self):
        pass

    def one_hot_encoded(self) -> np.array:
        pass


class BBoxesRecordComponent(RecordComponent):
    def __init__(self, task=tasks.detection):
        super().__init__(task=task)
        self.bboxes: List[BBox] = []

    def set_bboxes(self, bboxes: Sequence[BBox]):
        self.bboxes = list(bboxes)

    def add_bboxes(self, bboxes: Sequence[BBox]):
        self.bboxes.extend(bboxes)

    def _autofix(self) -> Dict[str, bool]:
        pass

    def _num_annotations(self) -> Dict[str, int]:
        pass

    def _remove_annotation(self, i):
        pass

    def _aggregate_objects(self) -> Dict[str, List[dict]]:
        pass

    def _repr(self) -> List[str]:
        pass

    def as_dict(self) -> dict:
        pass

    def setup_transform(self, tfm) -> None:
        pass

    def _builder_template(self) -> List[str]:
        pass


class BaseMasksRecordComponent(RecordComponent):
    def __init__(self, task):
        super().__init__(task=task)
        self.masks = self.mask_parts = []
        self.mask_array: MaskArray = None

    def add_masks(self, masks: Sequence[Mask]):
        pass

    def set_masks(self, masks: Sequence[Mask]):
        pass

    def set_mask(self, mask: Mask):
        pass

    def set_mask_array(self, mask_array: MaskArray):
        self.mask_array = mask_array

    def _load(self):
        pass

    def _unload(self):
        pass

    def setup_transform(self, tfm) -> None:
        pass

    def _repr(self) -> List[str]:
        pass

    def as_dict(self) -> dict:
        pass

    def _remove_annotation(self, i):
        raise NotImplementedError(
            "_remove_annotation does not work for mask component, if you're getting "
            "this on autofix, it's probably means this record has to be fixes manually"
        )


class SemanticMaskRecordComponent(BaseMasksRecordComponent):
    def __init__(self, task=tasks.segmentation):
        super().__init__(task=task)

    def _builder_template(self) -> List[str]:
        pass


class InstanceMasksRecordComponent(BaseMasksRecordComponent):
    def __init__(self, task=tasks.detection):
        super().__init__(task=task)

    def _builder_template(self) -> List[str]:
        pass


























class AreasRecordComponent(RecordComponent):
    def __init__(self, task=tasks.detection):
        super().__init__(task=task)
        self.areas: List[float] = []

    def set_areas(self, areas: Sequence[float]):
        pass

    def add_areas(self, areas: Sequence[float]):
        pass

    def setup_transform(self, tfm) -> None:
        pass

    def _num_annotations(self) -> Dict[str, int]:
        pass

    def _remove_annotation(self, i):
        pass

    def _repr(self) -> List[str]:
        pass

    def as_dict(self) -> dict:
        pass


class IsCrowdsRecordComponent(RecordComponent):
    def __init__(self, task=tasks.detection):
        super().__init__(task=task)
        self.iscrowds: List[bool] = []

    def set_iscrowds(self, iscrowds: Sequence[bool]):
        pass

    def add_iscrowds(self, iscrowds: Sequence[bool]):
        pass

    def setup_transform(self, tfm) -> None:
        pass

    def _num_annotations(self) -> Dict[str, int]:
        pass

    def _remove_annotation(self, i):
        pass

    def _aggregate_objects(self) -> Dict[str, List[dict]]:
        pass

    def _repr(self) -> List[str]:
        pass

    def as_dict(self) -> dict:
        pass


class KeyPointsRecordComponent(RecordComponent):
    def __init__(self, task=tasks.detection):
        super().__init__(task=task)
        self.keypoints: List[KeyPoints] = []

    def set_keypoints(self, keypoints: Sequence[KeyPoints]):
        pass

    def add_keypoints(self, keypoints: Sequence[KeyPoints]):
        self.keypoints.extend(keypoints)

    def setup_transform(self, tfm) -> None:
        pass

    def as_dict(self) -> dict:
        pass

    def _aggregate_objects(self) -> Dict[str, List[dict]]:
        pass

    def _repr(self) -> List[str]:
        pass


class ScoresRecordComponent(RecordComponent):
    def __init__(self, task=tasks.detection):
        super().__init__(task=task)
        self.scores = None

    def set_scores(self, scores: Sequence[float]):
        self.scores = scores

    def _repr(self) -> List[str]:
        pass

    def as_dict(self) -> dict:
        pass


class LossesRecordComponent(RecordComponent):
    def __init__(self, task=tasks.common):
        super().__init__(task=task)
        self.losses = None

    def set_losses(self, losses: Dict):
        self.losses = losses
