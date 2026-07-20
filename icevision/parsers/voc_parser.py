__all__ = ["voc", "VOCBBoxParser", "VOCMaskParser"]

import xml.etree.ElementTree as ET
from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.parsers.parser import *


def voc(
    annotations_dir: Union[str, Path],
    images_dir: Union[str, Path],
    class_map: Optional[ClassMap] = None,
    masks_dir: Optional[Union[str, Path]] = None,
    idmap: Optional[IDMap] = None,
):
    pass


class VOCBBoxParser(Parser):
    def __init__(
        self,
        annotations_dir: Union[str, Path],
        images_dir: Union[str, Path],
        class_map: Optional[ClassMap] = None,
        idmap: Optional[IDMap] = None,
    ):
        super().__init__(template_record=self.template_record(), idmap=idmap)
        self.class_map = class_map or ClassMap().unlock()
        self.images_dir = Path(images_dir)

        self.annotations_dir = Path(annotations_dir)
        self.annotation_files = get_files(self.annotations_dir, extensions=[".xml"])

    def __len__(self):
        return len(self.annotation_files)

    def __iter__(self):
        yield from self.annotation_files

    def template_record(self) -> BaseRecord:
        pass

    def record_id(self, o) -> Hashable:
        pass

    def prepare(self, o):
        pass

    def parse_fields(self, o, record, is_new):
        pass

    def filepath(self, o) -> Union[str, Path]:
        pass

    def img_size(self, o) -> ImgSize:
        pass

    def labels(self, o) -> List[Hashable]:
        pass

    def bboxes(self, o) -> List[BBox]:
        pass


class VOCMaskParser(VOCBBoxParser):
    def __init__(
        self,
        annotations_dir: Union[str, Path],
        images_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        class_map: Optional[ClassMap] = None,
        idmap: Optional[IDMap] = None,
    ):
        super().__init__(
            annotations_dir=annotations_dir,
            images_dir=images_dir,
            class_map=class_map,
            idmap=idmap,
        )
        self.masks_dir = masks_dir
        self.mask_files = get_image_files(masks_dir)

        self._record_id2maskfile = {self.record_id_mask(o): o for o in self.mask_files}

        masks_ids = frozenset(self._record_id2maskfile.keys())
        self._intersection = []
        for item in super().__iter__():
            super().prepare(item)
            if super().record_id(item) in masks_ids:
                self._intersection.append(item)

    def __len__(self):
        return len(self._intersection)

    def __iter__(self):
        yield from self._intersection

    def template_record(self) -> BaseRecord:
        pass

    def record_id_mask(self, o) -> Hashable:
        pass

    def parse_fields(self, o, record, is_new):
        pass

    def masks(self, o) -> List[Mask]:
        pass
