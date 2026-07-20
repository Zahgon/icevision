__all__ = ["via", "VIAParseError", "VIABaseParser", "VIABBoxParser"]

from icevision.imports import *
from icevision.core import *
from icevision.utils import *
from icevision.parsers.parser import *


def via(
    annotations_file: Union[str, Path],
    img_dir: Union[str, Path],
    class_map: ClassMap,
    label_field: str = "label",
) -> Parser:
    pass


class VIAParseError(Exception):
    pass


class VIABaseParser(Parser):
    def __init__(
        self,
        annotations_filepath: Union[str, Path, dict],
        img_dir: Union[str, Path],
        class_map: ClassMap,
        label_field: str = "label",
    ):
        super().__init__(template_record=self.template_record())
        if isinstance(annotations_filepath, dict):
            self.annotations_dict = annotations_filepath
        else:
            self.annotations_dict = json.loads(Path(annotations_filepath).read_bytes())
        self.img_dir = Path(img_dir)
        self.label_field = label_field
        self.class_map = class_map

    def template_record(self) -> BaseRecord:
        pass

    def __iter__(self):
        yield from self.annotations_dict.values()

    def __len__(self):
        return len(self.annotations_dict.values())

    def record_id(self, o) -> Hashable:
        pass

    def parse_fields(self, o, record, is_new):
        pass

    def filepath(self, o) -> Path:
        pass

    def image_width_height(self, o) -> Tuple[int, int]:
        pass

    def _get_label(self, o, region_attributes: dict) -> str:
        pass

    def labels(self, o) -> List[int]:
        pass


class VIABBoxParser(VIABaseParser):

    def parse_fields(self, o, record, is_new):
        pass

    def bboxes(self, o) -> List[BBox]:
        pass
