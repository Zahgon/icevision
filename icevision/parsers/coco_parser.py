__all__ = [
    "COCOBaseParser",
    "COCOBBoxParser",
    "COCOMaskParser",
    "COCOKeyPointsParser",
    "COCOKeypointsMetadata",
]

from icevision.imports import *
from icevision.core import *
from icevision.utils import *
from icevision.parsers import *


class COCOBaseParser(Parser):
    def __init__(
        self,
        annotations_filepath: Union[str, Path, dict],
        img_dir: Union[str, Path],
        idmap: Optional[IDMap] = None,
    ):

        if isinstance(annotations_filepath, dict):
            self.annotations_dict = annotations_filepath
        else:
            self.annotations_dict = json.loads(Path(annotations_filepath).read_bytes())
        self.img_dir = Path(img_dir)

        self._record_id2info = {o["id"]: o for o in self.annotations_dict["images"]}

        categories = self.annotations_dict["categories"]
        id2class = {o["id"]: o["name"] for o in categories}
        id2class[0] = BACKGROUND
        classes = [None for _ in range(max(id2class.keys()) + 1)]
        for i, name in id2class.items():
            classes[i] = name
        self.class_map = ClassMap(classes)

        super().__init__(template_record=self.template_record(), idmap=idmap)

    def __iter__(self):
        yield from self.annotations_dict["annotations"]

    def __len__(self):
        return len(self.annotations_dict["annotations"])

    def template_record(self) -> BaseRecord:
        pass

    def prepare(self, o):
        pass

    def record_id(self, o) -> int:
        pass

    def filepath(self, o) -> Path:
        pass

    def img_size(self, o) -> ImgSize:
        pass

    def labels_ids(self, o) -> List[Hashable]:
        pass

    def areas(self, o) -> List[float]:
        pass

    def iscrowds(self, o) -> List[bool]:
        pass

    def parse_fields(self, o, record, is_new):
        pass


class COCOBBoxParser(COCOBaseParser):
    def bboxes(self, o) -> List[BBox]:
        pass

    def template_record(self) -> BaseRecord:
        pass

    def parse_fields(self, o, record, is_new):
        pass


class COCOMaskParser(COCOBBoxParser):
    def masks(self, o) -> List[MaskArray]:
        pass

    def template_record(self) -> BaseRecord:
        pass

    def parse_fields(self, o, record, is_new):
        pass


class COCOKeyPointsParser(COCOBBoxParser):
    def template_record(self) -> BaseRecord:
        pass

    def keypoints(self, o) -> List[KeyPoints]:
        pass

    def labels_ids(self, o) -> List[Hashable]:
        pass

    def areas(self, o) -> List[float]:
        pass

    def iscrowds(self, o) -> List[bool]:
        pass

    def bboxes(self, o) -> List[BBox]:
        pass

    def parse_fields(self, o, record, is_new):
        pass


class COCOConnectionsColor:
    head = (220, 30, 200)
    torso = (100, 240, 100)
    right_arm = (170, 170, 100)
    left_arm = (120, 120, 230)
    left_leg = (230, 120, 130)
    right_leg = (230, 180, 190)


class COCOKeypointsMetadata(KeypointsMetadata):
    labels = (
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )

    connections = (
        KeypointConnection(0, 1, COCOConnectionsColor.head),
        KeypointConnection(0, 2, COCOConnectionsColor.head),
        KeypointConnection(1, 2, COCOConnectionsColor.head),
        KeypointConnection(1, 3, COCOConnectionsColor.head),
        KeypointConnection(2, 4, COCOConnectionsColor.head),
        KeypointConnection(3, 5, COCOConnectionsColor.head),
        KeypointConnection(4, 6, COCOConnectionsColor.head),
        KeypointConnection(5, 6, COCOConnectionsColor.torso),
        KeypointConnection(5, 7, COCOConnectionsColor.left_arm),
        KeypointConnection(5, 11, COCOConnectionsColor.torso),
        KeypointConnection(6, 8, COCOConnectionsColor.right_arm),
        KeypointConnection(6, 12, COCOConnectionsColor.torso),
        KeypointConnection(7, 9, COCOConnectionsColor.left_arm),
        KeypointConnection(8, 10, COCOConnectionsColor.right_arm),
        KeypointConnection(11, 12, COCOConnectionsColor.torso),
        KeypointConnection(13, 11, COCOConnectionsColor.left_leg),
        KeypointConnection(14, 12, COCOConnectionsColor.right_leg),
        KeypointConnection(15, 13, COCOConnectionsColor.left_leg),
        KeypointConnection(16, 14, COCOConnectionsColor.right_leg),
    )
