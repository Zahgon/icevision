__all__ = [
    "Mask",
    "MaskArray",
    "MaskFile",
    "VocMaskFile",
    "RLE",
    "Polygon",
    "EncodedRLEs",
    "SemanticMaskFile",
]

from icevision.imports import *
from icevision.utils import *
from PIL import Image


class Mask(ABC):
    @abstractmethod
    def to_mask(self, h, w) -> "MaskArray":
        pass

    @abstractmethod
    def to_erles(self, h, w) -> "EncodedRLEs":
        pass


class EncodedRLEs(Mask):
    def __init__(self, erles: List[dict] = None):
        self.erles = erles or []

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self)} objects>"

    def __len__(self):
        return len(self.erles)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.erles == other.erles
        return False

    def append(self, v: "EncodedRLEs"):
        self.erles.extend(v.erles)

    def extend(self, v: List["EncodedRLEs"]):
        for o in v:
            self.append(o)

    def pop(self, i: int):
        self.erles.pop(i)

    def to_mask(self, h, w) -> "MaskArray":
        pass

    def to_erles(self, h, w) -> "EncodedRLEs":
        pass


class MaskArray(Mask):

    def __init__(self, data: np.uint8):
        if len(data.shape) == 2:
            data = np.expand_dims(data, 0)
        self.data = data.astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return type(self)(self.data[i])

    def to_tensor(self):
        pass

    def to_mask(self, h, w):
        pass

    def to_erles(self, h, w) -> EncodedRLEs:
        pass

    def to_coco_rle(self, h, w) -> List[dict]:
        pass

    @property
    def shape(self):
        pass

    @classmethod
    def from_masks(cls, masks: Union[EncodedRLEs, Sequence[Mask]], h: int, w: int):
        pass


class MaskFile(Mask):

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)

    def to_mask(self, h, w):
        pass

    def to_coco_rle(self, h, w) -> List[dict]:
        pass

    def to_erles(self, h, w) -> EncodedRLEs:
        pass


class VocMaskFile(MaskFile):

    def __init__(self, filepath: Union[str, Path], drop_void: bool = True):
        super().__init__(filepath=filepath)
        self.drop_void = drop_void

    def to_mask(self, h, w) -> MaskArray:
        pass


class RLE(Mask):

    def __init__(self, counts: List[int]):
        self.counts = counts

    def to_mask(self, h, w) -> "MaskArray":
        pass

    def to_coco(self) -> List[int]:
        pass

    def to_erles(self, h, w) -> EncodedRLEs:
        pass

    @classmethod
    def from_string(cls, s, sep=" "):
        pass

    @classmethod
    def from_kaggle(cls, counts: Sequence[int]):
        pass

    @classmethod
    def from_coco(cls, counts: Sequence[int]):
        pass


class Polygon(Mask):

    def __init__(self, points: List[List[int]]):
        self.points = points

    def to_mask(self, h, w):
        pass

    def to_erles(self, h, w) -> EncodedRLEs:
        pass


class SemanticMaskFile(Mask):

    def __init__(self, filepath: Union[str, Path], binary=False):
        self.filepath = Path(filepath)
        self.binary = binary

    def to_mask(self, h, w):
        pass

    def to_coco_rle(self, h, w) -> List[dict]:
        pass

    def to_erles(self, h, w) -> EncodedRLEs:
        pass
