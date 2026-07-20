__all__ = ["Dataset"]

from icevision.imports import *
from icevision.core import *
from icevision.tfms import *


class Dataset:

    def __init__(
        self,
        records: List[dict],
        tfm: Optional[Transform] = None,
    ):
        self.records = records
        self.tfm = tfm

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        record = self.records[i].load()
        if self.tfm is not None:
            record = self.tfm(record)
        else:
            record.set_img(np.array(record.img))
        return record

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self.records)} items>"

    @classmethod
    def from_images(
        cls,
        images: Sequence[np.array],
        tfm: Transform = None,
        class_map: Optional[ClassMap] = None,
    ):
        pass
