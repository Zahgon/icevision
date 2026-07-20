__all__ = ["ImgPadStack"]

from icevision.imports import *
from icevision.core import *
from icevision.tfms.batch.batch_transform import BatchTransform


class ImgPadStack(BatchTransform):
    def __init__(self, pad_value: Union[float, Sequence[float]] = 0.0):
        self.pad_value = np.array(pad_value).reshape(-1)

    def apply(self, records: List[RecordType]) -> List[RecordType]:
        pass
