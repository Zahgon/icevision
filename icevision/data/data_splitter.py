__all__ = [
    "DataSplitter",
    "SingleSplitSplitter",
    "RandomSplitter",
    "FixedSplitter",
    "FuncSplitter",
]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *


class DataSplitter(ABC):

    def __call__(self, records: Sequence[BaseRecord]):
        return self.split(records=records)

    @abstractmethod
    def split(self, records: Sequence[BaseRecord]):
        """Splits `ids` into groups.

        # Arguments
            idmap: idmap used for getting ids.
        """
        pass


class SingleSplitSplitter(DataSplitter):

    def split(self, records: Sequence[BaseRecord]):
        pass


class RandomSplitter(DataSplitter):

    def __init__(self, probs: Sequence[int], seed: int = None):
        self.probs = probs
        self.seed = seed

    def split(self, records: Sequence[BaseRecord]):
        pass


class FixedSplitter(DataSplitter):

    def __init__(self, splits: Sequence[Sequence[Hashable]]):
        self.splits = splits

    def split(self, records: Sequence[BaseRecord]):
        pass




class FuncSplitter(DataSplitter):
    def __init__(self, func):
        self.func = func

    def split(self, records: Sequence[BaseRecord]):
        pass
