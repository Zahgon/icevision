__all__ = ["IDMap"]

from icevision.imports import *


class IDMap:

    def __init__(self, initial_names: Optional[Sequence[Hashable]] = None):
        names = initial_names or []
        self.id2name = OrderedDict((id, name) for id, name in enumerate(names))
        self.name2id = OrderedDict((name, id) for id, name in enumerate(names))

    def get_id(self, id: int) -> Hashable:
        pass

    def get_name(self, name: Hashable) -> int:
        pass

    def filter_ids(self, ids: List[int]) -> "IDMap":
        pass

    def get_ids(self) -> List[int]:
        pass

    def get_names(self) -> List[Hashable]:
        pass

    def __getitem__(self, record_id):
        return self.get_name(record_id)
