__all__ = ["ClassMap", "BACKGROUND"]

from icevision.imports import *

BACKGROUND = "background"


class ClassMap:

    def __init__(
        self,
        classes: Optional[Sequence[str]] = None,
        background: Optional[str] = BACKGROUND,
    ):
        self._lock = True

        self._id2class = copy(list(classes)) if classes else []
        self._background = background
        if self._background is not None:
            try:
                self._id2class.remove(self._background)
            except ValueError:
                pass
            self._id2class.insert(0, self._background)

        self._class2id = {name: i for i, name in enumerate(self._id2class)}

    @property
    def num_classes(self):
        pass

    def get_classes(self) -> Sequence[str]:
        pass

    def get_by_id(self, id: int) -> str:
        return self._id2class[id]

    def get_by_name(self, name: str) -> int:
        pass

    def add_name(self, name: str) -> int:
        pass

    def lock(self):
        pass

    def unlock(self):
        pass

    def __eq__(self, other) -> bool:
        if isinstance(other, ClassMap):
            return self.__dict__ == other.__dict__
        return False

    def __len__(self):
        return len(self._id2class)

    def __repr__(self):
        return f"<ClassMap: {self._class2id.__repr__()}>"
