__all__ = ["BackboneConfig"]

from abc import ABC, abstractmethod
from typing import Any


class BackboneConfig(ABC):
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> "BackboneConfig":
        """Completes configuration for this backbone.

        Called by the end user. All heavy lifting (such as creating a NN backbone)
        should be done here.
        """
