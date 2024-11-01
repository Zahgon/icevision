import functools
import re


class partial:
    """Wraps functools.partial, same functionality.

    Modifies the original partial `__repr__` and `__str__` in other to fix #270
    """

    def __init__(self, func, *args, **kwargs):
        self._partial = functools.partial(func, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._partial(*args, **kwargs)

    def __str__(self):
        name = self._partial.func.__name__
        partial_str = str(self._partial)
        return re.sub(r"<.+>", name, partial_str)

    def __repr__(self):
        return str(self)
