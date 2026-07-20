__all__ = ["get_files", "get_image_files"]

from icevision.imports import *

def _get_files(p, fs, extensions=None):
    pass


def get_files(
    path,
    extensions=None,
    recurse=True,
    folders=None,
    followlinks=True,
    sort: bool = True,
):
    pass


image_extensions = set(
    k for k, v in mimetypes.types_map.items() if v.startswith("image/")
)


def get_image_files(path, recurse=True, folders=None):
    pass
