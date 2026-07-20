__all__ = ["load_txt", "create_tmp_dir", "mkdir", "extract_files"]

import numpy as np
import shutil
from pathlib import Path
from .utils import pbar


def load_txt(file):
    pass


def create_tmp_dir(name: str, overwrite: bool = True) -> Path:
    pass


def mkdir(path, exist_ok=False, parents=False, overwrite=False) -> Path:
    path = Path(path)

    if path.exists() and overwrite:
        shutil.rmtree(path)

    path.mkdir(exist_ok=exist_ok, parents=parents)
    return path


def extract_files(files, extract_to_dir, show_pbar: bool = True):
    pass
