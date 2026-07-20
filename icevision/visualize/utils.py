__all__ = [
    "draw_label",
    "bbox_polygon",
    "draw_mask",
    "as_rgb_tuple",
    "get_default_font",
    "rand_cmap",
]

from icevision.imports import *
from icevision.utils import *
from matplotlib import patches
from PIL import Image, ImageFont, ImageDraw
import PIL


def draw_label(ax, x, y, name, color, fontsize=18):
    pass


def bbox_polygon(bbox):
    pass


def draw_mask(ax, mask, color):
    pass


def as_rgb_tuple(x: Union[np.ndarray, tuple, list, str]) -> tuple:
    pass


def get_default_font() -> str:
    import requests

    font_dir = get_root_dir() / "fonts"
    font_dir.mkdir(exist_ok=True)

    font_file = font_dir / "SpaceGrotesk-Medium.ttf"
    if not font_file.exists():
        URL = "https://raw.githubusercontent.com/airctic/storage/master/SpaceGrotesk-Medium.ttf"
        logger.info(
            "Downloading default `.ttf` font file - SpaceGrotesk-Medium.ttf from {} to {}",
            URL,
            font_file,
        )
        font_file.write_bytes(requests.get(URL).content)
    return str(font_file)


def rand_cmap(
    nlabels,
    type="bright",
    first_color_black=True,
    last_color_black=False,
    verbose=False,
):
    pass
