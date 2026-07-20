
__all__ = [
    "draw_sample",
    "draw_record",
    "draw_pred",
    "draw_bbox",
    "draw_mask",
    "draw_keypoints",
    "draw_label",
    "draw_segmentation_mask",
]

from icevision.imports import *
from icevision.data import *
from icevision.core import *
from icevision.visualize.utils import *
from matplotlib.colors import LinearSegmentedColormap

from PIL import Image, ImageFont, ImageDraw
import PIL

DEFAULT_FONT_PATH = get_default_font()


def draw_sample(
    sample,
    class_map: Optional[ClassMap] = None,
    denormalize_fn: Optional[callable] = None,
    display_label: bool = True,
    display_bbox: bool = True,
    display_score: bool = True,
    display_mask: bool = True,
    display_keypoints: bool = True,
    font_path: Optional[os.PathLike] = DEFAULT_FONT_PATH,
    font_size: Union[int, float] = None,
    label_color: Union[np.array, list, tuple, str] = "#C4C4C4",  # Mild Gray
    label_border_color: Union[np.array, list, tuple, str] = "#020303",  # Black,
    label_thin_border: bool = True,
    label_pad_width_factor: float = 0.02,
    label_pad_height_factor: float = 0.005,
    mask_blend: float = 0.5,
    mask_border_thickness: int = 7,
    color_map: Optional[dict] = None,  # label -> color mapping
    prettify: bool = True,
    prettify_func: Callable = str.capitalize,
    return_as_pil_img=False,
    exclude_labels: List[str] = [],
    include_only: List[str] = None,
    multiple_classification_spacing_factor: float = 1.05,
    dynamic_font_size_div_factor: float = 20.0,
    include_classification_task_names: bool = True,
    include_instances_task_names: bool = False,
) -> Union[np.ndarray, PIL.Image.Image]:
    pass


def draw_label(
    img: np.ndarray,
    label: Union[int, str],
    score: Optional[float],
    color: Union[np.ndarray, list, tuple],
    border_color: Union[np.ndarray, list, tuple],
    class_map: Optional[ClassMap] = None,
    bbox=None,
    mask=None,
    font: Union[int, os.PathLike, None] = None,
    font_size: Union[int, float] = 12,
    prettify: bool = True,
    prettify_func: Callable = str.capitalize,
    return_as_pil_img=False,
    pad_width_factor=0.02,
    pad_height_factor=0.005,
    thin_border=True,
    x: Optional[int] = None,
    y: Optional[int] = None,
    prefix: str = "",
) -> Union[np.ndarray, PIL.Image.Image]:
    pass


def _draw_label(
    img: np.ndarray,
    caption: str,
    x: int,
    y: int,
    color: Union[np.ndarray, list, tuple],
    border_color: Union[np.ndarray, list, tuple],
    font_path=DEFAULT_FONT_PATH,
    font_size: int = 20,
    return_as_pil_img: bool = False,
    pad_width_factor=0.02,
    pad_height_factor=0.005,
    thin_border=True,
) -> Union[PIL.Image.Image, np.ndarray]:
    pass


def draw_record(
    record,
    class_map: Optional[ClassMap] = None,
    display_label: bool = True,
    display_bbox: bool = True,
    display_mask: bool = True,
    display_score: bool = True,
    display_keypoints: bool = True,
    font_path: Optional[os.PathLike] = DEFAULT_FONT_PATH,
    font_size: Union[int, float] = 12,
    label_color: Union[np.array, list, tuple, str] = "#C4C4C4",  # Mild Gray
    mask_blend: float = 0.5,
    mask_border_thickness: int = 7,
    color_map: Optional[dict] = None,  # label -> color mapping
    prettify: bool = True,
    prettify_func: Callable = str.capitalize,
    return_as_pil_img=False,
    exclude_labels: List[str] = [],
    include_only: List[str] = None,
):
    pass


def draw_pred(
    pred: Prediction,
    denormalize_fn: Optional[callable] = None,
    display_label: bool = True,
    display_score: bool = True,
    display_bbox: bool = True,
    display_mask: bool = True,
    font_path: Optional[os.PathLike] = DEFAULT_FONT_PATH,
    font_size: Union[int, float] = 12,
    label_color: Union[np.array, list, tuple, str] = "#C4C4C4",  # Mild Gray
    mask_blend: float = 0.5,
    mask_border_thickness: int = 7,
    color_map: Optional[dict] = None,  # label -> color mapping
    prettify: bool = True,
    prettify_func: Callable = str.capitalize,
    return_as_pil_img=False,
    exclude_labels: List[str] = [],
    include_only: List[str] = None,
):
    pass


def draw_bbox(
    img: np.ndarray,
    bbox: BBox,
    color: Tuple[int, int, int],
    gap: bool = True,
):
    pass


def draw_mask(
    img: np.ndarray,
    mask: MaskArray,
    color: Tuple[int, int, int],
    blend: float = 0.5,
    border_thickness: int = 7,
):
    pass


def draw_segmentation_mask(
    img: np.ndarray,
    mask: MaskArray,
    cmap: LinearSegmentedColormap,
    display_mask: bool = True,
    alpha: float = 0.5,
):
    pass


def draw_keypoints(
    img: np.ndarray,
    kps: KeyPoints,
    color: Tuple[int, int, int],
):
    pass
