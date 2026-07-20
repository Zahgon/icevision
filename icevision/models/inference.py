__all__ = ["process_bbox_predictions", "_end2end_detect", "draw_img_and_boxes"]

from icevision.imports import *
from icevision.core import *
from icevision.data import *
from icevision.tfms.albumentations.albumentations_helpers import (
    get_size_without_padding,
)
from icevision.tfms.albumentations import albumentations_adapter

from icevision.utils.imageio import *
from icevision.visualize.draw_data import *
from icevision.visualize.utils import *

DEFAULT_FONT_PATH = get_default_font()


def _end2end_detect(
    img: Union[PIL.Image.Image, Path, str],
    transforms: albumentations_adapter.Adapter,
    model: torch.nn.Module,
    class_map: ClassMap,
    detection_threshold: float = 0.5,
    predict_fn: Callable = None,
    display_label: bool = True,
    display_bbox: bool = True,
    display_score: bool = True,
    font_path: Optional[os.PathLike] = DEFAULT_FONT_PATH,
    font_size: Union[int, float] = 12,
    label_color: Union[np.array, list, tuple, str] = ("#FF59D6"),  # Pink
    return_as_pil_img=True,
    return_img=True,
    **kwargs,
):
    pass


def process_bbox_predictions(
    pred: Prediction,
    img: PIL.Image.Image,
    transforms: List[Any],
) -> List[Dict[str, Any]]:
    pass


def postprocess_bbox(
    img: PIL.Image.Image, bbox: BBox, transforms: List[Any], h_after: int, w_after: int
) -> Tuple[int, int, int, int]:
    """
    Post-process predicted bbox to adjust coordinates to input image size.

    Parameters
    ----------
    img: original image, before any model-pre-processing done
    bbox: predicted bbox
    transforms: list of model-pre-processing transforms
    h_after: height of image after model-pre-processing transforms
    w_after: width of image after model-pre-processing transforms

    Returns
    -------
    Tuple with (xmin, ymin, xmax, ymax) rescaled and re-adjusted to match the original image size
    """
    w_before, h_before = img.size
    h_after, w_after = get_size_without_padding(transforms, img, h_after, w_after)
    pad = np.abs(h_after - w_after) // 2

    h_scale, w_scale = h_after / h_before, w_after / w_before
    if h_after < w_after:
        xmin, xmax, ymin, ymax = (
            int(bbox.xmin),
            int(bbox.xmax),
            int(bbox.ymin) - pad,
            int(bbox.ymax) - pad,
        )
    else:
        xmin, xmax, ymin, ymax = (
            int(bbox.xmin) - pad,
            int(bbox.xmax) - pad,
            int(bbox.ymin),
            int(bbox.ymax),
        )

    xmin, xmax, ymin, ymax = (
        max(xmin, 0),
        min(xmax, w_after),
        max(ymin, 0),
        min(ymax, h_after),
    )
    xmin, xmax, ymin, ymax = (
        int(xmin / w_scale),
        int(xmax / w_scale),
        int(ymin / h_scale),
        int(ymax / h_scale),
    )

    return xmin, ymin, xmax, ymax


def draw_img_and_boxes(
    img: Union[PIL.Image.Image, np.ndarray],
    bboxes: dict,
    class_map,
    display_score: bool = True,
    label_color: Union[np.array, list, tuple, str] = (255, 255, 0),
    label_border_color: Union[np.array, list, tuple, str] = (255, 255, 0),
) -> PIL.Image.Image:
    pass
