__all__ = [
    "ImgSize",
    "open_img",
    "get_image_size",
    "get_img_size",
    "show_img",
    "plot_grid",
]

from icevision.imports import *
from PIL import ExifTags

ImgSize = namedtuple("ImgSize", "width,height")

for _EXIF_ORIENTATION_TAG in ExifTags.TAGS.keys():
    if PIL.ExifTags.TAGS[_EXIF_ORIENTATION_TAG] == "Orientation":
        break



def open_img(fn, gray=False, ignore_exif: bool = False) -> PIL.Image.Image:
    pass


def open_gray_scale_image(fn):
    "Opens an radiographic/gray scale image, stacks the channel to represent a RGB image and returns is as a 32bit float array."
    img = np.array(PIL.Image.open(fn))
    img = np.dstack([img, img, img])
    img = img.astype(np.float32)
    return img


def get_image_size(filepath: Union[str, Path]) -> Tuple[int, int]:
    pass


def get_img_size(filepath: Union[str, Path]) -> ImgSize:
    pass


def show_img(img, ax=None, show: bool = False, **kwargs):
    pass


def plot_grid(
    fs: List[callable], ncols=1, figsize=None, show=False, axs_per_iter=1, **kwargs
):
    nrows = math.ceil(len(fs) * axs_per_iter / ncols)
    figsize = figsize or (12 * ncols, 12 * nrows)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    axs = np.asarray(axs)

    if axs_per_iter == 1:
        axs = axs.flatten()
    elif axs_per_iter > 1:
        axs = axs.reshape(-1, axs_per_iter)
    else:
        raise ValueError("axs_per_iter has to be greater than 1")

    for f, ax in zip(fs, axs):
        f(ax=ax)

    plt.tight_layout()
    if show:
        plt.show()
