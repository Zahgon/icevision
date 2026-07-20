__all__ = [
    "notnone",
    "ifnotnone",
    "last",
    "lmap",
    "allequal",
    "cleandict",
    "mergeds",
    "zipsafe",
    "np_local_seed",
    "pbar",
    "IMAGENET_STATS",
    "normalize",
    "denormalize",
    "normalize_imagenet",
    "denormalize_imagenet",
    "denormalize_mask",
    "patch_class_to_main",
]

from icevision.imports import *


def notnone(x):
    return x is not None


def ifnotnone(x, f):
    pass


def last(x):
    pass


def lmap(f, xs):
    return list(map(f, xs)) if notnone(xs) else None


def allequal(l):
    return l.count(l[0]) == len(l) if l else True


def cleandict(d):
    pass


def mergeds(ds):
    pass


def zipsafe(*its):
    if not allequal(lmap(len, its)):
        raise ValueError("The elements have different leghts")
    return zip(*its)


def pbar(iter, show=True, total: Optional[int] = None):
    return tqdm(iter, total=total) if show else iter


@contextmanager
def np_local_seed(seed):
    pass


def normalize(img, mean, std, max_pixel_value=255):
    pass


def denormalize(img, mean, std, max_pixel_value=255):
    pass


IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def denormalize_imagenet(img):
    pass


def normalize_imagenet(img):
    pass


def denormalize_mask(img):
    pass


def patch_class_to_main(cls):
    pass
