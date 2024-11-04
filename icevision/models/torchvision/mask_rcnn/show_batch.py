__all__ = ["show_batch"]

import numpy as np

from icevision.core.mask import MaskArray
from icevision.models.utils import unpack_batch
from icevision.utils.torch_utils import tensor_to_image
from icevision.visualize.show_data import show_samples


def show_batch(batch, ncols: int = 1, figsize=None, **show_samples_kwargs):
    """Show a single batch from a dataloader.

    # Arguments
        show_samples_kwargs: Check the parameters from `show_samples`
    """
    (tensor_images, labels), records = unpack_batch(batch)

    for tensor_image, label, record in zip(tensor_images, labels, records):
        image = tensor_to_image(tensor_image)
        record.set_img(image)

        if label is not None:
            mask = MaskArray(np.array(label["masks"].cpu().numpy()))
            record.detection.set_mask_array(mask)

    return show_samples(records, ncols=ncols, figsize=figsize, **show_samples_kwargs)
