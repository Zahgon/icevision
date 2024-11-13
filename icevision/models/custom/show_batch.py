import cv2
import numpy as np

from icevision.utils.torch_utils import tensor_to_image
from icevision.visualize.show_data import show_samples


def show_batch(batch, ncols: int = 1, figsize=None, **show_samples_kwargs):
    """Show a single batch from a dataloader.

    # Arguments
        show_samples_kwargs: Check the parameters from `show_samples`
    """
    (tensor_images, heatmaps), records = batch

    for tensor_image, heatmap, record in zip(tensor_images, heatmaps, records):
        image = tensor_to_image(tensor_image)
        hm = heatmap.cpu().numpy().squeeze()
        # 1. Resize mask to match image dimensions
        mask_resized = cv2.resize(hm, (image.shape[1], image.shape[0]))

        # 2. Create a colored overlay (e.g., red)
        # Assuming mask values are normalized between 0 and 1
        overlay = np.zeros_like(image)
        overlay[..., 2] = mask_resized * 255  # Red channel

        # 3. Blend images
        alpha_channel = mask_resized
        result = image * (1 - alpha_channel[:, :, np.newaxis]) + overlay * alpha_channel[:, :, np.newaxis]

        record.set_img(result)

    return show_samples(records, ncols=ncols, figsize=figsize, **show_samples_kwargs)
