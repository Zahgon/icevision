import torch
import torch.nn.functional as F
import numpy as np


class HeatmapDecoder:
    def __init__(self, output_stride=4, use_dark=True):
        """
        Initialize heatmap decoder

        Args:
            output_stride (int): Stride of the network output compared to input
            use_dark (bool): Whether to use DARK method for subpixel refinement
        """
        self.output_stride = output_stride
        self.use_dark = use_dark

    def _get_max_preds(self, heatmaps):
        """Get predictions from score maps"""
        batch_size, num_keypoints, height, width = heatmaps.shape

        # Flatten last two dimensions
        heatmaps_reshaped = heatmaps.reshape((batch_size, num_keypoints, -1))

        # Get max scores and locations
        maxvals, idx = torch.max(heatmaps_reshaped, dim=2)

        # Convert indices to coordinates
        preds = torch.zeros((batch_size, num_keypoints, 2)).to(device=heatmaps.device)
        preds[..., 0] = (idx % width)  # x coords
        preds[..., 1] = (idx // width)  # y coords

        return preds, maxvals

    def _dark_post_processing(self, heatmaps, coords):
        """
        Implement DARK post-processing for subpixel accuracy
        https://arxiv.org/abs/1910.06278
        """
        batch_size, num_keypoints, height, width = heatmaps.shape
        px = coords[..., 0].long()
        py = coords[..., 1].long()

        # Clip coordinates to valid range
        px = px.clamp(1, width - 2)
        py = py.clamp(1, height - 2)

        # For each keypoint, get 3x3 around max
        batch_idx = torch.arange(batch_size)[:, None].expand(batch_size, num_keypoints)
        kpt_idx = torch.arange(num_keypoints)[None, :].expand(batch_size, num_keypoints)

        # Get differential values
        diff_x = (heatmaps[batch_idx, kpt_idx, py, px + 1] -
                  heatmaps[batch_idx, kpt_idx, py, px - 1]) / 2
        diff_y = (heatmaps[batch_idx, kpt_idx, py + 1, px] -
                  heatmaps[batch_idx, kpt_idx, py - 1, px]) / 2

        # Get second order values
        diff_xx = (heatmaps[batch_idx, kpt_idx, py, px + 1] +
                   heatmaps[batch_idx, kpt_idx, py, px - 1] -
                   2 * heatmaps[batch_idx, kpt_idx, py, px])
        diff_yy = (heatmaps[batch_idx, kpt_idx, py + 1, px] +
                   heatmaps[batch_idx, kpt_idx, py - 1, px] -
                   2 * heatmaps[batch_idx, kpt_idx, py, px])
        diff_xy = ((heatmaps[batch_idx, kpt_idx, py + 1, px + 1] +
                    heatmaps[batch_idx, kpt_idx, py - 1, px - 1] -
                    heatmaps[batch_idx, kpt_idx, py + 1, px - 1] -
                    heatmaps[batch_idx, kpt_idx, py - 1, px + 1]) / 4)

        # Solve system of equations
        det = diff_xx * diff_yy - diff_xy * diff_xy

        # Avoid division by zero
        det = torch.where(det != 0, det, det + 1e-5)

        # Get offset
        offset_x = (diff_xy * diff_y - diff_yy * diff_x) / det
        offset_y = (diff_xy * diff_x - diff_xx * diff_y) / det

        # Add offsets to original coordinates
        coords[..., 0] += offset_x
        coords[..., 1] += offset_y

        return coords

    def __call__(self, heatmaps):
        """
        Decode heatmaps to keypoint coordinates

        Args:
            heatmaps (torch.Tensor): Predicted heatmaps (B, K, H, W)

        Returns:
            torch.Tensor: Coordinates in original image space (B, K, 2)
            torch.Tensor: Confidence scores (B, K) if return_confidence=True
        """
        # Get initial predictions
        coords, maxvals = self._get_max_preds(heatmaps)

        # Apply DARK post-processing if enabled
        if self.use_dark:
            coords = self._dark_post_processing(heatmaps, coords)

        # Scale coordinates to original image space
        coords = coords * self.output_stride
        return coords, maxvals
