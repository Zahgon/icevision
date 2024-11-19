import torch
import numpy as np


class KeypointHeatmapGenerator:
    def __init__(self, output_size, sigma=2):
        """
        Initialize heatmap generator

        Args:
            output_size (tuple): Size of output heatmap (H, W)
            sigma (int): Standard deviation for Gaussian kernel
        """
        self.output_size = output_size
        self.sigma = sigma
        self.generate_gaussian_kernel()

    def generate_gaussian_kernel(self):
        """Create a Gaussian kernel for heatmap generation"""
        size = 6 * self.sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = size // 2, size // 2

        # Generate 2D gaussian
        self.gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
        self.kernel_size = size

    def __call__(self, keypoints: np.ndarray):
        """
        Generate heatmaps for keypoints

        Args:
            keypoints (torch.Tensor): Keypoints of shape (N, K, 3) where N is batch size,
                                    K is number of keypoints, and last dim is (x, y, v)

        Returns:
            torch.Tensor: Heatmaps of shape (N, K, H, W)
        """
        batch_size, num_keypoints, _ = keypoints.shape
        heatmaps = torch.zeros((batch_size, num_keypoints, *self.output_size))
        visibility = torch.from_numpy(keypoints[:,:,2])

        for n in range(batch_size):
            for k in range(num_keypoints):
                x, y, visible = keypoints[n, k]
                # Convert to output space coordinates
                x = x * self.output_size[1]
                y = y * self.output_size[0]

                # Get top left corner of gaussian patch
                x0 = int(x - self.kernel_size // 2)
                y0 = int(y - self.kernel_size // 2)

                # Calculate gaussian range
                left, right = max(0, x0), min(self.output_size[1], x0 + self.kernel_size)
                top, bottom = max(0, y0), min(self.output_size[0], y0 + self.kernel_size)

                # Calculate gaussian patch range
                patch_left = max(0, -x0)
                patch_right = self.kernel_size - max(0, x0 + self.kernel_size - self.output_size[1])
                patch_top = max(0, -y0)
                patch_bottom = self.kernel_size - max(0, y0 + self.kernel_size - self.output_size[0])

                # Apply gaussian
                heatmaps[n, k, top:bottom, left:right] = torch.from_numpy(
                    self.gaussian[patch_top:patch_bottom, patch_left:patch_right]
                )

        return heatmaps, visibility
