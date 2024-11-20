import torch
import numpy as np


class KeypointHeatmapGenerator:
    def __init__(self, output_size):
        """
        Initialize heatmap generator

        Args:
            output_size (tuple): Size of output heatmap (H, W)
            sigma (int): Standard deviation for Gaussian kernel
        """
        self.output_size = output_size
        h, w = output_size
        self.sigma = w // 32
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
        visibility = torch.from_numpy(keypoints[:, :, 2]).clone()

        for n in range(batch_size):
            for k in range(num_keypoints):
                x, y, _ = keypoints[n, k]  # Ignore visibility flag

                # Convert to output space coordinates
                x = x * self.output_size[1]
                y = y * self.output_size[0]

                # Get top left corner of gaussian patch
                x0 = int(x - self.kernel_size // 2)
                y0 = int(y - self.kernel_size // 2)

                # Check if keypoint is completely out of bounds
                if (x0 >= self.output_size[1] or
                        y0 >= self.output_size[0] or
                        x0 + self.kernel_size <= 0 or
                        y0 + self.kernel_size <= 0):
                    visibility[n, k] = 0
                    continue

                # Calculate valid ranges for both the output heatmap and gaussian kernel
                left = max(0, x0)
                right = min(self.output_size[1], x0 + self.kernel_size)
                top = max(0, y0)
                bottom = min(self.output_size[0], y0 + self.kernel_size)

                patch_left = max(0, -x0)
                patch_right = min(self.kernel_size, self.output_size[1] - x0)
                patch_top = max(0, -y0)
                patch_bottom = min(self.kernel_size, self.output_size[0] - y0)

                if right > left and bottom > top:
                    try:
                        heatmaps[n, k, top:bottom, left:right] = torch.from_numpy(
                            self.gaussian[patch_top:patch_bottom, patch_left:patch_right]
                        )
                    except ValueError:
                        visibility[n, k] = 0

        return heatmaps, visibility


if __name__ == "__main__":
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    # Initialize generator with 32x32 output size
    generator = KeypointHeatmapGenerator(output_size=(512, 512))

    # Create test keypoints
    keypoints = np.array([
        # Within bounds (0.5, 0.5)
        [[0.5, 0.5, 0],
         # Partially visible (-0.1, 0.5)
         [-0.01, 0.5, 0],
         # Outside bounds (1.2, 1.2)
         [1.2, 1.2, 1]],
    ])

    # Generate heatmaps
    heatmaps, visibility = generator(keypoints)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Within bounds', 'Partially visible', 'Outside bounds']

    for i in range(3):
        axes[i].imshow(heatmaps[0, i].numpy(), cmap='hot')
        axes[i].set_title(f"{titles[i]}\nVisibility: {visibility[0, i].item()}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    print("Heatmap shapes:", heatmaps.shape)
    print("Max values:", [heatmaps[0, i].max().item() for i in range(3)])