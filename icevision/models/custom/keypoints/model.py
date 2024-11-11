import torch
import torch.nn as nn

from icevision.models.custom.backbones import TimmBackboneConfig


class KeypointNetwork(nn.Module):
    def __init__(self, backbone: TimmBackboneConfig, num_keypoints=17, pretrained=True):
        """
        Simple keypoint detection network using a timm backbone

        Args:
            backbone_name (str): Name of the timm backbone to use
            num_keypoints (int): Number of keypoints to detect
            pretrained (bool): Whether to use pretrained weights
        """
        super().__init__()

        self.backbone = backbone.backbone

        # Get backbone feature dimensions
        dummy_input = torch.randn(1, 3, 256, 256)
        features = self.backbone(dummy_input)
        feature_dims = [feat.shape[1] for feat in features]

        # Simple decoder using transposed convolutions
        self.decoder = nn.Sequential(
            # Start from smallest feature map
            nn.ConvTranspose2d(feature_dims[-1], 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Final layer to predict heatmaps
            nn.Conv2d(32, num_keypoints, kernel_size=1)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input image of shape (batch_size, 3, H, W)

        Returns:
            torch.Tensor: Keypoint heatmaps of shape (batch_size, num_keypoints, H/2, W/2)
        """
        # Get features from backbone
        features = self.backbone(x)

        # Use deepest features for decoding
        x = features[-1]

        # Decode to heatmaps
        heatmaps = self.decoder(x)

        return heatmaps


def model(
    backbone: TimmBackboneConfig,
    num_keypoints: int = 17,
    **kwargs,
) -> nn.Module:
    model = KeypointNetwork(backbone=backbone, num_keypoints=num_keypoints, pretrained=True)
    return model