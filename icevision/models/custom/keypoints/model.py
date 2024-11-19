import torch
import torch.nn as nn
from torch.nn import functional as F

from icevision.models.custom.backbones import TimmBackboneConfig


def Upsample(x, size=None, scale_factor=2, mode="bilinear", align_corners=True):
    """
    Wrapper Around the Upsample Call
    """
    if mode == "nearest":
        align_corners = None
    return F.upsample(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


class ConvTransposeBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, bias=False, **kwargs):
        super(ConvTransposeBNReLU, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, output_padding=output_padding, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.activation = act_layer(True) if act_layer else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""

    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.out_channels = out_channels

        # Lateral connections
        self.laterals = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])

        # FPN connections
        self.fpn = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(len(in_channels))])

    def forward(self, features):
        # Process deepest layer first
        laterals = [conv(feature) for feature, conv in zip(features, self.laterals)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")

        # Final convolution
        outputs = [conv(lateral) for lateral, conv in zip(laterals, self.fpn)]
        return outputs


class KeypointNetwork(nn.Module):
    def __init__(self, backbone: TimmBackboneConfig, num_keypoints=17, use_fpn=True, use_visibility=False, fpn_channels=256):
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

        # FPN if requested
        self.use_fpn = use_fpn
        if use_fpn:
            self.fpn = FPN(feature_dims, fpn_channels)
            in_channels = fpn_channels
        else:
            in_channels = feature_dims[-1]

        # Visibility classifier
        self.use_visibility = use_visibility
        if use_visibility:
            self.visibility_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, fpn_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(fpn_channels, num_keypoints),
            )

        # Simple decoder using transposed convolutions
        self.decoder = nn.Sequential(
            ConvTransposeBNReLU(in_channels, 256, 4, 2, 1),
            ConvTransposeBNReLU(256, 128, 4, 2, 1),
            ConvTransposeBNReLU(128, 64, 4, 2, 1),
            ConvTransposeBNReLU(64, 32, 4, 2, 1),
            # Final layer to predict heatmaps
            nn.Conv2d(32, num_keypoints, kernel_size=1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

        # Apply FPN if used
        if self.use_fpn:
            features = self.fpn(features)
            x = features[-1]  # Use deepest level
        else:
            x = features[-1]

        # Decode to heatmaps
        heatmaps = self.decoder(x)
        if self.use_visibility:
            visibility = self.visibility_head(x)
        else:
            visibility = torch.ones_like(heatmaps[:, :, 0, 0])

        return heatmaps, visibility


def model(backbone: TimmBackboneConfig, num_keypoints: int = 17, use_fpn: bool = True, use_visibility: bool = False, fpn_channels: int = 256) -> nn.Module:
    model = KeypointNetwork(backbone=backbone, num_keypoints=num_keypoints, use_fpn=use_fpn, use_visibility=use_visibility, fpn_channels=fpn_channels)
    return model
