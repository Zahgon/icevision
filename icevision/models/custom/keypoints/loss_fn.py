import torch
from torch import nn as nn
from torch.nn import functional as F


class KeypointLoss(nn.Module):
    def __init__(self, use_target_weight=True, focal_alpha=2, focal_beta=4, scale=1):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta
        self.scale = scale

    def forward(self, pred, target, target_weight=None):
        """Focal MSE Loss for heatmap regression"""
        # Scale up the predictions and targets for larger gradients
        mse_loss = self.criterion(pred * self.scale, target * self.scale)  # Shape: (B, K, H, W)

        # Compute focal weights
        focal_weight = ((1 - pred) ** self.focal_alpha) * ((1 - target) ** self.focal_beta)

        # Apply focal weighting to MSE
        loss = mse_loss * focal_weight

        # Average over spatial dimensions
        loss = loss.mean(dim=(2, 3))  # Shape: (B, K)

        if self.use_target_weight and target_weight is not None:
            loss = loss * target_weight

        return loss.mean()


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True, topk=8):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def forward(self, pred, target, target_weight=None):
        batch_size = pred.size(0)
        num_joints = pred.size(1)

        # Calculate MSE loss per joint
        loss = self.criterion(pred, target)
        loss = loss.mean(dim=(2, 3))  # Average over spatial dims

        if self.use_target_weight and target_weight is not None:
            loss = loss * target_weight

        # Online Hard Keypoint Mining
        if self.topk:
            # Sort losses per sample
            topk_vals, _ = torch.topk(loss, k=min(self.topk, num_joints),
                                      dim=1, largest=True)
            loss = topk_vals.mean()
        else:
            loss = loss.mean()

        return loss


class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets, visibility, weights=None):
        """
        Calculate Weighted Dice Loss for continuous heatmap values
        Args:
            predictions (torch.Tensor): Predicted heatmap (B, C, H, W)
            targets (torch.Tensor): Target heatmap (B, C, H, W)
            visibility (torch.Tensor, optional): sample visibility (B, C)
            weights (torch.Tensor, optional): Pixel-wise weights (B, C, H, W)
        """
        batch_size = predictions.size(0)
        predictions = predictions.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        if weights is not None:
            weights = weights.view(batch_size, -1)
            intersection = (predictions * targets * weights).sum(dim=1)
            union = (predictions * weights).sum(dim=1) + (targets * weights).sum(dim=1)
        else:
            intersection = (predictions * targets).sum(dim=1)
            union = predictions.sum(dim=1) + targets.sum(dim=1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - (dice * visibility).mean()


class KeypointHeatmapLoss(nn.Module):
    def __init__(self, ignore_invisible=True):
        super().__init__()
        self.ignore_invisible = ignore_invisible
        self.smooth_scale = 100
        self.dice_scale = 1/20
        self.visibility_scale = 10
        self.focal_gamma = 2.0
        self.smooth_loss = nn.SmoothL1Loss(reduction='none', beta=0.1)
        self.dice_loss = WeightedDiceLoss()

    def forward(self, pred_in, target_in):
        """
        Args:
            pred: (B, K, H, W) predicted heatmaps
            target: (B, K, H, W) target heatmaps
            pred_visibility: (B, K) keypoint visibility
            target_visibility: (B, K) keypoint visibility
        """
        # Apply log softmax over spatial dimensions
        pred, pred_visibility = pred_in
        target, target_visibility = target_in

        pred = torch.sigmoid(pred)

        # compute MSE loss
        smooth_loss = self.smooth_loss(pred, target)
        smooth_loss = smooth_loss.mean((2, 3))

        if self.ignore_invisible:
            smooth_loss = (smooth_loss * target_visibility).mean()

        # compute dice loss
        dice_loss = self.dice_loss(pred, target, visibility=target_visibility)

        # step 1: Class weights
        neg_ratio = 1 - target_visibility.mean()
        pos_weight = torch.tensor([neg_ratio / (1 - neg_ratio)], device=pred.device)
        weighted_visibility_loss = F.binary_cross_entropy_with_logits(
            pred_visibility,
            target_visibility,
            reduction='none',
            pos_weight=pos_weight,
        )

        # step 2: Add focal term
        p = torch.sigmoid(pred_visibility)
        pt = p * target_visibility + (1 - p) * (1 - target_visibility)
        focal_weight = (1 - pt) ** self.focal_gamma

        visibility_loss = (weighted_visibility_loss * focal_weight).mean()

        total_loss = smooth_loss * self.smooth_scale + visibility_loss * self.visibility_scale + dice_loss * self.dice_scale
        return dict(loss=total_loss, heatmap_loss=smooth_loss, visibility_loss=visibility_loss, dice_loss=dice_loss)
