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


class KeypointHeatmapLoss(nn.Module):
    def __init__(self, ignore_invisible=True):
        super().__init__()
        self.ignore_invisible = ignore_invisible
        self.heatmap_scale = 0.01
        self.visibility_scale = 1.0

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
        log_prob = F.log_softmax(pred.reshape(*pred.shape[:2], -1), dim=2)
        log_prob = log_prob.reshape_as(pred)

        # Compute cross entropy loss
        heatmap_loss = -(target * log_prob).sum(dim=(2, 3))

        if self.ignore_invisible:
            heatmap_loss = heatmap_loss * target_visibility

        visibility_loss = F.binary_cross_entropy_with_logits(
            pred_visibility,
            target_visibility,
            reduction='mean'
        )

        total_loss = heatmap_loss.mean() * self.heatmap_scale + visibility_loss * self.visibility_scale
        return dict(loss=total_loss, heatmap_loss=heatmap_loss.mean(), visibility_loss=visibility_loss)
