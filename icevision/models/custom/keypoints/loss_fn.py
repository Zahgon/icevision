from torch import nn as nn


class KeypointLoss(nn.Module):
    def __init__(self, use_target_weight=True):
        """
        Loss function for keypoint detection

        Args:
            use_target_weight (bool): Whether to use keypoint visibility as weights
        """
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight

    def forward(self, pred, target, target_weight=None):
        """
        Compute loss between predicted and target heatmaps

        Args:
            pred (torch.Tensor): Predicted heatmaps (N, K, H, W)
            target (torch.Tensor): Target heatmaps (N, K, H, W)
            target_weight (torch.Tensor, optional): Keypoint visibility weights (N, K)

        Returns:
            torch.Tensor: Computed loss
        """
        batch_size = pred.size(0)
        num_keypoints = pred.size(1)

        # Compute MSE loss for each keypoint
        loss = self.criterion(pred, target)
        loss = loss.mean(dim=(2, 3))  # Average over spatial dimensions

        if self.use_target_weight and target_weight is not None:
            # Apply keypoint visibility weights
            loss = loss * target_weight

        return loss.mean()
