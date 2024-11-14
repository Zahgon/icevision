from typing import List

import torch
from icevision.metrics import Metric


class KeypointMetrics(Metric):
    def __init__(self, distance_thresholds: List[float] = None):
        """
        Initialize keypoint metrics calculator

        Args:
            distance_thresholds List[float]: Distance threshold for PCK
        """
        super().__init__()
        self.pred = []
        self.target = []
        if distance_thresholds is None:
            distance_thresholds = [0.01, 0.05, 0.1, 0.2]
        self.distance_thresholds = distance_thresholds

    def _get_distance(self, pred, target):
        """Calculate Euclidean distance between predicted and target keypoints"""
        return torch.sqrt(((pred - target) ** 2).sum(dim=-1))

    def _get_normalization(self, keypoints, method='head'):
        """
        Get normalization factor for PCK

        Args:
            keypoints (torch.Tensor): Keypoints of shape (N, K, 2)
            method (str): 'head' or 'torso' for normalization
        """
        if method == 'head':
            # Assuming keypoints[..., 9:10] is head size
            # This should be adjusted based on your keypoint format
            head_size = torch.norm(keypoints[:, 9] - keypoints[:, 8], dim=1)
            return head_size
        elif method == 'torso':
            # Assuming keypoints[..., 5:6] is shoulder and keypoints[..., 11:12] is hip
            # This should be adjusted based on your keypoint format
            torso_size = torch.norm(
                (keypoints[:, 5] + keypoints[:, 6]) / 2 -
                (keypoints[:, 11] + keypoints[:, 12]) / 2,
                dim=1
            )
            return torso_size
        else:
            return torch.ones(keypoints.shape[0])

    def pck(self, pred, target, visible=None):
        """
        Calculate PCK (Percentage of Correct Keypoints)

        Args:
            pred (torch.Tensor): Predicted keypoints (N, K, 2)
            target (torch.Tensor): Target keypoints (N, K, 2)
            visible (torch.Tensor): Keypoint visibility (N, K)

        Returns:
            dict: PCK metrics at different thresholds
        """
        normalized_dist = self._get_distance(torch.tensor(pred), torch.tensor(target))

        if visible is None:
            visible = torch.ones_like(normalized_dist)

        metrics = {}
        for t in self.distance_thresholds:
            correct = (normalized_dist <= t) & (visible > 0)
            pck = (correct & (visible > 0)).sum() / (visible > 0).sum()
            metrics[f'PCK@{t}'] = pck.item()

        return metrics

    def accumulate(self, preds):
        for pred in preds:
            gt = pred.ground_truth
            x, y = gt.detection.keypoints[0].xy[0]
            x /= gt.img_size.width
            y /= gt.img_size.height
            self.target.append((x, y))
            p = pred.pred
            x, y = p.detection.keypoints[0].xy[0]
            x /= gt.img_size.width
            y /= gt.img_size.height
            self.pred.append((x, y))

    def finalize(self):
        """Calculate all metrics at once"""
        metrics = self.pck(self.pred, self.target)
        return metrics
