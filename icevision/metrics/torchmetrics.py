from typing import Dict

import torch

from icevision.core.mask import MaskArray
from icevision.metrics import Metric
import torchmetrics


class TorchmetricsWrapper(Metric):
    def __init__(self, metric: torchmetrics.Metric):
        super().__init__()
        self.metric = metric

    def finalize(self) -> Dict[str, float]:
        return self.metric.compute()

    def accumulate(self, preds):
        preds_list = []
        gt_list = []
        for prediction in preds:
            pred_record = prediction.pred
            pred_ma = pred_record.detection.mask_array
            pred_dict = {
                "masks": torch.from_numpy(pred_ma.data),
                "labels": torch.tensor(pred_record.detection.label_ids),
                "scores": torch.from_numpy(pred_record.detection.scores),
            }
            preds_list.append(pred_dict)

            gt_record = prediction.ground_truth
            gt_ma = MaskArray.from_masks(gt_record.detection.masks, gt_record.height, gt_record.width)
            gt_dict = {
                "masks": torch.from_numpy(gt_ma.data),
                "labels": torch.tensor(gt_record.detection.label_ids),
            }
            gt_list.append(gt_dict)

        self.metric.update(preds=preds_list, target=gt_list)

