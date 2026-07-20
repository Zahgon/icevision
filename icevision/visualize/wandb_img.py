__all__ = ["wandb_img_preds", "wandb_image"]


from typing import List

import wandb
from icevision import BaseRecord, BBox
from icevision.data.prediction import Prediction


def wandb_img_preds(
    preds: List[Prediction], add_ground_truth: bool = False
) -> List[wandb.Image]:
    pass


def bbox_wandb(bbox: BBox, label_id: int, label_name: str, score=None) -> dict:
    pass


def wandb_image(pred: Prediction, add_ground_truth: bool = False) -> wandb.Image:
    pass
