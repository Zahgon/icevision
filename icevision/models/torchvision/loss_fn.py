__all__ = ["loss_fn"]

import torch


def loss_fn(preds, targets) -> torch.Tensor:
    return sum(preds.values())
