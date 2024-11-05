import torch
from fastcore.xtras import dict2obj
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.loss import v8DetectionLoss


def loss_fn(preds, targets, model) -> torch.Tensor:
    "ultralytics loss_fn requires model arg"
    wrapped_model_dict = dict(model=model.model, args=DEFAULT_CFG_DICT, parameters=model.parameters)
    wrapped_model = dict2obj(wrapped_model_dict)
    result = v8DetectionLoss(wrapped_model)(preds, targets)
    return result[0]
