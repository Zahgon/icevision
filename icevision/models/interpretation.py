__all__ = [
    "sort_losses",
    "get_stats",
    "get_weighted_sum",
    "add_annotations",
    "get_samples_losses",
    "_move_to_device",
]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.visualize.show_data import show_preds
from icevision.core.record_components import LossesRecordComponent


def get_weighted_sum(sample, weights):
    pass


def sort_losses(
    samples: List[dict], preds: List[dict], by: Union[str, dict] = "loss_total"
) -> Tuple[List[dict], List[dict], List[str]]:
    pass


def get_stats(l: List) -> dict:
    pass


def _move_to_device(x, y, device):

    if isinstance(y, list):
        x = [o.to(device) for o in x]
        y = [
            {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in o.items()
            }
            for o in y
        ]
    elif isinstance(
        y, dict
    ):  # this covers the efficientdet case in which `y` is a dict of Union[list, tensor] and not a list of dicts and `x` is a Tensor and not a list of Tensors
        x = x.to(device) if x is not None else x
        for k in y.keys():
            if isinstance(y[k], list):
                y[k] = [
                    o.to(device) if isinstance(o, torch.Tensor) else o for o in y[k]
                ]
            else:
                y[k] = y[k].to(device) if isinstance(y[k], torch.Tensor) else y[k]
    else:
        return x.to(device), y.to(device)
    return x, y


def _prepend_str(d: dict, s: str):
    pass


class Interpretation:
    def __init__(self, losses_dict, valid_dl, infer_dl, predict_from_dl):
        self.losses_dict = losses_dict
        self.valid_dl = valid_dl
        self.infer_dl = infer_dl
        self.predict_from_dl = predict_from_dl

    def _rename_losses(self, losses_dict):
        pass

    def _sum_losses(self, losses_dict):
        pass

    def _loop(self, dl, model, losses_stats, device):
        pass

    def get_losses(
        self,
        model: nn.Module,
        dataset: Dataset,
    ) -> Tuple[List[dict], dict]:
        pass

    def plot_top_losses(
        self,
        model: nn.Module,
        dataset: Dataset,
        sort_by: str = "loss_total",
        n_samples: int = 5,
        batch_size: int = 8,
    ) -> Tuple[List[dict], List[dict], dict]:
        pass


def add_annotations(samples: List[dict]) -> List[dict]:
    pass


def get_samples_losses(samples_plus_losses):
    def _get_info(sample):
        d = {k: v for k, v in sample.losses.items() if "loss" in k}
        d["filepath"] = sample.filepath
        return d

    return [_get_info(l) for l in samples_plus_losses]
