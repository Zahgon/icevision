__all__ = [
    "predict",
    "predict_from_dl",
    "convert_raw_prediction",
    "convert_raw_predictions",
    "end2end_detect",
]

from typing import Sequence, Optional, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from icevision.core.bbox import BBox
from icevision.core.record import BaseRecord
from icevision.core.record_components import ScoresRecordComponent, ImageRecordComponent, \
    InstancesLabelsRecordComponent, BBoxesRecordComponent
from icevision.data.dataset import Dataset
from icevision.data.prediction import Prediction
from icevision.models.inference import _end2end_detect
from icevision.models.utils import _predict_from_dl
from icevision.models.torchvision.dataloaders import build_infer_batch
from icevision.utils.torch_utils import model_device, tensor_to_image
from icevision.utils.partial import partial


@torch.no_grad()
def _predict_batch(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    records: Sequence[BaseRecord],
    detection_threshold: float = 0.5,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
):
    model.eval()
    device = device or model_device(model)
    batch = [o.to(device) for o in batch]

    raw_preds = model(*batch)
    return convert_raw_predictions(
        batch=batch,
        raw_preds=raw_preds,
        records=records,
        detection_threshold=detection_threshold,
        keep_images=keep_images,
    )


def predict(
    model: nn.Module,
    dataset: Dataset,
    detection_threshold: float = 0.5,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:
    batch, records = build_infer_batch(dataset)
    return _predict_batch(
        model=model,
        batch=batch,
        records=records,
        detection_threshold=detection_threshold,
        keep_images=keep_images,
        device=device,
    )


def predict_from_dl(
    model: nn.Module,
    infer_dl: DataLoader,
    show_pbar: bool = True,
    keep_images: bool = False,
    **predict_kwargs,
):
    return _predict_from_dl(
        predict_fn=_predict_batch,
        model=model,
        infer_dl=infer_dl,
        show_pbar=show_pbar,
        keep_images=keep_images,
        **predict_kwargs,
    )


def convert_raw_predictions(
    batch,
    raw_preds,
    records: Sequence[BaseRecord],
    detection_threshold: float,
    keep_images: bool = False,
):
    return [
        convert_raw_prediction(
            sample=sample,
            raw_pred=raw_pred,
            record=record,
            detection_threshold=detection_threshold,
            keep_image=keep_images,
        )
        for sample, raw_pred, record in zip(zip(*batch), raw_preds, records)
    ]


def convert_raw_prediction(
    sample,
    raw_pred: dict,
    record: BaseRecord,
    detection_threshold: float,
    keep_image: bool = False,
):
    above_threshold = raw_pred["scores"] >= detection_threshold

    # convert predictions
    labels = raw_pred["labels"][above_threshold]
    labels = labels.detach().cpu().numpy()

    scores = raw_pred["scores"][above_threshold]
    scores = scores.detach().cpu().numpy()

    boxes = raw_pred["boxes"][above_threshold]
    bboxes = []
    for box_tensor in boxes:
        xyxy = box_tensor.cpu().numpy()
        bbox = BBox.from_xyxy(*xyxy)
        bboxes.append(bbox)

    # build prediction
    pred = BaseRecord(
        (
            ScoresRecordComponent(),
            ImageRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
        )
    )
    pred.detection.set_class_map(record.detection.class_map)
    pred.detection.set_scores(scores)
    pred.detection.set_labels_by_id(labels)
    pred.detection.set_bboxes(bboxes)
    pred.detection.above_threshold = above_threshold

    if keep_image:
        tensor_image, *_ = sample
        record.set_img(tensor_to_image(tensor_image))

    return Prediction(pred=pred, ground_truth=record)


end2end_detect = partial(_end2end_detect, predict_fn=predict)
