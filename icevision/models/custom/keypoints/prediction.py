from typing import Union, Sequence, Optional, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from icevision.core.keypoints import KeyPoints
from icevision.core.record import BaseRecord
from icevision.core.record_components import ScoresRecordComponent, ImageRecordComponent, \
    InstancesLabelsRecordComponent, KeyPointsRecordComponent
from icevision.data.dataset import Dataset
from icevision.data.prediction import Prediction
from icevision.models.custom.keypoints.heatmap_decoder import HeatmapDecoder
from icevision.models.custom.keypoints.model import KeypointNetwork
from icevision.models.utils import _predict_from_dl
from icevision.utils.torch_utils import model_device, tensor_to_image
from icevision.models.custom.keypoints.dataloaders import build_infer_batch


@torch.no_grad()
def _predict_batch(
    model: KeypointNetwork,
    batch: Sequence[torch.Tensor],
    records: Sequence[BaseRecord],
    detection_threshold: float = 0.5,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:
    device = device or model_device(model)

    imgs, _ = batch
    imgs = imgs.to(device)

    # bench = DetBenchPredict(unwrap_bench(model))
    # bench = bench.eval().to(device)

    raw_preds = model(x=imgs)
    preds = convert_raw_predictions(
        batch=batch,
        raw_preds=raw_preds,
        records=records,
        detection_threshold=detection_threshold,
        keep_images=keep_images,
    )

    return preds


def predict(
    model: KeypointNetwork,
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


def _construct_prediction(
    image_tensor,
    coord,
    visibility,
    score,
    record: BaseRecord,
    keep_images: bool = False,
) -> Prediction:
    # build prediction
    xyv = torch.cat((coord, visibility.unsqueeze(1)), dim=1)
    keypoints = KeyPoints.from_xyv(xyv.cpu().view(-1), None)
    pred = BaseRecord(
        (
            ScoresRecordComponent(),
            ImageRecordComponent(),
            InstancesLabelsRecordComponent(),
            KeyPointsRecordComponent(),
        )
    )
    pred.detection.set_class_map(record.detection.class_map)
    pred.detection.add_labels_by_id([1])
    pred.detection.set_scores(score)
    pred.detection.set_keypoints([keypoints])
    if keep_images:
        record.set_img(tensor_to_image(image_tensor))

    return Prediction(pred=pred, ground_truth=record)


def convert_raw_predictions(
    batch,
    raw_preds,
    records: Sequence[BaseRecord],
    detection_threshold: float,
    keep_images: bool = False,
) -> List[Prediction]:
    xb, yb = batch
    heatmap_decoder = HeatmapDecoder(output_stride=2)
    heatmap, visibility = raw_preds
    # postprocessing
    coords, confs = heatmap_decoder(heatmap)
    visibility = nn.functional.sigmoid(visibility)
    return [
        _construct_prediction(
            image_tensor=image_tensor,
            coord=coord,
            visibility=v,
            score=score,
            record=record,
            keep_images=keep_images,
        )
        for image_tensor, coord, v, score, record in zip(xb, coords, visibility, confs, records)
    ]


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