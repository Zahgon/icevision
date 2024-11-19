import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

from icevision.models.custom.keypoints.heatmap_generator import KeypointHeatmapGenerator
from icevision.models.utils import transform_dl


def process_train_record(record) -> tuple:
    """Extracts information from record and prepares a format required by the EffDet training"""
    image = to_tensor(record.img)
    # background and dummy if no label in record
    bboxes = (
        [bbox.yxyx for bbox in record.detection.bboxes]
        if len(record.detection.label_ids) > 0
        else [[0, 0, 0, 0]]
    )
    return image, bboxes


def train_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    """A `DataLoader` with a custom `collate_fn` that batches items as required for training the model.

    # Arguments
        dataset: Possibly a `Dataset` object, but more generally, any `Sequence` that returns records.
        batch_tfms: Transforms to be applied at the batch level.
        **dataloader_kwargs: Keyword arguments that will be internally passed to a Pytorch `DataLoader`.
        The parameter `collate_fn` is already defined internally and cannot be passed here.

    # Returns
        A Pytorch `DataLoader`.
    """
    return transform_dl(
        dataset=dataset,
        build_batch=build_train_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def valid_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    """A `DataLoader` with a custom `collate_fn` that batches items as required for validating the model.

    # Arguments
        dataset: Possibly a `Dataset` object, but more generally, any `Sequence` that returns records.
        batch_tfms: Transforms to be applied at the batch level.
        **dataloader_kwargs: Keyword arguments that will be internally passed to a Pytorch `DataLoader`.
        The parameter `collate_fn` is already defined internally and cannot be passed here.

    # Returns
        A Pytorch `DataLoader`.
    """
    return transform_dl(
        dataset=dataset,
        build_batch=build_valid_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )

def build_train_batch(records):
    """Builds a batch in the format required by the model when training.

    # Arguments
        records: A `Sequence` of records.

    # Returns
        A tuple with two items. The first will be a tuple like `(images, targets)`,
        in the input format required by the model. The second will be a list
        of the input records.

    # Examples

    Use the result of this function to feed the model.
    ```python
    batch, records = build_train_batch(records)
    outs = model(*batch)
    ```
    """
    batch_raw_keypoints = []
    batch_images = []
    for record in records:
        norm = np.array([record.img_size.height, record.img_size.width, 1.0])
        record_raw_keypoints = [keypoint.xyv/norm for keypoint in record.detection.keypoints]
        batch_raw_keypoints.extend(record_raw_keypoints)
        batch_images.append(to_tensor(record.img))

    heatmap_shape = record.img_size.height // 2, record.img_size.width // 2
    hmgen = KeypointHeatmapGenerator(heatmap_shape)
    # convert to tensors
    batch_data = hmgen(np.array(batch_raw_keypoints))
    batch_images = torch.stack(batch_images)

    return (batch_images, batch_data), records


def build_valid_batch(records):
    """Builds a batch in the format required by the model when validating.

    # Arguments
        records: A `Sequence` of records.

    # Returns
        A tuple with two items. The first will be a tuple like `(images, targets)`,
        in the input format required by the model. The second will be a list
        of the input records.

    # Examples

    Use the result of this function to feed the model.
    ```python
    batch, records = build_valid_batch(records)
    outs = model(*batch)
    ```
    """
    return build_train_batch(records)


def build_infer_batch(records):
    """Builds a batch in the format required by the model when doing inference.

    # Arguments
        records: A `Sequence` of records.

    # Returns
        A tuple with two items. The first will be a tuple like `(images, targets)`,
        in the input format required by the model. The second will be a list
        of the input records.
    Use the result of this function to feed the model.
    ```python
    batch, records = build_infer_batch(records)
    outs = model(*batch)
    ```
    """
    batch_images = []
    for record in records:
        batch_images.append(to_tensor(record.img))

    # convert to tensors
    batch_images = torch.stack(batch_images)
    # batch_sizes = torch.tensor(batch_sizes, dtype=torch.float32)

    # convert to EffDet interface

    return (batch_images, None), records