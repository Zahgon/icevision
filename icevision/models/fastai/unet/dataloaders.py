__all__ = ["train_dl", "valid_dl", "infer_dl", "build_train_batch", "build_infer_batch"]

from typing import Sequence

from torch.utils.data import DataLoader
import torch
from torchvision.transforms.functional import to_tensor

from icevision.core.record import BaseRecord
from icevision.models.utils import transform_dl


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
    return train_dl(dataset, batch_tfms=None, **dataloader_kwargs)


def infer_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    return transform_dl(
        dataset=dataset,
        build_batch=build_infer_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def build_train_batch(records: Sequence[BaseRecord]):
    tensor_images, tensor_masks = [], []
    for record in records:
        # can be optimzed to be converted to tensor once at the end
        tensor_images.append(to_tensor(record.img))
        tensor_masks.append(
            torch.tensor(record.segmentation.mask_array.data).long().squeeze()
        )

    tensor_images = torch.stack(tensor_images)
    tensor_masks = torch.stack(tensor_masks)

    return (tensor_images, tensor_masks), records


def build_infer_batch(records: Sequence[BaseRecord]):
    tensor_images = []
    for record in records:
        tensor_images.append(to_tensor(record.img))

    tensor_images = torch.stack(tensor_images)
    return (tensor_images,), records
