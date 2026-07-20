__all__ = [
    "train_dl",
    "valid_dl",
    "infer_dl",
    "build_train_batch",
    "build_valid_batch",
    "build_infer_batch",
]

from icevision.imports import *
from icevision.core import *
from icevision.models.utils import *
from icevision.models.torchvision.faster_rcnn.dataloaders import _build_train_sample
from icevision.models.torchvision.faster_rcnn.dataloaders import (
    build_infer_batch,
    infer_dl,
)


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


def _build_mask_train_sample(record: RecordType):
    pass


def build_train_batch(
    records: List[RecordType],
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    pass


def build_valid_batch(
    records: List[RecordType],
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    pass
