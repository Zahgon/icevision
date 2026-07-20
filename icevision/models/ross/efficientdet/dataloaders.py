__all__ = [
    "build_train_batch",
    "build_valid_batch",
    "build_infer_batch",
    "train_dl",
    "valid_dl",
    "infer_dl",
]

from icevision.imports import *
from icevision.models.utils import *


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


def infer_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    """A `DataLoader` with a custom `collate_fn` that batches items as required for inferring the model.

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
        build_batch=build_infer_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def build_train_batch(records):
    pass


def build_valid_batch(records):
    pass


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
    batch_images, batch_sizes, batch_scales = zip(
        *(process_infer_record(record) for record in records)
    )

    batch_images = torch.stack(batch_images)
    batch_sizes = tensor(batch_sizes, dtype=torch.float32)
    batch_scales = tensor(batch_scales, dtype=torch.float32)

    targets = dict(img_size=batch_sizes, img_scale=batch_scales)

    return (batch_images, targets), records


def process_train_record(record) -> tuple:
    pass


def process_infer_record(record) -> tuple:
    """Extracts information from record and prepares a format required by the EffDet inference"""
    image = im2tensor(record.img)
    n_channels, image_height, image_width = image.shape
    image_scale = 1.0
    return image, (image_width, image_height), image_scale
