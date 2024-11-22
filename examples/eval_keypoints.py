from pathlib import Path

import pandas as pd
import torch

from icevision import models

import lightning.pytorch as L
from matplotlib import pyplot as plt

from icevision.data.data_splitter import FolderSplitter
from icevision.data.dataset import Dataset
from icevision.parsers.vlp_parser import VLPParser
from icevision import tfms


def main():
    data_dir = Path.home() / "datasets/plate_localization/v2.2"
    parser = VLPParser(annotations_filepath=data_dir / "metadata.csv")

    train_records, valid_records = parser.parse(data_splitter=FolderSplitter(["train", "val"]), cache_filepath=data_dir/"cache_manual_visibility")

    # Create the parser
    image_size = 384
    noisy_ckpt = "/home/ppotrykus/Programs/icevision/icevision-2.0-keypoints/92yzbaf1/checkpoints/042350_loss=0.00_PCK@0.1=0.847.ckpt"
    subset_ckpt = "/home/ppotrykus/Programs/icevision/icevision-2.0-keypoints/9nxwzqxt/checkpoints/008400_loss=0.00_PCK@0.1=0.723.ckpt"
    torch_compile = False
    ignore_invisible = True
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])

    # Datasets
    valid_ds = Dataset(valid_records, valid_tfms)

    model_type = models.custom.keypoints
    backbone = model_type.backbones.tf_efficientnet_b0
    model = model_type.model(backbone=backbone(pretrained=False), num_keypoints=1, use_visibility=True)

    # Data Loaders
    num_workers = 0
    valid_dl = model_type.valid_dl(valid_ds, batch_size=16, num_workers=num_workers, shuffle=False)

    light_model = model_type.lightning.ModelAdapter.load_from_checkpoint(noisy_ckpt, model=model, torch_compile=torch_compile, ignore_invisible=ignore_invisible)
    trainer = L.Trainer(accelerator='gpu', precision="16-mixed", enable_checkpointing=False, limit_test_batches=1.0)
    # trainer.fit(light_model, valid_dl)
    trainer.test(light_model, valid_dl)


if __name__ == "__main__":
    main()