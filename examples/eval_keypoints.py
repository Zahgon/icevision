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
    data_dir = Path.home() / "datasets/plate_localization/v2"
    parser = VLPParser(annotations_filepath=data_dir / "metadata.csv")

    train_records, valid_records = parser.parse(data_splitter=FolderSplitter(["train", "val"]), cache_filepath=data_dir/"cache_manual")

    # Create the parser
    image_size = 512
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])

    # Datasets
    valid_ds = Dataset(valid_records, valid_tfms)

    model_type = models.custom.keypoints
    backbone = model_type.backbones.tf_efficientnet_b2
    model = model_type.model(backbone=backbone(pretrained=False), num_keypoints=1)

    # Data Loaders
    num_workers = 8
    valid_dl = model_type.valid_dl(valid_ds, batch_size=32, num_workers=num_workers, shuffle=False)

    ckpt_path = "/home/ppotrykus/Programs/icevision/icevision-2.0-keypoints/r1zl02br/checkpoints/056000_loss=1.66_PCK@0.1=0.879.ckpt"
    model = torch.compile(model)
    light_model = model_type.lightning.ModelAdapter.load_from_checkpoint(ckpt_path, model=model)
    trainer = L.Trainer(accelerator='gpu', precision="16-mixed",)
    trainer.test(light_model, valid_dl)


if __name__ == "__main__":
    main()