from collections import OrderedDict
from pathlib import Path

import torch
from matplotlib import pyplot as plt

from icevision import models, tfms
from icevision.data.data_splitter import FolderSplitter
from icevision.data.dataset import Dataset
from icevision.parsers.vlp_parser import VLPParser


def main():
    data_dir = Path.home() / "datasets/plate_localization/v1"
    parser = VLPParser(annotations_filepath=data_dir / "metadata.csv")

    train_records, valid_records = parser.parse(data_splitter=FolderSplitter(["train", "val"]))

    # Create the parser
    image_size = 384
    train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=image_size, presize=512, crop_fn=None), tfms.A.Normalize()])
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])

    # Datasets
    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)
    model_type = models.custom.keypoints
    backbone = model_type.backbones.resnet18
    model = model_type.model(backbone=backbone(pretrained=True), num_keypoints=1)

    # load model
    model_checkpoint = "/home/ppotrykus/Programs/icevision/examples/icevision-2.0-keypoints/5j0cpwrq/checkpoints/076200_loss=0.00_PCK@0.5=0.970.ckpt"
    light_model = model_type.lightning.ModelAdapter.load_from_checkpoint(model_checkpoint, model=model)
    samples_plus_losses, preds, losses_stats = model_type.interp.plot_top_losses(model=light_model, dataset=valid_ds, sort_by="loss_total", n_samples=4)
    plt.show()


if __name__ == '__main__':
    main()