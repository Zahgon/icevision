from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch
from matplotlib import pyplot as plt

from icevision import models, tfms
from icevision.data.data_splitter import FolderSplitter, SingleSplitSplitter
from icevision.data.dataset import Dataset
from icevision.parsers.vlp_parser import VLPParser


def main():
    data_dir = Path.home() / "datasets/plate_localization/v2.1"
    parser = VLPParser(annotations_filepath=data_dir / "metadata.csv")

    train_records, valid_records = parser.parse(data_splitter=FolderSplitter(["train", "val"]), cache_filepath=data_dir / "cache_manual_visibility")    # Create the parser
    # Create the parser
    image_size = 384
    ckpt_path = "/home/ppotrykus/Programs/icevision/icevision-2.0-keypoints/t74d6s3g/checkpoints/056000_loss=0.00_PCK@0.1=0.932.ckpt"
    torch_compile = False
    ignore_invisible = False
    n_samples = 20
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])

    outrecords = []
    for record in valid_records:
        if record.record_id in (
            "34469a08-d099-479a-af9e-15474d7e072f_000019_1.png",
            "2d593c5f-60d5-4aa4-b188-d298011311c7_000133_3.png",
            "6e5ebc4e-fb21-4feb-8b07-61f4bb435af2_000129_2.png",
            "6e5ebc4e-fb21-4feb-8b07-61f4bb435af2_000129_3.png",
            "a940d0a1-e81f-4157-bd74-4e8ed1983bbf_000143_4.png",
            "ca98a0f8-2e0b-41f2-a2d1-ea29215ceadb_000109_5.png",
            "e2caa6b3-a9d0-4a94-8104-8dcb6c588c39_000022_2.png"
        ):
            outrecords.append(record)
    # Datasets
    infer_ds = Dataset(outrecords, valid_tfms)

    model_type = models.custom.keypoints
    backbone = model_type.backbones.tf_efficientnet_b2
    model = model_type.model(backbone=backbone(pretrained=False), num_keypoints=1, use_visibility=True)

    # model = torch.compile(model)
    light_model = model_type.lightning.ModelAdapter.load_from_checkpoint(ckpt_path, model=model, torch_compile=torch_compile, ignore_invisible=ignore_invisible)

    samples_plus_losses, preds, losses_stats = model_type.interp.plot_top_losses(
        model=light_model,
        dataset=infer_ds,
        sort_by="heatmap_loss",
        n_samples=n_samples,
        ascending=True,
        color_map={"license_plate": (156.62, 160.5, 239.77)},
        show=True
    )


if __name__ == '__main__':
    main()