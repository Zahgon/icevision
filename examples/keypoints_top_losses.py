from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch
from matplotlib import pyplot as plt

from icevision import models, tfms
from icevision.data.data_splitter import FolderSplitter
from icevision.data.dataset import Dataset
from icevision.parsers.vlp_parser import VLPParser


def main():
    data_dir = Path.home() / "datasets/plate_localization/v2.1"
    parser = VLPParser(annotations_filepath=data_dir / "metadata.csv")

    train_records, valid_records = parser.parse(data_splitter=FolderSplitter(["train", "val"]), cache_filepath=data_dir/"cache_manual")

    # Create the parser
    image_size = 384
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])

    # Datasets
    valid_ds = Dataset(valid_records, valid_tfms)

    model_type = models.custom.keypoints
    backbone = model_type.backbones.tf_efficientnet_b0
    model = model_type.model(backbone=backbone(pretrained=False), num_keypoints=1)

    ckpt_path = "/home/ppotrykus/Programs/icevision/icevision-2.0-keypoints/hxvhcttu/checkpoints/056000_loss=1.59_PCK@0.1=0.879.ckpt"
    model = torch.compile(model)
    light_model = model_type.lightning.ModelAdapter.load_from_checkpoint(ckpt_path, model=model)

    # model = torch.compile(model)
    light_model = model_type.lightning.ModelAdapter.load_from_checkpoint(ckpt_path, model=model, torch_compile=torch_compile, ignore_invisible=ignore_invisible)

    samples_plus_losses, preds, losses_stats = model_type.interp.plot_top_losses(
        model=light_model,
        dataset=infer_ds,
        sort_by="loss_total",
        n_samples=n_samples,
        color_map={"license_plate": (156.62, 160.5, 239.77)},
        show=True
    )
    return
    preds_list = []
    for pred, sample in zip(preds, samples_plus_losses):
        p = pred.pred.detection.keypoints[0]

        gt = sample.detection.keypoints[0]
        h, w = sample.img_size.height, sample.img_size.width
        d = (
                    ((p.x - gt.x) / w) ** 2 + ((p.y - gt.y) / h) ** 2
            ) ** 0.5
        preds_list.append((sample.record_id, (p.x.item() / w, p.y.item() / h), d.item()))

    preds_df = pd.DataFrame.from_records(preds_list, columns=["image_filename", "lp_pred", "lp_distance"])
    labels = pd.read_csv(data_dir / "metadata.csv")
    run_id = Path(ckpt_path).parents[1].stem
    labels.merge(preds_df).to_csv(data_dir / f"preds/{run_id}.csv", index=False)

if __name__ == '__main__':
    main()