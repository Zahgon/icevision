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

    # Data Loaders
    num_workers = 8
    valid_dl = model_type.valid_dl(valid_ds, batch_size=32, num_workers=num_workers, shuffle=False)

    ckpt_path = "/home/ppotrykus/Programs/icevision/icevision-2.0-keypoints/hxvhcttu/checkpoints/056000_loss=1.59_PCK@0.1=0.879.ckpt"
    model = torch.compile(model)
    light_model = model_type.lightning.ModelAdapter.load_from_checkpoint(ckpt_path, model=model)
    trainer = L.Trainer(accelerator='gpu', precision="16-mixed",)
    trainer.test(light_model, valid_dl)
    # samples_plus_losses, preds, losses_stats = model_type.interp.plot_top_losses(model=light_model, dataset=valid_ds, sort_by="loss_total", n_samples=16)
    # preds_list = []
    # for pred, sample in zip(preds, samples_plus_losses):
    #     p = pred.pred.detection.keypoints[0]
    #
    #     gt = sample.detection.keypoints[0]
    #     h, w = sample.img_size.height, sample.img_size.width
    #     d = (
    #                 ((p.x - gt.x) / w) ** 2 + ((p.y - gt.y) / h) ** 2
    #         ) ** 0.5
    #     preds_list.append((sample.record_id, (p.x.item() / w, p.y.item() / h), d.item()))
    #
    # preds_df = pd.DataFrame.from_records(preds_list, columns=["image_filename", "lp_pred", "lp_distance"])
    # labels = pd.read_csv(data_dir / "metadata.csv")
    # run_id = Path(ckpt_path).parents[1].stem
    # labels.merge(preds_df).to_csv(data_dir / f"preds/{run_id}.csv", index=False)


if __name__ == "__main__":
    main()