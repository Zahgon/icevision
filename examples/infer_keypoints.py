from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from matplotlib import pyplot as plt

from icevision import models, tfms
from icevision.data.data_splitter import FolderSplitter, SingleSplitSplitter
from icevision.data.dataset import Dataset
from icevision.parsers.vlp_parser import VLPParser


def main():
    data_dir = Path.home() / "datasets/plate_localization/v2.1"
    parser = VLPParser(annotations_filepath=data_dir / "metadata.csv")

    train_records, valid_records = parser.parse(data_splitter=FolderSplitter(["train", "val"]), cache_filepath=data_dir / "cache_manual_visibility")    # Create the parser
    image_size = 384
    ckpt_path = "/home/ppotrykus/Programs/icevision/icevision-2.0-keypoints/wbc3wi4t/checkpoints/000560_loss=0.00_PCK@0.1=0.945.ckpt"
    torch_compile = False
    ignore_invisible = False

    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])
    infer_ds = Dataset(valid_records, valid_tfms)

    model_type = models.custom.keypoints
    backbone = model_type.backbones.tf_efficientnet_b2
    model = model_type.model(backbone=backbone(pretrained=False), num_keypoints=1, use_visibility=True)

    # model = torch.compile(model)
    light_model = model_type.lightning.ModelAdapter.load_from_checkpoint(ckpt_path, model=model, torch_compile=torch_compile, ignore_invisible=ignore_invisible)
    infer_dl = model_type.valid_dl(infer_ds, batch_size=32, shuffle=False)
    predictions = model_type.predict_from_dl(model, infer_dl, keep_images=True, detection_threshold=0.001)

    preds_list = []
    for prediction in predictions:
        pred = prediction.pred
        record = prediction.ground_truth

        p = pred.detection.keypoints[0]
        gt = record.detection.keypoints[0]

        h, w = record.img_size.height, record.img_size.width
        d = (
                    ((p.x - gt.x) / w) ** 2 + ((p.y - gt.y) / h) ** 2
            ) ** 0.5
        preds_list.append((record.record_id, (p.x.item() / w, p.y.item() / h), d.item(), p.visible.item()))

    preds_df = pd.DataFrame.from_records(preds_list, columns=["image_filename", "lp_pred", "lp_distance", "pred_visible"])
    labels = pd.read_csv(data_dir / "metadata.csv")
    (data_dir/"predictions").mkdir(exist_ok=True)
    run_id = Path(ckpt_path).parents[1].stem
    save_dir = data_dir / f"predictions/{run_id}.csv"
    logger.info(f"saved predictions to {save_dir}")
    labels.merge(preds_df).to_csv(save_dir, index=False)


if __name__ == '__main__':
    main()