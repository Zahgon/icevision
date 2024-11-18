from pathlib import Path

from icevision import models
from matplotlib import pyplot as plt
from fastcore.basics import first

import lightning.pytorch as L
import lightning.pytorch.loggers

from icevision.data.data_splitter import FolderSplitter
from icevision.data.dataset import Dataset
from icevision.parsers.vlp_parser import VLPParser
from icevision import tfms
from icevision.visualize.show_data import show_samples

def main():
    data_dir = Path.home() / "datasets/plate_localization/v2"
    parser = VLPParser(annotations_filepath=data_dir / "metadata.csv")

    train_records, valid_records = parser.parse(data_splitter=FolderSplitter(["train", "val"]), cache_filepath=data_dir/"cache_manual")

    # Create the parser
    image_size = 512
    train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=image_size, presize=512, crop_fn=None), tfms.A.Normalize()])
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])

    # Datasets
    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)

    model_type = models.custom.keypoints
    backbone = model_type.backbones.tf_efficientnet_b2
    model = model_type.model(backbone=backbone(pretrained=True), num_keypoints=1)

    # Data Loaders
    num_workers = 6
    max_epochs = 100
    torch_compile = True

    train_dl = model_type.train_dl(train_ds, batch_size=16, num_workers=num_workers, shuffle=True)
    valid_dl = model_type.valid_dl(valid_ds, batch_size=16, num_workers=num_workers, shuffle=False)

    logger = [L.loggers.WandbLogger(project="icevision-2.0-keypoints", group="fpn", notes=backbone.model_name, tags=["fp16", str(image_size)])]
    # logger = []
    light_model = model_type.lightning.ModelAdapter(model, learning_rate=1e-4, torch_compile=torch_compile)

    callbacks = [
        L.callbacks.ModelSummary(max_depth=2),
        L.callbacks.LearningRateMonitor(logging_interval='epoch'),
        L.callbacks.ModelCheckpoint(
            monitor="KeypointMetrics/PCK@0.1",
            auto_insert_metric_name=False,
            filename="{step:06d}_loss={val_loss:.2f}_PCK@0.1={KeypointMetrics/PCK@0.1:.3f}",
            verbose=True,
            mode="max",
        )
    ]
    trainer = L.Trainer(
        accelerator='gpu',
        max_epochs=max_epochs,
        logger=logger,
        callbacks=callbacks,
        precision="16-mixed",

    )
    trainer.fit(light_model, train_dl, valid_dl)
    #
    # model_type.show_results(model, valid_ds, detection_threshold=0.5)
    # plt.show()


if __name__ == "__main__":
    main()