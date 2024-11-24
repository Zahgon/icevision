from pathlib import Path

from icevision import models
from matplotlib import pyplot as plt
from fastcore.basics import first

import lightning.pytorch as L
import lightning.pytorch.loggers

from icevision.data.data_splitter import FolderSplitter
from icevision.data.dataset import Dataset
from icevision.models.utils import freeze, unfreeze
from icevision.parsers.vlp_parser import VLPParser
from icevision import tfms
from icevision.visualize.show_data import show_samples

def main():
    # hparams
    use_heavy_augs = False
    learning_rate = 1e-4
    max_epochs = 100

    image_size = 384
    batch_size = 32
    num_workers = 6

    skip_unaudited = True
    torch_compile = False
    ignore_invisible = True

    data_dir = Path.home() / "datasets/plate_localization/v2.2"
    parser = VLPParser(annotations_filepath=data_dir / "metadata.csv", skip_unaudited=skip_unaudited)

    train_records, valid_records = parser.parse(data_splitter=FolderSplitter(["train", "val"]), cache_filepath=data_dir/"cache_audited")

    # Create the parser
    train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=image_size, presize=512, crop_fn=None, include_heavy=use_heavy_augs), tfms.A.Normalize()])
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])

    # Datasets
    train_ds = Dataset(valid_records, train_tfms)
    valid_ds = Dataset(train_records, valid_tfms)

    model_type = models.custom.keypoints
    backbone = model_type.backbones.tf_efficientnet_b0
    model = model_type.model(backbone=backbone(pretrained=True), num_keypoints=1, use_visibility=True)

    train_dl = model_type.train_dl(train_ds, point_ratio=cfg.model.point_ratio, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_dl = model_type.valid_dl(valid_ds, point_ratio=cfg.model.point_ratio, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    logger = L.loggers.WandbLogger(
        project="icevision-2.0-keypoints",
        group="v2.2",
        notes=f"{backbone.model_name} + dice loss",
        tags=["fp16", str(image_size)]
    )

    # logger = None

    # ckpt_path = "/home/ppotrykus/Programs/icevision/icevision-2.0-keypoints/92yzbaf1/checkpoints/042350_loss=0.00_PCK@0.1=0.847.ckpt"
    # light_model = model_type.lightning.ModelAdapter.load_from_checkpoint(ckpt_path, model=model, torch_compile=torch_compile, ignore_invisible=ignore_invisible)
    # freeze(light_model.model.parameters())
    # unfreeze(light_model.model.visibility_head.parameters())

    light_model = model_type.lightning.ModelAdapter(model, learning_rate=learning_rate, torch_compile=torch_compile, ignore_invisible=ignore_invisible)

    callbacks = [
        L.callbacks.ModelSummary(max_depth=2),
        L.callbacks.LearningRateMonitor(logging_interval='epoch'),
        L.callbacks.ModelCheckpoint(
            dirpath=Path.home() / "Programs/icevision/icevision-2.0-keypoints" / logger.experiment.id,
            save_last=True,
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
        # enable_checkpointing=False,
        # limit_val_batches=0.1,
        # limit_train_batches=0.1,
        precision="16-mixed",

    )
    trainer.fit(light_model, train_dl, valid_dl)
    #
    # model_type.show_results(model, valid_ds, detection_threshold=0.5)
    # plt.show()


if __name__ == "__main__":
    main()