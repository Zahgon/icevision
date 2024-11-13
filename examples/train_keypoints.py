from pathlib import Path

from icevision import models
from matplotlib import pyplot as plt
from fastcore.basics import first

import lightning.pytorch as L
import lightning.pytorch.loggers
from torch.optim import AdamW
from torch.optim import lr_scheduler

from icevision.data.data_splitter import FolderSplitter
from icevision.data.dataset import Dataset
from icevision.metrics import KeypointMetrics
from icevision.parsers.vlp_parser import VLPParser
from icevision import tfms
from icevision.utils.schedulers import WarmupCosineScheduler
from icevision.visualize.show_data import show_samples

def main():
    data_dir = Path.home() / "datasets/plate_localization/v2"
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

    # Data Loaders
    num_workers = 6
    num_epochs = 100
    train_dl = model_type.train_dl(train_ds, batch_size=32, num_workers=num_workers, shuffle=True)
    valid_dl = model_type.valid_dl(valid_ds, batch_size=32, num_workers=num_workers, shuffle=False)

    class LightModel(model_type.lightning.ModelAdapter):
        def configure_optimizers(self):
            optimizer = AdamW(self.parameters(), lr=1e-4)
            scheduler = WarmupCosineScheduler(optimizer=optimizer, warmup_epochs=5, max_epochs=num_epochs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler},


    logger = [L.loggers.WandbLogger(project="icevision-2.0-keypoints", group="v2")]
    # logger = []
    # light_model = LightModel(model)
    ckpt_path = "/home/ppotrykus/Programs/icevision/examples/icevision-2.0-keypoints/mjosjx5b/checkpoints/epoch=99-step=6500.ckpt"
    light_model = LightModel.load_from_checkpoint(ckpt_path, model=model)

    monitor = "KeypointMetrics/PCK@0.5"
    callbacks = [
        L.callbacks.ModelSummary(max_depth=2),
        L.callbacks.LearningRateMonitor(logging_interval='epoch'),
        L.callbacks.ModelCheckpoint(
            monitor=monitor,
            auto_insert_metric_name=False,
            filename="{step:06d}_loss={val_loss:.2f}_PCK@0.5={KeypointMetrics/PCK@0.5:.3f}",
            verbose=True,
            mode="max",
        )
    ]
    trainer = L.Trainer(
        accelerator='gpu',
        max_epochs=num_epochs,
        logger=logger,
        # log_every_n_steps=10,
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
        callbacks=callbacks,

    )
    trainer.fit(light_model, train_dl, valid_dl)

    model_type.show_results(model, valid_ds, detection_threshold=0.5)
    plt.show()


if __name__ == "__main__":
    main()