from pathlib import Path
import icedata
from fastcore.utils import first
from icevision import parsers, tfms, models
from icevision.data.dataset import Dataset
from icevision.metrics.coco_metric.coco_metric import COCOMetricType, COCOMetric
from icevision.visualize.show_data import show_samples

import lightning.pytorch as L
import lightning.pytorch.loggers
from torch.optim import AdamW
from matplotlib import pyplot as plt



if __name__ == "__main__":
    data_dir = icedata.biwi.load_data()
    # Create the parser
    train_ds, valid_ds = icedata.biwi.dataset(data_dir)

    # image_size = 384
    # train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=image_size, presize=512), tfms.A.Normalize()])
    # valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])
    #
    # # Datasets
    # train_ds = Dataset(train_records, train_tfms)
    # valid_ds = Dataset(valid_records, valid_tfms)

    samples = [train_ds[0] for _ in range(3)]
    # show_samples(samples, ncols=3)

    model_type = models.custom.keypoints
    backbone = model_type.backbones.resnet18(pretrained=True)
    model = model_type.model(backbone=backbone, num_keypoints=1, num_classes=icedata.biwi.NUM_CLASSES)

    # Data Loaders
    num_workers = 8
    train_dl = model_type.train_dl(train_ds, batch_size=16, num_workers=num_workers, shuffle=True)
    valid_dl = model_type.valid_dl(valid_ds, batch_size=16, num_workers=num_workers, shuffle=False)

    # model_type.show_batch(first(valid_dl), ncols=4)
    # plt.show()


    class LightModel(model_type.lightning.ModelAdapter):
        def configure_optimizers(self):
            return AdamW(self.parameters(), lr=1e-4)


    metrics = []
    # logger = [L.loggers.WandbLogger(project="icevision-2.0-effdet")]
    logger = []
    light_model = LightModel(model, metrics=metrics)

    callbacks = L.callbacks.ModelSummary(max_depth=2)
    trainer = L.Trainer(max_epochs=5, logger=logger, log_every_n_steps=5, callbacks=callbacks)
    trainer.fit(light_model, train_dl, valid_dl)