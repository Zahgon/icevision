from pathlib import Path

from fastcore.utils import first
from torchmetrics.detection import MeanAveragePrecision

from icevision import parsers, tfms, models
from icevision.data.dataset import Dataset
from icevision.metrics.coco_metric.coco_metric import COCOMetricType, COCOMetric
from icevision.metrics import TorchmetricsWrapper
from icevision.visualize.show_data import show_samples

import icedata

import lightning.pytorch as L
import lightning.pytorch.loggers
from torch.optim import AdamW


if __name__ == "__main__":
    data_dir = Path('/home/ppotrykus/.icevision/data/PennFudanPed')
    parser = icedata.pennfudan.parser(data_dir)
    train_records, valid_records = parser.parse()

    image_size = 512
    train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=image_size, presize=1024), tfms.A.Normalize()])
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])

    # Datasets
    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)

    samples = [train_ds[0] for _ in range(3)]
    show_samples(samples, ncols=3)

    model_type = models.torchvision.mask_rcnn
    backbone = model_type.backbones.resnet18_fpn

    model = model_type.model(backbone=backbone(pretrained=True), num_classes=len(parser.class_map))

    # Data Loaders
    num_workers = 4
    train_dl = model_type.train_dl(train_ds, batch_size=16, num_workers=num_workers, shuffle=True)
    valid_dl = model_type.valid_dl(valid_ds, batch_size=16, num_workers=num_workers, shuffle=False)

    model_type.show_batch(first(valid_dl), ncols=4)


    class LightModel(model_type.lightning.ModelAdapter):
        def configure_optimizers(self):
            return AdamW(self.model.parameters(), lr=5e-5)

    metrics = [TorchmetricsWrapper(MeanAveragePrecision(iou_type="segm"))]
    # metrics = [COCOMetric(metric_type=COCOMetricType.mask)]
    logger = [L.loggers.WandbLogger(project="icevision-update-pennfudan", group="mask_rcnn")]
    light_model = LightModel(model, metrics=metrics)

    trainer = L.Trainer(max_epochs=100, logger=logger, log_every_n_steps=5, precision="16")
    trainer.fit(light_model, train_dl, valid_dl)