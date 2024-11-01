from icevision.models.torchvision.faster_rcnn import backbones
from icevision.models.torchvision.dataloaders import train_dl, valid_dl, infer_dl
from icevision.models.torchvision.faster_rcnn.model import model
from icevision.models.torchvision.faster_rcnn.prediction import *
from icevision.models.torchvision.faster_rcnn.show_batch import *
from icevision.models.torchvision.faster_rcnn.show_results import *


# Soft dependencies
from icevision.soft_dependencies import SoftDependencies

if SoftDependencies.fastai:
    import icevision.models.torchvision.faster_rcnn.fastai

if SoftDependencies.pytorch_lightning:
    import icevision.models.torchvision.faster_rcnn.lightning
