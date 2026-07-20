__all__ = ["Yolov5Callback"]

from icevision.engines.fastai import *
from icevision.models.ultralytics import yolov5


class Yolov5Callback(fastai.Callback):
    def before_batch(self):
        pass

    def after_pred(self):
        pass
