from icevision.models.utils import filter_params
from icevision.models.interpretation import get_stats, sort_losses, get_weighted_sum

from icevision.models import torchvision
from icevision.models import ross
from icevision.models.ross import efficientdet
from icevision.models import custom

# Soft dependencies
from icevision.soft_dependencies import SoftDependencies, _SoftDependencies


if SoftDependencies.mmdet:
    from icevision.models import mmdet
    from icevision.models.checkpoint import *

# if SoftDependencies.ultralytics:
#     # HACK: yolov5 changes matplotlib backend here: https://github.com/ultralytics/yolov5/blob/77415a42e5975ea356393c9f1d5cff0ae8acae2c/utils/plots.py#L26
#     import matplotlib
#     from IPython import get_ipython

#     backend = matplotlib.get_backend()
#     from icevision.models import ultralytics

#     matplotlib.use(backend)
#     matplotlib.rcdefaults()
#     session = get_ipython()
#     shell = session.__class__.__module__
#     # HACK: yolov5 breaks automatic setting of backend setting by notebook
#     if shell in ["google.colab._shell", "ipykernel.zmqshell"]:
#         session.run_line_magic("matplotlib", "inline")

if SoftDependencies.mmseg:
    from icevision.models import mmseg

if SoftDependencies.fastai:
    from icevision.models import fastai

if SoftDependencies.sahi:
    from icevision.models import inference_sahi
