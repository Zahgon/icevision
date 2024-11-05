__all__ = ["model"]

from types import MethodType
from typing import Optional, List

import torch
from torch import nn
from ultralytics import YOLO

from ultralytics.nn import DetectionModel
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.torch_utils import intersect_dicts

from icevision.models.ultralytics.yolov5.backbones import YoloV5BackboneConfig
from icevision.utils.data_dir import get_root_dir
from icevision.utils.torch_utils import check_all_model_params_in_groups2

yolo_dir = get_root_dir() / "yolo"
yolo_dir.mkdir(exist_ok=True)


def model(
    backbone: YoloV5BackboneConfig,
    num_classes: int,
    img_size: int,  # must be multiple of 32
    device: Optional[torch.device] = None,
) -> nn.Module:
    model_name = backbone.model_name
    pretrained = backbone.pretrained

    # this is to remove background from ClassMap as discussed
    # here: https://github.com/ultralytics/yolov5/issues/2950
    # and here: https://discord.com/channels/735877944085446747/782062040168267777/836692604224536646
    # so we should pass `num_classes=parser.class_map.num_classes`
    num_classes -= 1

    model = DetectionModel(f"{model_name}.yaml", nc=num_classes)

    if pretrained:
        model_fp = attempt_download_asset(f"{model_name}.pt", dir=yolo_dir)
        pretrained_model = YOLO(yolo_dir/model_fp)
        state_dict = intersect_dicts(pretrained_model.model.state_dict(), model.state_dict())  # intersect
        model.load_state_dict(state_dict, strict=False)  # load

    def param_groups_fn(model: nn.Module) -> List[List[nn.Parameter]]:
        spp_index = [
            i + 1
            for i, layer in enumerate(model.model.children())
            if layer._get_name() == "SPPF"
        ][0]
        backbone = list(model.model.children())[:spp_index]
        neck = list(model.model.children())[spp_index:-1]
        head = list(model.model.children())[-1]

        layers = [nn.Sequential(*backbone), nn.Sequential(*neck), nn.Sequential(head)]

        param_groups = [list(group.parameters()) for group in layers]
        check_all_model_params_in_groups2(model.model, param_groups)

        return param_groups

    model.param_groups = MethodType(param_groups_fn, model)

    return model
