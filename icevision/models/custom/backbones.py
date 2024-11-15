import timm
from icevision.models.backbone_config import BackboneConfig


class TimmBackboneConfig(BackboneConfig):
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, pretrained: bool = True, **kwargs):
        """Completes the configuration of the backbone

        # Arguments
            pretrained: If True, use a pretrained backbone (on COCO).
            By default it is set to True: this is generally used when training a new model (transfer learning).
            `pretrained = False`  is used during inference (prediction) for cases where the users have their own pretrained weights.
        """
        self.pretrained = pretrained
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )

        return self


resnet18 = TimmBackboneConfig("resnet18")
tf_efficientnet_b0 = TimmBackboneConfig("tf_efficientnet_b0")  # small
tf_efficientnet_b2 = TimmBackboneConfig("tf_efficientnet_b2")  # medium
tf_efficientnet_b4 = TimmBackboneConfig("tf_efficientnet_b4")  # large
