import timm

def _resnet_fpn(name: str, pretrained: bool = True, **kwargs):
    return timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )


def resnet18(pretrained: bool = True, **kwargs):
    return _resnet_fpn("resnet18", pretrained=pretrained, **kwargs)