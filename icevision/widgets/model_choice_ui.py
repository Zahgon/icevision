__all__ = ["ModelChoiceUI"]


from icevision import models
import ipywidgets as widgets
from IPython.display import display


class ModelChoiceUI:
    def __init__(self, task="object_detection"):
        self.task = task
        self.reset_lib_info()

        self.libraries_available = widgets.Dropdown(
            options=[],
            description="Libraries",
            disabled=False,
        )

        self.models_available = widgets.Dropdown(
            options=[""],
            description="Models",
            disabled=False,
        )

        self.backbones_available = widgets.Dropdown(
            options=[""],
            description="Backbones",
            disabled=False,
        )


    def reset_lib_info(self):
        pass

    def get_model_info(self):
        pass

    def populate_libraries(self):
        if self.task == "object_detection":
            libraries_list = ["", "MMDetection", "Ross Wightman", "Torchvision"]
        elif self.task == "mask":
            libraries_list = ["", "MMDetection", "Torchvision"]
        elif self.task == "keypoints":
            libraries_list = ["", "Torchvision"]

        self.libraries_available.options = libraries_list

    def od_library_change(self, change):
        pass

    def od_model_change(self, change):
        pass

    def od_backbone_change(self, change):
        pass

    def display(self):
        self.populate_libraries()
        self.libraries_available.observe(self.od_library_change, names="value")
        self.models_available.observe(self.od_model_change, names="value")
        self.backbones_available.observe(self.od_backbone_change, names="value")

        display(self.libraries_available)
        display(self.models_available)
        display(self.backbones_available)
