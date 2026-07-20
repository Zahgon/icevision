__all__ = ["IceSahiModel"]

from types import ModuleType
from icevision.imports import *
from icevision.imports import *
from icevision.core import *
from icevision.data import *
from icevision.utils.imageio import *
from icevision.visualize.draw_data import *
from icevision.visualize.utils import *
from icevision.tfms.albumentations import albumentations_adapter


from sahi.model import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.predict import get_sliced_prediction as sahi_get_sliced_prediction


class IceSahiModel(DetectionModel):
    def __init__(
        self,
        model_type: ModuleType,
        model: torch.nn.Module,
        class_map: ClassMap,
        tfms: albumentations_adapter.Adapter,
        confidence_threshold: float = 0.5,
    ):
        super().__init__(
            model_path="", confidence_threshold=confidence_threshold, load_at_init=False
        )

        self.model_type = model_type
        self.class_map = class_map
        self.confidence_threshold = confidence_threshold
        self.model = model
        self.tfms = tfms
        self.category_mapping = self.class_map._class2id

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        pass

    @property
    def num_categories(self):
        pass

    @property
    def has_mask(self):
        pass

    @property
    def category_names(self):
        pass

    def get_sliced_prediction(
        self,
        image: Union[PIL.Image.Image, Path, str],
        keep_sahi_format: bool = False,
        display_label: bool = True,
        display_bbox: bool = True,
        display_score: bool = True,
        font_path: Optional[os.PathLike] = get_default_font(),
        font_size: Union[int, float] = 10,
        return_as_pil_img=True,
        return_img=True,
        **kwargs
    ):
        pass

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        pass
