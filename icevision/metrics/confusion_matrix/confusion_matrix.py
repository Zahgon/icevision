__all__ = ["SimpleConfusionMatrix"]

from icevision.data.prediction import Prediction
from icevision.metrics.metric import Metric
from icevision.imports import *
from icevision.metrics.confusion_matrix.confusion_matrix_utils import *
import PIL


class MatchingPolicy(Enum):
    BEST_SCORE = 1
    BEST_IOU = 2


class SimpleConfusionMatrix(Metric):
    def __init__(
        self,
        iou_threshold: float = 0.5,
        policy: MatchingPolicy = MatchingPolicy.BEST_SCORE,
        print_summary: bool = False,
    ):
        super(SimpleConfusionMatrix, self).__init__()
        self.print_summary = print_summary
        self.target_labels = []
        self.predicted_labels = []
        self._iou_threshold = iou_threshold
        self._policy = policy
        self.class_map = None
        self.confusion_matrix: sklearn.metrics.confusion_matrix = None

    def _reset(self):
        pass

    def accumulate(self, preds: Collection[Prediction]):
        pass

    def finalize(self):
        pass

    def plot(
        self,
        normalize: Optional[str] = None,
        xticks_rotation="vertical",
        values_format: str = None,
        cmap: str = "PuBu",
        figsize: int = 11,
        **display_args,
    ):
        pass

    def _fig2img(self, fig):
        pass

    def _maybe_normalize(self, cm, normalize):
        pass

    def log(self, logger_object) -> None:
        return
