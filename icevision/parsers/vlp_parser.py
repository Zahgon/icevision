from pathlib import Path
from typing import List, Hashable
import pandas as pd
from fastcore.basics import ifnone

from icevision.core.class_map import ClassMap
from icevision.core.exceptions import AbortParseRecord
from icevision.core.keypoints import KeypointsMetadata, KeyPoints
from icevision.core.record import BaseRecord
from icevision.core.record_components import FilepathRecordComponent, InstancesLabelsRecordComponent, KeyPointsRecordComponent
from icevision.data.data_splitter import FolderSplitter
from icevision.parsers import Parser
from icevision.utils.imageio import get_img_size, ImgSize

from matplotlib import pyplot as plt
from icevision.visualize.show_data import show_records


class VLPKeypointsMetadata(KeypointsMetadata):
    labels = ["license_plate"]


class VLPParser(Parser):
    def __init__(self, annotations_filepath, idmap=None):
        super().__init__(template_record=self.template_record(), idmap=idmap)
        self.annotations = pd.read_csv(annotations_filepath)
        self.img_dir = Path(annotations_filepath.parent) / "images"
        self.class_map = ClassMap(VLPKeypointsMetadata.labels)

    def __iter__(self):
        yield from self.annotations.itertuples()

    def __len__(self):
        return len(self.annotations)

    def template_record(self) -> BaseRecord:
        return BaseRecord(
        (
            FilepathRecordComponent(),
            InstancesLabelsRecordComponent(),
            KeyPointsRecordComponent(),
        )
    )

    def record_id(self, o):
        return o.image_filename

    def filepath(self, o):
        return self.img_dir / o.split / o.image_filename

    def image_width_height(self, o) -> ImgSize:
        return get_img_size(self.filepath(o))

    def labels(self, o) -> List[Hashable]:
        return [1]

    def prepare(self, o):
        if o.source != "manual":
            raise AbortParseRecord("auto annotated")
        if not self.filepath(o).exists():
            raise AbortParseRecord("image not found")

    def parse_fields(self, o, record, is_new):
        if is_new:
            record.set_filepath(self.filepath(o))
            imsize = self.image_width_height(o)
            record.set_img_size(imsize)

        record.detection.set_class_map(self.class_map)
        record.detection.add_labels_by_id(self.labels(o))
        x, y = eval(o[3])
        x *= imsize.width
        y *= imsize.height
        visible = ifnone(o.visible, 0)  # switch visibility to 0 for None
        keypoints = [KeyPoints.from_xyv([x, y, visible], VLPKeypointsMetadata)]
        record.detection.add_keypoints(keypoints)


if __name__ == "__main__":
    data_dir = Path.home() / "datasets/plate_localization/v1"
    parser = VLPParser(annotations_filepath=data_dir / "metadata.csv")

    train_records, val_records = parser.parse(data_splitter=FolderSplitter(["train", "val"]))
    show_records(train_records[:9], ncols=3)
    plt.show()