__all__ = ["BBox"]

from icevision.imports import *
from icevision.utils import *
from .exceptions import *


class BBox:

    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} "
            f"(xmin:{self.xmin}, ymin:{self.ymin}, xmax:{self.xmax}, ymax:{self.ymax})>"
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, BBox):
            return self.xyxy == other.xyxy
        return False

    @property
    def width(self):
        pass

    @property
    def height(self):
        pass

    @property
    def area(self):
        pass

    def to_tensor(self):
        pass

    def autofix(self, img_w, img_h, record_id: Optional[Any] = None) -> bool:
        """Tries to automatically fix invalid coordinates.

        # Returns
        - False if nothing was fixed (data had no problems)
        - True if data was successfully fixed
        - Raises `InvalidDataError` if unables to automatically fix the data
        """
        if self.xmin < 0:
            autofix_log(
                "AUTOFIX-SUCCESS",
                f"Clipping bbox xmin from {self.xmin} to 0 (Before: {self})",
                record_id=record_id,
            )
            self.xmin = max(self.xmin, 0)

        if self.ymin < 0:
            autofix_log(
                "AUTOFIX-SUCCESS",
                f"Clipping bbox ymin from {self.ymin} to 0 (Before: ({self}))",
                record_id=record_id,
            )
            self.ymin = max(self.ymin, 0)

        if self.xmax > img_w:
            autofix_log(
                "AUTOFIX-SUCCESS",
                f"Clipping bbox xmax from {self.xmax} to image width {img_w} (Before: {self})",
                record_id=record_id,
            )
            self.xmax = min(self.xmax, img_w)

        if self.ymax > img_h:
            autofix_log(
                "AUTOFIX-SUCCESS",
                f"Clipping bbox ymax from {self.ymax} to image height {img_h} (Before: {self})",
                record_id=record_id,
            )
            self.ymax = min(self.ymax, img_h)

        if (self.xmin >= self.xmax) or (self.ymin >= self.ymax):
            msg = []
            if self.xmin >= self.xmax:
                msg += [
                    f"\tx_min:{self.xmin} is greater than or equal to x_max:{self.xmax}"
                ]
            if self.ymin >= self.ymax:
                msg += [
                    f"\ty_min:{self.ymin} is greater than or equal to y_max:{self.ymax}"
                ]

            msg = "\n".join(msg)
            raise InvalidDataError(f"Cannot auto-fix coordinates: {self}\n{msg}")

        if self.xmin < 0 or self.ymin < 0 or self.xmax > img_w or self.ymax > img_h:
            return True

        return False

    @property
    def xyxy(self):
        pass

    @property
    def yxyx(self):
        pass

    @property
    def xywh(self):
        pass

    def relative_xcycwh(self, img_width: int, img_height: int):
        scale = np.array([img_width, img_height, img_width, img_height])
        x, y, w, h = self.xywh / scale
        xc = x + 0.5 * w
        yc = y + 0.5 * h
        return (xc, yc, w, h)

    @classmethod
    def from_xywh(cls, x, y, w, h):
        pass

    @classmethod
    def from_xyxy(cls, xl, yu, xr, yb):
        return cls(xl, yu, xr, yb)

    @classmethod
    def from_relative_xcycwh(cls, xc, yc, bw, bh, img_width, img_height):
        pass

    @classmethod
    def from_rle(cls, rle, h, w):
        pass
