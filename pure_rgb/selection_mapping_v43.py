"""Exact source-boundary mapping for selections on cropped, D4-oriented previews.

Only output crop plus the shared orthogonal orientation is represented here.
Callers must use original-source positioning for unsupported post geometry.
Image pixels may be sampled; selection coordinates remain full-source edges.
"""

from dataclasses import dataclass
import math

from .output_orientation import checked_document, IDENTITY


def _clean(value):
    nearest = round(value)
    return float(nearest) if abs(value - nearest) < 1e-8 else float(value)


@dataclass(frozen=True)
class SelectionMapping:
    source_shape: tuple
    output_roi: tuple | None = None
    output_orientation: dict | None = None

    def __post_init__(self):
        shape = self.source_shape
        if (not isinstance(shape, (list, tuple)) or len(shape) not in (2, 3)
                or any(type(value) is not int or value <= 0 for value in shape)
                or (len(shape) == 3 and shape[2] != 3)):
            raise ValueError("选区映射需要完整源图像尺寸")
        height, width = shape[:2]
        roi = self.output_roi or (0, 0, width, height)
        if (not isinstance(roi, (list, tuple)) or len(roi) != 4
                or any(type(value) is not int for value in roi)):
            raise ValueError("选区映射需要整数输出裁切边界")
        x0, y0, x1, y1 = roi
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("选区映射的输出裁切超出完整源")
        object.__setattr__(self, "source_shape", tuple(shape))
        object.__setattr__(self, "output_roi", tuple(roi))
        object.__setattr__(self, "output_orientation", checked_document(self.output_orientation))

    @property
    def output_size(self):
        x0, y0, x1, y1 = self.output_roi
        width, height = x1 - x0, y1 - y0
        if (self.output_orientation or IDENTITY)["quarter_turns"] % 2:
            return height, width
        return width, height

    @staticmethod
    def _point(point):
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in point)):
            raise ValueError("选区坐标必须是有限二维数值")
        return float(point[0]), float(point[1])

    def source_to_output(self, point):
        x, y = self._point(point)
        x0, y0, x1, y1 = self.output_roi
        width, height = x1 - x0, y1 - y0
        x, y = x - x0, y - y0
        doc = self.output_orientation or IDENTITY
        quarter = doc["quarter_turns"]
        if quarter == 1:
            x, y = height - y, x
        elif quarter == 2:
            x, y = width - x, height - y
        elif quarter == 3:
            x, y = y, width - x
        output_width, output_height = self.output_size
        if doc["flip_horizontal"]:
            x = output_width - x
        if doc["flip_vertical"]:
            y = output_height - y
        return _clean(x), _clean(y)

    def output_to_source(self, point, *, clamp=False):
        x, y = self._point(point)
        output_width, output_height = self.output_size
        if clamp:
            x, y = max(0, min(output_width, x)), max(0, min(output_height, y))
        doc = self.output_orientation or IDENTITY
        if doc["flip_horizontal"]:
            x = output_width - x
        if doc["flip_vertical"]:
            y = output_height - y
        x0, y0, x1, y1 = self.output_roi
        width, height = x1 - x0, y1 - y0
        quarter = doc["quarter_turns"]
        if quarter == 1:
            x, y = y, height - x
        elif quarter == 2:
            x, y = width - x, height - y
        elif quarter == 3:
            x, y = width - y, x
        return _clean(x + x0), _clean(y + y0)

    def source_to_preview(self, point, preview_size):
        x, y = self.source_to_output(point)
        width, height = self.output_size
        return x * preview_size[0] / width, y * preview_size[1] / height

    def preview_to_source(self, point, preview_size, *, clamp=False):
        x, y = self._point(point)
        if (not isinstance(preview_size, (list, tuple)) or len(preview_size) != 2
                or any(type(value) is not int or value <= 0 for value in preview_size)):
            raise ValueError("选区映射需要实际显示图尺寸")
        width, height = self.output_size
        return self.output_to_source((x * width / preview_size[0], y * height / preview_size[1]), clamp=clamp)
