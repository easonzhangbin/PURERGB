"""Explicit camera-standard crop geometry, separate from decoder repair.

RawPy crop fields expose LibRaw raw_inset_crops[0] in absolute RAW coordinates.
The decoded input here must be full-resolution, sensor-layout RGB (user_flip=0,
half_size=False, square pixels). No padding, clipping or source mutation occurs.
This opt-in API does not certify all pixels outside that standard crop as bad.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from collections.abc import Mapping

import numpy as np


def _integer(value, name, *, minimum=0):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(f"{name} requires an explicit integer >= {minimum}")
    return int(value)


def _orientation(value):
    value = _integer(value, "EXIF orientation", minimum=1)
    if value > 8:
        raise ValueError("EXIF orientation must be 1..8")
    return value


def _rectangle(roi, width, height):
    if not isinstance(roi, (tuple, list)) or len(roi) != 4:
        raise ValueError("Explicit half-open xyxy rectangle required")
    x0, y0, x1, y1 = [_integer(v, "ROI coordinate") for v in roi]
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError("Rectangle outside its declared coordinate space")
    return x0, y0, x1, y1


def _oriented_shape(shape, orientation):
    height, width, channels = shape
    return (width, height, channels) if orientation >= 5 else shape


def _oriented_rectangle(roi, width, height, orientation):
    def point(x, y):
        return {
            1: (x, y),
            2: (width - x, y),
            3: (width - x, height - y),
            4: (x, height - y),
            5: (y, x),
            6: (height - y, x),
            7: (height - y, width - x),
            8: (y, width - x),
        }[orientation]

    x0, y0, x1, y1 = roi
    corners = [point(x, y) for x in (x0, x1) for y in (y0, y1)]
    return (
        min(x for x, _ in corners),
        min(y for _, y in corners),
        max(x for x, _ in corners),
        max(y for _, y in corners),
    )


def _orient(pixels, orientation):
    if orientation == 1:
        return pixels
    if orientation == 2:
        return pixels[:, ::-1]
    if orientation == 3:
        return pixels[::-1, ::-1]
    if orientation == 4:
        return pixels[::-1]
    if orientation == 5:
        return pixels.transpose(1, 0, 2)
    if orientation == 6:
        return np.rot90(pixels, -1)
    if orientation == 7:
        return pixels.transpose(1, 0, 2)[::-1, ::-1]
    return np.rot90(pixels, 1)


@dataclass(frozen=True)
class StandardCropPlan:
    source_shape: tuple[int, int, int]
    raw_shape: tuple[int, int]
    visible_origin_raw: tuple[int, int]
    raw_roi: tuple[int, int, int, int]
    visible_roi: tuple[int, int, int, int]

    def __post_init__(self):
        # Direct construction and dataclasses.replace must obey the same
        # coordinate contract as a plan made from an ImageSizes snapshot.
        for name, length, minimum in (
            ("source_shape", 3, 1),
            ("raw_shape", 2, 1),
            ("visible_origin_raw", 2, 0),
        ):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or len(value) != length:
                raise ValueError(f"Explicit {name} with {length} integers required")
            object.__setattr__(
                self,
                name,
                tuple(_integer(v, name, minimum=minimum) for v in value),
            )
        height, width, channels = self.source_shape
        raw_height, raw_width = self.raw_shape
        left, top = self.visible_origin_raw
        if channels != 3:
            raise ValueError("Full visible HWC RGB shape required")
        if left + width > raw_width or top + height > raw_height:
            raise ValueError("Visible canvas exceeds absolute RAW extent")
        raw_roi = _rectangle(self.raw_roi, raw_width, raw_height)
        visible_roi = _rectangle(self.visible_roi, width, height)
        expected = tuple(
            v + (left if index % 2 == 0 else top) for index, v in enumerate(visible_roi)
        )
        if raw_roi != expected:
            raise ValueError("Standard crop RAW and visible rectangles disagree")
        object.__setattr__(self, "raw_roi", raw_roi)
        object.__setattr__(self, "visible_roi", visible_roi)

    @property
    def crop_shape(self):
        x0, y0, x1, y1 = self.visible_roi
        return y1 - y0, x1 - x0, 3

    def oriented_full_roi(self, orientation):
        orientation = _orientation(orientation)
        height, width, _ = self.source_shape
        return _oriented_rectangle(self.visible_roi, width, height, orientation)

    def local_to_full_roi(self, roi, *, orientation):
        """Map crop-local oriented ROI into the oriented full-visible canvas."""
        orientation = _orientation(orientation)
        h, w, _ = _oriented_shape(self.crop_shape, orientation)
        x0, y0, x1, y1 = _rectangle(roi, w, h)
        ox, oy, _, _ = self.oriented_full_roi(orientation)
        return x0 + ox, y0 + oy, x1 + ox, y1 + oy

    def full_to_local_roi(self, roi, *, orientation):
        """Require full containment; never silently intersect or clamp an ROI."""
        orientation = _orientation(orientation)
        h, w, _ = _oriented_shape(self.source_shape, orientation)
        x0, y0, x1, y1 = _rectangle(roi, w, h)
        ox, oy, right, bottom = self.oriented_full_roi(orientation)
        if not ox <= x0 < x1 <= right or not oy <= y0 < y1 <= bottom:
            raise ValueError("Full-source ROI is not wholly within the standard crop")
        return x0 - ox, y0 - oy, x1 - ox, y1 - oy


def plan_standard_crop(sizes, decoded_shape, *, user_flip, half_size):
    """Validate a pre-postprocess size snapshot and decoded full sensor shape.

    Missing/zero crop dimensions are unavailable metadata, not full-frame
    fallback. RawPy maps unknown 0xffff origins to zero; a zero origin alone
    therefore cannot establish a usable standard crop.
    """
    if not isinstance(sizes, Mapping):
        if not hasattr(sizes, "_asdict"):
            raise ValueError("RawPy ImageSizes mapping/snapshot required")
        sizes = sizes._asdict()
    if (
        type(user_flip) is not int
        or user_flip != 0
        or type(half_size) is not bool
        or half_size
    ):
        raise ValueError(
            "Standard crop requires explicit user_flip=0 and half_size=False"
        )
    names = [
        "raw_width",
        "raw_height",
        "width",
        "height",
        "iwidth",
        "iheight",
        "left_margin",
        "top_margin",
        "crop_left_margin",
        "crop_top_margin",
        "crop_width",
        "crop_height",
        "pixel_aspect",
    ]
    if any(name not in sizes for name in names):
        raise ValueError("Incomplete pre-postprocess ImageSizes snapshot")
    dimensions = {
        name: _integer(sizes[name], name, minimum=1)
        for name in [
            "raw_width",
            "raw_height",
            "width",
            "height",
            "iwidth",
            "iheight",
            "crop_width",
            "crop_height",
        ]
    }
    margins = {
        name: _integer(sizes[name], name)
        for name in ["left_margin", "top_margin", "crop_left_margin", "crop_top_margin"]
    }
    aspect = sizes["pixel_aspect"]
    if (
        isinstance(aspect, (bool, np.bool_))
        or not isinstance(aspect, Real)
        or not math.isfinite(float(aspect))
        or float(aspect) != 1.0
    ):
        raise ValueError(
            "Non-square pixel resampling needs a separate coordinate contract"
        )
    if not isinstance(decoded_shape, (tuple, list)) or len(decoded_shape) != 3:
        raise ValueError("Explicit decoded HWC RGB shape required")
    shape = tuple(_integer(v, "decoded shape", minimum=1) for v in decoded_shape)
    if (
        shape != (dimensions["height"], dimensions["width"], 3)
        or (dimensions["iheight"], dimensions["iwidth"]) != shape[:2]
    ):
        raise ValueError(
            "Decoded shape must match unrotated full-resolution visible dimensions"
        )
    left, top = margins["left_margin"], margins["top_margin"]
    if (
        left + shape[1] > dimensions["raw_width"]
        or top + shape[0] > dimensions["raw_height"]
    ):
        raise ValueError("Visible canvas exceeds absolute RAW extent")
    x, y = margins["crop_left_margin"], margins["crop_top_margin"]
    if x == 65535 or y == 65535:
        raise ValueError("Unknown standard crop origin")
    raw_roi = (x, y, x + dimensions["crop_width"], y + dimensions["crop_height"])
    _rectangle(raw_roi, dimensions["raw_width"], dimensions["raw_height"])
    visible_roi = (
        raw_roi[0] - left,
        raw_roi[1] - top,
        raw_roi[2] - left,
        raw_roi[3] - top,
    )
    _rectangle(visible_roi, shape[1], shape[0])
    return StandardCropPlan(
        shape,
        (dimensions["raw_height"], dimensions["raw_width"]),
        (left, top),
        raw_roi,
        visible_roi,
    )


def standard_crop_view(pixels, plan, *, orientation, copy=False):
    """Return a read-only crop/orientation without interpolation or source edits.

    By default this is a strided view sharing the full source; ``copy=True``
    explicitly allocates a separate C-contiguous read-only array. Geometry only:
    sample values are neither clipped nor certified as calibrated scene data.
    """
    if not isinstance(plan, StandardCropPlan):
        raise ValueError("Validated StandardCropPlan required")
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.shape != plan.source_shape
        or pixels.dtype not in (np.uint16, np.float32, np.float64)
        or not pixels.flags.c_contiguous
    ):
        raise ValueError(
            "Matching C-contiguous full sensor-layout uint16/float RGB source required"
        )
    orientation = _orientation(orientation)
    if type(copy) is not bool:
        raise ValueError("copy requires an explicit boolean")
    x0, y0, x1, y1 = plan.visible_roi
    selected = _orient(pixels[y0:y1, x0:x1], orientation)
    result = np.array(selected, copy=True, order="C") if copy else selected.view()
    result.setflags(write=False)
    return result, {
        "schema": "pure-rgb-standard-raw-crop",
        "version": 1,
        "metadata_source": "LibRaw raw_inset_crops[0] exposed by RawPy; absolute RAW coordinates",
        "source_coordinate_space": "full-resolution sensor-layout visible RGB; postprocess user_flip=0, half_size=False, pixel_aspect=1",
        "source_shape": list(plan.source_shape),
        "raw_shape": list(plan.raw_shape),
        "visible_origin_raw_xy": list(plan.visible_origin_raw),
        "standard_crop_raw_absolute_xyxy": list(plan.raw_roi),
        "standard_crop_visible_xyxy": list(plan.visible_roi),
        "standard_crop_in_oriented_full_visible_xyxy": list(
            plan.oriented_full_roi(orientation)
        ),
        "output_coordinate_space": "standard-crop-local, explicitly EXIF-oriented RGB; half-open xyxy",
        "output_shape": list(result.shape),
        "orientation": orientation,
        "explicit_copy": copy,
        "output_read_only": True,
        "resampling_or_padding_used": False,
        "source_modified": False,
        "outside_standard_crop_all_invalid": False,
        "decoder_root_cause_repaired": False,
        "production_defaults_changed": False,
    }
