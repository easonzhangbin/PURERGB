"""Vector-defined, pixel-centre metering selections without an image/UI crop.

Included shapes are unioned; excluded shapes are then subtracted. A lasso is
stored as an even-odd polygon, not as painted RGB or a resized thumbnail mask.
The returned bool raster covers only the declared source-coordinate ROI.
"""

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real

import numpy as np

from .metering_regions import _integer, _rectangle

MAX_SHAPES = 64
MAX_POLYGON_POINTS = 2048
MAX_MASK_PIXELS = 128 * 1024 * 1024


@dataclass(frozen=True)
class MeteringMask:
    source_shape: tuple[int, ...]
    roi: tuple[int, int, int, int]
    mask: np.ndarray = field(repr=False, compare=False)
    spec: dict
    mask_sha256: str


def _digest(spec, mask):
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode())
    digest.update(memoryview(mask).cast("B"))
    return digest.hexdigest()


def _geometry(shape, roi):
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) not in (2, 3)
        or not all(_integer(v) and v > 0 for v in shape)
        or (len(shape) == 3 and shape[2] != 3)
    ):
        raise ValueError("Expected a positive source shape (H,W) or (H,W,3)")
    shape = tuple(int(v) for v in shape)
    roi = _rectangle(roi, shape, "Mask ROI")
    if (roi[3] - roi[1]) * (roi[2] - roi[0]) > MAX_MASK_PIXELS:
        raise ValueError("Metering mask exceeds the supported raster pixel limit")
    return shape, roi


def _spec_header(spec):
    if (
        not isinstance(spec, dict)
        or type(spec.get("version")) is not int
        or spec.get("version") != 1
        or spec.get("RGB_support_shared") is not True
        or spec.get("output_crop_changed") is not False
        or set(spec)
        != {
            "version",
            "source_shape",
            "roi",
            "include",
            "exclude",
            "coordinate_space",
            "raster_rule",
            "RGB_support_shared",
            "output_crop_changed",
        }
    ):
        raise ValueError("Unsupported or malformed metering mask metadata")


def _number(value):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError("Shape coordinates must be finite numbers, not booleans")
    return float(value)


def _shape(entry, source_shape):
    if not isinstance(entry, dict):
        raise ValueError("Metering shapes must be objects")
    h, w = source_shape[:2]
    kind = entry.get("type")
    if kind == "polygon":
        if set(entry) != {"type", "points"}:
            raise ValueError("Polygon requires only type and points")
        points = entry["points"]
        if (
            not isinstance(points, (list, tuple))
            or not 3 <= len(points) <= MAX_POLYGON_POINTS
        ):
            raise ValueError("Polygon requires 3..2048 source-coordinate points")
        result = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("Polygon points require an x,y pair")
            x, y = map(_number, point)
            if not (0 <= x <= w and 0 <= y <= h):
                raise ValueError("Polygon point outside its source")
            result.append([x, y])
        return {"type": kind, "points": result}
    if kind not in ("rectangle", "rounded_rectangle", "ellipse"):
        raise ValueError("Unsupported metering shape")
    keys = (
        {"type", "bounds", "radius"}
        if kind == "rounded_rectangle"
        else {"type", "bounds"}
    )
    if set(entry) != keys:
        raise ValueError("Unexpected or missing metering shape fields")
    bounds = entry["bounds"]
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 4:
        raise ValueError("Shape bounds require source x0,y0,x1,y1")
    x0, y0, x1, y1 = map(_number, bounds)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Shape bounds outside source or empty")
    result = {"type": kind, "bounds": [x0, y0, x1, y1]}
    if kind == "rounded_rectangle":
        radius = _number(entry["radius"])
        if not 0 <= radius <= min(x1 - x0, y1 - y0) / 2:
            raise ValueError("Rounded rectangle radius exceeds half its short side")
        result["radius"] = radius
    return result


def _raster(shape, roi):
    rx0, ry0, rx1, ry1 = roi
    height, width = ry1 - ry0, rx1 - rx0
    result = np.zeros((height, width), dtype=bool)
    if shape["type"] == "polygon":
        points = np.asarray(shape["points"], dtype=np.float64)
        following = np.roll(points, -1, axis=0)
        for row in range(height):
            y = ry0 + row + 0.5
            crosses = ((points[:, 1] <= y) & (y < following[:, 1])) | (
                (following[:, 1] <= y) & (y < points[:, 1])
            )
            first, second = points[crosses], following[crosses]
            if not first.size:
                continue
            intersections = np.sort(
                first[:, 0]
                + (y - first[:, 1])
                * (second[:, 0] - first[:, 0])
                / (second[:, 1] - first[:, 1])
            )
            if len(intersections) % 2:
                raise ValueError("Polygon has an ambiguous scanline intersection")
            for left, right in zip(intersections[::2], intersections[1::2]):
                start = max(0, min(width, math.ceil(left - rx0 - 0.5)))
                stop = max(0, min(width, math.ceil(right - rx0 - 0.5)))
                result[row, start:stop] = True
        return result
    x0, y0, x1, y1 = shape["bounds"]
    xx = (np.arange(width, dtype=np.float64) + rx0 + 0.5)[None, :]
    yy = (np.arange(height, dtype=np.float64) + ry0 + 0.5)[:, None]
    inside = (x0 <= xx) & (xx < x1) & (y0 <= yy) & (yy < y1)
    if shape["type"] == "rectangle":
        return inside
    if shape["type"] == "ellipse":
        return inside & (
            ((xx - (x0 + x1) / 2) / ((x1 - x0) / 2)) ** 2
            + ((yy - (y0 + y1) / 2) / ((y1 - y0) / 2)) ** 2
            <= 1
        )
    radius = shape["radius"]
    dx = np.maximum(np.maximum(x0 + radius - xx, xx - x1 + radius), 0)
    dy = np.maximum(np.maximum(y0 + radius - yy, yy - y1 + radius), 0)
    return inside & (dx**2 + dy**2 <= radius**2)


def build_metering_mask(shape, roi, *, include=(), exclude=()):
    """Rasterize a binary unfeathered selection at original source pixel centres.

    Empty include means the explicit whole ROI. Exclusions always win. Shapes
    may cross this ROI but must remain inside the declared source. No source
    pixels are read, resampled, blackened or modified. Store ``spec`` alongside
    the workflow's source hash/geometry token when persisting a selection.
    """
    shape, roi = _geometry(shape, roi)
    normalized = {}
    for role, values in (("include", include), ("exclude", exclude)):
        if not isinstance(values, (list, tuple)) or len(values) > MAX_SHAPES:
            raise ValueError("Each mask role supports at most 64 vector shapes")
        normalized[role] = [_shape(value, shape) for value in values]
    result = np.full(
        (roi[3] - roi[1], roi[2] - roi[0]), not normalized["include"], dtype=bool
    )
    def apply(entry, include):
        # Pixel-centre arithmetic still runs in absolute source coordinates.
        # Shapes with explicit bounds cannot affect pixels outside that box;
        # avoid a camera-sized float distance plane for a small exclusion.
        sub = roi
        if entry["type"] != "polygon":
            a, b, c, d = entry["bounds"]
            sub = (max(roi[0], math.floor(a) - 1), max(roi[1], math.floor(b) - 1),
                   min(roi[2], math.ceil(c) + 1), min(roi[3], math.ceil(d) + 1))
            if sub[0] >= sub[2] or sub[1] >= sub[3]:
                return
        target = result[sub[1]-roi[1]:sub[3]-roi[1], sub[0]-roi[0]:sub[2]-roi[0]]
        raster = _raster(entry, sub)
        if include:
            target |= raster
        else:
            target &= ~raster

    for entry in normalized["include"]:
        apply(entry, True)
    for entry in normalized["exclude"]:
        apply(entry, False)
    if not result.any():
        raise ValueError("Metering selection contains no source pixel centres")
    result.setflags(write=False)
    spec = {
        "version": 1,
        "source_shape": list(shape),
        "roi": list(roi),
        **normalized,
        "coordinate_space": "oriented merged source, half-open xyxy",
        "raster_rule": "binary original pixel centres; even-odd polygon scanlines, left/top inclusive and right/bottom exclusive; no feathering",
        "RGB_support_shared": True,
        "output_crop_changed": False,
    }
    return MeteringMask(shape, roi, result, spec, _digest(spec, result))


def mask_from_spec(spec):
    """Reconstruct a persisted vector selection, checking its declared version."""
    _spec_header(spec)
    result = build_metering_mask(
        spec["source_shape"],
        spec["roi"],
        include=spec["include"],
        exclude=spec["exclude"],
    )
    if result.spec != spec:
        raise ValueError(
            "Metering selection metadata does not match its vector definition"
        )
    return result


def validate_metering_mask(selection, shape, roi):
    """Validate exact source/ROI binding and the immutable raster/spec receipt.

    HW and HWC are distinct declarations; density callers pass the actual HWC
    source shape. A mask made for another coordinate space is never resized.
    """
    if not isinstance(selection, MeteringMask):
        raise ValueError("Expected a vector-defined MeteringMask")
    shape, roi = _geometry(shape, roi)
    _geometry(selection.source_shape, selection.roi)
    if selection.source_shape != shape or selection.roi != roi:
        raise ValueError("Metering mask belongs to another source shape or ROI")
    _spec_header(selection.spec)
    _geometry(selection.spec["source_shape"], selection.spec["roi"])
    if (
        not isinstance(selection.mask, np.ndarray)
        or selection.mask.dtype != np.bool_
        or selection.mask.shape != (roi[3] - roi[1], roi[2] - roi[0])
        or not selection.mask.flags.c_contiguous
        or selection.mask.flags.writeable
        or selection.spec.get("source_shape") != list(shape)
        or selection.spec.get("roi") != list(roi)
        or _digest(selection.spec, selection.mask) != selection.mask_sha256
    ):
        raise ValueError("Metering mask raster or vector receipt changed")
    return selection.mask
