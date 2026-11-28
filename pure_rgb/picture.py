"""Conservative picture-area metering through the existing NegPy detector."""

import math


# Scaling both dimensions by sqrt(0.8) leaves the central 80% of the
# detected picture area while retaining its aspect ratio (subject to pixels).
AUTO_PICTURE_AREA_FRACTION = 0.8
AUTO_PICTURE_INSET = (1.0 - math.sqrt(AUTO_PICTURE_AREA_FRACTION)) / 2.0


class NoReliablePictureRectangle(ValueError):
    """NegPy could not identify the film box for automatic metering."""


def normalized_box(rect, shape, inset):
    h, w = shape[:2]
    if not 0 <= inset < 0.2:
        raise ValueError("Picture inset must be between 0 and 0.2")
    if rect is None or len(rect) != 4 or not all(math.isfinite(v) for v in rect):
        raise NoReliablePictureRectangle(
            "No reliable picture rectangle; provide --roi explicitly"
        )
    x0, y0, x1, y1 = rect
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise NoReliablePictureRectangle("No reliable picture rectangle; detected bounds are invalid")
    box = (math.ceil(x0 * w), math.ceil(y0 * h), math.floor(x1 * w), math.floor(y1 * h))
    dx = math.ceil((box[2] - box[0]) * inset)
    dy = math.ceil((box[3] - box[1]) * inset)
    inner = (box[0] + dx, box[1] + dy, box[2] - dx, box[3] - dy)
    if inner[2] - inner[0] < 16 or inner[3] - inner[1] < 16:
        raise NoReliablePictureRectangle("No reliable picture rectangle; detected region is too small for metering")
    return box, inner



def metering_region_from_frame(bounds, shape):
    """Center the automatic metering region within a confirmed source frame."""
    h, w = shape[:2]
    if (len(bounds) != 4 or any(type(v) is not int for v in bounds)
            or not (0 <= bounds[0] < bounds[2] <= w and 0 <= bounds[1] < bounds[3] <= h)):
        raise ValueError("有效画幅范围无效")
    x0, y0, x1, y1 = bounds
    dx = math.ceil((x1 - x0) * AUTO_PICTURE_INSET)
    dy = math.ceil((y1 - y0) * AUTO_PICTURE_INSET)
    inner = [x0 + dx, y0 + dy, x1 - dx, y1 - dy]
    if inner[2] - inner[0] < 16 or inner[3] - inner[1] < 16:
        raise ValueError("有效画幅太小，无法测量中心区域")
    return inner

def detect_picture(image, inset=AUTO_PICTURE_INSET):
    from negpy.features.geometry.logic import resolve_meter_rect
    from negpy.features.geometry.models import GeometryConfig
    from negpy.kernel.system.config import APP_CONFIG

    rect = resolve_meter_rect(image, GeometryConfig(), APP_CONFIG.preview_render_size)
    box, inner = normalized_box(rect, image.shape, inset)
    return {
        "method": "NegPy 0.68 resolve_meter_rect, image mode",
        "coordinate_space": "EXIF-oriented merged linear source; half-open xyxy",
        "detected_picture_roi": box,
        "analysis_roi": inner,
        "inset_per_side": inset,
        "confidence": None,
        "limitations": "Axis-aligned, conservatively inset rectangle; not a pixelwise semantic mask",
    }


def _coherent_picture_bounds(gray, polarity, *, minimum_change=0.08, max_weak_edges=0):
    """Find four rebate-to-picture transitions, measured from outside in."""
    import cv2
    import numpy as np

    height, width = gray.shape
    depths, rebate_levels = [], []
    weak_edges = 0
    for image in (gray, gray.T, gray[:, ::-1], gray[::-1, :].T):
        breadth, extent = image.shape
        line = image[breadth // 4 : breadth - breadth // 4]
        radius = max(3, round(extent * 0.003))
        start, stop = max(radius, round(extent * 0.01)), round(extent * 0.28)
        outside = line[:, start - radius : stop - radius]
        inside = line[:, start + radius : stop + radius]
        change = np.quantile(polarity * (inside - outside), 0.25, axis=0)
        quiet_low, quiet_high = np.quantile(outside, [0.10, 0.90], axis=0)
        gradient = np.abs(cv2.Sobel(line, cv2.CV_32F, 1, 0, ksize=3)[:, start:stop]) / 4
        coherent = np.quantile(gradient, 0.25, axis=0)
        # A rebate leaves a quiet band and a transition with the right polarity
        # across most of the line, including on frames with sparse lettering.
        scores = np.where(
            (change >= minimum_change) & (coherent >= 0.025) & (quiet_high - quiet_low <= 0.15),
            change * coherent,
            0,
        )
        peak = int(np.argmax(scores))
        score = scores[peak]
        if not score:
            return None
        if change[peak] < 0.08:
            weak_edges += 1
            if weak_edges > max_weak_edges:
                return None
        scores[max(0, peak - 2 * radius) : peak + 2 * radius + 1] = 0
        if scores.max() >= 0.65 * score:
            return None
        depths.append(start + peak)
        rebate_levels.append(float(np.median(outside[:, peak])))

    left, top, right, bottom = depths
    x1, y1 = width - right, height - bottom
    if x1 - left < width * 0.55 or y1 - top < height * 0.55:
        return None
    if polarity < 0:
        # A real clear rebate has similar transmission on each side. Check
        # the interior also has scene signal, rather than an opaque holder.
        if max(rebate_levels) - min(rebate_levels) > 0.15:
            return None
        dx, dy = (x1 - left) // 8, (y1 - top) // 8
        picture = gray[top + dy : y1 - dy, left + dx : x1 - dx]
        low, high = np.quantile(
            cv2.resize(picture, (32, 32), interpolation=cv2.INTER_AREA), [0.05, 0.95]
        )
        if high < 0.12 or high - low < 0.04:
            return None
    return left, top, x1, y1


def detect_picture_from_locators(geometry, shape, inset=AUTO_PICTURE_INSET):
    """Accept a fallback only when both full-source locators agree on the frame."""
    from PIL import Image

    from .workflow_auto_crop import _gray

    height, width = shape[:2]
    if (geometry["height"], geometry["width"]) != (height, width):
        return None
    try:
        with Image.open(geometry["preview"]) as image:
            transmission = _gray(image)
        with Image.open(geometry["positive_preview"]) as image:
            positive = _gray(image)
    except (OSError, ValueError):
        return None
    if transmission.shape != positive.shape:
        return None
    ph, pw = transmission.shape
    if abs(pw / ph - width / height) > 0.01 * width / height:
        return None
    first = _coherent_picture_bounds(transmission, -1)
    second = _coherent_picture_bounds(positive, 1)
    if first is None and second is not None:
        # A bright film edge can have less contrast in the compressed
        # transmission locator than in the positive. Permit one weak edge
        # only; the positive still has to pass all four strict edge tests and
        # the two independent locator candidates must agree below.
        first = _coherent_picture_bounds(
            transmission, -1, minimum_change=0.055, max_weak_edges=1
        )
    if first is None or second is None:
        return None
    dimensions = (pw, ph, pw, ph)
    if any(
        abs(a - b) > max(2, 0.01 * size)
        for a, b, size in zip(first, second, dimensions)
    ):
        return None
    # Use the intersection, then apply the same metering inset as NegPy.
    common = (
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    )
    box, inner = normalized_box(
        tuple(edge / size for edge, size in zip(common, dimensions)), shape, inset
    )
    return {
        "method": "Coherent picture edges in transmission and positive source locators",
        "fallback_reason": "NegPy could not identify a reliable film box",
        "coordinate_space": "EXIF-oriented merged linear source; half-open xyxy",
        "detected_picture_roi": box,
        "analysis_roi": inner,
        "inset_per_side": inset,
        "confidence": "four corroborated edges",
        "limitations": "Axis-aligned, conservatively inset rectangle; not a pixelwise semantic mask",
    }
