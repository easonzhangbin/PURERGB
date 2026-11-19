"""Conservative picture-area metering through the existing NegPy detector."""

import math


class NoReliablePictureRectangle(ValueError):
    """NegPy could not identify the film box for automatic metering."""


def normalized_box(rect, shape, inset):
    h, w = shape[:2]
    if not 0 <= inset < 0.2:
        raise ValueError("Picture inset must be between 0 and 0.2")
    if rect is None:
        raise NoReliablePictureRectangle(
            "No reliable picture rectangle; provide --roi explicitly"
        )
    if len(rect) != 4 or not all(math.isfinite(v) for v in rect):
        raise ValueError("No reliable picture rectangle; provide --roi explicitly")
    x0, y0, x1, y1 = rect
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("Invalid detected picture rectangle")
    box = (math.ceil(x0 * w), math.ceil(y0 * h), math.floor(x1 * w), math.floor(y1 * h))
    dx = math.ceil((box[2] - box[0]) * inset)
    dy = math.ceil((box[3] - box[1]) * inset)
    inner = (box[0] + dx, box[1] + dy, box[2] - dx, box[3] - dy)
    if inner[2] - inner[0] < 16 or inner[3] - inner[1] < 16:
        raise ValueError("Detected picture is too small for metering")
    return box, inner


def detect_picture(image, inset=0.02):
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


def _coherent_picture_bounds(gray, polarity):
    """Find four rebate-to-picture transitions, measured from outside in."""
    import cv2
    import numpy as np

    height, width = gray.shape
    depths, rebate_levels = [], []
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
            (change >= 0.08) & (coherent >= 0.025) & (quiet_high - quiet_low <= 0.15),
            change * coherent,
            0,
        )
        peak = int(np.argmax(scores))
        score = scores[peak]
        if not score:
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


def detect_picture_from_locators(geometry, shape, inset=0.02):
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
