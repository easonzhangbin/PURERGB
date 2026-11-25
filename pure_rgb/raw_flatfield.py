"""Explicit post-merge flat-field gain in a standard-crop coordinate domain.

The reference gain still spans the original full visible sensor canvas. Match
the pinned Native order: orient gain, resize it to the full oriented canvas,
select the effective crop/common-valid window, then multiply the merged image.
This does not move correction to the individual LED exposures or repair RAWs.
"""

from numbers import Integral

import negpy_core
import numpy as np

from negpy.features.flatfield import logic as _native_flatfield
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.kernel.image.logic import apply_exif_orientation

from .raw_validity import StandardCropPlan


def _finite(array, *, positive=False):
    rows = max(1, 262144 // array.shape[1])
    for offset in range(0, len(array), rows):
        block = array[offset : offset + rows]
        if not np.isfinite(block).all() or (positive and np.any(block <= 0)):
            return False
    return True


def _reference_shape(value, expected):
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 3
        or any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 1
            for v in value
        )
    ):
        raise ValueError("Explicit full sensor-visible reference HWC shape required")
    shape = tuple(int(v) for v in value)
    if shape != expected:
        raise ValueError(
            "Flat-field reference shape must match full sensor-visible source"
        )
    return shape


def map_gain_to_sensor(gain, mapping):
    """Select a proven sensor region before any output crop or orientation.

    Full-field resolution changes keep the original low-resolution gain; the
    Native full-canvas resize remains the single interpolation. An explicit
    sensor crop is sampled once onto the target sensor canvas, with pixel-center
    bilinear coordinates. Neither an unknown crop nor extrapolation is accepted.
    The frozen gain file and input array remain unchanged.
    """
    target = mapping.get("target_sensor_shape")
    if (not isinstance(target, (list, tuple)) or len(target) != 3
            or any(type(v) is not int or v < 1 for v in target) or target[2] != 3):
        raise ValueError("Explicit target sensor-visible RGB shape required")
    region = mapping.get("reference_roi_normalized_xyxy")
    if (not isinstance(region, (list, tuple)) or len(region) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not np.isfinite(v) for v in region)):
        raise ValueError("Explicit normalized reference sensor rectangle required")
    x0, y0, x1, y1 = region
    if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
        raise ValueError("Target sensor rectangle is outside the flat-field reference")
    if (not isinstance(gain, np.ndarray) or gain.ndim != 3 or gain.shape[2] != 3
            or min(gain.shape[:2]) < 1 or gain.dtype.kind != "f"
            or not _finite(gain, positive=True)):
        raise ValueError("Gain must be finite positive floating HWC RGB")
    if list(region) == [0., 0., 1., 1.]:
        return gain
    height, width, _ = target
    gh, gw, _ = gain.shape
    # Match Native resize_linear's half-pixel convention and edge extension.
    x = np.clip((x0 + (np.arange(width, dtype=np.float64) + .5)
                 / width * (x1 - x0)) * gw - .5, 0, gw - 1)
    ix = np.floor(x).astype(np.intp)
    jx = np.minimum(ix + 1, gw - 1)
    wx = (x - ix).astype(np.float32)[None, :, None]
    result = np.empty(target, dtype=np.float32)
    rows = max(1, 262144 // width)
    for start in range(0, height, rows):
        end = min(start + rows, height)
        y = np.clip((y0 + (np.arange(start, end, dtype=np.float64) + .5)
                     / height * (y1 - y0)) * gh - .5, 0, gh - 1)
        iy = np.floor(y).astype(np.intp)
        jy = np.minimum(iy + 1, gh - 1)
        wy = (y - iy).astype(np.float32)[:, None, None]
        upper = gain[iy[:, None], ix[None, :]] * (1 - wx) + gain[iy[:, None], jx[None, :]] * wx
        lower = gain[jy[:, None], ix[None, :]] * (1 - wx) + gain[jy[:, None], jx[None, :]] * wx
        result[start:end] = upper * (1 - wy) + lower * wy
    if not _finite(result, positive=True):
        raise ValueError("Mapped sensor gain must remain finite and positive")
    return result


def apply_standard_flatfield(
    image, config, plan, *, orientation, full_sensor_shape=None
):
    """Return post-merge float32 pixels and an explicit coordinate receipt.

    ``full_sensor_shape`` must come from validated calibration provenance. It is
    the gain's bound full visible sensor-layout HWC shape (after any proven
    cross-resolution mapping), not RAW storage dimensions, an oriented shape,
    an output crop or the low-resolution gain raster.
    Active gain without that binding fails closed. Disabled gain is identity.

    The exact implementation may allocate a full-resolution RGB gain array,
    plus a contiguous selected window and the output. This is not a bounded-
    memory implementation and should not be used to claim large-image memory
    improvements. Source and gain arrays remain untouched; headroom is retained.
    """
    if not isinstance(config, FlatFieldConfig) or type(config.apply) is not bool:
        raise ValueError("Explicit Native FlatFieldConfig required")
    if not isinstance(plan, StandardCropPlan):
        raise ValueError("Validated StandardCropPlan required")
    roi = plan.oriented_full_roi(orientation)
    orientation = int(orientation)
    x0, y0, x1, y1 = roi
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.float32
        or image.shape != (y1 - y0, x1 - x0, 3)
        or not image.flags.c_contiguous
    ):
        raise ValueError(
            "Matching C-contiguous oriented merged float32 RGB window required"
        )
    full_shape = (
        (plan.source_shape[1], plan.source_shape[0], 3)
        if orientation >= 5
        else plan.source_shape
    )
    receipt = {
        "schema": "pure-rgb-standard-crop-flatfield",
        "version": 1,
        "applied": False,
        "identity": True,
        "source_full_sensor_visible_shape": list(plan.source_shape),
        "full_oriented_visible_shape": list(full_shape),
        "effective_oriented_full_roi_xyxy": list(roi),
        "output_shape": list(image.shape),
        "orientation": orientation,
        "coordinate_space": "EXIF-oriented common-valid window in the green-reference canvas",
        "source_modified": False,
        "gain_modified": False,
        "correction_stage": "Existing broadband post-merge gain; no per-LED correction",
        "decoder_root_cause_repaired": False,
        "physical_or_spectral_calibration_certified": False,
    }
    if not config.apply:
        return image, receipt
    reference_shape = _reference_shape(full_sensor_shape, plan.source_shape)
    if not isinstance(config.profile_id, str) or not config.profile_id:
        raise ValueError("Active flat-field profile must resolve explicitly")
    entry = _native_flatfield._resolve(config.profile_id)
    if entry is None:
        raise ValueError("Active flat-field gain could not be resolved")
    if entry.sensor_layout is not True:
        raise ValueError("Active flat-field gain requires known sensor_layout=True")
    gain = entry.gain
    if (
        not isinstance(gain, np.ndarray)
        or gain.ndim != 3
        or gain.shape[2] != 3
        or min(gain.shape[:2]) < 1
        or gain.dtype.kind != "f"
        or not _finite(gain, positive=True)
    ):
        raise ValueError("Gain must be finite positive floating HWC RGB")
    if not _finite(image):
        raise ValueError("Merged RGB image must contain finite samples")
    with np.errstate(over="ignore", invalid="ignore"):
        oriented_gain = np.ascontiguousarray(
            apply_exif_orientation(gain, orientation), dtype=np.float32
        )
    if not _finite(oriented_gain, positive=True):
        raise ValueError("Gain must remain finite positive float32")
    resize = oriented_gain.shape != full_shape
    full_gain = (
        negpy_core.resize_linear(oriented_gain, (full_shape[1], full_shape[0]))
        if resize
        else oriented_gain
    )
    if not _finite(full_gain, positive=True):
        raise ValueError("Full-canvas resized gain must remain finite and positive")
    selected = np.ascontiguousarray(full_gain[y0:y1, x0:x1], dtype=np.float32)
    if selected.shape != image.shape:
        raise ValueError("Selected full-canvas gain does not match the merged window")
    result = negpy_core.multiply(image, selected)
    if not _finite(result):
        raise ValueError("Flat-field multiplication produced nonfinite samples")
    return result, {
        **receipt,
        "applied": True,
        "identity": False,
        "profile_id": config.profile_id,
        "gain_token": entry.token,
        "gain_sensor_layout": True,
        "reference_full_sensor_shape": list(reference_shape),
        "reference_shape_explicitly_bound": True,
        "gain_original_shape": list(gain.shape),
        "gain_oriented_shape": list(oriented_gain.shape),
        "gain_resize_full_canvas": resize,
        "gain_full_canvas_bytes": full_gain.nbytes,
        "gain_window_bytes": selected.nbytes,
        "order": "gain EXIF orientation -> resize to full oriented visible canvas -> effective common-valid window -> multiply merged image",
        "gain_stretched_directly_to_cropped_image": False,
        "gain_resized_after_window_selection": False,
        "clipping_or_renormalization_applied": False,
        "memory_limitation": "May hold a full-resolution RGB gain, contiguous selected gain and output; bounded-memory remapping is not implemented",
    }
