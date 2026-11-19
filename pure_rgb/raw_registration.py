"""Selected linear RAW bands -> explicit translation registration and support.

The pinned NegPy phase/resize/warp primitives are reused. Unlike its unchanged
RGB-scan wrapper, this adapter retains diagnostics, rejects failed estimates,
and crops away replicated warp borders. Mean removal affects estimation only.
"""

import math

import negpy_core
import numpy as np

from .raw_registration_model import (
    POLICY as TRANSLATION_CONSISTENCY_V1,
    SOURCE_POLICY as TRANSLATION_CONSISTENCY_SOURCE_V2,
    RegistrationModelCheckError,
    check_translation_consistency,
    check_translation_consistency_source,
    pixel_sha,
    validate_registration_model_metadata as validate_registration_model_metadata,
)

EST_WIDTH = 1024
RESPONSE_THRESHOLD = 0.05
RELATIVE_STD_THRESHOLD = 1e-5
AXIS_GRADIENT_THRESHOLD = 1e-6
TEXTURE_EIGEN_RATIO_THRESHOLD = 1e-4


def _planes(red, green, blue):
    values = []
    for name, plane in zip(("red", "green", "blue"), (red, green, blue), strict=True):
        if (
            not isinstance(plane, np.ndarray)
            or plane.dtype != np.float32
            or plane.ndim != 2
            or min(plane.shape, default=0) < 2
            or not plane.flags.c_contiguous
        ):
            raise ValueError(
                f"{name}: require a C-contiguous float32 2D selected band, at least 2x2"
            )
        if not np.isfinite(plane).all():
            raise ValueError(f"{name}: selected band contains nonfinite samples")
        values.append(plane)
    if any(p.shape != values[0].shape for p in values[1:]):
        raise ValueError("Selected RAW bands must have identical shape")
    return values


def _estimator(plane, size, name):
    selected = (
        negpy_core.resize_area(plane, size) if plane.shape[1] > EST_WIDTH else plane
    )
    mean = float(np.mean(selected, dtype=np.float64))
    carrier64 = selected.astype(np.float64) - mean
    variance = float(np.mean(carrier64 * carrier64))
    rms = math.sqrt(variance + mean * mean)
    denominator = max(rms, float(np.finfo(np.float32).tiny))
    relative_std = math.sqrt(variance) / denominator
    gx = carrier64[1:, 1:] - carrier64[1:, :-1]
    gy = carrier64[1:, 1:] - carrier64[:-1, 1:]
    xx, yy, xy = (
        float(np.mean(gx * gx)),
        float(np.mean(gy * gy)),
        float(np.mean(gx * gy)),
    )
    eigen = np.linalg.eigvalsh([[xx, xy], [xy, yy]])
    eigen_ratio = max(float(eigen[0]), 0.0) / max(float(eigen[1]), np.finfo(float).tiny)
    axes = [math.sqrt(xx) / denominator, math.sqrt(yy) / denominator]
    diagnostic = {
        "mean_before_estimation_centering": mean,
        "variance_before_window": variance,
        "relative_std": relative_std,
        "relative_gradient_rms_xy": axes,
        "gradient_structure_eigen_ratio": eigen_ratio,
    }
    if relative_std < RELATIVE_STD_THRESHOLD:
        raise ValueError(
            f"{name}: constant or insufficient contrast for registration (relative_std={relative_std:.6g})"
        )
    if (
        min(axes) < AXIS_GRADIENT_THRESHOLD
        or eigen_ratio < TEXTURE_EIGEN_RATIO_THRESHOLD
    ):
        raise ValueError(
            f"{name}: insufficient two-axis texture for registration (eigen_ratio={eigen_ratio:.6g})"
        )
    carrier = np.ascontiguousarray(carrier64, dtype=np.float32)
    if not np.isfinite(carrier).all():
        raise ValueError(f"{name}: centered estimation carrier exceeds finite float32")
    return carrier, diagnostic


def _support(shape, dx, dy):
    """Conservative original support for cubic inverse sampling at (x+dx,y+dy).

    Centre samples are restricted to [2, W-3] and [2, H-3]. This protects all
    four cubic taps and the backend's 1/32-pixel coordinate quantization; no
    replicated edge value is certified as an original observation.
    """
    h, w = shape
    return (
        max(0, math.ceil(2 - dx)),
        max(0, math.ceil(2 - dy)),
        min(w, math.floor(w - 3 - dx) + 1),
        min(h, math.floor(h - 3 - dy) + 1),
    )


def register_planes(red, green, blue, *, align=True, model_check=None):
    """Return (merged_float32, receipt) in standard-crop-local sensor axes.

    All input bands are already standard-cropped and LibRaw-black-subtracted;
    they must be finite, same-shape, contiguous float32. There is no clipping,
    uint16 interpolation, gain/colour change or EXIF orientation here. Failure
    is explicit, never a silent return of unregistered channels.
    """
    planes = _planes(red, green, blue)
    if type(align) is not bool:
        raise ValueError("align must explicitly be bool")
    if model_check is not None and (
        type(model_check) is not str
        or model_check
        not in (TRANSLATION_CONSISTENCY_V1, TRANSLATION_CONSISTENCY_SOURCE_V2)
    ):
        raise ValueError("Unknown RAW registration model check")
    if model_check is not None and not align:
        raise ValueError("Translation model check requires alignment enabled")
    source_hashes = (
        {
            name: pixel_sha(plane)
            for name, plane in zip(("red", "green", "blue"), planes)
        }
        if model_check is not None
        else None
    )
    model_proofs = {}
    h, w = green.shape
    scale = w / EST_WIDTH if w > EST_WIDTH else 1.0
    size = (EST_WIDTH, max(2, round(h / scale))) if w > EST_WIDTH else (w, h)
    scale_x, scale_y = w / size[0], h / size[1]
    max_shift = max(16.0, 0.02 * w)
    receipt = {
        "schema": "pure-rgb-raw-registration",
        "version": 1,
        "status": "registered" if align else "disabled",
        "align": align,
        "reference_channel": "green",
        "input_shape_h_w": [h, w],
        "coordinate_space": "standard-crop-local sensor xy; before EXIF orientation",
        "phase_estimation_shape_h_w": [size[1], size[0]] if align else None,
        "phase_estimation_width_limit": EST_WIDTH,
        "phase_estimation_to_source_scale_xy": [scale_x, scale_y] if align else None,
        "estimator_mean_removed_independently": align,
        "original_signal_centered_or_rescaled": False,
        "difference_from_pinned_rgbscan_wrapper": "Estimator mean removal, actual per-axis resize scales, explicit quality failures and retained diagnostics/support; unchanged vendor wrapper is not replaced",
        "response_threshold": RESPONSE_THRESHOLD,
        "max_shift_per_axis_source_pixels": max_shift,
        "quality_thresholds_are_heuristics": True,
        "texture_thresholds": {
            "relative_std_min": RELATIVE_STD_THRESHOLD,
            "relative_gradient_rms_each_axis_min": AXIS_GRADIENT_THRESHOLD,
            "gradient_structure_min_over_max_eigenvalue_min": TEXTURE_EIGEN_RATIO_THRESHOLD,
        },
        "motion_model": "translation only",
        "interpolation": "cubic" if align else None,
        "warp_border": "replicate" if align else None,
        "warp_quantization_pixels": 1 / 32 if align else None,
        "support_policy": "Each warped source centre inside [2,W-3]x[2,H-3], protecting the 4x4 footprint; retain only intersection with unwarped green",
        "channels": {},
        "exif_orientation_applied": False,
        "input_samples_modified": False,
        "output_clipped_or_quantized": False,
    }
    if not align:
        merged = np.stack(planes, axis=-1)
        for name in ("red", "green", "blue"):
            receipt["channels"][name] = {
                "status": "disabled",
                "shift_xy": [0.0, 0.0],
                "response": None,
                "texture": None,
            }
        common = (0, 0, w, h)
    else:
        reference, texture = _estimator(green, size, "green")
        receipt["channels"]["green"] = {
            "status": "reference",
            "shift_xy": [0.0, 0.0],
            "response": None,
            "texture": texture,
        }
        shifts = []
        supports = [(0, 0, w, h)]
        for name, plane in (("red", red), ("blue", blue)):
            carrier, texture = _estimator(plane, size, name)
            (dx, dy), response = negpy_core.phase_correlate(reference, carrier)
            dx, dy, response = float(dx * scale_x), float(dy * scale_y), float(response)
            if not all(math.isfinite(v) for v in (dx, dy, response)):
                raise ValueError(
                    f"{name}: registration returned nonfinite shift/response"
                )
            if response < RESPONSE_THRESHOLD:
                raise ValueError(
                    f"{name}: registration response {response:.6g} is below heuristic {RESPONSE_THRESHOLD}"
                )
            if max(abs(dx), abs(dy)) > max_shift:
                raise ValueError(
                    f"{name}: registration shift ({dx:.6g},{dy:.6g}) exceeds {max_shift:.6g}px"
                )
            if model_check is not None:
                try:
                    check = (
                        check_translation_consistency_source
                        if model_check == TRANSLATION_CONSISTENCY_SOURCE_V2
                        else check_translation_consistency
                    )
                    model_proofs[name] = check(
                        reference,
                        carrier,
                        (dx / scale_x, dy / scale_y),
                        (scale_x, scale_y),
                        name,
                    )
                except RegistrationModelCheckError as error:
                    error.diagnostic["source_plane_sha256"] = source_hashes
                    raise
            support = _support((h, w), dx, dy)
            receipt["channels"][name] = {
                "status": "registered",
                "shift_xy": [dx, dy],
                "response": response,
                "texture": texture,
                "original_support_roi_sensor_xyxy": list(support),
            }
            shifts.append((dx, dy))
            supports.append(support)
        common = (
            max(s[0] for s in supports),
            max(s[1] for s in supports),
            min(s[2] for s in supports),
            min(s[3] for s in supports),
        )
        x0, y0, x1, y1 = common
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                "No common original cubic support remains after registration"
            )
        merged = np.empty((y1 - y0, x1 - x0, 3), dtype=np.float32)
        merged[..., 1] = green[y0:y1, x0:x1]
        for output_channel, plane, (dx, dy) in zip(
            (0, 2), (red, blue), shifts, strict=True
        ):
            matrix = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float64)
            warped = negpy_core.warp_affine(
                plane, matrix, (w, h), interpolation="cubic", border="replicate"
            )
            merged[..., output_channel] = warped[y0:y1, x0:x1]
    if not np.isfinite(merged).all():
        raise ValueError("Registered output contains nonfinite interpolation values")
    receipt["common_valid_roi_sensor_xyxy"] = list(common)
    receipt["output_shape_h_w_c"] = list(merged.shape)
    if model_check is not None:
        source_check = model_check == TRANSLATION_CONSISTENCY_SOURCE_V2
        receipt["version"] = 3 if source_check else 2
        receipt["model_check"] = {
            "schema": "pure-rgb-registration-model-check",
            "version": 2 if source_check else 1,
            "policy": model_check,
            "status": "passed",
            "source_plane_sha256": source_hashes,
            "channels": model_proofs,
        }
    return merged, receipt
