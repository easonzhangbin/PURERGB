"""Original Mac ApplyYGrad arithmetic, distinct from generic brightness.

The stage consumes post-ColorMap RGB14 codes. Its Y is the original fixed
333/1000 arithmetic proxy, not ICC luminance. The separate four-point curve
depends on an external original global word; that state is explicit here.
No preset, input calibration or production default is selected by this API.
"""

import hashlib

import numpy as np

from .flex_curves import _build


INPUT_DOMAIN = "flex-post-colormap-rgb14"
SOURCE_SHA256 = "cf14e1086e731ee641ba3975524e8e23476f2922ca589453708bcb566921570e"


def _request(lightness, anchor_preference, enabled, curve_present):
    if type(lightness) is not int or not -32768 <= lightness <= 32767:
        raise ValueError("Lightness requires an explicit signed-short integer")
    if type(enabled) is not bool or type(curve_present) is not bool:
        raise ValueError("ApplySliders and curve presence require explicit booleans")
    if anchor_preference is not None and (
        type(anchor_preference) is not int or not 0 <= anchor_preference <= 65535
    ):
        raise ValueError("Original external anchor word requires uint16 or None")
    active = curve_present and lightness > 0 and enabled
    if active and anchor_preference is None:
        raise ValueError(
            "Active Lightness requires explicit original external anchor state; "
            "do not infer it from a film profile"
        )
    return active


def lightness_curve_parameters(
    lightness, *, anchor_preference=None, enabled=True, curve_present=True
):
    """Original setter's byte narrowing and gate, with no guessed global state.

    The original global pointer at 0x9b7020 supplies a uint16 word at +0x18.
    Values 1..255 are used; all other values fall back to 50. Its application
    preference/UI name and live state are not recovered by this API.
    """
    active = _request(lightness, anchor_preference, enabled, curve_present)
    anchor = (
        anchor_preference
        if anchor_preference is not None and 1 <= anchor_preference <= 255
        else 50
    )
    product = 250 * lightness
    quotient = product // 100 if product >= 0 else -((-product) // 100)
    points = ((0, 0), (2, (2 + quotient) & 255), (anchor, anchor), (255, 255))
    return {
        "schema": "pure-rgb-flex-lightness-parameters-v1",
        "source_sha256": SOURCE_SHA256,
        "original_apply_function": "0x2e8b1c",
        "original_setter_function": "0x2ea2d8",
        "lightness": lightness,
        "enabled": enabled,
        "enabled_field": "ApplySliders (ic+0x5e8)",
        "curve_present": curve_present,
        "stage_active": active,
        "bypass_proven": not active,
        "external_anchor_word": anchor_preference,
        "effective_anchor": anchor if active else None,
        "anchor_fallback_applied": active and not 1 <= anchor_preference <= 255,
        "external_state_binding": "uint16 at *(0x9b7020)+0x18; live state/UI preference name not inferred",
        "curve_points": [list(p) for p in points] if active else None,
        "curve_is_separate_from_seven_Gradations": True,
        "curve_field": "ic+0x84",
        "setter_byte_formula": "point1=(2,(2+trunc(250*Lightness/100)) & 255); point2=(anchor,anchor)",
        "Y_formula": "floor(333*(R+G+B)/1000)",
        "Y_is_ICC_luminance": False,
        "CMS_required_for_this_stage": False,
        "camera_to_scanner_calibration_verified": False,
        "production_defaults_changed": False,
    }


def lightness_lut(
    lightness, *, anchor_preference=None, enabled=True, curve_present=True
):
    """Return immutable original four-point curve LUT and explicit receipt.

    Original anchor values 1/2/255 can repeat or reorder adjacent control X.
    The original parametric B-spline still accepts them. They are derived
    internal points, so the existing public user-point validation is not used;
    the same independently checked numerical builder is reused directly.
    """
    parameters = lightness_curve_parameters(
        lightness,
        anchor_preference=anchor_preference,
        enabled=enabled,
        curve_present=curve_present,
    )
    points = (
        tuple(tuple(p) for p in parameters["curve_points"])
        if parameters["stage_active"]
        else ((0, 0), (255, 255))
    )
    table = _build(points, 14)
    return table, parameters | {
        "table_sha256": hashlib.sha256(
            table.astype("<u2", copy=False).tobytes()
        ).hexdigest(),
        "table_dtype": "uint16",
        "table_entries": 16384,
        "original_Darwin_numeric_service_replaced": False,
    }


def _pixels(pixels, domain):
    if domain != INPUT_DOMAIN:
        raise ValueError("Require explicit flex-post-colormap-rgb14 input domain")
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.uint16
        or pixels.ndim not in (2, 3)
        or pixels.shape[-1] != 3
        or not pixels.size
        or not pixels.flags.c_contiguous
        or pixels.max() > 16383
    ):
        raise ValueError("Require nonempty contiguous N×3/H×W×3 uint16 RGB14")
    return pixels


def apply_lightness14(
    pixels,
    *,
    lightness,
    anchor_preference=None,
    enabled=True,
    curve_present=True,
    input_domain=INPUT_DOMAIN,
    out=None,
    chunk_pixels=32768,
):
    """Apply original Y-curve offset; return (pixels, numerical receipt).

    Original: delta=int16(curve[Y]-Y), v=RGB+delta; v>=16384 becomes16383,
    otherwise store low16(v), including negative wrap. No lower clamp is added.
    Extreme positive Lightness can wrap the setter byte and therefore produce
    values >16383; the NEXT RGB14/ICC stage must validate its input domain.

    The original sign-extends its loop counter and uses an unsigned comparison:
    it processes at most32768 pixels, so a longer call leaves its tail unchanged.
    This wrapper splits larger images into complete segments of at most32768.
    Original ApplyYGrad is in-place; separate output here is an
    explicit convenience copy and is initialized even when the stage bypasses.
    """
    source = _pixels(pixels, input_domain)
    if type(chunk_pixels) is not int or not 1 <= chunk_pixels <= 32768:
        raise ValueError("Lightness segments require 1..32768 pixels")
    table, receipt = lightness_lut(
        lightness,
        anchor_preference=anchor_preference,
        enabled=enabled,
        curve_present=curve_present,
    )
    if out is None:
        out = source.copy()
    elif (
        not isinstance(out, np.ndarray)
        or out.dtype != np.uint16
        or out.shape != source.shape
        or not out.flags.c_contiguous
        or not out.flags.writeable
    ):
        raise ValueError("Output requires writable contiguous same-shape uint16 RGB")
    elif np.shares_memory(out, source) and out.ctypes.data != source.ctypes.data:
        raise ValueError("Only exact in-place or disjoint output buffers supported")
    else:
        np.copyto(out, source)
    clamped = np.zeros(3, dtype=np.int64)
    wrapped = np.zeros(3, dtype=np.int64)
    delta_min = delta_max = 0
    flat_source, destination = source.reshape(-1, 3), out.reshape(-1, 3)
    if receipt["stage_active"]:
        delta_min, delta_max = 32767, -32768
        for start in range(0, len(flat_source), chunk_pixels):
            rgb = flat_source[start : start + chunk_pixels].astype(np.int32)
            y = (333 * rgb.sum(1, dtype=np.int64) // 1000).astype(np.int32)
            delta = ((table[y].astype(np.int32) - y + 32768) & 65535) - 32768
            value = rgb + delta[:, None]
            clamped += (value >= 16384).sum(0)
            wrapped += (value < 0).sum(0)
            delta_min = min(delta_min, int(delta.min()))
            delta_max = max(delta_max, int(delta.max()))
            destination[start : start + len(rgb)] = np.where(
                value >= 16384, 16383, value & 65535
            ).astype(np.uint16)
    return out, receipt | {
        "input_domain": input_domain,
        "order": "post-final Gradations -> ColorMap -> ApplyYGrad -> optional USM -> final world2 ICC",
        "maximum_segment_pixels": chunk_pixels,
        "original_maximum_complete_call_pixels": 32768,
        "long_original_calls_leave_tail_unprocessed": True,
        "original_count_unit": "RGB pixels, not channel words",
        "delta_minimum_maximum": [delta_min, delta_max],
        "upper_clamped_sample_fraction_rgb": (clamped / len(flat_source)).tolist(),
        "negative_wrapped_sample_fraction_rgb": (wrapped / len(flat_source)).tolist(),
        "output_outside_RGB14_sample_fraction_rgb": (destination > 16383)
        .mean(0)
        .tolist(),
        "next_RGB14_stage_must_validate_domain": True,
        "output_is_unbounded_float_or_scene_recovery": False,
        "extra_gamma_or_ICC_transform_applied": False,
        "separate_output_is_explicit_wrapper_copy": True,
    }
