"""Verified FlexColor 4.8.13 post-gradation RGB14 color-correction stage.

This is the original six-color difference model, not HSL saturation or a
device-back LUT. It does not establish camera/scanner input calibration.
"""

import numpy as np


# Each row is an original PrepareColorMap coefficient, RGB × R/G/B/C/M/Y.
# Entries 0..35 index XML ColorCorr; entry 36 is signed Saturation.
_COEFFICIENT_TERMS = (
    (1, 2, 3),
    (7, 8, 9, 36),
    (13, 14, 15, 36),
    (19, 20, 21, 36),
    (25, 26, 27),
    (31, 32, 33),
    (0, 2, 4, 36),
    (6, 8, 10),
    (12, 14, 16, 36),
    (18, 20, 22),
    (24, 26, 28, 36),
    (30, 32, 34),
    (0, 1, 5, 36),
    (6, 7, 11, 36),
    (12, 13, 17),
    (18, 19, 23),
    (24, 25, 29),
    (30, 31, 35, 36),
)


def lightness_stage_request(lightness, *, curve_present=True, enabled=True):
    """Describe original ApplyYGrad's gate (enabled=original ApplySliders)."""
    if type(lightness) is not int or not -32768 <= lightness <= 32767:
        raise ValueError("Lightness需明确signed-short整数")
    if type(curve_present) is not bool or type(enabled) is not bool:
        raise ValueError("Lightness gate开关需明确布尔值")
    active = curve_present and lightness > 0 and enabled
    return {
        "lightness": lightness,
        "curve_present": curve_present,
        "enabled": enabled,
        "enabled_field": "ApplySliders (ic+0x5e8)",
        "stage_active": active,
        "bypass_proven": not active,
        "active_positive_transform_implemented": True,
        "requires_explicit_external_anchor_word": active,
        "CMS_required_for_this_stage": False,
        "scope": "Original ApplyYGrad gate; the separate four-point Y curve and fixed 333/1000 RGB proxy are implemented in flex_lightness. Active execution requires an explicit external anchor word; no generic brightness substitute or inferred preference state.",
    }


def color_map_parameters(*, color_corr, saturation, apply_cc):
    """Return original signed-word coefficients and gates without touching pixels."""
    if not isinstance(color_corr, (list, tuple)) or len(color_corr) != 36:
        raise ValueError("ColorCorr 需原始36项一维列表，不可解释为3×3矩阵")
    values = [*color_corr, saturation]
    if any(type(v) is not int or not -32768 <= v <= 32767 for v in values):
        raise ValueError("ColorCorr/Saturation需明确的signed-short整数")
    if type(apply_cc) is not bool:
        raise ValueError("ApplyCC需明确布尔值")
    not_empty = any(value + saturation != 0 for value in color_corr)
    coefficients = [
        ((sum(values[index] for index in terms) + 32768) & 65535) - 32768
        for terms in _COEFFICIENT_TERMS
    ]
    return {
        "apply_cc": apply_cc,
        "not_empty": not_empty,
        "stage_active": apply_cc and not_empty,
        "coefficients_rgb_by_sector": [coefficients[i : i + 6] for i in (0, 6, 12)],
        "sector_order": ["R", "G", "B", "C", "M", "Y"],
        "saturation_zero_is_neutral_with_zero_color_corr": True,
        "saturation_15_is_neutral": False,
        "position": "after three final Gradation LUTs; before output ICC",
        "input_domain": "flex-post-gradation-rgb14",
        "original_input_count_units": "channel words (three per RGB pixel)",
        "bypass_contract": "original function returns without writing destination",
        "extra_gamma_or_oetf_applied": False,
        "device_colormap_or_raw_camera_correction_applied": False,
    }


def color_map14(
    pixels,
    *,
    color_corr,
    saturation,
    apply_cc,
    input_domain,
    out=None,
    chunk_pixels=262_144,
):
    """Apply original six-color corrections to explicitly declared RGB14 codes.

    ``out=None`` initializes a copy, then applies the original stage. ``out=pixels``
    supports the original in-place call. A separately supplied output follows the
    original no-write bypass contract; the caller must initialize it as needed.
    Strict RGB14 bounds are required before this stage and no ICC is assigned.
    """
    if input_domain != "flex-post-gradation-rgb14":
        raise ValueError("只接受明确的flex-post-gradation-rgb14域，不能隐式解释RAW")
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.uint16
        or pixels.ndim not in (2, 3)
        or pixels.shape[-1] != 3
        or not pixels.size
        or not pixels.flags.c_contiguous
    ):
        raise ValueError("需非空、C-contiguous、uint16的N×3/H×W×3 RGB")
    if pixels.max() > 16383:
        raise ValueError("RGB14需0..16383；不能将16384哨兵或RGB16直接送入")
    if type(chunk_pixels) is not int or not 1 <= chunk_pixels <= 4_194_304:
        raise ValueError("分块需1..4194304像素")
    params = color_map_parameters(
        color_corr=color_corr, saturation=saturation, apply_cc=apply_cc
    )
    if out is None:
        out = pixels.copy()
    elif (
        not isinstance(out, np.ndarray)
        or out.shape != pixels.shape
        or out.dtype != np.uint16
        or not out.flags.c_contiguous
        or not out.flags.writeable
    ):
        raise ValueError("输出需同形状、可写、C-contiguous uint16 RGB缓冲区")
    if np.shares_memory(out, pixels) and out.ctypes.data != pixels.ctypes.data:
        raise ValueError("仅支持同一原位缓冲区或互不重叠的独立输出")
    if not params["stage_active"]:
        return out
    source, destination = pixels.reshape(-1, 3), out.reshape(-1, 3)
    coefficients = params["coefficients_rgb_by_sector"]
    for offset in range(0, len(source), chunk_pixels):
        # Snapshot all input channels before any in-place output writes.
        rgb = source[offset : offset + chunk_pixels].astype(np.int32)
        r, g, b = rgb.T
        sectors = (
            np.maximum(r - np.maximum(g, b), 0),
            np.maximum(g - np.maximum(r, b), 0),
            np.maximum(b - np.maximum(r, g), 0),
            np.maximum(np.minimum(g, b) - r, 0),
            np.maximum(np.minimum(r, b) - g, 0),
            np.maximum(np.minimum(r, g) - b, 0),
        )
        basis = tuple(v.astype(np.float32) for v in sectors)
        for channel, terms in enumerate(coefficients):
            correction = basis[0] * np.float32(terms[0])
            for sector in range(1, 6):
                correction = correction + basis[sector] * np.float32(terms[sector])
            correction = correction / np.float32(-100)
            value = rgb[:, channel].astype(np.float32) + correction
            # Original SSE truncates to int32, then tests the signed low word,
            # before its unsigned 14-bit upper clamp; this is not generic clip.
            word = np.trunc(value).astype(np.int32) & 65535
            word = np.where(word >= 32768, 0, np.minimum(word, 16383))
            destination[offset : offset + len(rgb), channel] = word.astype(np.uint16)
    return out
