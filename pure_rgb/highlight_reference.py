"""Opt-in working-linear highlight luminance from a second Native render.

The caller must establish that both renders use the same source, geometry and
working colour space. This module borrows only reference luminance: it cannot
reconstruct clipped source data or the base render's lost highlight chroma.
Its EV limit describes a burn amount, never recoverable scene exposure range.
"""

import math

import numpy as np

from .positive_float import LUMA, positive_domain, working_icc_bytes


SCHEMA = "pure-rgb-highlight-reference"
VERSION = 1
LUMINANCE_EPSILON = 1e-12
BLOCK_PIXELS = 262144
DEFAULTS = {
    "version": VERSION,
    "strength": 0.0,
    "range_start": 0.08,
    "range_end": 0.50,
    "max_burn_ev": 6.0,
}


def checked_highlight_reference(document=None):
    """Return independent finite JSON parameters; absent means disabled."""
    if document is None:
        document = {}
    if not isinstance(document, dict) or set(document) - set(DEFAULTS):
        raise ValueError("Highlight reference expects only its versioned controls")
    result = {**DEFAULTS, **document}
    if type(result["version"]) is not int or result["version"] != VERSION:
        raise ValueError("Unsupported highlight reference version")
    for key in ("strength", "range_start", "range_end", "max_burn_ev"):
        value = result[key]
        if type(value) not in (int, float):
            raise ValueError("Highlight reference controls require finite JSON numbers")
        try:
            scalar = float(value)
        except OverflowError as error:
            raise ValueError(
                "Highlight reference controls require finite JSON numbers"
            ) from error
        if not math.isfinite(scalar):
            raise ValueError("Highlight reference controls require finite JSON numbers")
        result[key] = scalar
    if not 0 <= result["strength"] <= 1:
        raise ValueError("Highlight reference strength must be in [0,1]")
    if not 0 <= result["range_start"] < result["range_end"] <= 1:
        raise ValueError(
            "Highlight reference requires 0 <= range_start < range_end <= 1"
        )
    if not 0 <= result["max_burn_ev"] <= 8:
        raise ValueError("Highlight reference max_burn_ev must be in [0,8]")
    return result


def _blocks(image):
    rows = max(1, BLOCK_PIXELS // image.shape[1])
    for start in range(0, image.shape[0], rows):
        yield start, image[start : start + rows]


def _inspect(image):
    if (
        not isinstance(image, np.ndarray)
        or image.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or image.ndim != 3
        or image.shape[-1] != 3
        or image.size == 0
        or not image.flags.c_contiguous
    ):
        raise ValueError("Expected C-contiguous float32/64 HWC RGB in [0,1]")
    minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
    for _, block in _blocks(image):
        if not np.isfinite(block).all() or np.any(block < 0) or np.any(block > 1):
            raise ValueError("Highlight reference requires finite bounded [0,1] RGB")
        minimum = np.minimum(minimum, block.min((0, 1)))
        maximum = np.maximum(maximum, block.max((0, 1)))
    return {"minimum_rgb": minimum.tolist(), "maximum_rgb": maximum.tolist()}


def _domain(document):
    expected = positive_domain("linear", working_icc_bytes())
    if (
        not isinstance(document, dict)
        or set(document) != set(expected)
        or type(document.get("version")) is not int
        or document != expected
    ):
        raise ValueError(
            "Highlight reference requires the explicit linear working ICC domain"
        )
    return expected


def _weight(y, start, end):
    # Only divide interior samples. Tiny but valid finite range widths must
    # not overflow while evaluating pixels already outside the transition.
    weight = np.zeros_like(y)
    weight[y >= end] = 1
    inside = (y > start) & (y < end)
    t = (y[inside] - start) / (end - start)
    weight[inside] = t * t * (3 - 2 * t)
    return weight


def apply_highlight_reference(base_linear, reference_linear, document=None, *, domain):
    """Borrow a bounded reference's brightness structure without lifting pixels.

    Both arrays are positive working-linear RGB, not normalized film density
    or working-OETF codes. A neutral call returns the original base object.
    Active results have the base dtype and independent read-only storage.
    """
    controls = checked_highlight_reference(document)
    checked_domain = _domain(domain)
    base_range, reference_range = _inspect(base_linear), _inspect(reference_linear)
    if (
        base_linear.shape != reference_linear.shape
        or base_linear.dtype != reference_linear.dtype
    ):
        raise ValueError("Base and reference must have identical shape and dtype")
    count = base_linear.shape[0] * base_linear.shape[1]
    receipt = {
        "schema": SCHEMA,
        "version": VERSION,
        "controls": controls,
        "domain": checked_domain,
        "shape": list(base_linear.shape),
        "dtype": base_linear.dtype.str,
        "pixel_count": count,
        "luminance_weights_rgb": LUMA.tolist(),
        "ratio_luminance_epsilon": LUMINANCE_EPSILON,
        "base_input_range": base_range,
        "reference_input_range": reference_range,
        "formula": "dEV=clip(log2(max(Yref,eps)/max(Ybase,eps)),-limit,0); gain=2**(strength*smoothstep(Ybase,start,end)*dEV); RGBout=RGBbase*gain",
        "reference_role": "luminance only; reference RGB/chroma is not transferred",
        "colour_policy": "one common gain preserves base RGB ratios apart from dtype rounding; clipped base chroma is not reconstructed",
        "source_geometry_certified": False,
        "source_geometry_requirement": "caller must separately bind the same source, full-frame geometry and working ICC for both renders",
        "pixel_sha256_computed": False,
        "signal_hard_clip_performed": False,
        "integer_encoding_performed": False,
        "source_recovery_certified": False,
        "burn_limit_is_recovered_scene_EV": False,
    }
    if controls["strength"] == 0:
        receipt.update(
            status="disabled",
            identity=True,
            output_range={key: list(value) for key, value in base_range.items()},
            affected_pixel_count=0,
            affected_pixel_fraction=0.0,
            applied_burn_ev={"minimum": 0.0, "mean": 0.0, "maximum": 0.0},
            gain_range=[1.0, 1.0],
        )
        return base_linear, receipt

    output = np.empty_like(base_linear)
    affected = gated = black = reference_not_darker = limit_clamped = 0
    burn_sum, burn_min, burn_max = 0.0, np.inf, 0.0
    gain_min, gain_max = 1.0, 0.0
    out_minimum, out_maximum = np.full(3, np.inf), np.full(3, -np.inf)
    for start, block in _blocks(base_linear):
        base = block.astype(np.float64, copy=False)
        reference = reference_linear[start : start + block.shape[0]].astype(
            np.float64, copy=False
        )
        y_base = np.einsum("...c,c->...", base, LUMA)
        y_reference = np.einsum("...c,c->...", reference, LUMA)
        weight = _weight(y_base, controls["range_start"], controls["range_end"])
        raw_ev = np.log2(
            np.maximum(y_reference, LUMINANCE_EPSILON)
            / np.maximum(y_base, LUMINANCE_EPSILON)
        )
        ev = np.clip(raw_ev, -controls["max_burn_ev"], 0)
        burn = -controls["strength"] * weight * ev
        burn[y_base == 0] = 0
        gain = np.exp2(-burn)
        result = (base * gain[..., None]).astype(base_linear.dtype)
        output[start : start + block.shape[0]] = result
        affected += int(np.count_nonzero(np.any(result != block, axis=-1)))
        gated += int(np.count_nonzero(weight > 0))
        black += int(np.count_nonzero(y_base == 0))
        reference_not_darker += int(np.count_nonzero(y_reference >= y_base))
        limit_clamped += int(
            np.count_nonzero((weight > 0) & (raw_ev < -controls["max_burn_ev"]))
        )
        burn_sum += float(burn.sum(dtype=np.float64))
        burn_min, burn_max = (
            min(burn_min, float(burn.min())),
            max(burn_max, float(burn.max())),
        )
        gain_min, gain_max = (
            min(gain_min, float(gain.min())),
            max(gain_max, float(gain.max())),
        )
        out_minimum = np.minimum(out_minimum, result.min((0, 1)))
        out_maximum = np.maximum(out_maximum, result.max((0, 1)))
    output.setflags(write=False)
    receipt.update(
        status="applied",
        identity=affected == 0,
        output_range={
            "minimum_rgb": out_minimum.tolist(),
            "maximum_rgb": out_maximum.tolist(),
        },
        affected_pixel_count=affected,
        affected_pixel_fraction=affected / count,
        positive_range_weight_pixel_count=gated,
        black_base_pixel_count=black,
        reference_not_darker_pixel_count=reference_not_darker,
        requested_burn_beyond_limit_and_positive_weight_pixel_count=limit_clamped,
        applied_burn_ev={
            "minimum": burn_min,
            "mean": burn_sum / count,
            "maximum": burn_max,
        },
        gain_range=[gain_min, gain_max],
    )
    return output, receipt
