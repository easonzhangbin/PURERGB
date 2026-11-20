"""Guarded version-2 reference-luminance burn with explicit protection.

For a fixed reference/base luminance ratio above the luminance floor in both
images and a fixed protection value, the ideal real-valued mapping has log
slope at least min_log_slope. A spatially
varying reference or mask does not have this global ordering guarantee. Neither
source geometry nor semantic person detection is certified by this module.
"""

import math

import numpy as np

from .positive_float import LUMA, positive_domain, working_icc_bytes


SCHEMA = "pure-rgb-highlight-reference-guarded"
VERSION = 2
BLOCK_PIXELS = 262144
LUMINANCE_EPSILON = 1e-12
DEFAULTS = {
    "version": VERSION,
    "strength": 0.0,
    "range_start": 0.08,
    "budget_reference": 0.50,
    "max_burn_ev": 6.0,
    "min_log_slope": 0.2,
    "protection_strength": 1.0,
}


def checked_highlight_reference_guarded(document=None):
    """Canonical JSON controls; budget_reference sets curvature, not full effect."""
    if document is None:
        document = {}
    if not isinstance(document, dict) or set(document) - set(DEFAULTS):
        raise ValueError("Guarded highlight reference expects only version-2 controls")
    result = {**DEFAULTS, **document}
    if type(result["version"]) is not int or result["version"] != VERSION:
        raise ValueError("Unsupported guarded highlight reference version")
    for key in set(DEFAULTS) - {"version"}:
        value = result[key]
        if type(value) not in (int, float):
            raise ValueError("Guarded controls require finite JSON numbers")
        try:
            scalar = float(value)
        except OverflowError as error:
            raise ValueError("Guarded controls require finite JSON numbers") from error
        if not math.isfinite(scalar):
            raise ValueError("Guarded controls require finite JSON numbers")
        result[key] = scalar
    if not 0 < result["range_start"] < result["budget_reference"] <= 1:
        raise ValueError("Require 0 < range_start < budget_reference <= 1")
    if not 0 <= result["max_burn_ev"] <= 8:
        raise ValueError("max_burn_ev must be in [0,8]")
    for key in ("strength", "min_log_slope", "protection_strength"):
        if not 0 <= result[key] <= 1:
            raise ValueError(f"{key} must be in [0,1]")
    return result


def _blocks(image):
    rows = max(1, BLOCK_PIXELS // image.shape[1])
    for start in range(0, image.shape[0], rows):
        yield start, image[start : start + rows]


def _inspect(image, *, mask=False):
    valid_shape = (
        isinstance(image, np.ndarray)
        and image.ndim == (2 if mask else 3)
        and (mask or image.shape[-1] == 3)
    )
    if (
        not valid_shape
        or image.size == 0
        or image.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or not image.flags.c_contiguous
    ):
        raise ValueError("Expected finite C-order float32/64 HW mask or HWC RGB")
    minimum, maximum = np.inf, -np.inf
    for _, block in _blocks(image):
        if not np.isfinite(block).all() or np.any(block < 0) or np.any(block > 1):
            raise ValueError("Guarded inputs must be finite and bounded [0,1]")
        minimum, maximum = (
            min(minimum, float(block.min())),
            max(maximum, float(block.max())),
        )
    return {"minimum": minimum, "maximum": maximum}


def _domain(document):
    expected = positive_domain("linear", working_icc_bytes())
    if (
        not isinstance(document, dict)
        or set(document) != set(expected)
        or type(document.get("version")) is not int
        or document != expected
    ):
        raise ValueError("Explicit linear working ICC domain required")
    return expected


def _tau(start, reference):
    if reference <= 2 * start:
        return math.log1p((reference - start) / start) / (3 * math.log(2))
    return (math.log2(reference) - math.log2(start)) / 3


def _budget(y, start, tau, minimum_slope):
    z = np.zeros_like(y)
    near = (y > start) & (y <= 2 * start)
    z[near] = np.log1p((y[near] - start) / start) / math.log(2)
    far = y > 2 * start
    z[far] = np.log2(y[far]) - math.log2(start)
    u = z / tau
    h = np.empty_like(z)
    small = u < 1e-3
    t = u[small]
    # u + expm1(-u) starts at u^2/2. This positive series avoids
    # subtracting two nearly equal first-order terms at the C1 onset.
    h[small] = (
        tau * t * t * (0.5 + t * (-1 / 6 + t * (1 / 24 + t * (-1 / 120 + t / 720))))
    )
    h[~small] = z[~small] + tau * np.expm1(-u[~small])
    return (1 - minimum_slope) * h


def apply_highlight_reference_guarded(
    base_linear, reference_linear, document=None, *, domain, protection_mask=None,
    effect_mask=None,
):
    """Return (working-linear RGB, receipt), with RGB-common burn/protection.

    Protection is one HW float mask of the image dtype: 1 protects, 0 permits
    the effect. The caller binds its source/geometry; no mask means zero
    protection. An optional effect mask uses 1 for full effect and 0 for no
    effect. Disabled/zero-budget calls return the original base object.
    """
    controls = checked_highlight_reference_guarded(document)
    checked_domain = _domain(domain)
    base_range, reference_range = _inspect(base_linear), _inspect(reference_linear)
    if (
        base_linear.shape != reference_linear.shape
        or base_linear.dtype != reference_linear.dtype
    ):
        raise ValueError("Base/reference shape and dtype must match")
    mask_range = None
    if protection_mask is not None:
        mask_range = _inspect(protection_mask, mask=True)
        if (
            protection_mask.shape != base_linear.shape[:2]
            or protection_mask.dtype != base_linear.dtype
        ):
            raise ValueError("Protection must be HW with the base shape and dtype")
    effect_range = None
    if effect_mask is not None:
        effect_range = _inspect(effect_mask, mask=True)
        if effect_mask.shape != base_linear.shape[:2] or effect_mask.dtype != base_linear.dtype:
            raise ValueError("Effect area must be HW with the base shape and dtype")
    local_start = max(controls["range_start"], 0.35) if effect_mask is not None else controls["range_start"]
    local_reference = max(controls["budget_reference"], 0.75) if effect_mask is not None else controls["budget_reference"]
    tau = _tau(local_start, local_reference)
    count = base_linear.shape[0] * base_linear.shape[1]
    receipt = {
        "schema": SCHEMA,
        "version": VERSION,
        "controls": controls,
        "domain": checked_domain,
        "shape": list(base_linear.shape),
        "dtype": base_linear.dtype.str,
        "pixel_count": count,
        "base_input_range": base_range,
        "reference_input_range": reference_range,
        "protection_mask_present": protection_mask is not None,
        "protection_mask_range": mask_range,
        "protection_is_automatic_person_detection": False,
        "luminance_weights_rgb": LUMA.tolist(),
        "ratio_luminance_epsilon": LUMINANCE_EPSILON,
        "budget_tau_log2_luminance": tau,
        "budget_reference_role": "sets tau=log2(budget_reference/range_start)/3; not a full-effect endpoint or semantic highlight classifier",
        "formula": "z=max(log2(Ybase/start),0); h=z+tau*expm1(-z/tau); b=(1-min_log_slope)*h; D=strength*clip(log2(max(Ybase,eps)/max(Yref,eps)),0,max_burn_ev); burn=-b*expm1(-D/b), with b=0 ->0; burn*=1-protection_strength*mask; RGBout=RGBbase*2**(-burn)",
        "ordering_guarantee": "ideal real-valued dlog2(Yout)/dlog2(Ybase)>=min_log_slope only for constant reference/base luminance ratio, with both luminances above ratio_luminance_epsilon, and constant protection value; dtype rounding may tie values",
        "global_spatial_ordering_certified": False,
        "ordering_limitations": "spatially changing reference ratio or protection can reorder brightness and form boundaries; no halo-free or semantic guarantee",
        "colour_policy": "RGB-common gain preserves base channel ratios apart from dtype rounding; reference chroma and clipped base chroma are not reconstructed",
        "source_geometry_certified": False,
        "source_geometry_requirement": "caller separately binds both renders and the optional mask to one source/full-frame geometry/working ICC",
        "pixel_sha256_computed": False,
        "signal_hard_clip_performed": False,
        "integer_encoding_performed": False,
        "burn_limit_is_recovered_scene_EV": False,
    }
    if effect_mask is not None:
        receipt.update(effect_mask_present=True, effect_mask_range=effect_range)
        receipt["formula"] += "; local budget uses the recorded local start and reference; burn*=effect_mask"
        receipt["local_highlight_start"] = local_start
        receipt["local_highlight_reference"] = local_reference
        receipt["budget_reference_role"] = "local_highlight_reference sets the local budget curvature"
        receipt["ordering_guarantee"] += "; the effect mask must also be constant"
    if effect_mask is not None or protection_mask is not None:
        receipt["local_selection_policy"] = "highlight-luminance-soft-v2"
    if (
        controls["strength"] == 0
        or controls["max_burn_ev"] == 0
        or controls["min_log_slope"] == 1
    ):
        receipt.update(
            status="disabled" if controls["strength"] == 0 else "zero-budget",
            identity=True,
            output_range=dict(base_range),
            affected_pixel_count=0,
            affected_pixel_fraction=0.0,
            applied_burn_ev={"minimum": 0.0, "mean": 0.0, "maximum": 0.0},
            gain_range=[1.0, 1.0],
        )
        return base_linear, receipt
    output = np.empty_like(base_linear)
    affected = fully_protected = 0
    burn_sum, burn_min, burn_max = 0.0, np.inf, 0.0
    gain_min, gain_max = 1.0, 0.0
    out_minimum, out_maximum = np.inf, -np.inf
    for start, block in _blocks(base_linear):
        base = block.astype(np.float64, copy=False)
        reference = reference_linear[start : start + block.shape[0]].astype(
            np.float64, copy=False
        )
        y_base = np.einsum("...c,c->...", base, LUMA)
        y_reference = np.einsum("...c,c->...", reference, LUMA)
        budget = _budget(y_base, local_start, tau, controls["min_log_slope"])
        desired = controls["strength"] * np.clip(
            np.log2(
                np.maximum(y_base, LUMINANCE_EPSILON)
                / np.maximum(y_reference, LUMINANCE_EPSILON)
            ),
            0,
            controls["max_burn_ev"],
        )
        burn = np.zeros_like(budget)
        positive = budget > 0
        # D/b may overflow only toward +infinity for a vanishing positive
        # budget. expm1(-infinity)=-1 is the correct finite soft-limit value.
        with np.errstate(over="ignore", under="ignore"):
            burn[positive] = -budget[positive] * np.expm1(
                -desired[positive] / budget[positive]
            )
        if protection_mask is not None:
            mask = protection_mask[start : start + block.shape[0]].astype(
                np.float64, copy=False
            )
            protected = controls["protection_strength"] * mask
            burn *= 1 - protected
            fully_protected += int(np.count_nonzero(protected == 1))
        if effect_mask is not None:
            burn *= effect_mask[start : start + block.shape[0]]
        gain = np.exp2(-burn)
        result = (base * gain[..., None]).astype(base_linear.dtype)
        output[start : start + block.shape[0]] = result
        affected += int(np.count_nonzero(np.any(result != block, axis=-1)))
        burn_sum += float(burn.sum(dtype=np.float64))
        burn_min, burn_max = (
            min(burn_min, float(burn.min())),
            max(burn_max, float(burn.max())),
        )
        gain_min, gain_max = (
            min(gain_min, float(gain.min())),
            max(gain_max, float(gain.max())),
        )
        out_minimum, out_maximum = (
            min(out_minimum, float(result.min())),
            max(out_maximum, float(result.max())),
        )
    output.setflags(write=False)
    receipt.update(
        status="applied",
        identity=affected == 0,
        output_range={"minimum": out_minimum, "maximum": out_maximum},
        affected_pixel_count=affected,
        affected_pixel_fraction=affected / count,
        fully_protected_pixel_count=fully_protected,
        applied_burn_ev={
            "minimum": burn_min,
            "mean": burn_sum / count,
            "maximum": burn_max,
        },
        gain_range=[gain_min, gain_max],
    )
    return output, receipt
