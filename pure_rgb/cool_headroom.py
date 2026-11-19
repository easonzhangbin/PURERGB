"""Independent active-only density headroom experiment for frozen Cool fits.

Selected sensor-band signal is not calibrated transmittance. Its signed
scene-relative log coordinate is not positive linear light. This opt-in route
compresses that coordinate before the original frozen shared power/scale and
then bounds its maximum with a common RGB gain. It neither executes a factory
CMS nor reconstructs absent/clipped source data. Disabled callers must retain
the complete historical route, including its intermediate RGB16 boundary.
"""

import math

import numpy as np

from .engines import Analysis
from .rgb_curve_study import SharedCurveFit


SCHEMA = "pure-rgb-cool-headroom"
VERSION = 1
SIGNAL_DOMAIN = "selected-band-linear-signal"
DENSITY_DOMAIN = "signed-scene-relative-log-density-fraction"
SOURCE_FLOOR = 1e-6
DENSITY_KNEE = 0.8
MAXIMUM_KNEE = 0.8
BLOCK_PIXELS = 262144
DEFAULTS = {"version": VERSION, "strength": 0.0}


def checked_cool_headroom(document=None):
    """Versioned finite controls; zero requires a complete old-route bypass."""
    if document is None:
        document = {}
    if not isinstance(document, dict) or set(document) - set(DEFAULTS):
        raise ValueError("Cool headroom expects only version and strength")
    result = {**DEFAULTS, **document}
    value = result["strength"]
    if type(result["version"]) is not int or result["version"] != VERSION:
        raise ValueError("Unsupported Cool headroom version")
    if type(value) not in (int, float):
        raise ValueError("Cool headroom strength must be a finite JSON number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError("Cool headroom strength must be finite") from error
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Cool headroom strength must be in [0,1]")
    if value > 0 and 1.0 - value * (1.0 - DENSITY_KNEE) == 1.0:
        raise ValueError("Nonzero strength is too small for a representable shoulder")
    result["strength"] = value
    return result


def _active(document):
    controls = checked_cool_headroom(document)
    if controls["strength"] == 0:
        raise ValueError(
            "Disabled Cool headroom must bypass to the complete historical route; "
            "new floating mathematics is not an exact replacement for its RGB16 path"
        )
    return controls


def _blocks(image):
    rows = max(1, BLOCK_PIXELS // image.shape[1])
    for start in range(0, image.shape[0], rows):
        yield start, image[start : start + rows]


def _pixels(image, *, signed=False):
    if (
        not isinstance(image, np.ndarray)
        or image.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or image.ndim != 3
        or image.shape[-1] != 3
        or image.size == 0
        or not image.flags.c_contiguous
    ):
        raise ValueError("Expected C-order float32/64 HWC RGB samples")
    for _, block in _blocks(image):
        if not np.isfinite(block).all() or (not signed and np.any(block < 0)):
            raise ValueError("Expected finite nonnegative selected-band signal")


def _domain(actual, expected):
    if not isinstance(actual, str) or actual != expected:
        raise ValueError("Explicit matching signal/density domain is required")


def _analysis(analysis, shape):
    if not isinstance(analysis, Analysis):
        raise ValueError("Frozen Analysis required; no reanalysis or fit")
    dark = np.asarray(analysis.dark, dtype=np.float64)
    base = np.asarray(analysis.base, dtype=np.float64)
    roi = analysis.roi
    if (
        dark.shape != (3,)
        or base.shape != (3,)
        or not np.isfinite(dark).all()
        or not np.isfinite(base).all()
        or np.any(dark < 0)
        or np.any(base - np.maximum(dark, SOURCE_FLOOR) < 1e-6)
        or not isinstance(roi, (tuple, list))
        or len(roi) != 4
        or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in roi)
    ):
        raise ValueError("Invalid frozen density endpoints or ROI")
    x0, y0, x1, y1 = (int(value) for value in roi)
    h, w = shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Frozen Analysis ROI must lie inside the actual source shape")
    percentiles = analysis.percentiles
    if percentiles is not None:
        if (
            not isinstance(percentiles, (tuple, list))
            or len(percentiles) != 2
            or any(isinstance(v, (bool, np.bool_)) for v in percentiles)
        ):
            raise ValueError("Invalid frozen percentile metadata")
        try:
            percentiles = [float(value) for value in percentiles]
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError("Invalid frozen percentile metadata") from error
        if not all(math.isfinite(v) for v in percentiles) or not 0 <= percentiles[0] < percentiles[1] <= 100:
            raise ValueError("Invalid frozen percentile metadata")
    log_base = np.log10(base)
    # Subtract logs rather than divide huge finite endpoints before log10.
    span = log_base - np.log10(np.maximum(dark, SOURCE_FLOOR))
    if not np.isfinite(span).all() or np.any(span <= 0):
        raise ValueError("Frozen density span must be finite and positive")
    frozen = {
        "dark_rgb": dark.tolist(),
        "base_rgb": base.tolist(),
        "roi_xyxy": [x0, y0, x1, y1],
        "percentiles": percentiles,
        "log10_span_rgb": span.tolist(),
        "dark_endpoint_floor_applied_rgb": (dark < SOURCE_FLOOR).tolist(),
    }
    return log_base, span, frozen


def _fit(fit):
    if not isinstance(fit, SharedCurveFit):
        raise ValueError("Frozen SharedCurveFit required; no new fitting")
    values = (fit.power, fit.scale, fit.valid_fraction, fit.span_residual_stops)
    if any(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError("Invalid finite frozen shared-fit numbers")
    try:
        power, scale, valid_fraction, residual = (float(value) for value in values)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("Invalid finite frozen shared-fit numbers") from error
    if (
        not math.isfinite(power)
        or not math.isfinite(scale)
        or not 0.1 <= power <= 12
        or scale <= 0
    ):
        raise ValueError("Invalid finite frozen shared power/scale")
    if (
        type(fit.grid_stride) is not int
        or fit.grid_stride < 1
        or not math.isfinite(valid_fraction)
        or not math.isfinite(residual)
        or not 0 <= valid_fraction <= 1
        or residual < 0
    ):
        raise ValueError("Invalid frozen shared-fit metadata")
    return {
        "power": float(power), "scale": float(scale),
        "grid_stride": fit.grid_stride,
        "valid_fraction": valid_fraction,
        "span_residual_stops": residual,
    }


def _density(source, log_base, span):
    return (log_base - np.log10(np.maximum(source, SOURCE_FLOOR))) / span


def _shoulder(values, knee):
    """C1 increasing rational shoulder in exact math; finite rounding can merge."""
    output = values.copy()
    selected = values > knee
    d = values[selected] - knee
    width = 1.0 - knee
    output[selected] = knee + width * (d / (width + d))
    return output


def _map_density(values, controls):
    # Explicit black policy for negative density coordinates. This is not an
    # input physical-RAW black calibration and not a positive-light clamp.
    positive = np.maximum(values, 0.0)
    knee = 1.0 - controls["strength"] * (1.0 - DENSITY_KNEE)
    return _shoulder(positive, knee), knee


def unclipped_density_fraction(source, analysis, *, domain):
    """Return signed scene-relative density u in float64, including u>1/u<0."""
    _domain(domain, SIGNAL_DOMAIN)
    _pixels(source)
    log_base, span, frozen = _analysis(analysis, source.shape)
    output = np.empty(source.shape, dtype=np.float64)
    low = np.zeros(3, np.int64)
    for start, block in _blocks(source):
        output[start : start + block.shape[0]] = _density(block.astype(np.float64), log_base, span)
        low += (block <= SOURCE_FLOOR).sum((0, 1))
    output.setflags(write=False)
    return output, {
        "schema": SCHEMA, "version": VERSION, "output_domain": DENSITY_DOMAIN,
        "formula": "u=(log10(base)-log10(max(signal,1e-6)))/(log10(base)-log10(max(dark,1e-6)))",
        "physical_transmittance_or_positive_linear_light": False,
        "frozen_analysis": frozen, "source_floor": SOURCE_FLOOR,
        "source_at_or_below_floor_count_rgb": low.tolist(),
        "u_below_zero_count_rgb": (output < 0).sum((0, 1)).tolist(),
        "u_above_one_count_rgb": (output > 1).sum((0, 1)).tolist(),
        "density_hard_clip_performed": False,
    }


def map_density_headroom(density, document=None, *, domain):
    """Active bounded density coordinate; no power, OETF, integer or ICC step."""
    _domain(domain, DENSITY_DOMAIN)
    controls = _active(document)
    _pixels(density, signed=True)
    output = np.empty(density.shape, dtype=np.float64)
    for start, block in _blocks(density):
        mapped, knee = _map_density(block.astype(np.float64), controls)
        output[start : start + block.shape[0]] = mapped
    output.setflags(write=False)
    return output, {
        "schema": SCHEMA, "version": VERSION, "controls": controls,
        "input_domain": DENSITY_DOMAIN,
        "output_domain": "bounded-density-rendering-coordinate",
        "base_density_knee": DENSITY_KNEE, "effective_density_knee": knee,
        "formula": "u<=k: max(u,0); u>k: k+(1-k)*(u-k)/((1-k)+(u-k)); k=1-strength*(1-.8)",
        "negative_u_policy": "u<0 is rendered black before frozen positive power",
        "negative_u_black_count_rgb": (density < 0).sum((0, 1)).tolist(),
        "mapped_exact_one_rounding_count_rgb": (output == 1).sum((0, 1)).tolist(),
        "preserves_input_rgb_chromaticity": False,
        "no_strict_distinct_float_value_or_physical_EV_claim": True,
    }


def render_cool_headroom(source, analysis, fit, document=None, *, domain):
    """Active-only floating density -> frozen fit -> common-maximum shoulder.

    The result is a density-derived rendering value assigned to the existing
    working primaries by the caller, not measured positive scene radiometry.
    The caller also owns the unchanged disabled route and final OETF/encoding.
    """
    _domain(domain, SIGNAL_DOMAIN)
    controls = _active(document)
    _pixels(source)
    log_base, span, frozen_analysis = _analysis(analysis, source.shape)
    frozen_fit = _fit(fit)
    output = np.empty_like(source)
    low = np.zeros(3, np.int64)
    negative = np.zeros(3, np.int64)
    above_one = np.zeros(3, np.int64)
    rounded_density_one = np.zeros(3, np.int64)
    rounded_output_one = np.zeros(3, np.int64)
    cast_to_zero = np.zeros(3, np.int64)
    density_changed = output_shoulder = actual_changed = 0
    maximum_pre_fit = 0.0
    minimum_out, maximum_out = np.full(3, np.inf), np.full(3, -np.inf)
    for start, block in _blocks(source):
        signal = block.astype(np.float64, copy=False)
        density = _density(signal, log_base, span)
        mapped, effective_knee = _map_density(density, controls)
        with np.errstate(under="ignore"):
            rendered = frozen_fit["scale"] * np.power(mapped, frozen_fit["power"])
        maximum = np.max(rendered, axis=-1)
        mapped_maximum = _shoulder(maximum, MAXIMUM_KNEE)
        gain = np.divide(mapped_maximum, maximum, out=np.ones_like(maximum), where=maximum > 0)
        with np.errstate(under="ignore"):
            bounded = rendered * gain[..., None]
            result = bounded.astype(source.dtype)
        if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
            raise ValueError("Active Cool headroom produced invalid bounded values")
        output[start : start + block.shape[0]] = result
        low += (block <= SOURCE_FLOOR).sum((0, 1))
        negative += (density < 0).sum((0, 1))
        above_one += (density > 1).sum((0, 1))
        rounded_density_one += (mapped == 1).sum((0, 1))
        rounded_output_one += (result == 1).sum((0, 1))
        cast_to_zero += ((bounded > 0) & (result == 0)).sum((0, 1))
        density_changed += int(np.count_nonzero(np.any(mapped != density, axis=-1)))
        output_shoulder += int(np.count_nonzero(maximum > MAXIMUM_KNEE))
        actual_changed += int(np.count_nonzero(np.any(result.astype(np.float64) != rendered, axis=-1)))
        maximum_pre_fit = max(maximum_pre_fit, float(maximum.max()))
        minimum_out = np.minimum(minimum_out, result.min((0, 1)))
        maximum_out = np.maximum(maximum_out, result.max((0, 1)))
    output.setflags(write=False)
    count = source.shape[0] * source.shape[1]
    receipt = {
        "schema": SCHEMA, "version": VERSION, "status": "active-experimental",
        "controls": controls, "source_domain": SIGNAL_DOMAIN,
        "density_domain": DENSITY_DOMAIN,
        "output_domain": "density-derived-positive-rendering-values; working-primaries assignment is a presentation assumption",
        "shape": list(source.shape), "dtype": source.dtype.str,
        "pixel_count": count, "frozen_analysis": frozen_analysis, "frozen_shared_fit": frozen_fit,
        "source_floor": SOURCE_FLOOR,
        "source_at_or_below_floor_count_rgb": low.tolist(),
        "negative_u_black_policy": "u<0 maps to black before fractional power; no signed positive-light interpretation",
        "negative_u_black_count_rgb": negative.tolist(),
        "previous_upper_density_clip_candidate_count_rgb": above_one.tolist(),
        "density_mapping_changed_pixel_count": density_changed,
        "base_density_knee": DENSITY_KNEE, "effective_density_knee": effective_knee,
        "density_mapping": "Identity on nonnegative u<=effective knee; C1 rational upper shoulder before frozen power/scale",
        "maximum_shoulder_knee": MAXIMUM_KNEE,
        "maximum_shoulder_formula": "M<=.8: M; M>.8: .8+.2*(M-.8)/(.2+(M-.8)); RGBout=RGBfit*(mappedM/M)",
        "maximum_shoulder_pixel_count": output_shoulder,
        "post_fit_or_dtype_changed_pixel_count": actual_changed,
        "post_fit_maximum_before_output_shoulder": maximum_pre_fit,
        "output_range": {"minimum_rgb": minimum_out.tolist(), "maximum_rgb": maximum_out.tolist()},
        "mapped_density_exact_one_due_to_rounding_count_rgb": rounded_density_one.tolist(),
        "output_exact_one_due_to_rounding_count_rgb": rounded_output_one.tolist(),
        "positive_float_cast_to_zero_count_rgb": cast_to_zero.tolist(),
        "new_fit_performed": False, "intermediate_integer_encoding_performed": False,
        "OETF_or_ICC_transform_performed": False,
        "signal_upper_hard_clip_performed": False,
        "preserves_input_rgb_chromaticity": False,
        "common_output_shoulder_preserves_post_fit_RGB_ratios_except_rounding": True,
        "all_midtone_values_unchanged": False,
        "source_geometry_certified": False, "source_recovery_certified": False,
        "distinct_codes_are_recovered_bits_or_scene_EV": False,
        "disabled_route_requirement": "Caller must execute the complete old route; strength0 is rejected by this active-only function",
        "limits": [
            "Different channel density curves and the frozen shared power alter colour relationships.",
            "Even u below the density knee can change through removed intermediate quantization or the post-fit maximum shoulder.",
            "Source values at/below the1e-6 floor retain no distinctions in this transform.",
            "Finite precision can round a far-tail value to1 or a tiny value to0; changed/selected values are not recovered detail.",
            "No factory CMS, measured camera/scanner colour mapping, physical scene linearity or recovered dynamic-range EV is claimed.",
        ],
    }
    return output, receipt
