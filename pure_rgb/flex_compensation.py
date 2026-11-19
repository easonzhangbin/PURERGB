"""Explicit density-reference and span choices for offline Flex adaptation.

These coordinates separate two different operations: removing a three-channel
reference offset in log space, and dividing by three different scene spans.
They do not identify the response of a scanner, an orange mask or LED energies.
No production recipe or automatic choice is made by this research API.
"""

from dataclasses import asdict
import numbers

import numpy as np

from .stage_trace import clip_observation, rgb_array, stage_statistics


def _positive_scalar(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, numbers.Real)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} requires an explicit finite positive scalar")
    return float(value)


def density_carrier(
    source,
    analysis,
    *,
    anchor,
    normalization,
    film_base=None,
    shared_span=None,
    log_floor=1e-6,
):
    """Return unbounded positive-density coordinates and their exact contract.

    ``anchor`` is ``scene-high`` or ``film-base``. The latter requires an
    explicit same-capture reference and does not assert physical film density.
    ``normalization`` is ``scene-channel-span`` or ``shared-span``. Both use
    the SAME frozen scene endpoints H/L; changing anchor never refits a span.
    A shared span must be supplied, so its selection cannot be an implicit fit.

    d_A=log10(A/max(S,floor)); s_c=log10(H/max(L,floor)); f=d_A/s.
    Every result remains signed and unbounded until a separate code boundary.
    Multiplying S/H/L/A by equal positive gain within each channel cancels when
    no value enters the fixed absolute floor. An additive offset does not.
    """
    values = rgb_array(source)
    if values.dtype.kind != "f" or np.any(values < 0):
        raise ValueError("Density carrier requires nonnegative floating linear RGB")
    if anchor not in ("scene-high", "film-base"):
        raise ValueError("Require explicit scene-high or film-base anchor")
    if normalization not in ("scene-channel-span", "shared-span"):
        raise ValueError("Require explicit scene-channel-span or shared-span")
    floor = _positive_scalar(log_floor, "log_floor")
    high = np.asarray(analysis.base, dtype=np.float64)
    low = np.asarray(analysis.dark, dtype=np.float64)
    if (
        high.shape != (3,)
        or low.shape != (3,)
        or not np.isfinite([high, low]).all()
        or np.any(low < 0)
        or np.any(high <= np.maximum(low, floor))
    ):
        raise ValueError("Invalid frozen logarithmic scene endpoints")
    spans = np.log10(high / np.maximum(low, floor))
    if normalization == "shared-span":
        denominator = np.repeat(_positive_scalar(shared_span, "shared_span"), 3)
    else:
        if shared_span is not None:
            raise ValueError("Do not silently ignore shared_span in channel mode")
        denominator = spans
    if anchor == "film-base":
        if film_base is None:
            raise ValueError("Film-base anchor requires an explicit reference")
        reference = np.asarray(film_base.values, dtype=np.float64)
        if (
            reference.shape != (3,)
            or not np.isfinite(reference).all()
            or np.any(reference <= floor)
        ):
            raise ValueError("Invalid explicit film-base reference")
        base_receipt = asdict(film_base)
    else:
        if film_base is not None:
            raise ValueError("Do not silently ignore film_base in scene-high mode")
        reference, base_receipt = high, None
    source64 = values.astype(np.float64)
    density = np.log10(reference / np.maximum(source64, floor))
    result = density / denominator
    near = np.log10(reference / high)
    far = np.log10(reference / np.maximum(low, floor))
    return result, {
        "schema": "pure-rgb-explicit-density-carrier-v1",
        "anchor": anchor,
        "normalization": normalization,
        "formula": "d_A=log10(A/max(S,floor)); f=d_A/denominator_rgb",
        "reference_rgb": reference.tolist(),
        "scene_high_rgb": high.tolist(),
        "scene_low_rgb": low.tolist(),
        "scene_span_rgb": spans.tolist(),
        "denominator_rgb": denominator.tolist(),
        "density_at_scene_high_rgb": near.tolist(),
        "density_at_scene_low_rgb": far.tolist(),
        "channel_base_offsets_are_not_channel_span_normalization": True,
        "scene_renormalization_cancels_any_base_reference": True,
        "scene_high_is_not_measured_film_base": True,
        "film_base": base_receipt,
        "log_floor": floor,
        "source_floor_fraction_rgb": (source64 < floor).mean((0, 1)).tolist(),
        "scene_low_floor_applied_rgb": (low < floor).tolist(),
        "gain_invariance_requires_source_and_references_scaled_together": True,
        "fixed_absolute_floor_can_break_gain_invariance": True,
        "absolute_LED_energy_and_open_path_unknown": True,
        "camera_to_scanner_calibration_verified": False,
        "white_balance_or_exposure_fit_performed": False,
        "signed_float_retained_by_caller": True,
        "carrier": stage_statistics(
            result, domain="signed density fraction research carrier", bounds=(0, 1)
        ),
    }


def encode_density_carrier(carrier):
    """Observe the one bounded RGB14 boundary, with no gamma or inversion."""
    values = rgb_array(carrier)
    if values.dtype.kind != "f":
        raise ValueError("Density encoding requires an explicit floating carrier")
    bounded = np.clip(values.astype(np.float64), 0, 1)
    codes = np.rint(bounded * 16383).astype(np.uint16)
    return codes, {
        "formula": "rint(clip(signed_density_fraction,0,1)*16383)",
        "quantization": "binary64 nearest ties-to-even; PURE research boundary",
        "gamma_applied": False,
        "negative_inversion_applied": False,
        "input_clip": clip_observation(values, bounded),
        "codes": stage_statistics(codes, domain="density RGB14", bounds=(0, 16383)),
        "original_signed_float_retained_by_caller": True,
    }
