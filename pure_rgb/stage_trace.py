"""Read-only numeric observations for explicit research data domains.

These helpers preserve signed/out-of-range samples. A scene endpoint is never
silently promoted to a measured film base or a scanner calibration.
"""

from dataclasses import asdict
import hashlib

import numpy as np


QUANTILES = (0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1)


def u14_to_u16(codes):
    """Explicit research scale conversion, binary64 division and ties-to-even.

    A 16384 sentinel is not a legal 14-bit sample and must be handled upstream.
    This function performs no transfer function or color-space conversion.
    """
    values = np.asarray(codes)
    if (
        values.dtype.kind not in "ui"
        or not values.size
        or values.min() < 0
        or values.max() > 16383
    ):
        raise ValueError("14-bit mapping requires nonempty integer codes in 0..16383")
    return np.rint((values.astype(np.float64) / 16383) * 65535).astype(np.uint16)


def rgb_array(samples):
    array = np.asarray(samples)
    if (
        array.ndim != 3
        or array.shape[2] != 3
        or not array.size
        or array.dtype.kind not in "fui"
        or not np.isfinite(array).all()
    ):
        raise ValueError("Stage tracing requires a finite nonempty HWC RGB array")
    return array


def pixel_digest(samples):
    array = rgb_array(samples)
    # Canonical byte order is recorded along with dtype/shape in the receipt.
    little = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return hashlib.sha256(np.ascontiguousarray(little).tobytes()).hexdigest()


def stage_statistics(samples, *, domain, bounds=None, icc=None):
    array = rgb_array(samples)
    if not isinstance(domain, str) or not domain:
        raise ValueError("An explicit stage domain is required")
    result = {
        "domain": domain,
        "shape": list(array.shape),
        "dtype": str(array.dtype.newbyteorder("<")),
        "pixels_sha256": pixel_digest(array),
        "quantile_positions": list(QUANTILES),
        "per_channel_quantiles": np.quantile(array, QUANTILES, axis=(0, 1)).tolist(),
        "per_channel_mean": array.mean((0, 1), dtype=np.float64).tolist(),
        "icc_sha256": hashlib.sha256(icc).hexdigest() if icc else None,
        "statistics_are_not_scene_colorimetry": True,
    }
    if bounds is not None:
        low, high = bounds
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError("Stage bounds must be finite and increasing")
        below, above = array < low, array > high
        result["declared_bounds"] = [float(low), float(high)]
        result["below_fraction_rgb"] = below.mean((0, 1)).tolist()
        result["above_fraction_rgb"] = above.mean((0, 1)).tolist()
        result["outside_any_channel_pixel_fraction"] = float(
            np.any(below | above, axis=2).mean()
        )
        result["at_low_fraction_rgb"] = (array == low).mean((0, 1)).tolist()
        result["at_high_fraction_rgb"] = (array == high).mean((0, 1)).tolist()
    return result


def clip_observation(before, after, *, low=0.0, high=1.0):
    """Count actual out-of-range truncation separately from natural endpoints."""
    before, after = rgb_array(before), rgb_array(after)
    if before.shape != after.shape or not np.isfinite([low, high]).all() or low >= high:
        raise ValueError(
            "Clipping observations require equal geometry and valid bounds"
        )
    expected = np.clip(before, low, high).astype(after.dtype)
    if not np.array_equal(expected, after):
        raise ValueError("Observed result is not the declared clipping operation")
    below, above = before < low, before > high
    return {
        "operation": "clip only; no tone mapping or range recovery",
        "bounds": [float(low), float(high)],
        "lower_truncated_fraction_rgb": below.mean((0, 1)).tolist(),
        "upper_truncated_fraction_rgb": above.mean((0, 1)).tolist(),
        "truncated_any_channel_pixel_fraction": float(
            np.any(below | above, axis=2).mean()
        ),
        "natural_low_endpoint_fraction_rgb": (before == low).mean((0, 1)).tolist(),
        "natural_high_endpoint_fraction_rgb": (before == high).mean((0, 1)).tolist(),
        "preclip_minimum_rgb": before.min((0, 1)).tolist(),
        "preclip_maximum_rgb": before.max((0, 1)).tolist(),
        "preclip_values_retained_by_caller": True,
    }


def common_input_planes(source, analysis, *, film_base_reference=None):
    """Expose the current two adapters' preparation before their clipping.

    `analysis.base` is the scene high-transmission endpoint, not physical film
    base. Absolute/relative film-base density is emitted only for an explicit
    reference. Source-scale/black-level calibration is outside this function.
    """
    source = rgb_array(source)
    if source.dtype.kind != "f":
        raise ValueError("Common input requires linear floating samples")
    x = source.astype(np.float64)
    dark, high = np.asarray(analysis.dark), np.asarray(analysis.base)
    dark_for_log = np.maximum(dark, 1e-6)
    span = np.log10(high / dark_for_log)
    if np.any(span <= 0) or np.any(high <= dark) or not np.isfinite(span).all():
        raise ValueError("Invalid frozen density window")
    safe = np.maximum(x, 1e-6)
    density = np.log10(high / safe)
    fraction = density / span
    transmission = (x - dark) / (high - dark)
    planes = {
        "source_linear": source,
        "scene_high_relative_density_signed": density,
        "scene_density_fraction_unclipped": fraction,
        "scene_density_fraction_clipped": np.clip(fraction, 0, 1).astype(np.float32),
        "scene_transmission_unclipped": transmission,
        "scene_transmission_clipped": np.clip(transmission, 0, 1),
    }
    receipt = {
        "input_domain": "neutral-WB selected camera sensor bands on the frozen LibRaw scale",
        "camera_to_scanner_mapping_verified": False,
        "analysis": asdict(analysis),
        "scene_high_is_not_measured_film_base": True,
        "density_formula": "log10(scene_high / max(source,1e-6))",
        "fraction_formula": "density / log10(scene_high/max(scene_low,1e-6))",
        "transmission_formula": "(source-scene_low)/(scene_high-scene_low)",
        "log_floor": 1e-6,
        "log_floor_fraction_rgb": (x < 1e-6).mean((0, 1)).tolist(),
        "film_base_reference": None,
        "density_clip": clip_observation(
            fraction, planes["scene_density_fraction_clipped"]
        ),
        "transmission_clip": clip_observation(
            transmission, planes["scene_transmission_clipped"]
        ),
    }
    if film_base_reference is not None:
        base = np.asarray(film_base_reference.values, dtype=np.float64)
        planes["explicit_film_base_relative_density_signed"] = np.log10(base / safe)
        receipt["film_base_reference"] = asdict(film_base_reference)
    return planes, receipt
