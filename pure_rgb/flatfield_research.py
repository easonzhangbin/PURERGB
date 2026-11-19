"""Read-only flat-field diagnostics; never certify a single RAW for three LEDs.

These helpers do not activate a NegPy profile. Camera channel responses are not
spectral power distributions, and a self-flattening reference is not a color
calibration or proof of matching target illumination.
"""

import math

import numpy as np


def capture_match(reference, target):
    """Separate mechanical, optical and unknown spectral matching evidence."""
    hard = []
    unknown = []
    for key in ("make", "model", "sensor_shape", "cfa_pattern"):
        a, b = reference.get(key), target.get(key)
        if a is None or b is None or a == "" or b == "":
            unknown.append(key)
        elif (a.casefold() if isinstance(a, str) else a) != (
            b.casefold() if isinstance(b, str) else b
        ):
            hard.append({"field": key, "reference": a, "target": b})
    optical = []
    exposure = []
    for key, bucket in (
        ("aperture", optical),
        ("focal_length", optical),
        ("iso", exposure),
        ("exposure_seconds", exposure),
    ):
        a, b = reference.get(key), target.get(key)
        if a is None or b is None:
            unknown.append(key)
        elif not math.isfinite(float(a)) or not math.isfinite(float(b)):
            raise ValueError("Capture values must be finite")
        elif not math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-8):
            bucket.append({"field": key, "reference": a, "target": b})
    return {
        "mechanical_compatibility": "incompatible"
        if hard
        else "unknown"
        if any(k in unknown for k in ("make", "model", "sensor_shape", "cfa_pattern"))
        else "compatible",
        "mechanical_mismatches": hard,
        "optical_exact_match": "incompatible"
        if optical
        else "unknown"
        if any(k in unknown for k in ("aperture", "focal_length"))
        else "compatible",
        "optical_mismatches": optical,
        "exposure_differences": exposure,
        "missing_metadata": unknown,
        "spectral_compatibility": "unknown",
        "spectral_reason": "A single broadband RAW does not identify the three target LED spectra or wavelength-dependent shading",
        "overall_physical_calibration": "incompatible" if hard else "unknown",
        "can_certify_production_calibration": False,
        "normalized_gain_scope": "Exposure differences do not directly scale mean(blur)/blur gain; this does not prove linearity or matched shading",
    }


def gain_for_source(gain, shape, *, orientation=1):
    """Use NegPy's EXIF orientation then full-canvas linear resize contract."""
    import negpy_core
    from negpy.kernel.image.logic import apply_exif_orientation

    x = np.asarray(gain)
    if (
        x.ndim != 3
        or x.shape[2] != 3
        or min(x.shape[:2]) < 1
        or x.dtype.kind != "f"
        or not np.isfinite(x).all()
        or np.any(x <= 0)
    ):
        raise ValueError("Gain must be finite positive floating HWC RGB")
    if (
        len(shape) != 3
        or shape[2] != 3
        or any(type(v) is not int or v < 1 for v in shape)
        or orientation not in range(1, 9)
    ):
        raise ValueError("Explicit source HWC shape and EXIF orientation required")
    x = np.ascontiguousarray(apply_exif_orientation(x, orientation), dtype=np.float32)
    if x.shape != tuple(shape):
        x = negpy_core.resize_linear(x, (shape[1], shape[0]))
    return x


def apply_spatial_gain(source, gain):
    """Return independent signed float32 samples, without clipping or renormalizing."""
    x, g = np.asarray(source), np.asarray(gain)
    if (
        x.shape != g.shape
        or x.ndim != 3
        or x.shape[2] != 3
        or not x.size
        or x.dtype.kind != "f"
        or g.dtype.kind != "f"
        or not np.isfinite(x).all()
        or not np.isfinite(g).all()
        or np.any(g <= 0)
    ):
        raise ValueError(
            "Matching finite floating HWC source and positive gain required"
        )
    return np.multiply(x, g, dtype=np.float32)


def response_summary(source):
    """Camera-code channel distribution; no inferred white balance or spectrum."""
    x = np.asarray(source)
    if x.ndim != 3 or x.shape[2] != 3 or not x.size or not np.isfinite(x).all():
        raise ValueError("Finite HWC RGB samples required")
    median = np.median(x, axis=(0, 1)).astype(float)
    return {
        "median_rgb": median.tolist(),
        "relative_channel_medians": (median / max(float(median.sum()), 1e-12)).tolist(),
        "quantiles_p001_p01_p50_p99_p999_rgb": np.quantile(
            x, [0.001, 0.01, 0.5, 0.99, 0.999], axis=(0, 1)
        ).tolist(),
        "at_or_above_0_98_fraction_rgb": (x >= 0.98).mean((0, 1)).tolist(),
        "above_1_fraction_rgb": (x > 1).mean((0, 1)).tolist(),
        "negative_fraction_rgb": (x < 0).mean((0, 1)).tolist(),
        "scope": "Linear camera code samples; channel ratios are not spectral measurements or a white-balance target",
    }
