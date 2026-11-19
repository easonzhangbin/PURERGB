"""Read-only numeric diagnostics for Nikon's unbound response hypotheses.

These helpers expose the coordinate required by the recovered builder. They
do not estimate exposure, white balance, a film base or camera calibration.
Device-code observations are not a color-managed photograph.
"""

import math

import numpy as np

from .nikon_lut import BuilderParameters, build_lut, curve
from .stage_trace import rgb_array


def _rgb_vector(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"{name} requires three finite values")
    return values


def scanner_response(density, *, exposure_ratio):
    """Explicit model D -> unbounded linear DN, 65535 * gain * 10**(-D).

    This is a constructed numerical response. No camera data are silently
    reclassified as measured optical density. Preserve out-of-ADC-range values
    for observation instead of applying an implicit gain, clip or gamma.
    """
    density = rgb_array(density).astype(np.float64)
    ratio = _rgb_vector(exposure_ratio, "exposure_ratio")
    if np.any(ratio <= 0):
        raise ValueError("exposure_ratio must be positive")
    with np.errstate(over="ignore", under="ignore"):
        result = 65535.0 * ratio * np.power(10.0, -density)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("Synthetic response exceeds finite positive DN domain")
    return result


def recover_builder_coordinate(final_codes, *, g):
    """Return log10(65535/fine_DN) + g, without clipping or exposure fitting."""
    final_codes = rgb_array(final_codes).astype(np.float64)
    if np.any(final_codes <= 0):
        raise ValueError("Builder coordinate requires positive DN")
    return np.log10(65535.0 / final_codes) + _rgb_vector(g, "g")


def inspect_builder_response(codes, parameters, control_x, control_y):
    """Apply the supplied tables and count prefix, wrap and slope events.

    Counts use occupied input codes, then weight by observed samples. The
    continuous expression is inspected *after* the source's explicit white
    prefix. Output RGB is device code, with no ICC attached or inferred.
    """
    codes = rgb_array(codes)
    if codes.dtype != np.uint16 or not all(
        len(v) == 3 for v in (parameters, control_x, control_y)
    ):
        raise ValueError("Require RGB16 codes and three explicit builder channels")
    output = np.empty_like(codes)
    reports = []
    for c, (p, x, y) in enumerate(zip(parameters, control_x, control_y, strict=True)):
        if not isinstance(p, BuilderParameters):
            raise ValueError("Require explicit BuilderParameters")
        table = build_lut(*p.arguments(), x, y)
        output[..., c] = table[codes[..., c]]
        span = min(max((p.A - ((1 - p.a) * p.B + p.E * p.a)) / p.C, 0.01), 4.0)
        prefix = math.trunc(65535 * 10 ** (-(p.f - p.g))) + 1
        occupied, counts = np.unique(codes[..., c], return_counts=True)
        after_prefix = occupied >= prefix
        unwrapped = np.full(occupied.shape, 65535.0, dtype=float)
        valid = occupied[after_prefix].astype(float)
        if valid.size:
            t = np.log10(65535.0 / valid) + p.g
            unwrapped[after_prefix] = 65535 * np.power(
                10.0, (p.A - curve(x, y, t)) * (-1.0 / span)
            )
        wraps = after_prefix & (unwrapped >= 65536)
        upward = np.diff(table.astype(np.int32)) > 0
        input_median = float(np.median(codes[..., c]))
        reports.append(
            {
                "channel": "RGB"[c],
                "density_range_D": span,
                "prefix_count": prefix,
                "prefix_max_code_inclusive": prefix - 1,
                "observed_prefix_fraction": float(
                    counts[~after_prefix].sum() / counts.sum()
                ),
                "observed_white_fraction": float(np.mean(output[..., c] == 65535)),
                "observed_zero_fraction": float(np.mean(output[..., c] == 0)),
                "occupied_input_codes_wrapping_after_prefix": int(wraps.sum()),
                "observed_wrap_fraction": float(counts[wraps].sum() / counts.sum()),
                "table_upward_steps": int(upward.sum()),
                "input_median_DN": input_median,
                "input_median_apparent_density_before_g": math.log10(
                    65535.0 / input_median
                )
                if input_median
                else None,
                "input_median_builder_coordinate_after_g": math.log10(
                    65535.0 / input_median
                )
                + p.g
                if input_median
                else None,
                "output_median_device_DN": float(np.median(output[..., c])),
                "unwrapped_max_occupied_after_prefix": float(
                    unwrapped[after_prefix].max()
                )
                if after_prefix.any()
                else None,
            }
        )
    return output, {
        "schema": "pure-rgb-nikon-response-stage-probe",
        "version": 1,
        "input_domain": "caller-supplied unbound RGB16 hypothesis",
        "output_domain": "pre-fixed Nikon device RGB16; no display ICC inferred",
        "source_response_or_exposure_calibrated": False,
        "source_unchanged_by_this_function": True,
        "channels": reports,
    }
