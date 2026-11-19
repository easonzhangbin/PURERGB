"""Explicit research bridge from analyzer exposure to builder input codes.

LS5000 4.0.3 analyzer 0x10011840 emits exposure[c] = low32(trunc(base[c]
* 10**g[c])) at 0x100126F3..0x10012773. Builder 0x10010C30 evaluates
log10(65535/i)+g. Under an explicitly assumed linear sensor, scaling analyzer
codes by final/analyzer exposure therefore puts i in the intended relative
density domain. This arithmetic is not a captured final scan or calibration.
No camera shutter time, guessed black offset, per-channel fit, driver clamp,
white balance or gamma is introduced here.
"""

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from . import nikon_metering as meter
from .nikon_parameters import ParameterResult

MODES = ("ideal-linear-ratio", "original-output-u32", "explicit-final-ticks")
ORIGINS = (
    "published-scanner-fixture",
    "synthetic-common-tick-grid",
    "caller-supplied-unverified",
)


@dataclass(frozen=True)
class ExposureBridgePlan:
    ratios: tuple[float, float, float]
    receipt: dict


@dataclass(frozen=True)
class ExposureBridgeResult:
    codes: np.ndarray
    preclip_codes: np.ndarray
    receipt: dict


def exposure_bridge_plan(
    parameters, *, mode, exposure_origin, tick_multiplier=1, final_exposures=None
):
    """Plan an explicit ideal, original integer-output or supplied-tick bridge.

    ``tick_multiplier`` is one common positive integer, never an RGB fit. A
    camera hypothesis may use 65536 as an artificial tick grid, clearly labeled
    synthetic. Original integer output keeps low32; it does not silently clamp
    to the separate scanner controller's 0x3ffffff limit.
    """
    if not isinstance(parameters, ParameterResult):
        raise ValueError("Explicit Nikon ParameterResult required")
    if mode not in MODES:
        raise ValueError("Choose explicit exposure bridge mode")
    if exposure_origin not in ORIGINS:
        raise ValueError("Choose explicit exposure metadata origin")
    if type(tick_multiplier) is not int or tick_multiplier <= 0:
        raise ValueError("tick_multiplier must be one positive common integer")
    if tick_multiplier != 1 and exposure_origin != "synthetic-common-tick-grid":
        raise ValueError("A rescaled tick grid must be explicitly synthetic")
    base = meter._exposures(parameters.receipt["base_exposures"], "base exposure")
    taken = meter._exposures(
        parameters.receipt["analyzer_exposures"], "analyzer exposure"
    )
    base_grid = tuple(v * tick_multiplier for v in base)
    taken_grid = tuple(v * tick_multiplier for v in taken)
    if max(base_grid + taken_grid) > 0xFFFFFFFF:
        raise ValueError("Scaled descriptor inputs exceed original uint32 fields")
    g = tuple(float(p.g) for p in parameters.channels)
    if len(g) != 3 or not all(math.isfinite(v) for v in g):
        raise ValueError("Require three finite builder g densities")
    try:
        ideal = tuple(float(b) * 10.0**d for b, d in zip(base_grid, g, strict=True))
    except OverflowError as exc:
        raise ValueError("Original ftol signed64 conversion range exceeded") from exc
    if not all(0 <= v < 2**63 for v in ideal):
        raise ValueError("Original ftol signed64 conversion range exceeded")
    integer_full = tuple(math.trunc(v) for v in ideal)
    original_u32 = tuple(v & 0xFFFFFFFF for v in integer_full)
    if mode == "ideal-linear-ratio":
        if final_exposures is not None:
            raise ValueError("Ideal mode does not consume supplied final ticks")
        chosen = ideal
    elif mode == "original-output-u32":
        if final_exposures is not None:
            raise ValueError("Original-output mode derives ticks from g")
        chosen = original_u32
    else:
        chosen = meter._exposures(final_exposures, "explicit final exposure")
    ratios = tuple(float(f) / t for f, t in zip(chosen, taken_grid, strict=True))
    log_residual = [
        math.log10(expected / selected) if selected > 0 and expected > 0 else None
        for expected, selected in zip(ideal, chosen, strict=True)
    ]
    return ExposureBridgePlan(
        ratios,
        {
            "schema": "pure-rgb-nikon-exposure-bridge-plan",
            "version": 1,
            "mode": mode,
            "exposure_origin": exposure_origin,
            "tick_multiplier": tick_multiplier,
            "base_descriptor_values": list(base),
            "analyzer_descriptor_values": list(taken),
            "base_tick_grid": list(base_grid),
            "analyzer_tick_grid": list(taken_grid),
            "g_density_rgb": list(g),
            "ideal_final_tick_values": list(ideal),
            "original_truncated_signed64": list(integer_full),
            "original_output_u32": list(original_u32),
            "selected_final_tick_values": list(chosen),
            "integer_math_policy": "portable float64 pow/trunc/low32; exact integer boundaries may differ from original x87 CRT",
            "u32_wrapped_channels": [
                v != w for v, w in zip(integer_full, original_u32, strict=True)
            ],
            "controller_limit_0x3ffffff_exceeded": [v > 0x3FFFFFF for v in chosen],
            "driver_or_hardware_limit_applied": False,
            "zero_exposure_channels": [v == 0 for v in chosen],
            "final_over_analyzer_ratios": list(ratios),
            "density_residual_from_tick_quantization_rgb": log_residual,
            "relation": "i=analyzer_code*(final_exposure/analyzer_exposure)",
            "ideal_density_identity": "log10(65535/i)+g=log10(65535/(analyzer_code*base/analyzer_exposure))",
            "source_analyzer_pixels_sha256": parameters.receipt["source_pixels_sha256"],
            "source_analyzer_domain": parameters.receipt["input_domain"],
            "assumed_linear_sensor": True,
            "black_offset_assumed_zero": True,
            "camera_shutter_times_used": False,
            "per_channel_gain_fit": False,
            "actual_final_scan_captured": False,
            "acquisition_binding_proven": False,
            "camera_to_scanner_calibrated": False,
            "production_defaults_changed": False,
        },
    )


def apply_exposure_bridge(codes, plan, *, bits, source_relation):
    """Produce a retained-float, clipped/quantized final-like numeric proxy.

    This cannot recreate an actual final acquisition, sensor noise or clipped
    prescan samples. Quantization is explicitly NumPy ties-to-even, not a claim
    about Nikon ADC rounding. Preclip float codes are retained for diagnostics.
    """
    pixels = meter._codes(codes, bits)
    if not isinstance(plan, ExposureBridgePlan):
        raise ValueError("Require explicit ExposureBridgePlan")
    if source_relation not in (
        "same-analyzer-raster",
        "same-synthetic-carrier-other-resolution",
    ):
        raise ValueError("Declare analyzer/raster source relation")
    source_hash = hashlib.sha256(pixels.astype("<u2").tobytes()).hexdigest()
    if (
        source_relation == "same-analyzer-raster"
        and source_hash != plan.receipt["source_analyzer_pixels_sha256"]
    ):
        raise ValueError("Analyzer pixel hash does not match bridge parameters")
    ratios = np.asarray(plan.ratios, dtype=np.float64)
    if ratios.shape != (3,) or not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise ValueError(
            "Nonpositive/invalid exposure: no final-like raster can be claimed"
        )
    widened = meter._widen(pixels, bits)
    preclip = widened.astype(np.float64) * ratios
    result = np.rint(np.clip(preclip, 0, 65535)).astype(np.uint16)
    return ExposureBridgeResult(
        result,
        preclip,
        {
            "schema": "pure-rgb-nikon-exposure-bridge-raster",
            "version": 1,
            "plan": plan.receipt,
            "source_relation": source_relation,
            "source_pixels_sha256": source_hash,
            "source_valid_bits": bits,
            "result_pixels_sha256": hashlib.sha256(
                result.astype("<u2").tobytes()
            ).hexdigest(),
            "source_unchanged": source_hash
            == hashlib.sha256(pixels.astype("<u2").tobytes()).hexdigest(),
            "source_adc_upper_endpoint_fraction_rgb": np.mean(
                pixels == (1 << bits) - 1, axis=(0, 1)
            ).tolist(),
            "preclip_minimum_rgb": preclip.min(axis=(0, 1)).tolist(),
            "preclip_maximum_rgb": preclip.max(axis=(0, 1)).tolist(),
            "upper_clipped_fraction_rgb": np.mean(
                preclip > 65535, axis=(0, 1)
            ).tolist(),
            "lower_clipped_fraction_rgb": np.mean(preclip < 0, axis=(0, 1)).tolist(),
            "float_preclip_retained": True,
            "quantization": "uint16 rint ties-to-even after explicit 0..65535 clip",
            "status": "unbound-final-like-linear-exposure-proxy",
            "not_an_actual_fine_scan": True,
            "upstream_clipping_recoverable": False,
            "gamma_or_oetf_applied": False,
            "production_defaults_changed": False,
        },
    )
