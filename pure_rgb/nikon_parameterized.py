"""Opt-in unbound analyzer -> inversion tables -> optional fixed -> Nikon CMS.

This closes the numerical research chain. A caller must state what the final
raster codes mean. No final-scan exposure or camera-to-scanner calibration is
inferred from successfully deriving or applying a LUT. Existing strict scanner
interfaces and all application defaults are unchanged.
"""

from dataclasses import asdict, dataclass
import hashlib

import numpy as np

from .nikon_cms import CMSResult, _model
from .nikon_cms_research import _strength, apply_fixed_strength
from .nikon_lut import build_lut
from .nikon_parameters import ParameterResult
from .stage_trace import pixel_digest, rgb_array

INPUT_HYPOTHESES = {
    "unbound-scanner-final-rgb16": "Scanner-like RGB16 raster, final exposure/acquisition ownership unverified",
    "camera-transmission-as-scanner-rgb16": "Camera transmission explicitly treated as scanner codes; response and exposure uncalibrated",
    "synthetic-rgb16-probe": "Constructed numeric ramp or fixture, no physical acquisition",
}


@dataclass(frozen=True)
class DerivedTables:
    tables: tuple[np.ndarray, ...]
    receipt: dict


def compile_parameter_tables(parameters):
    """Compile three 65536-entry tables from explicit analyzer outputs."""
    if not isinstance(parameters, ParameterResult):
        raise ValueError("Require an explicit Nikon ParameterResult")
    if not all(
        len(v) == 3
        for v in (parameters.channels, parameters.control_x, parameters.control_y)
    ):
        raise ValueError("Require three complete analyzer channels")
    tables = []
    for p, x, y in zip(
        parameters.channels, parameters.control_x, parameters.control_y, strict=True
    ):
        table = build_lut(*p.arguments(), x, y)
        table.setflags(write=False)
        tables.append(table)
    return DerivedTables(
        tuple(tables),
        {
            "schema": "pure-rgb-nikon-derived-tables",
            "version": 1,
            "analyzer": parameters.receipt,
            "parameters": [asdict(p) for p in parameters.channels],
            "controls": [
                {"x": list(x), "y": list(y)}
                for x, y in zip(parameters.control_x, parameters.control_y, strict=True)
            ],
            "tables_sha256": [
                hashlib.sha256(t.astype("<u2", copy=False).tobytes()).hexdigest()
                for t in tables
            ],
            "math_policy": "Portable float64 libm; exact integer thresholds can differ from original x87 CRT by one entry; no captured-prefix override",
            "acquisition_binding_proven": False,
            "camera_to_scanner_calibrated": False,
        },
    )


def evaluate_parameterized_codes(
    pixels, parameters, *, input_hypothesis, fixed_strength=0.0, threads=1
):
    """Recompute tables and explicitly run the downstream captured CMS once.

    Fixed strength defaults to zero per the selected PURE RGB research mode;
    full original fixed placement is available with strength=1. Partial strength
    is a PURE adaptation. Neither scanner provenance nor white balance is fit.
    """
    x = rgb_array(pixels)
    if x.dtype != np.uint16:
        raise ValueError("Parameterized Nikon input requires RGB16 codes")
    if (
        not isinstance(input_hypothesis, str)
        or input_hypothesis not in INPUT_HYPOTHESES
    ):
        raise ValueError(
            "Choose an explicit unbound scanner/camera/synthetic hypothesis"
        )
    if type(threads) is not int or not 1 <= threads <= 16:
        raise ValueError("Nikon research CMS threads must be 1..16")
    if (
        not isinstance(parameters, ParameterResult)
        or parameters.receipt.get("film_kind") != "color-negative"
    ):
        raise ValueError(
            "This captured CMS model is C-41 only; monochrome numerical LUTs remain separate"
        )
    strength = _strength(fixed_strength)
    source_hash = pixel_digest(x)
    compiled = compile_parameter_tables(parameters)
    # Independent per-channel lookup; do not apply a working OETF to scanner codes.
    inverted = np.empty_like(x)
    for c, table in enumerate(compiled.tables):
        inverted[..., c] = table[x[..., c]]
    device = apply_fixed_strength(inverted, strength)
    model = _model(threads)
    device_hash = pixel_digest(device)
    output = model.evaluate(device)
    if pixel_digest(x) != source_hash or pixel_digest(device) != device_hash:
        raise ValueError("Nikon numerical chain mutated its source")
    return CMSResult(
        output,
        model.profile,
        {
            "schema": "pure-rgb-nikon-parameterized-research",
            "version": 1,
            "input_hypothesis": input_hypothesis,
            "hypothesis_description": INPUT_HYPOTHESES[input_hypothesis],
            "source_pixels_sha256": source_hash,
            "compiled": compiled.receipt,
            "inverted_pixels_sha256": pixel_digest(inverted),
            "device_before_cms_pixels_sha256": device_hash,
            "output_pixels_sha256": pixel_digest(output),
            "fixed_strength": strength,
            "fixed_bypassed": strength == 0,
            "fixed_is_partial_pure_adaptation": 0 < strength < 1,
            "cms": model.bundle_receipt(),
            "source_unchanged": True,
            "film_kind": "color-negative",
            "final_raster_bound_to_prescan_or_thumbnail": False,
            "acquisition_binding_proven": False,
            "camera_to_scanner_calibrated": False,
            "production_defaults_changed": False,
            "scope": "Numerical analyzer/LUT/fixed/CMS research join; not a complete calibrated original scan",
        },
    )
