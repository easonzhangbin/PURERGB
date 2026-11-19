"""Same frozen engine adjustments for interactive previews and final renders."""

from dataclasses import replace
import hashlib
import numpy as np

from .cli import inspect_tiff
from .engines import flexcolor_research
from .endpoint_study import apply_endpoint_curve
from .native import encode_experiment
from .professional_controls import checked_engine
from .rgb_curve_study import apply_shared_curve
from .tone_study import GAMMA, ICC_SHA256

CURVE_BLOCK_PIXELS = 262144


def encode_positive_rows(linear):
    """Keep historical float arithmetic while bounding quantization temporaries."""
    encoded = np.empty(linear.shape, dtype=np.uint16)
    rows = max(1, 262144 // linear.shape[1])
    for start in range(0, linear.shape[0], rows):
        block = linear[start : start + rows]
        encoded[start : start + rows] = np.rint(
            np.minimum(block, 1) ** (1 / GAMMA) * 65535
        ).astype(np.uint16)
    return encoded


def _curve_codes(source, mode, fit, *, density_gamma=1.0, encoded_source=False):
    """Apply the frozen pointwise fit and quantize without a full float result.

    Analysis and fitting stay outside this function. The same scalar curve,
    working transfer function and rounding operate on each original pixel;
    only temporary-array lifetime changes. A resource-rendered Flex RGB16
    source is decoded here in blocks, after its unchanged encoder boundary.
    """
    source = np.asarray(source)
    if source.ndim != 3 or source.shape[2] != 3 or source.size == 0:
        raise ValueError("Expected finite linear working RGB in [0,1]")
    encoded = np.empty(source.shape, dtype=np.uint16)
    rows = max(1, CURVE_BLOCK_PIXELS // source.shape[1])
    total_pixels = source.shape[0] * source.shape[1]
    limited_pixels = 0
    metadata = {}
    for start in range(0, source.shape[0], rows):
        block = source[start : start + rows]
        if encoded_source:
            block = (block.astype(float) / 65535) ** GAMMA
        if mode == "cool-shared-rgb":
            if density_gamma != 1:
                block = block ** (1 / density_gamma)
            transformed, metadata = apply_shared_curve(block, fit)
            # Recover the integer count rather than averaging chunk fractions.
            # Counts are at most the bounded block size, so rounding its own
            # mean×count is exact and reproduces the whole-image fraction.
            limited_pixels += round(
                metadata["gamut_limited_pixel_fraction"]
                * block.shape[0]
                * block.shape[1]
            )
        else:
            transformed = apply_endpoint_curve(block, fit)
        encoded[start : start + rows] = encode_positive_rows(transformed)
    if mode == "cool-shared-rgb" and total_pixels:
        metadata = {
            **metadata,
            "gamut_limited_pixel_fraction": limited_pixels / total_pixels,
        }
    return encoded, metadata


def experimental_positive(
    mode,
    linear,
    fit,
    document,
    source,
    analysis,
    processor,
    export,
    profile=None,
    *,
    expected_icc=None,
):
    """Apply frozen adjustments, validating any new resource encoder boundary.

    Resource RGB16 must retain the exact working ICC whose transfer is GAMMA.
    An explicit baseline ICC must also match; omitting it permits only that
    same pinned working profile. Final-curve-only routes do not re-encode a
    resource or inspect an ICC and retain their original arithmetic.
    """
    values = checked_engine(mode, document or {"version": 1, "values": {}})["values"]
    profile_info = None
    pipeline_info = None
    encoded_source = False
    if mode == "cool-shared-rgb":
        gamma = values.get("density_gamma", 1.0)
        adjusted = replace(
            fit,
            power=float(
                np.clip(fit.power * 2 ** values.get("curve_contrast", 0), 0.1, 12)
            ),
            scale=fit.scale * 2 ** values.get("curve_gain_ev", 0),
        )
        encoded, meta = _curve_codes(linear, mode, adjusted, density_gamma=gamma)
        effective = {"power": adjusted.power, "scale": adjusted.scale}
    else:
        if profile or any(
            k in values
            for k in (
                "film_gamma",
                "brightness",
                "contrast",
                "black_code",
                "white_code",
                "curve_kernel",
                "sliders_kernel",
            )
        ):
            defaults = {
                "film_gamma": 2.0,
                "contrast": 0.0,
                "brightness": 0.0,
                "black_code": 0.0,
                "white_code": 255.0,
            }
            if profile:
                from .engine_resources import require_flex_profile

                _, profile_info = require_flex_profile(profile)
                defaults.update(profile_info["defaults"])
            samples, pipeline_info = flexcolor_research(
                np.maximum(source, 0),
                analysis,
                profile=profile,
                film_gamma=values.get("film_gamma", defaults["film_gamma"]),
                contrast_override=values.get("contrast", defaults["contrast"]),
                brightness_override=values.get("brightness", defaults["brightness"]),
                endpoints_override=(
                    values.get("black_code", defaults["black_code"]),
                    values.get("white_code", defaults["white_code"]),
                ),
                curve_kernel=values.get("curve_kernel", "legacy-linear"),
                sliders_kernel=values.get("sliders_kernel", "legacy-linear"),
            )
            payload = encode_experiment(
                processor, samples, export, transfer="working-encoded"
            )
            codes, actual_icc = inspect_tiff(payload)
            if (
                not isinstance(actual_icc, bytes)
                or hashlib.sha256(actual_icc).hexdigest() != ICC_SHA256
            ):
                raise ValueError(
                    "Flex resource output ICC does not match the supported working Gamma"
                )
            if expected_icc is not None and (
                not isinstance(expected_icc, bytes) or actual_icc != expected_icc
            ):
                raise ValueError(
                    "Flex resource output ICC differs from the baseline ICC"
                )
            # Keep the original full-image resource renderer/encoder contract,
            # then release its temporaries before the frozen final curve.
            del samples, payload
            linear, encoded_source = codes, True
        adjusted = replace(
            fit,
            slope=float(
                np.clip(fit.slope * 2 ** values.get("curve_contrast", 0), 0.05, 12)
            ),
            offset=fit.offset + values.get("curve_offset", 0),
        )
        encoded, meta = _curve_codes(
            linear, mode, adjusted, encoded_source=encoded_source
        )
        effective = {"slope": adjusted.slope, "offset": adjusted.offset}
    metadata = {"engine_controls": document, "effective_curve": effective, **meta}
    if pipeline_info:
        metadata["engine_pipeline"] = pipeline_info
    if profile_info:
        metadata["profile_inspection"] = profile_info
    return encoded, metadata
