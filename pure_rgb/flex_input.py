"""Explicit camera-to-Flex research carriers, preserving pre-quantization data.

A reversible NegPy density coordinate is not a second negative inversion.
These mappings are hypotheses; no scanner response or original exposure scale
is inferred from camera shutter metadata or a scene percentile.
"""

from dataclasses import asdict

import numpy as np

from .stage_trace import clip_observation, rgb_array, stage_statistics


def camera_transmission(
    source, analysis, *, normalization, film_base=None, via_density=False
):
    """Return unbounded transmission and a reversible-density receipt.

    ``scene-window`` preserves the existing endpoint-window definition.
    ``film-base`` divides by an explicit same-capture reference, and does not
    subtract the scene's darkest pixel as though it were sensor black.
    via_density reconstructs the same signal from signed, UNCLIPPED density.
    It never uses the clipped positive-density result of ``density_auto``.
    """
    values = rgb_array(source)
    if values.dtype.kind != "f" or np.any(values < 0):
        raise ValueError("Camera transmission requires nonnegative linear floating RGB")
    if type(via_density) is not bool:
        raise ValueError("via_density must be explicit boolean")
    if normalization not in ("scene-window", "film-base"):
        raise ValueError("Unknown transmission normalization")
    high = np.asarray(analysis.base, dtype=np.float64)
    low = np.asarray(analysis.dark, dtype=np.float64)
    if (
        high.shape != (3,)
        or low.shape != (3,)
        or not np.isfinite([high, low]).all()
        or np.any(high <= low)
        or np.any(low < 0)
    ):
        raise ValueError("Invalid frozen scene density window")
    span = np.log10(high / np.maximum(low, 1e-6))
    if not np.isfinite(span).all() or np.any(span <= 0):
        raise ValueError("Invalid frozen logarithmic span")
    x = values.astype(np.float64)
    safe = np.maximum(x, 1e-6)
    density = np.log10(high / safe)
    density_fraction = density / span
    reconstructed = high * np.power(10.0, -density_fraction * span)
    # The logarithmic floor cannot reconstruct a true zero. Preserve that
    # branch explicitly rather than presenting the floor as recovered signal.
    reconstructed = np.where(x < 1e-6, x, reconstructed)
    carrier = reconstructed if via_density else x
    base_receipt = None
    if normalization == "film-base":
        if film_base is None:
            raise ValueError("Film-base normalization requires an explicit reference")
        base = np.asarray(film_base.values, dtype=np.float64)
        if base.shape != (3,) or not np.isfinite(base).all() or np.any(base <= 0):
            raise ValueError("Invalid explicit film-base reference")
        result = carrier / base
        base_receipt = asdict(film_base)
    else:
        if film_base is not None:
            raise ValueError(
                "Do not silently ignore a film-base reference in scene mode"
            )
        result = (carrier - low) / (high - low)
    return result, {
        "normalization": normalization,
        "route": "unclipped NegPy density coordinate -> reconstructed transmission"
        if via_density
        else "direct camera linear signal -> transmission",
        "density_formula": "d=log10(scene_high/max(signal,1e-6)); f=d/log10(scene_high/max(scene_low,1e-6))",
        "inverse_formula": "signal=scene_high*10**(-f*span); original sub-floor signal retained separately",
        "density_fraction_is_not_positive_RGB": True,
        "clipped_density_auto_used": False,
        "scene_high_is_not_film_base": True,
        "film_base": base_receipt,
        "scene_low_subtracted_only_in_scene_window_mode": True,
        "log_floor_fraction_rgb": (x < 1e-6).mean((0, 1)).tolist(),
        "density_roundtrip_max_abs_error_rgb": np.max(
            np.abs(reconstructed - x), axis=(0, 1)
        ).tolist(),
        "camera_to_scanner_calibration_verified": False,
        "transmission": stage_statistics(
            result, domain="unclipped research transmission", bounds=(0, 1)
        ),
    }


def encode_transmission(transmission, *, encoding):
    """Quantize an explicit linear or sqrt carrier once to original 14-bit codes.

    Clipping is observed at the scanner-code boundary. The supplied floating
    array is neither changed nor discarded; it remains the source for later
    exposure/highlight tests. sqrt is an encoding hypothesis, not Gamma=2.
    """
    values = rgb_array(transmission)
    if values.dtype.kind != "f" or encoding not in ("linear", "sqrt"):
        raise ValueError(
            "Require floating transmission and explicit linear/sqrt encoding"
        )
    clipped = np.clip(values.astype(np.float64), 0, 1)
    encoded = clipped if encoding == "linear" else np.sqrt(clipped)
    codes = np.rint(encoded * 16383).astype(np.uint16)
    return codes, {
        "encoding": encoding,
        "formula": "rint(clip(T,0,1)*16383)"
        if encoding == "linear"
        else "rint(sqrt(clip(T,0,1))*16383)",
        "profile_Gamma_is_separate": True,
        "negative_inversion_applied_at_input": False,
        "input_clip": clip_observation(values, clipped),
        "quantization": "binary64, nearest ties-to-even, unsigned 14-bit; PURE research adapter",
        "codes": stage_statistics(
            codes, domain="assumed scanner carrier u14", bounds=(0, 16383)
        ),
        "original_float_input_retained_by_caller": True,
        "camera_to_scanner_calibration_verified": False,
    }


def meter_after_gamma_negative(codes, gamma_negative, *, gray_controls):
    """Explicit PURE full-range meter in HighShadow's actual input domain.

    Stored profile histograms are not universal film characteristics. This
    candidate remeasures the current ROI AFTER GammaNeg, never on raw linear T
    or after the whole negative graph. Full min/max intentionally removes no
    percentile tails. This does not reproduce original AutoHistogram logic.
    """
    data = rgb_array(codes)
    lut = np.asarray(gamma_negative)
    if (
        data.dtype != np.uint16
        or data.min() < 0
        or data.max() > 16383
        or lut.shape != (16384,)
        or lut.dtype.kind not in "ui"
        or lut.max() > 16383
    ):
        raise ValueError(
            "Metering requires RGB14 and an explicit bounded GammaNeg table"
        )
    if (
        not isinstance(gray_controls, (tuple, list))
        or len(gray_controls) != 4
        or any(type(v) is not int or not 1 <= v <= 255 for v in gray_controls)
    ):
        raise ValueError(
            "Explicit Gray controls required; no automatic neutral inference"
        )
    after = lut[data]
    shadow = after.min((0, 1)).astype(int).tolist()
    highlight = after.max((0, 1)).astype(int).tolist()
    if any(a >= b for a, b in zip(shadow, highlight)):
        raise ValueError("Degenerate post-GammaNeg metering range")
    return {
        "Shadow": [min(shadow), *shadow],
        "Highlight": [max(highlight), *highlight],
        "Gray": list(gray_controls),
    }, {
        "method": "PURE explicit same-ROI post-GammaNeg full min/max; not original AutoHistogram",
        "domain": "GammaNeg RGB14 output, immediately before HighShadow(inner)",
        "percentile_tails_removed": False,
        "gray_controls_explicit": list(gray_controls),
        "metering_roi_is_not_output_crop_operation": True,
        "post_gamma_negative": stage_statistics(
            after, domain="post-GammaNeg metering RGB14", bounds=(0, 16383)
        ),
        "factory_auto_histogram_reproduced": False,
    }
