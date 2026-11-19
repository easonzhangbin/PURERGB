"""Opt-in observations before ColorMap's integer/wrap/clamp boundary.

Final pixels still come from the verified original ColorMap port. The temporary
float32 observation reconstructs the same SSE expression before integer storage;
it is not a scene-linear color space or a new color engine. Keeping it can help
locate information loss, but cannot reverse the earlier Gradation lookup.
"""

from dataclasses import dataclass
import hashlib

import numpy as np

from .flex_color_adjustments import color_map14, color_map_parameters


@dataclass(frozen=True)
class ColorMapTraceResult:
    pixels: np.ndarray
    precast: np.ndarray | None
    receipt: dict


def _digest(pixels):
    return hashlib.sha256(pixels.astype("<u2", copy=False).tobytes()).hexdigest()


def trace_color_map14(
    pixels,
    *,
    color_corr,
    saturation,
    apply_cc,
    input_domain,
    retain_precast=False,
    chunk_pixels=262144,
):
    """Run original output and separately account for its finite code boundary.

    ``retain_precast=True`` allocates one float32 image in caller code units.
    It is a diagnostic checkpoint only; do not assign an ICC or feed negative
    values to the RGB14 ICC bridge. Default tracing keeps only small counters.
    A bypass records input codes as a bookkeeping identity, not executed SSE.
    """
    if type(retain_precast) is not bool:
        raise ValueError("retain_precast requires an explicit boolean")
    before = (
        _digest(pixels)
        if isinstance(pixels, np.ndarray) and pixels.dtype == np.uint16
        else None
    )
    # Reuse the authoritative domain/buffer/parameter validation before access.
    output = color_map14(
        pixels,
        color_corr=color_corr,
        saturation=saturation,
        apply_cc=apply_cc,
        input_domain=input_domain,
        chunk_pixels=chunk_pixels,
    )
    params = color_map_parameters(
        color_corr=color_corr, saturation=saturation, apply_cc=apply_cc
    )
    retained = np.empty(pixels.shape, np.float32) if retain_precast else None
    kept = retained.reshape(-1, 3) if retained is not None else None
    source, actual = pixels.reshape(-1, 3), output.reshape(-1, 3)
    counts = {
        k: np.zeros(3, np.int64)
        for k in [
            "precast_below_zero",
            "precast_above_16383",
            "output_differs_from_simple_clip",
            "nonzero_input_now_zero",
            "nonmax_input_now_maximum",
            "output_zero",
            "output_maximum",
        ]
    }
    minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
    coefficients = params["coefficients_rgb_by_sector"]
    for offset in range(0, len(source), chunk_pixels):
        rgb = source[offset : offset + chunk_pixels].astype(np.int32)
        values = rgb.astype(np.float32)
        if params["stage_active"]:
            r, g, b = rgb.T
            basis = tuple(
                v.astype(np.float32)
                for v in (
                    np.maximum(r - np.maximum(g, b), 0),
                    np.maximum(g - np.maximum(r, b), 0),
                    np.maximum(b - np.maximum(r, g), 0),
                    np.maximum(np.minimum(g, b) - r, 0),
                    np.maximum(np.minimum(r, b) - g, 0),
                    np.maximum(np.minimum(r, g) - b, 0),
                )
            )
            for channel, terms in enumerate(coefficients):
                correction = basis[0] * np.float32(terms[0])
                for sector in range(1, 6):
                    correction = correction + basis[sector] * np.float32(terms[sector])
                correction = correction / np.float32(-100)
                values[:, channel] = rgb[:, channel].astype(np.float32) + correction
        truncated = np.trunc(values).astype(np.int32)
        word = truncated & 65535
        reconstructed = np.where(word >= 32768, 0, np.minimum(word, 16383)).astype(
            np.uint16
        )
        block = actual[offset : offset + len(rgb)]
        if not np.array_equal(block, reconstructed):
            raise ValueError(
                "Precast observation does not reproduce authoritative integer output"
            )
        if kept is not None:
            kept[offset : offset + len(rgb)] = values
        ordinary = np.clip(truncated, 0, 16383).astype(np.uint16)
        masks = {
            "precast_below_zero": values < 0,
            "precast_above_16383": values > 16383,
            "output_differs_from_simple_clip": block != ordinary,
            "nonzero_input_now_zero": (rgb != 0) & (block == 0),
            "nonmax_input_now_maximum": (rgb != 16383) & (block == 16383),
            "output_zero": block == 0,
            "output_maximum": block == 16383,
        }
        for key, mask in masks.items():
            counts[key] += mask.sum(axis=0)
        minimum = np.minimum(minimum, values.min(axis=0))
        maximum = np.maximum(maximum, values.max(axis=0))
    if before != _digest(pixels):
        raise ValueError("ColorMap tracing changed source codes")
    if retained is not None:
        retained.setflags(write=False)
    return ColorMapTraceResult(
        output,
        retained,
        {
            "schema": "pure-rgb-flex-colormap-precast-trace",
            "version": 1,
            "parameters": params,
            "input_shape": list(pixels.shape),
            "input_dtype": str(pixels.dtype),
            "source_pixels_sha256": before,
            "output_pixels_sha256": _digest(output),
            "source_unchanged": True,
            "precast_retained": retain_precast,
            "precast_domain": "Reconstructed SSE float32 before integer conversion, in RGB14 code units; not linear light or ICC-encoded output",
            "precast_minimum_rgb": minimum.tolist(),
            "precast_maximum_rgb": maximum.tolist(),
            "per_channel_sample_counts": {k: v.tolist() for k, v in counts.items()},
            "per_channel_sample_fractions": {
                k: (v / len(source)).tolist() for k, v in counts.items()
            },
            "integer_output_matches_verified_original_port": True,
            "bypass_precast_is_bookkeeping_identity": not params["stage_active"],
            "integer_policy": "trunc float32 to int32 -> low16 -> negative signed word to0 -> unsigned14 upper ceiling",
            "physical_scene_clipping_or_recoverability_certified": False,
            "earlier_LUT_code_collisions_reversed": False,
            "production_defaults_changed": False,
        },
    )
