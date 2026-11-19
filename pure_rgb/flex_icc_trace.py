"""Explicit floating ICC boundary probe alongside the authoritative RGB16 path.

A float output here is in the target ICC's encoded domain, not linear light.
The floating LittleCMS driver may take a different precision path from RGB16.
It locates numerical headroom; it does not recreate old ColorSync or guarantee
that the profile's internal tables preserve arbitrary negative or HDR inputs.
"""

from dataclasses import dataclass
import hashlib

import imagecodecs
import numpy as np

from .flex_color_management import scanner_rgb14_icc_bridge
from .stage_trace import stage_statistics


@dataclass(frozen=True)
class ICCFloatTraceResult:
    pixels: np.ndarray
    icc: bytes
    integer_pixels: np.ndarray
    receipt: dict


def probe_icc_float14(
    codes,
    source_icc,
    target_icc,
    *,
    source_sha256,
    target_sha256,
    assumption,
    input_domain,
    intent=1,
    chunk_pixels=262144,
):
    """Compare float/uint16 ICC calls using the same original u14<<2 scale."""
    authoritative = scanner_rgb14_icc_bridge(
        codes,
        source_icc,
        target_icc,
        source_sha256=source_sha256,
        output_sha256=target_sha256,
        assumption=assumption,
        input_domain=input_domain,
        intent=intent,
        chunk_pixels=chunk_pixels,
    )
    output = np.empty(codes.shape, np.float32)
    src, dst = codes.reshape(-1, 3), output.reshape(-1, 3)
    before = hashlib.sha256(codes.tobytes()).hexdigest()
    for start in range(0, len(src), chunk_pixels):
        block = src[start : start + chunk_pixels]
        # Match the original integer interface scaling. Full-range 16383->1
        # would be a different hypothesis, and applying an OETF would double it.
        normalized = (block.astype(np.uint32) << 2).astype(np.float32) / np.float32(
            65535
        )
        converted = imagecodecs.cms_transform(
            normalized.reshape(1, -1, 3),
            source_icc,
            target_icc,
            colorspace="RGB",
            outcolorspace="RGB",
            intent=intent,
            flags=0x0100,
        )
        if converted.dtype != np.float32 or not np.isfinite(converted).all():
            raise ValueError("ICC float driver returned invalid output")
        dst[start : start + len(block)] = converted.reshape(-1, 3)
    if before != hashlib.sha256(codes.tobytes()).hexdigest():
        raise ValueError("ICC trace modified RGB14 source")
    quantized = np.rint(np.clip(output.astype(np.float64), 0, 1) * 65535).astype(
        np.uint16
    )
    difference = quantized.astype(np.int32) - authoritative.pixels.astype(np.int32)
    output.setflags(write=False)
    return ICCFloatTraceResult(
        output,
        authoritative.icc,
        authoritative.pixels,
        {
            "schema": "pure-rgb-flex-floating-icc-probe",
            "version": 1,
            "authoritative_integer_bridge": authoritative.receipt,
            "source_pixels_sha256": before,
            "float_input_formula": "float32(uint32(rgb14)<<2)/float32(65535); no OETF",
            "LCMS_flags": {
                "float": 256,
                "integer": 256,
                "policy": "NOOPTIMIZE; same flags and no black-point-compensation toggle between drivers",
            },
            "float_output_domain": "Target ICC encoded RGB float32; not linear scene or working light",
            "float_output": stage_statistics(
                output,
                domain="target ICC encoded floating output",
                bounds=(0, 1),
                icc=target_icc,
            ),
            "clipped_quantized_float_vs_authoritative_integer": {
                "maximum_abs_code_difference": int(np.max(np.abs(difference))),
                "different_channel_sample_fraction": float(np.mean(difference != 0)),
                "interpretation": "Driver precision/interpolation and quantization comparison, not a factory-equivalence tolerance",
            },
            "source_unchanged": True,
            "old_colorsync_equivalence_claimed": False,
            "physical_scene_recoverability_certified": False,
            "production_defaults_changed": False,
        },
    )
