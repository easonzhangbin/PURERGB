"""Bit-exact CPU LUT for separable positive RGB16 output controls."""

from functools import lru_cache

import numpy as np

from .tone_study import GAMMA


@lru_cache(maxsize=16)
def _table(key):
    common_ev, exposure_ev, red_ev, green_ev, blue_ev, contrast = key
    ramp = np.arange(65536, dtype=np.float64)[:, None]
    gains = 2 ** np.array(
        [common_ev + exposure_ev + value for value in (red_ev, green_ev, blue_ev)]
    )
    linear = (ramp / 65535) ** GAMMA * gains
    if contrast != 1:
        linear = 0.18 * (linear / 0.18) ** contrast
    codes = np.rint(np.clip(linear, 0, 1) ** (1 / GAMMA) * 65535).astype(np.uint32)
    result = codes | ((linear > 1).astype(np.uint32) << 16)
    result.flags.writeable = False
    return result


def exact_output_lut(document, common_ev=0.0):
    from .professional_controls import OUTPUT_SPECS, checked_controls

    overrides = checked_controls(document, OUTPUT_SPECS)["values"]
    values = {spec.key: overrides.get(spec.key, spec.default) for spec in OUTPUT_SPECS}
    if values["saturation"] != 1:
        raise ValueError("精确一维LUT不适用于耦合饱和度")
    return _table(
        (
            common_ev,
            values["exposure_ev"],
            values["red_ev"],
            values["green_ev"],
            values["blue_ev"],
            values["contrast"],
        )
    )


def exact_cpu_lut_codes(pixels, document, common_ev=0.0):
    """Keep the reference float64 arithmetic for every possible input code."""
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.uint16
        or pixels.ndim != 3
        or pixels.shape[2] != 3
        or not pixels.size
    ):
        raise ValueError("精确正像LUT仅接受非空RGB16像素")
    lut = exact_output_lut(document, common_ev)
    output = np.empty_like(pixels, order="C")
    rows = max(1, 262144 // pixels.shape[1])
    above = 0
    for start in range(0, pixels.shape[0], rows):
        for channel in range(3):
            values = lut[:, channel][pixels[start : start + rows, :, channel]]
            output[start : start + rows, :, channel] = values & 65535
            above += np.count_nonzero(values >> 16)
    return output, dict(
        identity=False,
        domain="Positive linear Adobe RGB after inversion; no density recovery",
        above_one_fraction=above / pixels.size,
        below_zero_fraction=0.0,
        common_exposure_ev=common_ev,
    )


@lru_cache(maxsize=8)
def _temperature_lut(kelvin):
    from .positive_post import TEMPERATURE_SPEC, post_codes

    # With no additional post edits, the original positive transfer is
    # separable by channel. Run that same implementation over all RGB16 codes.
    ramp = np.arange(65536, dtype=np.uint16).reshape(256, 256, 1)
    ramp = np.repeat(ramp, 3, axis=2)
    adjusted, _ = post_codes(
        ramp, {"version": 1, "values": {TEMPERATURE_SPEC.key: kelvin}},
        collect_ranges=False,
    )
    result = adjusted.reshape(65536, 3)
    result.flags.writeable = False
    return result


def fit_point_codes(pixels, output_document, common_ev, post_document):
    """Apply exact separable output and temperature edits without GPU warmup."""
    from .positive_post import POST_SPECS, TEMPERATURE_SPEC
    from .professional_controls import OUTPUT_SPECS, checked_controls

    output = output_document or {"version": 1, "values": {}}
    values = checked_controls(output, OUTPUT_SPECS)["values"]
    post = checked_controls(post_document, POST_SPECS)["values"] if post_document else {}
    if values.get("saturation", 1) != 1 or post.keys() - {TEMPERATURE_SPEC.key}:
        return None
    lut = _temperature_lut(post[TEMPERATURE_SPEC.key]) if post else None
    if values or common_ev:
        output_lut = exact_output_lut(output, common_ev) & 65535
        # Preserve the intermediate RGB16 encoder; reordering the two edits
        # or combining their float arithmetic changes values near clipping.
        lut = lut[output_lut, np.arange(3)] if lut is not None else output_lut
    if lut is None:
        return pixels
    result = np.empty_like(pixels, order="C")
    if pixels.shape[0] * pixels.shape[1] >= 256 * 1024:
        try:
            from negpy_core import lut_u16
        except ImportError:
            pass
        else:
            # Use the worker's existing Rayon budget for larger fit grids.
            # Each table still addresses the exact 16-bit input code.
            tables = np.ascontiguousarray(lut.T, dtype=np.uint16)
            for channel, table in enumerate(tables):
                plane = np.ascontiguousarray(pixels[..., channel])
                result[..., channel] = lut_u16(plane, table)
            return result
    for channel in range(3):
        result[..., channel] = lut[:, channel][pixels[..., channel]]
    return result
