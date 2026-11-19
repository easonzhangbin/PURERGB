"""Explicit-LUT FlexColor curve graph and verified Mac scalar components.

Research interfaces only; no factory preset, scanner-domain conversion, CMS,
automatic histogram or original preference state is inferred. The caller must
provide every stage. These functions are not connected to the accepted renderer.
"""

import math

import numpy as np

from .flex_routing import MAXIMUM, _table, aggregate_lut


def _immutable(values):
    return np.frombuffer(np.asarray(values, dtype="<u2").tobytes(), dtype="<u2")


def _three(tables, label):
    if not isinstance(tables, (list, tuple)) or len(tables) != 3:
        raise ValueError(f"{label} 需显式提供 R/G/B 三条完整查找表")
    return [_table(table) for table in tables]


def gray_no_cms_parameters(gray):
    """Return three (X,Y) pairs from the original no-CMS Gray branch.

    Gray contains four serialized control values, here restricted to 0..255.
    Gray[0] is retained in the input contract but does not affect this branch.
    GetGray14's name does not make these already scaled pixel codes: the
    SinglePoint stage converts its control coordinates to code units later.
    """
    if (
        not isinstance(gray, (list, tuple))
        or len(gray) != 4
        or any(type(value) is not int or not 0 <= value <= 255 for value in gray)
    ):
        raise ValueError("Gray 需四个 0..255 整数控制值")
    target = int(np.float32(sum(gray[1:]) / 3.0))
    return tuple((float(value), float(target)) for value in gray[1:])


def single_point_lut(x, y, *, enabled=True):
    """Original 4.8.13 i386 SinglePoint arithmetic for explicit control X/Y.

    X/Y are in the approximately 8-bit control domain, not 14-bit pixels.
    Each is converted to float32 and multiplied by 64. SSE float32 operation
    order and the original asymmetric out-of-range branches are preserved.
    Degenerate X and coordinates outside the verified 1..255 X / 0..255 Y
    control domain are rejected.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled 必须为布尔值")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        for value in (x, y)
    ):
        raise ValueError("灰点坐标需有限数值")
    if not (1 <= float(x) <= 255 and 0 <= float(y) <= 255):
        raise ValueError("灰点需 1 ≤ X ≤ 255、0 ≤ Y ≤ 255 控制坐标")
    codes = np.arange(MAXIMUM + 1, dtype=np.float32)
    if not enabled:
        return _immutable(codes)
    x14, y14 = np.float32(x) * np.float32(64), np.float32(y) * np.float32(64)
    # Every assignment corresponds to an original SSE rounding boundary.
    low_a = np.float32(np.float32(x14 - y14) / np.float32(x14 * x14))
    low_b = np.float32(np.float32(x14 + x14) + np.float32(y14 * np.float32(-2)))
    low_b = np.float32(np.float32(1) - np.float32(low_b / x14))
    difference = float(MAXIMUM) - float(x14)
    high_a = np.float32(
        (float(np.float32(np.float32(MAXIMUM) - y14)) - difference)
        / (difference * difference)
    )
    low = codes < x14
    result = np.empty(MAXIMUM + 1, dtype=np.int64)
    lower = np.float32(
        np.float32(np.float32(low_a * codes[low]) * codes[low])
        + np.float32(low_b * codes[low])
    )
    lower_valid = (lower > 0) & (lower < MAXIMUM + 1)
    result[low] = np.where(lower_valid, np.trunc(lower), 0).astype(np.int64)
    offset = np.float32(codes[~low] - x14)
    upper = np.float32(
        np.float32(np.float32(np.float32(high_a * offset) * offset) + offset) + y14
    )
    upper_valid = (upper > 0) & (upper < MAXIMUM + 1)
    result[~low] = np.where(upper_valid, np.trunc(upper), MAXIMUM).astype(np.int64)
    return _immutable(result)


def rise_limiter_lut(table):
    """Original final-table forward limiter; decreasing segments are untouched.

    This does not recover clipped highlights or enforce a monotonic curve.
    """
    result = _table(table).copy()
    for index in range(1, MAXIMUM + 1):
        if result[index] > result[index - 1] + 64:
            result[index] = result[index - 1] + 64
    return _immutable(result)


def compose_inner_luts(
    *, gamma, gradations, negative_common, negative_channels, contrast, gray
):
    """Compose the verified inner graph from caller-provided tables only.

    Add/subtract Gradations sample the same saved reference after the channel
    Negative stage. Seven serial Gradations would produce a different graph.
    No None/omitted stage is replaced with an assumed default or identity.
    """
    if not isinstance(gradations, (list, tuple)) or len(gradations) != 7:
        raise ValueError("需显式提供七条 Gradations 完整查找表")
    points = [_table(table) for table in gradations]
    gamma, negative_common, contrast = map(_table, (gamma, negative_common, contrast))
    negatives, grays = _three(negative_channels, "Negative"), _three(gray, "Gray")
    return tuple(
        aggregate_lut(
            [
                gamma,
                points[0],
                negative_common,
                negatives[channel],
                points[1 + channel],
                points[4 + channel],
                contrast,
                grays[channel],
            ],
            [0, 0, 0, 0, 1, 2, 0, 0],
        )
        for channel in range(3)
    )


def compose_outer_luts(
    *, gamma_negative, high_shadow_wrapped_inner, apply_rise_limiter
):
    """Compose GammaNeg -> explicitly supplied HighShadow(inner) -> limiter.

    high_shadow_wrapped_inner is already the range-wrapped inner table for each
    channel. This interface does not create a histogram table from guessed
    endpoints or mix Windows x87 and Mac SSE intermediate rounding semantics.
    """
    if type(apply_rise_limiter) is not bool:
        raise ValueError("需明确选择是否应用最终曲线上升限幅")
    gamma = _three(gamma_negative, "GammaNeg")
    wrapped = _three(high_shadow_wrapped_inner, "HighShadow(inner)")
    result = tuple(
        aggregate_lut([gamma[channel], wrapped[channel]], [0, 0])
        for channel in range(3)
    )
    return (
        tuple(rise_limiter_lut(table) for table in result)
        if apply_rise_limiter
        else result
    )
