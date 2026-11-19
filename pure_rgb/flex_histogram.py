"""Isolated FlexColor CHighShadowCurve 14-bit range mapping.

Shadow/Highlight are input code positions, DotColor supplies output black/white
codes, and EndPoints contains two extrapolation policies (not black/white code
positions). Original parameter derivation and the full 16384-code evaluator are
checked independently. This is a scalar range wrapper, not automatic sampling,
Gray correction, scanner calibration or the complete factory pipeline.
"""

import numpy as np

from .flex_routing import MAXIMUM, _table


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def high_shadow_lut(
    nested_lut,
    shadow=0,
    highlight=MAXIMUM,
    output_black=0,
    output_white=255,
    endpoints=(1, 1),
    apply_histogram=True,
):
    """Wrap a caller-supplied 14-bit LUT in original histogram range arithmetic.

    Endpoint policies: 0 holds the respective output endpoint; 1 linearly
    connects toward code zero / 16383; 2 returns hard zero / 16383 outside.
    The high extension denominator is 16384-highlight, as in the original.
    Disabling histogram leaves the nested LUT active; its final code is still
    constrained by the wrapper's endpoint branch.
    """
    table = _table(nested_lut)
    if table.max() > MAXIMUM:
        raise ValueError("高光暗部曲线的嵌套表需保持在 14-bit 输出域内")
    if not (
        _integer(shadow, 0, MAXIMUM)
        and _integer(highlight, 0, MAXIMUM)
        and shadow < highlight
        and _integer(output_black, 0, 255)
        and _integer(output_white, 0, 255)
        and output_black <= output_white
    ):
        raise ValueError("需有效的 Shadow < Highlight 及 0..255 输出黑白点")
    if (
        not isinstance(endpoints, (list, tuple))
        or len(endpoints) != 2
        or any(not _integer(policy, 0, 2) for policy in endpoints)
    ):
        raise ValueError("EndPoints 需两个 0..2 范围的外延策略")
    if type(apply_histogram) is not bool:
        raise ValueError("ApplyHistogram 必须为布尔值")
    if apply_histogram:
        black = output_black * MAXIMUM // 255
        white = output_white * MAXIMUM // 255
        input_scale = float(np.float32(MAXIMUM / (highlight - shadow)))
        output_scale = float(np.float32((white - black) / MAXIMUM))
        low_policy, high_policy = endpoints
    else:
        shadow, highlight, black, white = 0, MAXIMUM, 0, MAXIMUM
        input_scale = output_scale = 1.0
        low_policy = high_policy = 0
    codes = np.arange(MAXIMUM + 1, dtype=np.int64)
    result = np.empty(MAXIMUM + 1, dtype=np.int64)
    low = codes < shadow
    high = codes >= highlight
    inside = ~(low | high)
    if low_policy == 0:
        result[low] = black
    elif low_policy == 1:
        # The branch is empty when shadow is zero; no divide-by-zero occurs.
        result[low] = black * codes[low] // shadow if shadow else 0
    else:
        result[low] = 0
    if high_policy == 0:
        result[high] = white
    elif high_policy == 1:
        result[high] = white + (
            (codes[high] - highlight) * (MAXIMUM - white) // (MAXIMUM + 1 - highlight)
        )
    else:
        result[high] = MAXIMUM
    normalized = np.trunc((codes[inside] - shadow) * input_scale).astype(np.int64)
    result[inside] = np.trunc(table[normalized] * output_scale + black).astype(np.int64)
    return np.frombuffer(result.astype("<u2").tobytes(), dtype="<u2")


def high_shadow_lut_mac(
    nested_lut,
    shadow=0,
    highlight=MAXIMUM,
    output_black=0,
    output_white=255,
    endpoints=(1, 1),
    apply_histogram=True,
):
    """Mac 4.8.13 i386 HighShadow with original SSE float32 boundaries.

    Kept separate from the previously verified Windows/x87 implementation.
    The supplied nested LUT is the entire explicit inner graph, not a single
    Gradation. Input/output quantization and endpoint policy are not recovery.
    """
    table = _table(nested_lut)
    if table.max() > MAXIMUM:
        raise ValueError("Mac 高光暗部曲线的嵌套表需保持在 14-bit 输出域内")
    if not (
        _integer(shadow, 0, MAXIMUM)
        and _integer(highlight, 0, MAXIMUM)
        and shadow < highlight
        and _integer(output_black, 0, 255)
        and _integer(output_white, 0, 255)
        and output_black <= output_white
    ):
        raise ValueError("需有效的 Shadow < Highlight 及 0..255 输出黑白点")
    if (
        not isinstance(endpoints, (list, tuple))
        or len(endpoints) != 2
        or any(not _integer(policy, 0, 2) for policy in endpoints)
    ):
        raise ValueError("EndPoints 需两个 0..2 范围的外延策略")
    if type(apply_histogram) is not bool:
        raise ValueError("ApplyHistogram 必须为布尔值")
    if apply_histogram:
        black, white = output_black * MAXIMUM // 255, output_white * MAXIMUM // 255
        input_scale = np.float32(MAXIMUM / (highlight - shadow))
        output_scale = np.float32((white - black) / MAXIMUM)
        low_policy, high_policy = endpoints
    else:
        shadow, highlight, black, white = 0, MAXIMUM, 0, MAXIMUM
        input_scale = output_scale = np.float32(1)
        low_policy = high_policy = 0
    codes = np.arange(MAXIMUM + 1, dtype=np.int64)
    result = np.empty(MAXIMUM + 1, dtype=np.int64)
    low, high = codes < shadow, codes >= highlight
    inside = ~(low | high)
    if low_policy == 0:
        result[low] = black
    elif low_policy == 1:
        result[low] = black * codes[low] // shadow if shadow else 0
    else:
        result[low] = 0
    if high_policy == 0:
        result[high] = white
    elif high_policy == 1:
        result[high] = white + (codes[high] - highlight) * (MAXIMUM - white) // (
            MAXIMUM + 1 - highlight
        )
    else:
        result[high] = MAXIMUM
    # Original cvtsi2ss/mulss/cvttss2si, then mulss and addss with an
    # intervening float32 store. Algebraic combination would lose these spills.
    normalized = np.trunc(
        (codes[inside] - shadow).astype(np.float32) * input_scale
    ).astype(np.int64)
    scaled = np.float32(table[normalized].astype(np.float32) * output_scale)
    output = np.float32(scaled + np.float32(black))
    result[inside] = np.trunc(output).astype(np.int64)
    return np.frombuffer(result.astype("<u2").tobytes(), dtype="<u2")
