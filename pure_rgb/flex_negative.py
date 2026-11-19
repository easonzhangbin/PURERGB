"""Independent FlexColor 4.8.13 i386 negative-curve scalar research kernels.

Inputs are explicit integer 14-bit table codes, never implicit camera RAW,
density, RGB16, ICC or gamma-encoded colors. Negative is a separate piecewise
polynomial stage; it does not invert an image by itself. GammaNeg reads global
preferences, not the film profile's Gamma field. Neither implements the full
seven-curve graph or camera-to-scanner calibration.

Negative reconstruction is checked over every code against original Mac and
Windows instructions. GammaNeg's original i386 loop is checked with a declared
host math.pow substitute for its external Darwin _pow import; original libm
bit equivalence and the unavailable Windows MSVCR90!_CIpow remain unverified.
"""

from functools import lru_cache
import math
import struct

import numpy as np

MAXIMUM = 16383


def _f32_bits(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# PrepareGradations i386 original float immediates/constants, not fitted values.
PREPARE_NEGATIVE_PIVOTS = {
    "common": (_f32_bits(0x431EB333), _f32_bits(0x42C93333)),
    "red": (107.5, _f32_bits(0x4311199A)),
    "green": (_f32_bits(0x430BE666), 124.5),
    "blue": (_f32_bits(0x4312B333), _f32_bits(0x42B4CCCD)),
}


def _numeric(value, name):
    if type(value) not in (int, float):
        raise ValueError(name + " 需数值，不能是布尔值或隐式字符串")
    return float(value)


def _film_type(film_type):
    if type(film_type) is not int or film_type not in (0, 1, 2):
        raise ValueError("当前研究只接受 FilmType 0／1／2")


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(name + " 需明确布尔值")


def _immutable(values):
    return np.frombuffer(np.asarray(values, dtype="<u2").tobytes(), dtype="<u2")


def gamma_negative_parameters(var_gamma=0.2, *, enhanced_shadow=False, stretch=1.0):
    """Return effective original preference fallbacks and code normalizer.

    NegVarGamma must be strictly within (0.099,10.1), else 0.2 is selected.
    StretchNegGamma must be strictly within (1.01,1.11), else no stretch is applied.
    These are GenGamma14's own fallback branches, not a user's saved preferences.
    """
    _boolean(enhanced_shadow, "EnhancedShadow")
    gamma = _numeric(var_gamma, "NegVarGamma") if var_gamma is not None else 0.2
    factor = _numeric(stretch, "StretchNegGamma") if stretch is not None else 1.0
    if not 0.099 < gamma < 10.1:
        gamma = 0.2
    if not 1.01 < factor < 1.11:
        factor = 1.0
    scale = (17700.0 if enhanced_shadow else 16383.0) * factor
    return {
        "var_gamma": gamma,
        "exponent": 1.0 / gamma,
        "stretch": factor,
        "input_scale": scale,
    }


@lru_cache(maxsize=64)
def _gamma(scale, exponent, active):
    if not active:
        return _immutable(np.arange(MAXIMUM + 1))
    values = []
    for code in range(MAXIMUM + 1):
        normalized = code / scale
        base = 1.0 - normalized * normalized
        # Separate Python binary64 operations mirror SSE2 spills. External pow
        # is the same explicitly declared service used by the Mac loop oracle.
        value = math.pow(base, exponent) * 16383.0
        values.append(math.trunc(value) & 65535)
    return _immutable(values)


def gamma_negative_lut(
    var_gamma=0.2, *, enhanced_shadow=False, stretch=1.0, film_type=1, enabled=True
):
    """16384-entry GammaNeg table; enabled nonzero FilmType inverts tone.

    The original table truncates binary64 products toward zero then stores a word.
    FilmType=0 or disabled produces identity. Gamma controls here are global
    NegVarGamma preference values; profile Gamma=2 is another stage.
    """
    _film_type(film_type)
    _boolean(enabled, "enabled")
    p = gamma_negative_parameters(
        var_gamma, enhanced_shadow=enhanced_shadow, stretch=stretch
    )
    return _gamma(p["input_scale"], p["exponent"], enabled and film_type != 0)


def negative_parameters(pivot_x, pivot_y):
    """Derive original Mac SSE2 coefficients with explicit float32 spills."""
    x, y = _numeric(pivot_x, "pivot_x"), _numeric(pivot_y, "pivot_y")
    if not math.isfinite(x) or not 0 < x < MAXIMUM / 64:
        raise ValueError("pivot_x 需在 (0,16383/64) 控制域内；不推断退化除零行为")
    if not math.isfinite(y) or not 0 <= y <= MAXIMUM / 64:
        raise ValueError("pivot_y 需在 [0,16383/64] 控制域内")
    f = np.float32
    x, y = f(f(x) * f(64)), f(f(y) * f(64))
    lower_a = f(f(x - y) / f(x * x))
    lower_b = f(f(1) - f(f(f(x + x) + f(y * f(-2))) / x))
    remaining = 16383.0 - float(x)
    upper_a = f((float(f(f(16383) - y)) - remaining) / (remaining * remaining))
    return {
        "pivot_x14": float(x),
        "pivot_y14": float(y),
        "lower_a": float(lower_a),
        "lower_b": float(lower_b),
        "upper_a": float(upper_a),
    }


@lru_cache(maxsize=64)
def _negative(x, y, a, b, c, active):
    if not active:
        return _immutable(np.arange(MAXIMUM + 1))
    f = np.float32
    codes = np.arange(MAXIMUM + 1, dtype=np.float32)
    low = f(f(f(a) * codes) * codes) + f(codes * f(b))
    delta = f(codes - f(x))
    high = f(f(f(f(c) * delta) * delta) + delta) + f(y)
    low_result = np.where((low > 0) & (low < 16384), np.trunc(low), 0)
    # This is an original upper-branch sentinel, NOT a symmetric 0..16383 clip.
    high_result = np.where((high > 0) & (high < 16384), np.trunc(high), 16384)
    return _immutable(np.where(codes < f(x), low_result, high_result))


def negative_lut(pivot_x, pivot_y, *, film_type=1, disabled=False):
    """16384-entry original piecewise Negative stage from two control values.

    Only FilmType=1 and NoNegCurve=false execute the polynomial. Some parameter
    combinations return the original 16384 sentinel; callers must not silently use
    it as an index into a 16384-entry downstream table or silently clip it away.
    No default pivot is invented: PrepareGradations common/R/G/B values are exposed
    separately and their full graph placement is another research contract.
    """
    _film_type(film_type)
    _boolean(disabled, "NoNegCurve")
    p = negative_parameters(pivot_x, pivot_y)
    return _negative(
        p["pivot_x14"],
        p["pivot_y14"],
        p["lower_a"],
        p["lower_b"],
        p["upper_a"],
        film_type == 1 and not disabled,
    )


def apply_negative_table(codes, table):
    """Lookup explicit 14-bit integer codes; never infer a RAW/output encoding."""
    values, lut = np.asarray(codes), np.asarray(table)
    if (
        lut.shape != (16384,)
        or lut.dtype.kind not in "ui"
        or lut.min() < 0
        or lut.max() > 65535
    ):
        raise ValueError("LUT 需含 16384 项无符号16位编码")
    if values.dtype.kind not in "ui" or (
        values.size and (values.min() < 0 or values.max() > MAXIMUM)
    ):
        raise ValueError("输入需0..16383整数编码，不能隐式解释RAW／RGB16／浮点")
    return lut[values]
