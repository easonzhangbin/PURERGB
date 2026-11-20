"""Nikon prescan exposure parameters, not an inversion LUT builder.

Adapted from activexray/nkscan, commit
f971cfd8efe89f6e02507c80b0945206fdaf0bcd, src/scan/meter/{mod,negative}.rs
and src/scan/thumbnail.rs, under its MIT option (MIT OR Apache-2.0).
https://github.com/activexray/nkscan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Inputs require measured scanner metadata or an explicit research hypothesis.
No camera exposure, film-base density, input calibration, or builder is inferred.
"""

from dataclasses import dataclass
import hashlib
import math
from numbers import Real
import struct

import numpy as np

REFERENCE_COMMIT = "f971cfd8efe89f6e02507c80b0945206fdaf0bcd"
SCANNER_DOMAIN = "nikon-scanner-prescan-codes"
CAMERA_HYPOTHESIS = "camera-linear-codes-unvalidated-scanner-hypothesis"
MAX_EXPOSURE = 0x3FFFFFF
DEFAULT_BASE_DENSITY = (0.316, 0.737, 0.886)
SEED_FACTORS = {
    "color-negative": (1.65959, 3.98107, 5.37032),
    "monochrome-negative": (1.86209, 2.0893, 2.0893),
}
CLIPPED_FACTORS = {
    "color-negative": (
        (1.65959, 1.15942, 0.9, 0.9),
        (3.98107, 2.84605, 1.4264, 0.9),
        (5.37032, 3.19332, 1.60045, 0.9),
    ),
    "monochrome-negative": (
        (1.86209, 0.94242, 0.9),
        (2.0893, 1.04713, 0.9),
        (2.0893, 1.04713, 0.9),
    ),
}


@dataclass(frozen=True)
class MeteringResult:
    values: tuple | None
    receipt: dict


def required_missing_inputs():
    """Metadata a camera RAW cannot supply merely by being an RGB image."""
    return (
        "scanner start-up base exposures for R/G/B, in native exposure ticks",
        "actual prescan exposures for R/G/B, bound to these prescan pixels",
        "film-base density measured on the thumbnail with its actual exposure ratio",
        "valid sample bits and independently chosen prescan analysis ROI",
        "scanner/device input domain or an explicitly unvalidated camera hypothesis",
    )


def _domain(domain):
    if domain not in (SCANNER_DOMAIN, CAMERA_HYPOTHESIS):
        raise ValueError("需明确 scanner 预扫域或未验证的 camera-to-scanner 假设")
    return domain


def _kind(kind):
    if kind not in SEED_FACTORS:
        raise ValueError(
            "需明确 color-negative 或 monochrome-negative；工艺不决定影像类型"
        )
    return kind


def _exposures(values, name):
    if (
        not isinstance(values, (tuple, list))
        or len(values) != 3
        or any(type(v) is not int or not 1 <= v <= 0xFFFFFFFF for v in values)
    ):
        raise ValueError(name + "需三个正 uint32 原生曝光计数，不能由 RAW 自动虚构")
    return tuple(values)


def _density(values):
    if (
        not isinstance(values, (tuple, list))
        or len(values) != 3
        or any(
            not isinstance(v, Real)
            or isinstance(v, (bool, np.bool_))
            or not math.isfinite(v)
            for v in values
        )
    ):
        raise ValueError("片基密度需三个明确有限值；缺少实测时必须单列研究假设")
    return tuple(float(v) for v in values)


def _codes(pixels, bits):
    if type(bits) is not int or not 1 <= bits <= 16:
        raise ValueError("有效位深需 1..16 整数")
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.uint16
        or pixels.ndim != 3
        or pixels.shape[2] != 3
        or not pixels.size
        or int(pixels.max()) > (1 << bits) - 1
    ):
        raise ValueError("输入需非空 H×W×3 uint16，码值不得超过声明位深")
    return pixels


def _roi(pixels, roi):
    h, w = pixels.shape[:2]
    if roi is None:
        return (0, 0, w, h)
    if (
        not isinstance(roi, (list, tuple))
        or len(roi) != 4
        or any(type(v) is not int for v in roi)
        or not 0 <= roi[0] < roi[2] <= w
        or not 0 <= roi[1] < roi[3] <= h
    ):
        raise ValueError("ROI 需原始像素 x0,y0,x1,y1 半开区间，仅用于分析")
    return tuple(roi)


def _percentile(value):
    if (
        not isinstance(value, Real)
        or isinstance(value, (bool, np.bool_))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("percentile 需 0..1 有限数值")
    # Rust's public Metering stores f32 before promoting in the histogram.
    return float(np.float32(value))


def _hash(pixels):
    return hashlib.sha256(
        np.asarray(pixels, dtype="<u2", order="C").tobytes()
    ).hexdigest()


def _receipt(domain, pixels, bits, stage):
    return {
        "schema": "pure-rgb-nikon-metering",
        "version": 1,
        "stage": stage,
        "source_commit": REFERENCE_COMMIT,
        "license_option": "MIT",
        "source_pixels_sha256": _hash(pixels),
        "input_domain": domain,
        "valid_bits": bits,
        "unvalidated_camera_domain": domain == CAMERA_HYPOTHESIS,
        "scanner_acquisition_metadata_independently_verified": False,
        "source_pixels_changed": False,
        "inversion_lut_generated": False,
        "factory_cms_executed": False,
        "production_defaults_changed": False,
    }


def seed_exposures(base_exposures, *, film_kind):
    """The public negative prescan seed only; counts are not photographic EV."""
    base = _exposures(base_exposures, "start-up exposure")
    factors = SEED_FACTORS[_kind(film_kind)]
    return tuple(
        min(int(b * f), 0xFFFFFFFF) for b, f in zip(base, factors, strict=True)
    )


def base_density_from_thumbnail(
    thumbnail_codes, *, bits, base_exposures, thumbnail_exposures, input_domain
):
    """Port GetBaseDensity: sensor rows × feed columns, 75..224 feed columns.

    A too-short or wholly bright/empty thumbnail returns None. Out-of-range
    channel density returns the declared Nikon default, recorded as fallback.
    This is not a generic film-edge picker for arbitrary camera thumbnails.
    """
    pixels = _codes(thumbnail_codes, bits)
    base = _exposures(base_exposures, "start-up exposure")
    taken = _exposures(thumbnail_exposures, "thumbnail exposure")
    domain = _domain(input_domain)
    receipt = _receipt(domain, pixels, bits, "thumbnail-base-density")
    receipt.update(
        base_exposures=base,
        thumbnail_exposures=taken,
        orientation="axis0 sensor rows; axis1 feed columns; not display orientation",
        measured_feed_columns_half_open=[75, 225],
    )
    if pixels.shape[1] < 226:
        return MeteringResult(None, {**receipt, "reason": "thumbnail-too-short"})
    full = (1 << bits) - 1
    means = pixels[:, 75:225].astype(np.float64).mean(axis=0)
    brightest = np.where(means < 0.9 * full, means, 0).max(axis=0)
    if np.any(brightest == 0):
        return MeteringResult(None, {**receipt, "reason": "no-unclipped-column-means"})
    actual = tuple(
        math.log10(full / (float(v) * (b / t)))
        for v, b, t in zip(brightest, base, taken, strict=True)
    )
    ranges = ((0.121, 0.526), (0.459, 0.969), (0.654, 1.137))
    fallback = tuple(
        not lo <= v <= hi for v, (lo, hi) in zip(actual, ranges, strict=True)
    )
    values = tuple(
        DEFAULT_BASE_DENSITY[c] if fallback[c] else actual[c] for c in range(3)
    )
    return MeteringResult(
        values,
        {
            **receipt,
            "raw_density": actual,
            "default_fallback_channels": fallback,
            "brightest_column_means": brightest.tolist(),
        },
    )


def _widen(pixels, bits):
    # Rust f64::round for nonnegative codes is half-away, not numpy round-even.
    return np.floor(
        pixels.astype(np.float64) * 65535.0 / ((1 << bits) - 1) + 0.5
    ).astype(np.uint16)


def _hist_ends(hist, percentile):
    n = int(hist.sum())
    if not n:
        return None
    tail = int(n * 0.0042)
    lo = int(np.searchsorted(hist.cumsum(), tail, side="left"))
    top = int((1.0 - percentile) * n)
    hi = 65535 - int(np.searchsorted(hist[::-1].cumsum(), top, side="right"))
    return (float(lo), float(hi)) if hi >= 0 else None


def _end_points(codes, scale, base_density, percentile):
    v = codes.reshape(-1, 3)
    scaled = v.astype(np.float64) * scale
    d = base_density
    masks = (10 ** (d[1] - d[0]), 10 ** (d[2] - d[0]), 10 ** (d[2] - d[1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = [
            scaled[:, 0] / (masks[0] * scaled[:, 1]),
            scaled[:, 0] / (masks[1] * scaled[:, 2]),
            scaled[:, 1] / (masks[2] * scaled[:, 2]),
        ]
    # NaN (0/0) is neutral in the original !(q<=1/3 || 3<=q).
    rg, rb, gb = [~((q <= 1 / 3) | (q >= 3)) for q in ratios]
    near = (rg | rb, rg | gb, rb | gb)
    all_ends, gray_ends, gray_counts = [], [], []
    for c in range(3):
        bad = ~near[c]
        before_bad = bad.cumsum() - bad
        accepted = near[c] | (before_bad.astype(float) / len(v) >= 0.9)
        ah = np.bincount(v[:, c], minlength=65536)
        gh = np.bincount(v[accepted, c], minlength=65536)
        all_ends.append(_hist_ends(ah, percentile))
        gray_ends.append(_hist_ends(gh, percentile))
        gray_counts.append(int(accepted.sum()))
    if any(x is None for x in all_ends + gray_ends):
        return None
    return all_ends, gray_ends, gray_counts


def _interpolate(xs, ys, x):
    # The low side extrapolates; the high side holds. Duplicate plateau X
    # returns the preceding Y. np.interp would change this behavior.
    for i in range(1, 32):
        if x < xs[i]:
            if xs[i] == xs[i - 1]:
                return ys[i - 1]
            return (x - xs[i - 1]) * (ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1]) + ys[
                i - 1
            ]
    return ys[31]


def _sample_density(value):
    return math.log2(65535.0 / value) * math.log10(2.0) if value else math.inf


def _rust_u32(value):
    # Rust float->uint casts truncate, saturate infinities, and map NaN to 0.
    if math.isnan(value) or value < 0:
        return 0
    return int(min(value, 0xFFFFFFFF))


def _off_curve(all_ends, neutral, scale, base_density, film_kind):
    t, templates = CHARACTERISTIC_CURVES[film_kind]
    all_scaled = [
        (max(lo * scale[c], 1.0), max(hi * scale[c], 1.0))
        for c, (lo, hi) in enumerate(all_ends)
    ]
    ns = [_sample_density(neutral[c][0] * scale[c]) for c in range(3)]
    nh = [_sample_density(neutral[c][1] * scale[c]) for c in range(3)]
    base = [min(base_density[c], nh[c]) for c in range(3)]
    curve = [[v + (base[c] - templates[c][0]) for v in templates[c]] for c in range(3)]
    fog = [b + 0.15 for b in base]

    def at(c, x):
        return _interpolate(curve[c], t, x)

    se = [at(c, ns[c]) for c in range(3)]
    he = [at(c, nh[c]) for c in range(3)]
    fe = [at(c, fog[c]) for c in range(3)]
    ranges = [(se[c] - fe[c], he[c] - fe[c]) for c in range(3)]
    (lr, hr), (lg, hg), (lb, hb) = ranges
    g = lg - hg
    nr, nb = 10 ** -abs(hr - hg), 10 ** -abs(hb - hg)
    sr, sb = 3 ** -abs(lr - hr - g), 3 ** -abs(g - (lb - hb))
    dense = min(10 ** -(0.8 - max(lr, lg, lb)), 1.0)
    wr, wb = (1 - nr + sr * nr) * dense, (1 - nb + sb * nb) * dense
    shift_r = (se[0] - se[1]) * wr + (he[0] - he[1]) * (1 - wr) * nr
    shift_b = (se[1] - se[2]) * wb + (he[1] - he[2]) * (1 - wb) * nb
    shifted = [[v - shift_r for v in t], t, [v + shift_b for v in t]]

    def at(c, x):
        return _interpolate(curve[c], shifted[c], x)

    sd = [_sample_density(x[0]) for x in all_scaled]
    hd = [_sample_density(x[1]) for x in all_scaled]
    fe = [at(c, fog[c]) for c in range(3)]
    shadow_over = max(at(c, sd[c]) - fe[c] for c in range(3))
    highlight_over = min(at(c, hd[c]) - fe[c] for c in range(3))
    shadow = max(at(c, min(sd[c], curve[c][31])) for c in range(3))
    highlight = min(at(c, max(hd[c], curve[c][0])) for c in range(3))
    weight = min(max(min(10 ** (-(r[1] + 0.23) * 4), 1.0) for r in ranges), 1.0)
    was = shadow
    if shadow - highlight < 0.6:
        highlight -= (1 - weight) * (0.6 - (shadow - highlight))
        shadow = max(highlight + 0.6, was + 0.05)
    else:
        shadow += 0.05
    span = shadow - highlight
    reach = (
        max(min(0.4 ** (highlight_over + 0.23) * 1.9, 1.9), span) if span < 1.9 else 1.9
    )
    if shadow_over < 1.2:
        shadow += (1.2 - shadow_over) * (span / reach)
    densities = [_interpolate(shifted[c], curve[c], highlight) for c in range(3)]
    return densities, {
        "effective_base_density": base,
        "fog_density": fog,
        "red_log_exposure_shift": shift_r,
        "blue_log_exposure_shift": shift_b,
        "final_shadow_log_exposure": shadow,
        "final_highlight_log_exposure": highlight,
        "output_target_density": densities,
    }


def negative_exposures(
    prescan_codes,
    *,
    bits,
    base_exposures,
    prescan_exposures,
    base_density,
    film_kind,
    input_domain,
    roi=None,
    percentile=1.0,
):
    """CNegaAlgorithm::Process exposure calculation only, no builder params/LUT.

    Input H×W: H is feed lines, W is sensor position in increasing crop-X order.
    ROI is half-open x0,y0,x1,y1 and never crops a rendered output.
    """
    pixels = _codes(prescan_codes, bits)
    base = _exposures(base_exposures, "start-up exposure")
    taken = _exposures(prescan_exposures, "prescan exposure")
    density = _density(base_density)
    kind, domain = _kind(film_kind), _domain(input_domain)
    percent = _percentile(percentile)
    box = _roi(pixels, roi)
    selected = _widen(pixels[box[1] : box[3], box[0] : box[2]], bits)
    scale = tuple(b / p for b, p in zip(base, taken, strict=True))
    ends = _end_points(selected, scale, density, percent)
    receipt = _receipt(domain, pixels, bits, "negative-exposure-parameters")
    receipt.update(
        base_exposures=base,
        prescan_exposures=taken,
        base_density=density,
        film_kind=kind,
        development_chemistry="not inferred from image type",
        curve_sha256=curve_sha256(kind),
        analysis_roi_half_open=box,
        percentile_f32=percent,
        required_scanner_metadata=required_missing_inputs(),
    )
    if ends is None:
        return MeteringResult(None, {**receipt, "reason": "no-channel-endpoints"})
    densities, trace = _off_curve(ends[0], ends[1], scale, density, kind)
    values = tuple(_rust_u32(base[c] * 10 ** densities[c]) for c in range(3))
    receipt.update(
        all_endpoints=ends[0],
        neutral_endpoints=ends[1],
        neutral_counts=ends[2],
        target_exposures_unlimited=values,
        **trace,
    )
    return MeteringResult(values, receipt)


def prescan_step(
    prescan_codes,
    *,
    bits,
    base_exposures,
    prescan_exposures,
    base_density,
    film_kind,
    input_domain,
    pass_number,
    roi=None,
    percentile=1.0,
    lock_white_balance=False,
    exposure_limits=(1, MAX_EXPOSURE),
):
    """Three-visible-channel negative step; no IR/device crop/hardware actions."""
    pixels = _codes(prescan_codes, bits)
    base = _exposures(base_exposures, "start-up exposure")
    current = _exposures(prescan_exposures, "prescan exposure")
    _density(base_density)
    kind, domain = _kind(film_kind), _domain(input_domain)
    percent = _percentile(percentile)
    box = _roi(pixels, roi)
    if type(pass_number) is not int or not 1 <= pass_number <= 3:
        raise ValueError("本接口仅支持负片三次预扫 pass_number=1..3")
    if type(lock_white_balance) is not bool:
        raise ValueError("lock_white_balance 需 bool")
    if (
        not isinstance(exposure_limits, tuple)
        or len(exposure_limits) != 2
        or any(type(v) is not int for v in exposure_limits)
        or not 1 <= exposure_limits[0] <= exposure_limits[1] <= MAX_EXPOSURE
    ):
        raise ValueError("曝光限制需明确正整数范围，不超过 Nikon 最大曝光")
    codes = _widen(pixels[box[1] : box[3], box[0] : box[2]], bits).reshape(-1, 3)
    ends = [
        _hist_ends(np.bincount(codes[:, c], minlength=65536), percent) for c in range(3)
    ]
    highs = [e[1] if e is not None else 0 for e in ends]
    next_values = list(current)
    again = False
    if pass_number < 3:
        for c, level in enumerate(highs):
            if level <= 0x7FFF and current[c] != MAX_EXPOSURE:
                next_values[c] = _rust_u32(
                    min(current[c] * (65535.0 / max(level, 1) * 0.9), MAX_EXPOSURE)
                )
                again = True
            elif level == 65535:
                factor = CLIPPED_FACTORS[kind][c][
                    min(pass_number, len(CLIPPED_FACTORS[kind][c]) - 1)
                ]
                next_values[c] = _rust_u32(min(base[c] * factor, MAX_EXPOSURE))
                again = True
    if again:
        receipt = _receipt(domain, pixels, bits, "negative-prescan-step")
    else:
        result = negative_exposures(
            pixels,
            bits=bits,
            base_exposures=base,
            prescan_exposures=current,
            base_density=base_density,
            film_kind=kind,
            input_domain=domain,
            roi=box,
            percentile=percent,
        )
        receipt = result.receipt
        if result.values is not None:
            next_values = list(result.values)
            if lock_white_balance:
                k = min(next_values[c] / base[c] for c in range(3))
                next_values = [_rust_u32(min(b * k, MAX_EXPOSURE)) for b in base]
    values = tuple(
        max(exposure_limits[0], min(v, exposure_limits[1])) for v in next_values
    )
    return MeteringResult(
        values,
        {
            **receipt,
            "step": "again" if again else "done",
            "pass_number": pass_number,
            "measured_high_codes": highs,
            "lock_white_balance": lock_white_balance,
            "exposure_limits": exposure_limits,
            "infrared_metering_included": False,
            "adapter_crop_inferred": False,
        },
    )


def curve_sha256(film_kind):
    t, d = CHARACTERISTIC_CURVES[_kind(film_kind)]
    return hashlib.sha256(
        struct.pack("<128d", *t, *[v for c in d for v in c])
    ).hexdigest()


# Filled from the pinned Rust literals; these are original characteristic data,
# not named C-41 film-stock profiles. Tuples prevent accidental table mutation.
CHARACTERISTIC_CURVES = {
    "color-negative": (
        (
            -3.5,
            -3.4142,
            -3.2142,
            -3.0,
            -2.9435,
            -2.9173,
            -2.7692,
            -2.6892,
            -2.6142,
            -2.4142,
            -2.2142,
            -2.0,
            -1.7,
            -1.4,
            -1.1,
            -0.8,
            -0.5,
            -0.2,
            0.1,
            0.4,
            0.7,
            1.0,
            1.3,
            1.6,
            1.9,
            2.2,
            2.5,
            2.8,
            3.1,
            3.4,
            3.7,
            4.0,
        ),
        (
            (
                0.316,
                0.316,
                0.316,
                0.316,
                0.316,
                0.316,
                0.3893,
                0.4289,
                0.466,
                0.565,
                0.664,
                0.77,
                0.9185,
                1.067,
                1.2155,
                1.364,
                1.5125,
                1.661,
                1.8095,
                1.958,
                2.1065,
                2.255,
                2.4035,
                2.552,
                2.7005,
                2.849,
                2.9975,
                3.146,
                3.2945,
                3.443,
                3.5915,
                3.74,
            ),
            (
                0.737,
                0.737,
                0.737,
                0.737,
                0.737,
                0.7525,
                0.8398,
                0.887,
                0.9313,
                1.0493,
                1.1673,
                1.2936,
                1.4706,
                1.6476,
                1.8246,
                2.0016,
                2.1786,
                2.3556,
                2.5326,
                2.7096,
                2.8866,
                3.0636,
                3.2406,
                3.4176,
                3.5946,
                3.7716,
                3.9486,
                4.1256,
                4.3026,
                4.4796,
                4.6566,
                4.8336,
            ),
            (
                0.886,
                0.886,
                0.886,
                0.886,
                0.9227,
                0.9398,
                1.036,
                1.088,
                1.1368,
                1.2668,
                1.3968,
                1.536,
                1.731,
                1.926,
                2.121,
                2.316,
                2.511,
                2.706,
                2.901,
                3.096,
                3.291,
                3.486,
                3.681,
                3.876,
                4.071,
                4.266,
                4.461,
                4.656,
                4.851,
                5.046,
                5.241,
                5.436,
            ),
        ),
    ),
    "monochrome-negative": (
        (
            -1.5,
            -1.4,
            -1.3,
            -1.2,
            -1.1,
            -1.0,
            -0.9,
            -0.8,
            -0.7,
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
        ),
        (
            (
                0.295,
                0.37,
                0.42,
                0.46,
                0.505,
                0.56,
                0.61,
                0.65,
                0.705,
                0.75,
                0.81,
                0.86,
                0.91,
                0.96,
                1.01,
                1.07,
                1.13,
                1.19,
                1.25,
                1.31,
                1.38,
                1.46,
                1.54,
                1.62,
                1.71,
                1.81,
                1.9,
                2.03,
                2.16,
                2.32,
                2.52,
                2.78,
            ),
            (
                0.313,
                0.388,
                0.438,
                0.478,
                0.523,
                0.578,
                0.628,
                0.668,
                0.723,
                0.768,
                0.828,
                0.878,
                0.928,
                0.978,
                1.028,
                1.088,
                1.148,
                1.208,
                1.268,
                1.328,
                1.398,
                1.478,
                1.558,
                1.638,
                1.728,
                1.828,
                1.918,
                2.048,
                2.178,
                2.338,
                2.538,
                2.798,
            ),
            (
                0.247,
                0.34,
                0.39,
                0.43,
                0.475,
                0.53,
                0.58,
                0.62,
                0.675,
                0.72,
                0.78,
                0.83,
                0.88,
                0.93,
                0.98,
                1.04,
                1.1,
                1.16,
                1.22,
                1.28,
                1.35,
                1.43,
                1.51,
                1.59,
                1.68,
                1.78,
                1.87,
                2.0,
                2.13,
                2.29,
                2.49,
                2.75,
            ),
        ),
    ),
}
