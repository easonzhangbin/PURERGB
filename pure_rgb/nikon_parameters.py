"""Research Nikon per-frame analyzer parameters, separate from acquisition.

The shared metering arithmetic is adapted from activexray/nkscan (MIT option),
commit f971cfd8efe89f6e02507c80b0945206fdaf0bcd, negative.rs; see the full
MIT notice in nikon_metering.py. The additional builder parameter/control
mapping is independently checked against isolated LS5000 4.0.3 numerical
instructions, source SHA-256 45afd6fb61a9517ff95d1896dc4257779c319310c2e8bbe75f3b4f3dada920af.

An analyzer image is not a final scan, and its exposure ticks are not the
thumbnail/density or final-scan ticks. This module does not claim or create an
acquisition binding, camera calibration, factory capture or runnable scanner.
"""

from dataclasses import dataclass
import hashlib

import numpy as np

from . import nikon_metering as meter
from .nikon_lut import BuilderParameters

SCANNER_DOMAIN = meter.SCANNER_DOMAIN
CAMERA_HYPOTHESIS = meter.CAMERA_HYPOTHESIS
ANALYZER_VA = 0x10011840
SOURCE_SHA256 = "45afd6fb61a9517ff95d1896dc4257779c319310c2e8bbe75f3b4f3dada920af"


@dataclass(frozen=True)
class ParameterResult:
    channels: tuple[BuilderParameters, ...]
    control_x: tuple[tuple[float, ...], ...]
    control_y: tuple[tuple[float, ...], ...]
    receipt: dict

    def require_bound_acquisition(self):
        """Numeric success alone cannot authorize a strict scanner render."""
        raise ValueError(
            "Numeric parameters are unbound: final-scan exposure, raster, "
            "thumbnail ownership and acquisition oracle remain required"
        )


def _histogram_ends(hist, population):
    # Original Tanten uses 65536 uint16 histogram cells, not uint32 counts.
    counts = np.asarray(hist, dtype=np.uint64) & 0xFFFF
    tail = int(population * 0.0042)
    cumulative = counts.cumsum()
    if not counts.any() or cumulative[-1] < tail:
        raise ValueError("Original uint16 histogram cannot supply endpoints")
    lo = int(np.searchsorted(cumulative, tail, side="left"))
    hi = int(np.flatnonzero(counts)[-1])
    return float(lo), float(hi)


def _endpoints(codes, scale, density):
    pixels = codes.reshape(-1, 3)
    n = len(pixels)
    scaled = pixels.astype(float) * scale
    masks = (
        10 ** (density[1] - density[0]),
        10 ** (density[2] - density[0]),
        10 ** (density[2] - density[1]),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = (
            scaled[:, 0] / (masks[0] * scaled[:, 1]),
            scaled[:, 0] / (masks[1] * scaled[:, 2]),
            scaled[:, 1] / (masks[2] * scaled[:, 2]),
        )
    rg, rb, gb = [~((q <= 1 / 3) | (q >= 3)) for q in ratios]
    all_ends, neutral_ends, neutral_counts = [], [], []
    for c, near in enumerate((rg | rb, rg | gb, rb | gb)):
        rejected = ~near
        before = rejected.cumsum() - rejected
        accepted = near | (before / n >= 0.9)
        count = int(accepted.sum())
        all_ends.append(_histogram_ends(np.bincount(pixels[:, c], minlength=65536), n))
        neutral_ends.append(
            _histogram_ends(np.bincount(pixels[accepted, c], minlength=65536), count)
        )
        neutral_counts.append(count)
    return all_ends, neutral_ends, neutral_counts


def derive_parameters(
    prescan_codes,
    *,
    bits,
    base_exposures,
    prescan_exposures,
    base_density,
    film_kind,
    input_domain,
    roi=None,
    auto_film_balance,
):
    """Explicit scanner analyzer inputs -> 3×7 scalars and 3×32 X/Y controls.

    ROI is half-open (sensor x0, feed y0, sensor x1, feed y1), analysis only.
    ``auto_film_balance`` explicitly selects descriptor +0x6C=0/1. No factory
    setting is guessed. Input codes are widened explicitly from declared bits.
    The result remains unbound to a final scanner acquisition.
    """
    pixels = meter._codes(prescan_codes, bits)
    base = meter._exposures(base_exposures, "start-up exposure")
    taken = meter._exposures(prescan_exposures, "analyzer exposure")
    density = meter._density(base_density)
    kind, domain = meter._kind(film_kind), meter._domain(input_domain)
    if type(auto_film_balance) is not bool:
        raise ValueError("auto_film_balance must explicitly be True or False")
    box = meter._roi(pixels, roi)
    selected = meter._widen(pixels[box[1] : box[3], box[0] : box[2]], bits)
    scale = tuple(b / t for b, t in zip(base, taken, strict=True))
    all_ends, neutral, counts = _endpoints(selected, scale, density)
    master, templates = meter.CHARACTERISTIC_CURVES[kind]
    all_scaled = [
        (max(lo * scale[c], 1.0), max(hi * scale[c], 1.0))
        for c, (lo, hi) in enumerate(all_ends)
    ]
    ns = [meter._sample_density(neutral[c][0] * scale[c]) for c in range(3)]
    nh = [meter._sample_density(neutral[c][1] * scale[c]) for c in range(3)]
    # A small neutral subset can have tail=int(n*.0042)==0, producing a zero
    # low endpoint and +infinite density. The original high-side curve holds
    # its last Y, so this does not itself make the derivation invalid.
    effective_base = [min(density[c], nh[c]) for c in range(3)]
    x = [
        [v + (effective_base[c] - templates[c][0]) for v in templates[c]]
        for c in range(3)
    ]
    fog = [v + 0.15 for v in effective_base]

    def initial(c, value):
        return meter._interpolate(x[c], master, value)

    se = [initial(c, ns[c]) for c in range(3)]
    he = [initial(c, nh[c]) for c in range(3)]
    fe = [initial(c, fog[c]) for c in range(3)]
    ranges = [(se[c] - fe[c], he[c] - fe[c]) for c in range(3)]
    (lr, hr), (lg, hg), (lb, hb) = ranges
    green_span = lg - hg
    nr, nb = 10 ** -abs(hr - hg), 10 ** -abs(hb - hg)
    sr = 3 ** -abs(lr - hr - green_span)
    sb = 3 ** -abs(green_span - (lb - hb))
    dense = min(10 ** -(0.8 - max(lr, lg, lb)), 1.0)
    wr, wb = (1 - nr + sr * nr) * dense, (1 - nb + sb * nb) * dense
    shift_r = (se[0] - se[1]) * wr + (he[0] - he[1]) * (1 - wr) * nr
    shift_b = (se[1] - se[2]) * wb + (he[1] - he[2]) * (1 - wb) * nb
    if not auto_film_balance:
        shift_r = shift_b = 0.0
    y = [[v - shift_r for v in master], list(master), [v + shift_b for v in master]]

    def at(c, value):
        return meter._interpolate(x[c], y[c], value)

    sd = [meter._sample_density(v[0]) for v in all_scaled]
    hd = [meter._sample_density(v[1]) for v in all_scaled]
    shifted_fog = [at(c, fog[c]) for c in range(3)]
    shadow_over = max(at(c, sd[c]) - shifted_fog[c] for c in range(3))
    channel_highlight_over = [at(c, hd[c]) - shifted_fog[c] for c in range(3)]
    highlight_over = min(channel_highlight_over)
    shadow = max(at(c, min(sd[c], x[c][31])) for c in range(3))
    channel_highlights = [at(c, max(hd[c], x[c][0])) for c in range(3)]
    highlight = min(channel_highlights)
    # The builder's three 'a' values use ALL-pixel highlights, after the
    # control shift. They are not the neutral highlights used for film balance.
    channel_weights = [
        min(10 ** (-(h + 0.23) * 4), 1.0) for h in channel_highlight_over
    ]
    weight = min(max(channel_weights), 1.0)
    initial_shadow = shadow
    if shadow - highlight < 0.6:
        highlight -= (1 - weight) * (0.6 - (shadow - highlight))
        shadow = max(highlight + 0.6, initial_shadow + 0.05)
    else:
        shadow += 0.05
    span = shadow - highlight
    reach = (
        max(min(0.4 ** (highlight_over + 0.23) * 1.9, 1.9), span) if span < 1.9 else 1.9
    )
    if shadow_over < 1.2:
        shadow += (1.2 - shadow_over) * (span / reach)
    f = [meter._interpolate(y[c], x[c], shadow) for c in range(3)]
    g = [meter._interpolate(y[c], x[c], highlight) for c in range(3)]
    parameters = tuple(
        BuilderParameters(
            shadow,
            highlight,
            reach,
            channel_highlights[c],
            channel_weights[c],
            f[c],
            g[c],
        )
        for c in range(3)
    )
    return ParameterResult(
        parameters,
        tuple(map(tuple, x)),
        tuple(map(tuple, y)),
        {
            "schema": "pure-rgb-nikon-parameters",
            "version": 1,
            "source_module_sha256": SOURCE_SHA256,
            "original_analyzer_va": hex(ANALYZER_VA),
            "public_reference_commit": meter.REFERENCE_COMMIT,
            "input_domain": domain,
            "valid_bits": bits,
            "source_pixels_sha256": hashlib.sha256(
                pixels.astype("<u2").tobytes()
            ).hexdigest(),
            "analysis_roi_half_open": list(box),
            "analysis_only_no_output_crop": True,
            "base_exposures": list(base),
            "analyzer_exposures": list(taken),
            "thumbnail_density_exposures_inferred": False,
            "final_scan_exposures_inferred": False,
            "base_density": list(density),
            "effective_base_density": effective_base,
            "film_kind": kind,
            "curve_sha256": meter.curve_sha256(kind),
            "auto_film_balance": auto_film_balance,
            "all_endpoints": all_ends,
            "neutral_endpoints": neutral,
            "neutral_counts": counts,
            "red_log_exposure_shift": shift_r,
            "blue_log_exposure_shift": shift_b,
            "shadow_log_exposure_A": shadow,
            "highlight_log_exposure_B": highlight,
            "density_span_C": reach,
            "highlight_blend_a_rgb": channel_weights,
            "shared_expansion_highlight_weight": weight,
            "parameter_order": ["A", "B", "C", "E", "a", "f", "g"],
            "status": "numeric-parameters-only-unbound",
            "acquisition_binding_proven": False,
            "factory_capture_oracle": False,
            "camera_to_scanner_calibrated": False,
            "inversion_lut_generated": False,
            "production_defaults_changed": False,
        },
    )
