"""Floating-point research engines with frozen analysis parameters."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import sys

import numpy as np

from . import BUNDLED_VENDOR, VENDOR_ROOT, WORKSPACE_ROOT

ROOT = WORKSPACE_ROOT


def check_bundled_revision(folder, revision):
    # The marker and reference source are sealed together by the app signature.
    if (VENDOR_ROOT / folder / ".pure-rgb-revision").read_text().strip() != revision:
        raise ValueError(f"Packaged reference revision is invalid: {folder}")

COOL_SHA = "74ffceb79e6e976858d9bffddfeefddba626378f"
FLEX_SHA = "04e745a2fde3559acb2d4f194b7a192150b26f5c"
CAPABILITIES = {
    "native": {
        "available": True,
        "experimental": False,
        "input": "NegPy linear source",
        "commit": "d8123a9b2a18fdaab5941347a91e402fa936d3dc",
    },
    "cool-colors-auto": {
        "available": True,
        "experimental": True,
        "input": "linear transmission",
        "commit": COOL_SHA,
        "factory_equivalent": False,
        "description": "Generic auto density algorithm from the Nikon research repository",
        "output_transfer": "linear density fraction; one explicit working OETF at export",
        "normalization_modes": ["scene", "base-anchored (explicit reference required)"],
    },
    "flexcolor-research": {
        "available": True,
        "experimental": True,
        "input": "linear bands, explicit scene-range / base-relative / raw-linear preparation",
        "commit": FLEX_SHA,
        "factory_equivalent": False,
        "description": "Supported reconstructed stages; synthetic neutral profile by default",
        "output_transfer": "reference post-neg_gamma codes; working TRC assignment is an explicit presentation assumption",
        "verified_scalar_components": {
            "CPointCurve": "optional point-14 research input stage",
            "ContrastBrightnessCurve": "optional flex-14 research input stage",
            "CAggregateCurve": "independent component; full factory graph not connected",
            "CHighShadowCurve": "independent component; full factory graph not connected",
        },
    },
    "nikon-exact": {
        "available": False,
        "reason": "No acquisition-bound builders or camera-to-scanner domain adapter for the supplied RW2; CML4 downstream is independently verified but not integrated into this RAW route",
    },
    "flexcolor-exact": {
        "available": False,
        "reason": "Original gamma/negative/gray tables, complete graph/CMS and scanner input adaptation are not yet verified together",
    },
}


def check_reference_sources():
    import subprocess

    for folder, sha in (("cool-colors", COOL_SHA), ("flexcolor-tool", FLEX_SHA)):
        if BUNDLED_VENDOR:
            check_bundled_revision(folder, sha)
            continue
        path = VENDOR_ROOT / folder
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=path,
            text=True,
        )
        if actual != sha or dirty:
            raise ValueError(
                f"Reference source must match its clean pinned commit: {folder}"
            )


def rgb_input(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Expected a nonempty HWC RGB image")
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError(
            "Linear input requires floating-point samples with explicit scaling"
        )
    if not np.isfinite(image).all() or np.any(image < 0):
        raise ValueError("Input must be finite, nonnegative linear RGB")
    return image


@dataclass(frozen=True)
class Analysis:
    dark: tuple[float, float, float]
    base: tuple[float, float, float]
    roi: tuple[int, int, int, int]
    percentiles: tuple[float, float] | None

    def __post_init__(self):
        dark, base = np.asarray(self.dark), np.asarray(self.base)
        if (
            dark.shape != (3,)
            or base.shape != (3,)
            or not np.isfinite([dark, base]).all()
            or np.any(dark < 0)
            or np.any(base - np.maximum(dark, 1e-6) < 1e-6)
        ):
            raise ValueError("Invalid frozen analysis endpoints")


def analyze(image, roi=None, dark_pct=0.5, base_pct=99.5):
    x = rgb_input(image)
    h, w = x.shape[:2]
    x0, y0, x1, y1 = roi or (0, 0, w, h)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("ROI is outside the image")
    if not 0 <= dark_pct < base_pct <= 100:
        raise ValueError("Invalid percentile range")
    dark, base = np.percentile(x[y0:y1, x0:x1], (dark_pct, base_pct), axis=(0, 1))
    if np.any(base - np.maximum(dark, 1e-6) < 1e-6):
        raise ValueError("Insufficient per-channel dynamic range")
    return Analysis(tuple(dark), tuple(base), (x0, y0, x1, y1), (dark_pct, base_pct))


@dataclass(frozen=True)
class FilmBaseReference:
    values: tuple[float, float, float]
    roi: tuple[int, int, int, int]
    relative_mad: tuple[float, float, float]
    provenance: str = (
        "same-capture manually selected rebate candidate; not independently certified"
    )

    def __post_init__(self):
        values = np.asarray(self.values)
        mad = np.asarray(self.relative_mad)
        if (
            values.shape != (3,)
            or not np.isfinite(values).all()
            or np.any(values <= 1e-6)
        ):
            raise ValueError(
                "Film-base reference requires three finite positive signals"
            )
        if mad.shape != (3,) or not np.isfinite(mad).all() or np.any(mad < 0):
            raise ValueError("Invalid film-base variability metrics")
        object.__setattr__(self, "values", tuple(float(v) for v in values))
        object.__setattr__(self, "relative_mad", tuple(float(v) for v in mad))
        object.__setattr__(self, "roi", tuple(self.roi))


def sample_film_base(image, roi):
    x = rgb_input(image)
    h, w = x.shape[:2]
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h) or (x1 - x0) * (y1 - y0) < 64:
        raise ValueError("Invalid film-base ROI")
    samples = x[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
    median = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - median), axis=0) / np.maximum(median, 1e-6)
    if np.any(mad > 0.1) or np.any(np.mean(samples >= 0.98, axis=0) > 0.001):
        raise ValueError("Film-base ROI is too variable or near sensor saturation")
    return FilmBaseReference(tuple(median), tuple(roi), tuple(mad))


def density_contract(image, analysis, reference=None):
    x0, y0, x1, y1 = analysis.roi
    x = rgb_input(image)[y0:y1, x0:x1]
    b = np.asarray(reference.values if reference else analysis.base, dtype=np.float64)
    near = np.log10(b / analysis.base)
    far = np.log10(b / np.maximum(analysis.dark, 1e-6))
    return {
        "input_domain": "neutral-WB linear selected sensor bands, fixed LibRaw scale; not absolute transmittance",
        "scene_low_transmission": analysis.dark,
        "scene_high_transmission": analysis.base,
        "base_reference": tuple(b),
        "base_provenance": reference.provenance
        if reference
        else "scene-high percentile proxy, not measured film base",
        "relative_density_formula": "D=log10(B/max(signal,1e-6)); signed until rendering",
        "density_at_scene_high": tuple(near),
        "density_at_scene_low": tuple(far),
        "scene_density_span": tuple(far - near),
        "negative_density_fraction": [
            float(np.mean(x[..., c] > b[c])) for c in range(3)
        ],
        "floor_fraction": [float(np.mean(x[..., c] < 1e-6)) for c in range(3)],
        "scene_normalization_formula": "(D-D_near)/(D_far-D_near), equivalent to log10(high/signal)/log10(high/low); B cancels",
        "base_normalization_formula": "D/D_far; explicit film-base reference required",
    }


def density_auto(image, analysis, reference=None, normalization="scene"):
    """Adapt cool-colors invert_auto without its uint16/output-gamma boundary."""
    x = rgb_input(image).astype(np.float64)
    if normalization not in ("scene", "base-anchored"):
        raise ValueError("Unknown density normalization")
    if normalization == "base-anchored" and reference is None:
        raise ValueError(
            "Base-anchored normalization requires an explicit film-base ROI"
        )
    # Scene-relative normalization cancels B analytically; avoid subtracting large logs.
    base = np.array(
        reference.values if normalization == "base-anchored" else analysis.base
    )
    dark = np.maximum(analysis.dark, 1e-6)
    span = np.log10(base / dark)
    if np.any(span <= 0) or not np.isfinite(span).all():
        raise ValueError("Invalid frozen density span")
    return np.clip(np.log10(base / np.maximum(x, 1e-6)) / span, 0, 1).astype(np.float32)


def flexcolor_research(
    image,
    analysis,
    profile=None,
    film_gamma=None,
    reference=None,
    input_policy="scene-range",
    contrast_override=None,
    brightness_override=None,
    endpoints_override=None,
    curve_kernel="legacy-linear",
    sliders_kernel="legacy-linear",
):
    """Apply the supported reconstructed stages; exclude the placeholder sensor LUT."""
    sys.path.insert(0, str(VENDOR_ROOT / "flexcolor-tool"))
    from flexcolor.profiles import gradations
    from flexcolor.render import build_lut, contrast_brightness, neg_gamma
    from .engine_resources import inspect_flex_profile, require_flex_profile

    profile_report = None
    if curve_kernel not in ("legacy-linear", "point-14"):
        raise ValueError("Unknown FlexColor point curve kernel")
    if sliders_kernel not in ("legacy-linear", "flex-14"):
        raise ValueError("Unknown FlexColor slider kernel")
    if profile:
        if isinstance(profile, dict):
            ic, profile_report = require_flex_profile(profile)
        else:
            ic, profile_report = inspect_flex_profile(profile)
            if not profile_report["executable"]:
                raise ValueError(
                    "Unsupported profile stages: "
                    + ", ".join(profile_report["unsupported_active_stages"])
                )
    else:
        ic = {
            "Gamma": 2.0,
            "Gradations": [{"Points": [{"X": 0, "Y": 0}, {"X": 255, "Y": 255}]}] * 3,
        }
    if not isinstance(ic, dict):
        raise ValueError("Invalid ImageCorrection block")
    if ic.get("ApplyCC") or ic.get("ApplyHistogram"):
        raise ValueError("Profile requests unsupported color/histogram stages")
    if float(ic.get("Saturation", 0)) not in (0.0,):
        raise ValueError("Profile saturation semantics are not implemented")
    curves = gradations(ic)
    if len(curves) not in (1, 3, 7):
        raise ValueError("Expected 1, 3, or 7 gradation curves")
    if curve_kernel == "point-14" and len(curves) == 7:
        raise ValueError(
            "7 条 Gradations 的原始路由尚未验证；重建内核只接受明确的 1 条共用或 3 条通道曲线"
        )
    curves = curves[:3] if len(curves) >= 3 else curves * 3
    for points in curves:
        p = np.asarray(points, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] != 3 or not np.isfinite(p).all():
            raise ValueError("Invalid gradation points")
        if (
            np.any(np.diff(p[:, 0]) <= 0)
            or np.any(p[:, :2] < 0)
            or np.any(p[:, :2] > 255)
        ):
            raise ValueError("Gradations require increasing X and 0..255 coordinates")
    gamma = float(film_gamma if film_gamma is not None else ic.get("Gamma", 2.0))
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Film gamma must be finite and positive")
    endpoints = np.asarray(
        endpoints_override
        if endpoints_override is not None
        else ic.get("EndPoints", (0, 255)),
        dtype=float,
    )
    if (
        endpoints.shape != (2,)
        or not np.isfinite(endpoints).all()
        or not 0 <= endpoints[0] < endpoints[1] <= 255
    ):
        raise ValueError("Invalid EndPoints")
    x = rgb_input(image).astype(np.float64)
    if input_policy == "scene-range":
        out = np.clip(
            (x - analysis.dark) / (np.array(analysis.base) - analysis.dark), 0, 1
        )
    elif input_policy == "base-relative":
        if reference is None:
            raise ValueError(
                "Base-relative FlexColor input requires an explicit film-base ROI"
            )
        out = np.clip(x / reference.values, 0, 1)
    elif input_policy == "raw-linear":
        out = np.clip(x, 0, 1)
    else:
        raise ValueError("Unknown FlexColor input policy")
    grid = np.linspace(0, 1, 65536)
    if ic.get("ApplyCurves", True):
        for channel, points in enumerate(curves):
            if curve_kernel == "point-14":
                from .flex_curves import apply_point_curve

                # Explicit research-input adapter; this is not a scanner input
                # calibration or evidence of original seven-curve routing.
                codes = np.rint(out[..., channel] * 16383).astype(np.uint16)
                out[..., channel] = apply_point_curve(codes, points, bits=14) / 16383.0
            else:
                # Preserve the accepted legacy path byte for byte.
                out[..., channel] = np.interp(
                    out[..., channel], grid, build_lut(points, size=65536)
                )
    contrast, brightness = float(ic.get("Contrast", 0)), float(ic.get("Brightness", 0))
    if not ic.get("ApplySliders", True):
        contrast, brightness = 0.0, 0.0
    if contrast_override is not None:
        contrast = float(contrast_override)
    if brightness_override is not None:
        brightness = float(brightness_override)
    if not np.isfinite([contrast, brightness]).all():
        raise ValueError("Invalid contrast/brightness")
    slider_info = None
    if sliders_kernel == "flex-14":
        from .flex_sliders import apply_sliders, slider_lut

        table = slider_lut(contrast, brightness)
        codes = np.rint(out * 16383).astype(np.uint16)
        out = apply_sliders(codes, contrast, brightness) / 16383.0
        slider_info = {
            "table_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
            "input_adapter": "rint(clip(prepared transmission,0,1)*16383)",
            "output_adapter": "original 14-bit code / 16383",
            "arithmetic": "original signed-int16 narrowing before 0..16383 clamp",
            "table_nondecreasing": bool(np.all(np.diff(table.astype(np.int32)) >= 0)),
            "scope": "Verified scalar slider stage; placement in this research adapter is not complete factory routing",
        }
    else:
        out = contrast_brightness(out, contrast, brightness)
    out = neg_gamma(out, gamma, endpoints)
    metadata = {
        "profile": profile_report["resource"]["path"]
        if profile_report
        else "synthetic-neutral (not a factory profile)",
        "profile_sha256": profile_report["resource"]["sha256"]
        if profile_report
        else None,
        "film_gamma": gamma,
        "curve_kernel": curve_kernel,
        "sliders_kernel": sliders_kernel,
        "curve_routing": "Explicit shared/per-channel research adapter; not verified original seven-curve routing",
        "contrast": contrast,
        "brightness": brightness,
        "endpoints": endpoints.tolist(),
        "reconstructed_curve_exponent": 1 / gamma,
        "input_policy": input_policy,
        "density_log_applied": False,
        "input_preparation_is_adapter_choice": True,
        "identity_curve_base_relative_formula": "code=(1-10**(-D))**(1/Gamma); not normalized optical density",
        "upstream_commit": FLEX_SHA,
        "stages": [
            "frozen-percentile-range",
            "gradation-" + curve_kernel,
            "contrast-brightness",
            "neg-gamma",
        ],
        "limitations": [
            "not engine-equivalent",
            "complete seven-curve routing not reconstructed",
            "no sensor LUT or factory CMS",
        ],
    }
    if slider_info is not None:
        metadata["slider_stage"] = slider_info
    if profile_report:
        metadata["profile_inspection"] = profile_report
        if (
            hashlib.sha256(
                Path(profile_report["resource"]["path"]).read_bytes()
            ).hexdigest()
            != metadata["profile_sha256"]
        ):
            raise ValueError("FlexColor 配置处理期间发生变化")
    return np.asarray(out, dtype=np.float32), metadata
