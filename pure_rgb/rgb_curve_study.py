"""Shared RGB tone-curve diagnostic after the rejected luminance-only study."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import uuid

import numpy as np
from PIL import Image, ImageCms, ImageDraw
import tifffile

from .engines import ROOT
from .native import sha256
from .study import preview, read_rgb
from .tone_study import (
    GAMMA,
    EPS,
    checked_crop,
    linear_samples,
    luminance,
    brightness_statistics,
)


@dataclass(frozen=True)
class SharedCurveFit:
    power: float
    scale: float
    grid_stride: int
    valid_fraction: float
    span_residual_stops: float


def fit_shared_curve(source, reference, stride=8):
    source, reference = linear_samples(source), linear_samples(reference)
    if source.shape != reference.shape or not isinstance(stride, int) or stride < 1:
        raise ValueError(
            "Matching geometry and a positive integer grid stride required"
        )
    s, r = source[::stride, ::stride], reference[::stride, ::stride]
    sy, ry = luminance(s), luminance(r)
    valid = (sy > EPS) & (ry > EPS) & (s.max(2) < 1 - 1e-8) & (r.max(2) < 1 - 1e-8)
    if np.count_nonzero(valid) < 64:
        raise ValueError("Insufficient common pixels for RGB curve fit")
    rgb = s[valid]
    tq = np.quantile(ry[valid], [0.1, 0.5, 0.9])
    target_span = np.log2(tq[2] / tq[0])

    def measure(power):
        q = np.quantile(
            np.einsum("...c,c->...", rgb**power, [0.2973769, 0.6273491, 0.0752741]),
            [0.1, 0.5, 0.9],
        )
        return float(np.log2(q[2] / q[0])), float(tq[1] / q[1])

    lo, hi = 0.1, 12.0
    if target_span < 0.05 or not measure(lo)[0] < target_span < measure(hi)[0]:
        raise ValueError("Degenerate or unbracketed RGB tone curve")
    for _ in range(28):
        mid = (lo + hi) / 2
        if measure(mid)[0] < target_span:
            lo = mid
        else:
            hi = mid
    power = (lo + hi) / 2
    measured, scale = measure(power)
    residual = abs(measured - target_span)
    if residual > 0.005 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("RGB curve fit did not converge")
    return SharedCurveFit(power, scale, stride, float(valid.mean()), residual)


def apply_shared_curve(source, fit):
    source = linear_samples(source)
    if (
        not np.isfinite([fit.power, fit.scale]).all()
        or not 0.1 <= fit.power <= 12
        or fit.scale <= 0
    ):
        raise ValueError("Invalid shared RGB curve")
    out = fit.scale * source**fit.power
    maximum = out.max(2)
    limited = maximum > 1
    out /= np.maximum(1, maximum)[..., None]
    return out, {
        "gamut_limited_pixel_fraction": float(limited.mean()),
        "formula": "RGB'=scale*RGB^power, same scalar curve for R/G/B; common pixel scale if max exceeds 1",
        "preserves_input_rgb_chromaticity": False,
    }


def run(previous, destination):
    previous = Path(previous).resolve()
    old = json.loads((previous / "manifest.json").read_text())
    if old["state"] != "complete":
        raise ValueError("Requires completed luminance-only study")
    out = Path(destination).resolve() / ("rgb-curve-study-" + uuid.uuid4().hex[:12])
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "state": "running",
        "previous_study": str(previous),
        "purpose": "Diagnostic shared RGB rendering curve; changes color relationships, not factory reproduction",
        "samples": [],
    }
    canvas = Image.new("RGB", (2000, 340 * len(old["samples"])), "#202020")
    protected = {}
    try:
        for row, s in enumerate(old["samples"]):
            roi = s["roi"]
            for rec in s["inputs"]:
                protected[rec["path"]] = rec["sha256"]
            protected[s["reference"]["path"]] = s["reference"]["sha256"]
            ref, icc, _ = checked_crop(s["reference"], roi)
            reference = (ref.astype(float) / 65535) ** GAMMA
            sample = {
                "inputs": s["inputs"],
                "roi": roi,
                "reference_brightness": brightness_statistics(reference),
                "engines": {},
            }
            label = f"{row + 1:02d} " + Path(s["inputs"][0]["path"]).stem

            def panel(col, pixels, title):
                canvas.paste(
                    preview(pixels, icc, (400, 305)), (col * 400, row * 340 + 30)
                )
                ImageDraw.Draw(canvas).text(
                    (col * 400 + 5, row * 340 + 8), label + " | " + title, fill="white"
                )

            panel(0, ref, "Native reference")
            for col, engine in enumerate(
                ("cool-density-linear", "flex-research-codes")
            ):
                rec = s["engines"][engine]
                protected[rec["source"]["path"]] = rec["source"]["sha256"]
                px, _, dpi = checked_crop(rec["source"], roi, icc)
                source = (px.astype(float) / 65535) ** GAMMA
                rejected = rec["outputs"]["tone"]
                if sha256(rejected["path"]) != rejected["sha256"]:
                    raise ValueError("Rejected-control hash mismatch")
                gray, gi = read_rgb(rejected["path"])
                if gi != icc or gray.shape != px.shape:
                    raise ValueError("Rejected-control domain mismatch")
                fit = fit_shared_curve(source, reference)
                result, diagnostics = apply_shared_curve(source, fit)
                encoded = np.rint(np.minimum(result, 1) ** (1 / GAMMA) * 65535).astype(
                    np.uint16
                )
                path = out / f"{row + 1:02d}__{engine}__shared-rgb.tiff"
                tifffile.imwrite(
                    path,
                    encoded,
                    photometric="rgb",
                    compression="deflate",
                    metadata=None,
                    extratags=[(34675, 7, len(icc), icc, False), (274, 3, 1, 1, False)],
                    **dpi,
                )
                check, ci = read_rgb(path)
                if not np.array_equal(check, encoded) or ci != icc:
                    raise ValueError("Shared RGB output failed roundtrip")
                diagnostics.update(
                    path=str(path),
                    sha256=sha256(path),
                    fit=asdict(fit),
                    brightness=brightness_statistics(
                        (encoded.astype(float) / 65535) ** GAMMA
                    ),
                )
                sample["engines"][engine] = diagnostics
                panel(1 + col * 2, gray, engine + " rejected Y-only")
                panel(2 + col * 2, encoded, engine + " shared RGB curve")
                del source, result, encoded, check, gray, px
            report["samples"].append(sample)
            print(label, "complete", flush=True)
        canvas.save(
            out / "comparison.png",
            icc_profile=ImageCms.ImageCmsProfile(
                ImageCms.createProfile("sRGB")
            ).tobytes(),
        )
        if any(sha256(path) != digest for path, digest in protected.items()):
            raise ValueError("A protected input changed")
        report.update(state="complete", protected_input_hashes_unchanged=len(protected))
    except BaseException as error:
        report.update(state="failed", error=str(error))
        raise
    finally:
        (out / "manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("previous")
    parser.add_argument("--output", default=str(ROOT / "outputs"))
    args = parser.parse_args()
    print(run(args.previous, args.output))
