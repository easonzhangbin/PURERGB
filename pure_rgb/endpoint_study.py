"""Endpoint-preserving shared RGB curve for the Flex research branch."""

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
    LUMA,
    checked_crop,
    linear_samples,
    luminance,
    brightness_statistics,
)

FIT_REFERENCE_STRIDE = 8


@dataclass(frozen=True)
class EndpointFit:
    slope: float
    offset: float
    grid_stride: int
    median_error_stops: float
    span_error_stops: float


def logit(values):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(values) - np.log1p(-values)


def from_logits(values):
    return np.exp(-np.logaddexp(0, -values))


def apply_endpoint_curve(source, fit):
    source = linear_samples(source)
    if (
        not np.isfinite([fit.slope, fit.offset]).all()
        or not 0.05 <= fit.slope <= 12
        or abs(fit.offset) > 30
    ):
        raise ValueError("Invalid endpoint curve")
    return from_logits(fit.slope * logit(source) + fit.offset)


def fit_endpoint_curve(source, reference, stride=8):
    source, reference = linear_samples(source), linear_samples(reference)
    if source.shape != reference.shape or not isinstance(stride, int) or stride < 1:
        raise ValueError("Matching geometry and positive integer stride required")
    s, r = source[::stride, ::stride], reference[::stride, ::stride]
    ys, yr = luminance(s), luminance(r)
    valid = (ys > EPS) & (yr > EPS) & (s.max(2) < 1 - 1e-8) & (r.max(2) < 1 - 1e-8)
    if valid.sum() < 64:
        raise ValueError("Insufficient common pixels")
    sq, rq = [np.quantile(y[valid], [0.1, 0.5, 0.9]) for y in (ys, yr)]
    span = np.log2(rq[2] / rq[0])
    if span < 0.05 or np.log2(sq[2] / sq[0]) < 0.05:
        raise ValueError("Degenerate endpoint-curve range")
    logits = logit(s[valid])
    slope = float((logit(rq[2]) - logit(rq[0])) / (logit(sq[2]) - logit(sq[0])))
    slope = np.clip(slope, 0.05, 12)
    params = np.array(
        [np.log(slope), np.clip(logit(rq[1]) - slope * logit(sq[1]), -30, 30)]
    )

    def residual(p):
        transformed = from_logits(np.exp(p[0]) * logits + p[1])
        q = np.quantile(np.einsum("...c,c->...", transformed, LUMA), [0.1, 0.5, 0.9])
        return np.array(
            [
                np.log2(max(q[1], EPS) / rq[1]),
                np.log2(max(q[2], EPS) / max(q[0], EPS)) - span,
            ]
        )

    for _ in range(30):
        error = residual(params)
        if np.max(np.abs(error)) < 1e-5:
            break
        jac = np.column_stack(
            [
                (residual(params + np.eye(2)[i] * 0.001) - error) / 0.001
                for i in range(2)
            ]
        )
        delta = np.linalg.lstsq(jac, error, rcond=None)[0]
        improved = False
        for damping in (1, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            trial = params - damping * delta
            trial[0] = np.clip(trial[0], np.log(0.05), np.log(12))
            trial[1] = np.clip(trial[1], -30, 30)
            if np.linalg.norm(residual(trial)) < np.linalg.norm(error):
                params = trial
                improved = True
                break
        if not improved:
            break
    error = residual(params)
    if not np.isfinite(error).all() or np.max(np.abs(error)) > 0.005:
        raise ValueError("Endpoint fit did not converge; no silent fallback")
    return EndpointFit(
        float(np.exp(params[0])),
        float(params[1]),
        stride,
        float(error[0]),
        float(error[1]),
    )


def run(previous, destination):
    previous = Path(previous).resolve()
    old = json.loads((previous / "manifest.json").read_text())
    original = json.loads((Path(old["previous_study"]) / "manifest.json").read_text())
    if old["state"] != "complete" or original["state"] != "complete":
        raise ValueError("Completed source studies required")
    out = Path(destination).resolve() / ("endpoint-study-" + uuid.uuid4().hex[:12])
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "state": "running",
        "previous_study": str(previous),
        "formula": "f(x)=sigmoid(slope*logit(x)+offset); shared R/G/B curve; f(0)=0, f(1)=1",
        "scope": "Flex research presentation only, not factory rendering; source-clipped information cannot be recovered",
        "samples": [],
    }
    canvas = Image.new("RGB", (1200, 340 * len(old["samples"])), "#202020")
    protected = {}
    try:
        for row, (s, o) in enumerate(
            zip(original["samples"], old["samples"], strict=True)
        ):
            if s["inputs"] != o["inputs"] or s["roi"] != o["roi"]:
                raise ValueError("Study capture mismatch")
            for record in s["inputs"]:
                protected[record["path"]] = record["sha256"]
            ref, icc, _ = checked_crop(s["reference"], s["roi"])
            protected[s["reference"]["path"]] = s["reference"]["sha256"]
            rec = s["engines"]["flex-research-codes"]["source"]
            src, _, dpi = checked_crop(rec, s["roi"], icc)
            protected[rec["path"]] = rec["sha256"]
            oldrec = o["engines"]["flex-research-codes"]
            protected[oldrec["path"]] = oldrec["sha256"]
            if sha256(oldrec["path"]) != oldrec["sha256"]:
                raise ValueError("Old curve output changed")
            oldpx, oldicc = read_rgb(oldrec["path"])
            if oldicc != icc or oldpx.shape != src.shape:
                raise ValueError("Old output domain mismatch")
            linear = (src.astype(float) / 65535) ** GAMMA
            reference = (ref.astype(float) / 65535) ** GAMMA
            fit = fit_endpoint_curve(linear, reference)
            transformed = apply_endpoint_curve(linear, fit)
            encoded = np.rint(transformed ** (1 / GAMMA) * 65535).astype(np.uint16)
            path = out / f"{row + 1:02d}__flex__endpoint.tiff"
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
            if ci != icc or not np.array_equal(check, encoded):
                raise ValueError("Endpoint output failed roundtrip")
            sample = {
                "inputs": s["inputs"],
                "roi": s["roi"],
                "fit": asdict(fit),
                "path": str(path),
                "sha256": sha256(path),
                "reference_brightness": brightness_statistics(reference),
                "brightness": brightness_statistics(
                    (encoded.astype(float) / 65535) ** GAMMA
                ),
                "source_white_sample_fraction": float(np.mean(src == 65535)),
                "source_black_sample_fraction": float(np.mean(src == 0)),
                "quantization_additional_white_sample_fraction": float(
                    np.mean((encoded == 65535) & (src < 65535))
                ),
                "quantization_additional_black_sample_fraction": float(
                    np.mean((encoded == 0) & (src > 0))
                ),
                "white_inputs_remain_white": bool(
                    np.all(encoded[src == 65535] == 65535)
                ),
                "black_inputs_remain_black": bool(np.all(encoded[src == 0] == 0)),
                "output_channel_max": encoded.max(axis=(0, 1)).tolist(),
            }
            report["samples"].append(sample)
            label = f"{row + 1:02d} " + Path(s["inputs"][0]["path"]).stem
            for col, (px, title) in enumerate(
                (
                    (ref, "Native reference"),
                    (oldpx, "Flex old: lowered white ceiling"),
                    (encoded, "Flex endpoint curve"),
                )
            ):
                canvas.paste(preview(px, icc, (400, 305)), (col * 400, row * 340 + 30))
                ImageDraw.Draw(canvas).text(
                    (col * 400 + 5, row * 340 + 8), label + " | " + title, fill="white"
                )
            print(label, "complete", flush=True)
            del linear, reference, transformed, encoded, check, src, oldpx, ref
        canvas.save(
            out / "comparison.png",
            icc_profile=ImageCms.ImageCmsProfile(
                ImageCms.createProfile("sRGB")
            ).tobytes(),
        )
        if any(sha256(p) != digest for p, digest in protected.items()):
            raise ValueError("Protected input changed")
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
