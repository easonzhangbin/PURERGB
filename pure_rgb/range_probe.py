"""Offline signal-range probes; no UI controls or default renderer changes."""

import argparse
from dataclasses import asdict, replace
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw
import tifffile

from .cli import inspect_tiff
from .density_matrix import DensityWindow, fixed_process, negpy_window, render_variant
from .endpoint_study import EndpointFit, apply_endpoint_curve
from .engines import Analysis, check_reference_sources
from .native import native_export, open_triplet, sha256
from .rgb_curve_study import SharedCurveFit, apply_shared_curve
from .study import preview, read_rgb
from .tone_study import GAMMA, ICC_SHA256, brightness_statistics

SIGNAL_FLOOR = 1e-6
TAIL_FRACTION = 0.0001
PRIMARY = (Path.home() / "Pictures/makina/rgb测试/rgb2").resolve()


def signed_coordinate(image, window):
    """Unclipped positive-density coordinate; not absolute optical density."""
    x = np.asarray(image)
    if (
        x.ndim != 3
        or x.shape[2] != 3
        or not np.issubdtype(x.dtype, np.floating)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Expected finite floating RGB source")
    span = np.subtract(window.log_thin, window.log_dense)
    return (np.asarray(window.log_thin) - np.log10(np.maximum(x, SIGNAL_FLOOR))) / span


def coverage_windows(image, baseline, tail_fraction=TAIL_FRACTION):
    if not 0 < tail_fraction < 0.01:
        raise ValueError("Explicit small diagnostic tail fraction required")
    u = signed_coordinate(image, baseline)
    tails = []
    for ch in range(3):
        valid = (image[..., ch] > SIGNAL_FLOOR) & (image[..., ch] < 1)
        if np.count_nonzero(valid) < 64:
            raise ValueError("Insufficient finite in-range source samples")
        tails.append(
            np.quantile(u[..., ch][valid], [tail_fraction, 1 - tail_fraction]).tolist()
        )
    near = min(0.0, min(t[0] for t in tails))
    far = max(1.0, max(t[1] for t in tails))
    # Keep the selected quantile just inside, including tied sensor code values.
    if near < 0:
        near -= 1e-6
    if far > 1:
        far += 1e-6
    span = np.subtract(baseline.log_thin, baseline.log_dense)
    windows = {"baseline": baseline}
    for name, lower, upper in (
        ("highlight-only", 0.0, far),
        ("shadow-only", near, 1.0),
        ("both", near, far),
    ):
        if lower == 0 and upper == 1:
            windows[name] = baseline
            continue
        dense = tuple(np.asarray(baseline.log_thin) - upper * span)
        thin = tuple(np.asarray(baseline.log_thin) - lower * span)
        analysis = Analysis(
            tuple(10 ** np.array(dense)),
            tuple(10 ** np.array(thin)),
            baseline.analysis.roi,
            None,
        )
        windows[name] = DensityWindow(
            "coverage-probe",
            analysis,
            dense,
            thin,
            {
                "policy": "Common normalized extension across channels; per-channel quantile coverage, not calibrated recoverable bounds",
                "tail_fraction": tail_fraction,
                "original_coordinate_near": lower,
                "original_coordinate_far": upper,
            },
        )
    return windows, {
        "tail_fraction": tail_fraction,
        "per_channel_coordinate_quantiles": tails,
        "common_near": near,
        "common_far": far,
        "excluded_from_bound_estimate": "signal <=1e-6 or >=1; variation may include noise/grain, not certified scene detail",
    }


def signal_diagnostics(image, window):
    u = signed_coordinate(image, window)
    out = {}
    for label, mask in (("positive_highlight", u >= 1), ("positive_shadow", u <= 0)):
        counts = []
        unique = []
        ranges = []
        for ch in range(3):
            samples = image[..., ch][mask[..., ch]]
            counts.append(float(len(samples) / image.shape[0] / image.shape[1]))
            unique.append(int(len(np.unique(samples))))
            ranges.append(
                [float(samples.min()), float(samples.max())] if len(samples) else None
            )
        out[label] = {
            "sample_fraction_by_channel": counts,
            "sample_fraction": float(np.mean(counts)),
            "distinct_source_values_by_channel": unique,
            "source_value_ranges": ranges,
        }
    out["floor_or_below_fraction"] = float(np.mean(image <= SIGNAL_FLOOR))
    out["ceiling_or_above_fraction"] = float(np.mean(image >= 1))
    out["coordinate_extrema"] = [float(u.min()), float(u.max())]
    return out


def normalization_metrics(source, process):
    from negpy.domain.interfaces import PipelineContext
    from negpy.features.exposure.processor import NormalizationProcessor
    from negpy.kernel.system.config import render_scale_factor

    ctx = PipelineContext(
        original_size=source.shape[:2],
        scale_factor=render_scale_factor(source.shape),
        process_mode=process.process_mode,
        wants_uv_grid=False,
        cache_stages=False,
    )
    NormalizationProcessor(process).process(source, ctx)
    names = (
        "norm_density_range",
        "metered_anchor",
        "textural_range",
        "highlight_point",
        "shadow_point",
        "neutral_axis_refs",
    )

    def plain(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (tuple, list)):
            return [plain(x) for x in v]
        if isinstance(v, np.generic):
            return v.item()
        return v

    return {k: plain(ctx.metrics.get(k)) for k in names}


def run_case(baseline_path, destination):
    baseline_path = Path(baseline_path).resolve()
    old = json.loads((baseline_path / "manifest.json").read_text())
    if old["state"] != "complete" or any(
        Path(r["path"]).resolve().parent != PRIMARY for r in old["inputs"]
    ):
        raise ValueError("This experiment is restricted to the original rgb2 cohort")
    out = Path(destination).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["NEGPY_USER_DIR"] = str(out / "negpy-state")
    report = {
        "state": "running",
        "baseline_case": str(baseline_path),
        "scope": "Offline range coverage probes on original rgb2, no UI slider, no renderer default changes",
        "fit_policy": "Original cool/Flex post-curves frozen; no new fitting to Native. Native input-dependent automatic quantities recompute and are logged.",
        "results": {},
    }
    try:
        check_reference_sources()
        processor, config, source, records = open_triplet(
            [r["path"] for r in old["inputs"]]
        )
        if records != old["inputs"]:
            raise ValueError("Primary source records changed")
        roi = tuple(old["picture_region"]["analysis_roi"])
        x0, y0, x1, y1 = roi
        base = negpy_window(source, config, roi)
        if (
            list(base.log_dense) != old["density_window"]["log_dense"]
            or list(base.log_thin) != old["density_window"]["log_thin"]
        ):
            raise ValueError("Baseline density window drift")
        crop = source[y0:y1, x0:x1]
        windows, policy = coverage_windows(crop, base)
        report.update(
            inputs=records,
            roi=roi,
            coverage_policy=policy,
            windows={k: asdict(v) for k, v in windows.items()},
            range_diagnostics={
                k: signal_diagnostics(crop, v) for k, v in windows.items()
            },
        )
        cool_fit = SharedCurveFit(**old["results"]["cool-shared-rgb"]["fit"])
        flex_fit = EndpointFit(**old["results"]["flex-endpoint"]["fit"])
        canvas = Image.new("RGB", (1560, 4 * 520), "#202020")
        draw = ImageDraw.Draw(canvas)
        common_icc = None
        baseline_pixels = {}
        baseline_post = old["results"]
        for row, (name, window) in enumerate(windows.items()):
            process = fixed_process(config.process, window, source.shape)
            report.setdefault("native_automatic_metrics", {})[name] = (
                normalization_metrics(source, process)
            )
            frozen = replace(config, process=process)
            native_payload, export = native_export(processor, frozen, records)
            native_full, icc = inspect_tiff(native_payload)
            if hashlib.sha256(icc).hexdigest() != ICC_SHA256:
                raise ValueError("Unexpected ICC")
            if common_icc is None:
                common_icc = icc
            if common_icc != icc:
                raise ValueError("ICC changed between probes")
            with tifffile.TiffFile(io.BytesIO(native_payload)) as tif:
                pg = tif.pages[0]
                dpi = {
                    "resolution": (pg.tags[282].value, pg.tags[283].value),
                    "resolutionunit": int(pg.tags[296].value),
                }
            native = native_full[y0:y1, x0:x1].copy()
            del native_full, native_payload
            samples = {"native": (native, {})}
            for renderer, mode in (
                ("cool-density-linear", "cool-shared-rgb"),
                ("flex-research-codes", "flex-endpoint"),
            ):
                payload, _ = render_variant(
                    renderer, processor, config, records, source, window, export
                )
                px, other = inspect_tiff(payload)
                if other != icc:
                    raise ValueError("Intermediate ICC changed")
                px = px[y0:y1, x0:x1].copy()
                linear = (px.astype(float) / 65535) ** GAMMA
                if mode == "cool-shared-rgb":
                    result, meta = apply_shared_curve(linear, cool_fit)
                else:
                    result = apply_endpoint_curve(linear, flex_fit)
                    meta = {}
                encoded = np.rint(np.minimum(result, 1) ** (1 / GAMMA) * 65535).astype(
                    np.uint16
                )
                meta.update(
                    intermediate_white_sample_fraction=float(np.mean(px == 65535)),
                    intermediate_black_sample_fraction=float(np.mean(px == 0)),
                    curve_fit_frozen=asdict(
                        cool_fit if mode == "cool-shared-rgb" else flex_fit
                    ),
                )
                samples[mode] = (encoded, meta)
                del px, linear, result, payload
            for col, (mode, (px, meta)) in enumerate(samples.items()):
                previous = baseline_post[mode]
                if name == "baseline":
                    if sha256(previous["path"]) != previous["sha256"]:
                        raise ValueError("Historical output changed")
                    original, oi = read_rgb(previous["path"])
                    if oi != icc or not np.array_equal(px, original):
                        raise ValueError("Zero-change path not pixel/ICC exact")
                    baseline_pixels[mode] = px.copy()
                path = out / (name + "__" + mode + ".tiff")
                tifffile.imwrite(
                    path,
                    px,
                    photometric="rgb",
                    compression="deflate",
                    metadata=None,
                    extratags=[(34675, 7, len(icc), icc, False), (274, 3, 1, 1, False)],
                    **dpi,
                )
                check, ci = read_rgb(path)
                if ci != icc or not np.array_equal(check, px):
                    raise ValueError("Range-probe roundtrip failed")
                delta = (
                    np.mean(np.abs(px.astype(np.int32) - baseline_pixels[mode]))
                    / 65535
                    * 100
                )
                report["results"][name + "__" + mode] = {
                    "path": str(path),
                    "sha256": sha256(path),
                    "mode": mode,
                    "probe": name,
                    "baseline_pixel_icc_exact": True if name == "baseline" else None,
                    "brightness": brightness_statistics(
                        (px.astype(float) / 65535) ** GAMMA
                    ),
                    "black_sample_fraction": float(np.mean(px == 0)),
                    "white_sample_fraction": float(np.mean(px == 65535)),
                    "mean_absolute_encoded_difference_percent": float(delta),
                    **meta,
                }
                thumb = preview(px, icc, (520, 480))
                canvas.paste(thumb, (col * 520, row * 520 + 35))
                draw.text(
                    (col * 520 + 5, row * 520 + 10), name + " | " + mode, fill="white"
                )
            del samples, native
            print(name, "complete", flush=True)
        canvas.save(
            out / "comparison.png",
            icc_profile=ImageCms.ImageCmsProfile(
                ImageCms.createProfile("sRGB")
            ).tobytes(),
        )
        if any(sha256(r["path"]) != r["sha256"] for r in records):
            raise ValueError("RAW hash changed")
        report.update(state="complete", raw_hashes_unchanged=True)
    except BaseException as error:
        report.update(state="failed", error=str(error))
        raise
    finally:
        (out / "manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    return str(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("baseline")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(run_case(a.baseline, a.output))
