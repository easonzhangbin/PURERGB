"""Exposure/tone matching for research outputs in the declared working space."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
from PIL import Image, ImageCms, ImageDraw
import tifffile

from .engines import ROOT
from .native import sha256
from .study import preview, read_rgb

GAMMA = 563 / 256
LUMA = np.array([0.2973769, 0.6273491, 0.0752741], dtype=np.float64)
ICC_SHA256 = "1e35b53d118eba6835a7bac06137ea87cd5ad6eee97b20a88b29ab6356b00e43"
EPS = 1e-10


def linear_samples(image):
    x = np.asarray(image)
    if (
        x.ndim != 3
        or x.shape[2] != 3
        or not np.issubdtype(x.dtype, np.floating)
        or not np.isfinite(x).all()
        or np.any(x < 0)
        or np.any(x > 1)
    ):
        raise ValueError("Expected finite linear working RGB in [0,1]")
    return x


def luminance(image):
    return np.einsum("...c,c->...", image, LUMA)


@dataclass(frozen=True)
class ToneFit:
    exposure_ev: float
    slope: float
    pivot: float
    valid_fraction: float


def fit_luminance(source, reference):
    source, reference = linear_samples(source), linear_samples(reference)
    if source.shape != reference.shape:
        raise ValueError("Tone comparison requires the same pixel geometry")
    ys, yr = luminance(source), luminance(reference)
    valid = (
        (ys > EPS)
        & (yr > EPS)
        & (np.max(source, axis=2) < 1 - 1e-8)
        & (np.max(reference, axis=2) < 1 - 1e-8)
    )
    if np.count_nonzero(valid) < 64:
        raise ValueError("Insufficient common unclipped pixels to fit brightness")
    s, r = [np.quantile(y[valid], (0.1, 0.5, 0.9)) for y in (ys, yr)]
    ss, rs = np.log2(s[2] / s[0]), np.log2(r[2] / r[0])
    if ss < 0.05 or rs < 0.05:
        raise ValueError("Insufficient luminance range for a stable tone fit")
    ev, slope = float(np.log2(r[1] / s[1])), float(rs / ss)
    if abs(ev) > 16 or not 0.1 <= slope <= 10:
        raise ValueError("Extreme tone fit requires explicit investigation")
    return ToneFit(ev, slope, float(r[1]), float(np.mean(valid)))


def apply_luminance(source, fit, mode):
    source = linear_samples(source)
    if mode not in ("exposure", "tone"):
        raise ValueError("Choose exposure or tone matching")
    if (
        not np.isfinite([fit.exposure_ev, fit.slope, fit.pivot]).all()
        or abs(fit.exposure_ev) > 16
        or not 0.1 <= fit.slope <= 10
        or not 0 < fit.pivot <= 1
    ):
        raise ValueError("Invalid tone fit")
    y = luminance(source)
    target = y * 2**fit.exposure_ev
    if mode == "tone":
        target = fit.pivot * (target / fit.pivot) ** fit.slope
    gain = np.divide(target, y, out=np.ones_like(y), where=y > EPS)
    # A single scale per pixel, including gamut limiting, preserves RGB ratios.
    maximum = np.max(source, axis=2)
    ceiling = np.divide(1, maximum, out=np.full_like(y, np.inf), where=maximum > 0)
    capped = gain > ceiling
    gain = np.minimum(gain, ceiling)
    out = source * gain[..., None]
    totals_before, totals_after = source.sum(axis=2), out.sum(axis=2)
    valid = (totals_before > EPS) & (totals_after > EPS)
    max_error = 0.0
    for channel in range(3):
        before = source[..., channel][valid] / totals_before[valid]
        after = out[..., channel][valid] / totals_after[valid]
        max_error = max(max_error, float(np.max(np.abs(before - after), initial=0)))
    return out, {
        "gamut_limited_pixel_fraction": float(np.mean(capped)),
        "maximum_float_rgb_chromaticity_change": max_error,
        "black_pixels_preserved": bool(np.all(out[maximum == 0] == 0)),
        "mapping": "Y'=Y*2^EV"
        if mode == "exposure"
        else "Y'=pivot*((Y*2^EV)/pivot)^slope",
        "gamut_policy": "Cap the common pixel gain at 1/max(RGB); no independent channel clipping",
    }


def brightness_statistics(linear):
    y = luminance(linear)
    q = np.quantile(y, (0.1, 0.5, 0.9))
    return {
        "linear_y_p10_p50_p90": q.tolist(),
        "p90_p10_range_stops": float(np.log2(max(q[2], EPS) / max(q[0], EPS))),
        "black_pixel_fraction": float(np.mean(np.max(linear, axis=2) == 0)),
        "any_channel_at_ceiling_pixel_fraction": float(
            np.mean(np.max(linear, axis=2) >= 1)
        ),
    }


def quantized_chromaticity_change(before, after):
    valid = (luminance(before) > 1e-4) & (luminance(after) > 1e-4)
    total_before, total_after = before.sum(axis=2), after.sum(axis=2)
    maximum = 0.0
    for channel in range(3):
        delta = (
            before[..., channel][valid] / total_before[valid]
            - after[..., channel][valid] / total_after[valid]
        )
        maximum = max(maximum, float(np.max(np.abs(delta), initial=0)))
    return {
        "maximum_rgb_chromaticity_change_above_y_1e_4": maximum,
        "evaluated_pixel_fraction": float(np.mean(valid)),
        "unit": "absolute normalized linear RGB coordinate difference; not DeltaE",
    }


def checked_crop(record, roi, expected_icc=None):
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise ValueError("Source TIFF hash mismatch")
    pixels, icc = read_rgb(path)
    if hashlib.sha256(icc).hexdigest() != ICC_SHA256:
        raise ValueError("This study only accepts the pinned Adobe RGB working profile")
    if expected_icc is not None and icc != expected_icc:
        raise ValueError("Profiles differ")
    if pixels.dtype != np.uint16 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("Expected RGB16 source")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= pixels.shape[1] and 0 <= y0 < y1 <= pixels.shape[0]):
        raise ValueError("ROI outside source")
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        dpi = {
            "resolution": (page.tags[282].value, page.tags[283].value),
            "resolutionunit": int(page.tags[296].value),
        }
    return pixels[y0:y1, x0:x1].copy(), icc, dpi


def run(batch, destination, density):
    batch = Path(batch).resolve()
    index = json.loads((batch / "index.json").read_text())
    if index["state"] != "complete":
        raise ValueError("Complete source matrices before tone matching")
    out = Path(destination).resolve() / ("tone-study-" + uuid.uuid4().hex[:12])
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "state": "running",
        "source_batch": str(batch),
        "density_provider": density,
        "reference_renderer": "native-renderer",
        "purpose": "Match apparent brightness to compare experimental color relationships; not factory reproduction or color calibration",
        "working_interpretation": "Decode pinned Adobe RGB 563/256 TRC, transform linear working RGB, encode once; experimental primaries/TRC remain a presentation assumption",
        "fit_domain": "Full-resolution common picture ROI; shared positive, non-ceiling pixel set; luminance distribution p10/p50/p90",
        "color_scope": "Preserves linear RGB ratios before quantization; perceived color appearance can change with brightness",
        "samples": [],
    }
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    count = len(index["matrices"])
    sheets = {
        "before-after": Image.new("RGB", (2000, count * 340), "#202020"),
        "exposure-only": Image.new("RGB", (1200, count * 340), "#202020"),
    }
    protected = {}
    try:
        for row, matrix in enumerate(index["matrices"]):
            m = json.loads((ROOT / "outputs" / matrix / "manifest.json").read_text())
            if m["state"] != "complete":
                raise ValueError("Incomplete source matrix")
            roi = m["picture_detection"]["analysis_roi"]
            for record in m["inputs"]:
                protected[record["path"]] = record["sha256"]
                if sha256(record["path"]) != record["sha256"]:
                    raise ValueError("RAW hash changed")
            reference_record = m["results"][density + "__native-renderer"]
            protected[reference_record["path"]] = reference_record["sha256"]
            reference, icc, _ = checked_crop(reference_record, roi)
            linear_reference = (reference.astype(np.float64) / 65535) ** GAMMA
            sample = {
                "matrix": matrix,
                "inputs": m["inputs"],
                "roi": roi,
                "reference": reference_record,
                "reference_brightness": brightness_statistics(linear_reference),
                "engines": {},
            }
            label = f"{row + 1:02d} " + Path(m["inputs"][0]["path"]).stem

            def panel(sheet, column, pixels, text):
                sheets[sheet].paste(
                    preview(pixels, icc, (400, 305)), (column * 400, row * 340 + 30)
                )
                ImageDraw.Draw(sheets[sheet]).text(
                    (column * 400 + 5, row * 340 + 8),
                    label + " | " + text,
                    fill="white",
                )

            for sheet in sheets:
                panel(sheet, 0, reference, "Native common ROI")
            for engine_index, engine in enumerate(
                ("cool-density-linear", "flex-research-codes")
            ):
                original = m["results"][density + "__" + engine]
                protected[original["path"]] = original["sha256"]
                pixels, _, dpi = checked_crop(original, roi, icc)
                linear = (pixels.astype(np.float64) / 65535) ** GAMMA
                fit = fit_luminance(linear, linear_reference)
                result = {
                    "source": original,
                    "fit": asdict(fit),
                    "before": brightness_statistics(linear),
                    "outputs": {},
                }
                panel("before-after", 1 + engine_index * 2, pixels, engine + " before")
                for mode in ("exposure", "tone"):
                    transformed, diagnostics = apply_luminance(linear, fit, mode)
                    encoded = np.rint(
                        np.minimum(transformed, 1) ** (1 / GAMMA) * 65535
                    ).astype(np.uint16)
                    path = out / f"{row + 1:02d}__{engine}__{mode}.tiff"
                    tifffile.imwrite(
                        path,
                        encoded,
                        photometric="rgb",
                        compression="deflate",
                        metadata=None,
                        extratags=[
                            (34675, 7, len(icc), icc, False),
                            (274, 3, 1, 1, False),
                        ],
                        **dpi,
                    )
                    reopened, output_icc = read_rgb(path)
                    if output_icc != icc or not np.array_equal(encoded, reopened):
                        raise ValueError("Tone output failed pixel/ICC roundtrip")
                    decoded_output = (encoded.astype(np.float64) / 65535) ** GAMMA
                    diagnostics.update(
                        path=str(path),
                        sha256=sha256(path),
                        icc_sha256=hashlib.sha256(icc).hexdigest(),
                        brightness=brightness_statistics(decoded_output),
                        quantization=quantized_chromaticity_change(
                            linear, decoded_output
                        ),
                    )
                    if diagnostics["maximum_float_rgb_chromaticity_change"] > 1e-12:
                        raise ValueError(
                            "Tone adjustment changed linear RGB chromaticity"
                        )
                    result["outputs"][mode] = diagnostics
                    if mode == "exposure":
                        panel(
                            "exposure-only",
                            1 + engine_index,
                            encoded,
                            engine + " EV only",
                        )
                    else:
                        panel(
                            "before-after",
                            2 + engine_index * 2,
                            encoded,
                            engine + " EV + tone",
                        )
                    del transformed, encoded, reopened, decoded_output
                sample["engines"][engine] = result
            report["samples"].append(sample)
            print(label, "complete", flush=True)
        for name, sheet in sheets.items():
            sheet.save(out / (name + ".png"), icc_profile=srgb)
        if any(sha256(path) != digest for path, digest in protected.items()):
            raise ValueError("A protected input changed during tone study")
        report["protected_input_hashes_unchanged"] = len(protected)
        report["state"] = "complete"
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
    parser.add_argument("batch")
    parser.add_argument("--output", default=str(ROOT / "outputs"))
    parser.add_argument(
        "--density", choices=("negpy-0.68", "percentile-0.5-99.5"), default="negpy-0.68"
    )
    args = parser.parse_args()
    print(run(args.batch, args.output, args.density))
