"""Verify primary-cohort range probes and separate coverage from appearance."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw

from .native import sha256
from .range_probe import PRIMARY
from .sample_study import code_difference
from .study import preview, read_rgb
from .tone_study import GAMMA


def color_change(before, after):
    from negpy.kernel.image.logic import rgb_to_lab_working

    a = rgb_to_lab_working((before[::8, ::8].astype(float) / 65535) ** GAMMA)
    b = rgb_to_lab_working((after[::8, ::8].astype(float) / 65535) ** GAMMA)
    ca, cb = np.hypot(a[..., 1], a[..., 2]), np.hypot(b[..., 1], b[..., 2])
    valid = (
        (ca >= 5)
        & (cb >= 5)
        & (a[..., 0] >= 10)
        & (a[..., 0] <= 90)
        & (b[..., 0] >= 10)
        & (b[..., 0] <= 90)
    )
    result = {
        "sample_stride": 8,
        "valid_pixel_fraction": float(valid.mean()),
        "domain": "Declared Adobe RGB/D65 Lab diagnostics, not calibrated accuracy; excludes near-neutral and extreme lightness",
    }
    if not np.any(valid):
        return {
            **result,
            "median_abs_hue_degrees": None,
            "p95_abs_hue_degrees": None,
            "median_chroma_ratio": None,
        }
    delta = np.arctan2(b[..., 2], b[..., 1]) - np.arctan2(a[..., 2], a[..., 1])
    degrees = np.abs(np.rad2deg(np.arctan2(np.sin(delta), np.cos(delta))))[valid]
    return {
        **result,
        "median_abs_hue_degrees": float(np.median(degrees)),
        "p95_abs_hue_degrees": float(np.quantile(degrees, 0.95)),
        "median_chroma_ratio": float(np.median(cb[valid] / ca[valid])),
    }


def summarize(folder):
    folder = Path(folder).resolve()
    index = json.loads((folder / "index.json").read_text())
    if index["state"] != "complete":
        raise ValueError("Complete all primary probes before final verification")
    os.environ["NEGPY_USER_DIR"] = str(folder / "summary-state")
    report = {
        "state": "running",
        "primary_root": str(PRIMARY),
        "cases": [],
        "coverage_monotonic": True,
        "baseline_regressions_exact": True,
        "raw_paths": [],
        "output_count": 0,
    }
    common_icc = None
    raw_paths = set()
    modes = ("native", "cool-shared-rgb", "flex-endpoint")
    accepted = []

    def checked(record):
        nonlocal common_icc
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("TIFF hash mismatch")
        px, icc = read_rgb(record["path"])
        if px.dtype != np.uint16 or px.ndim != 3 or px.shape[2] != 3:
            raise ValueError("RGB16 contract violated")
        if common_icc is None:
            common_icc = icc
        if icc != common_icc:
            raise ValueError("ICC mismatch")
        return px, icc

    for c in index["cases"]:
        m = json.loads((Path(c["directory"]) / "manifest.json").read_text())
        old = json.loads((Path(m["baseline_case"]) / "manifest.json").read_text())
        if m["state"] != "complete":
            raise ValueError("Case incomplete")
        for rec in m["inputs"]:
            if (
                Path(rec["path"]).resolve().parent != PRIMARY
                or sha256(rec["path"]) != rec["sha256"]
            ):
                raise ValueError("Wrong cohort or changed RAW")
            raw_paths.add(rec["path"])
        sample = {
            "id": c["id"],
            "directory": c["directory"],
            "coverage_policy": m["coverage_policy"],
            "range_diagnostics": m["range_diagnostics"],
            "native_automatic_metrics": m["native_automatic_metrics"],
            "results": {},
        }
        for probe in ("highlight-only", "shadow-only", "both"):
            for end in ("positive_highlight", "positive_shadow"):
                current = np.array(
                    m["range_diagnostics"][probe][end]["sample_fraction_by_channel"]
                )
                baseline = np.array(
                    m["range_diagnostics"]["baseline"][end][
                        "sample_fraction_by_channel"
                    ]
                )
                if np.any(current > baseline + 1e-12):
                    raise ValueError(
                        "Expanded window increased excluded source fraction"
                    )
        for mode in modes:
            baseline, icc = checked(m["results"]["baseline__" + mode])
            previous, oi = checked(old["results"][mode])
            if icc != oi or not np.array_equal(baseline, previous):
                raise ValueError("Zero-change historical regression failed")
            report["output_count"] += 1
            for probe in ("baseline", "highlight-only", "shadow-only", "both"):
                rec = m["results"][probe + "__" + mode]
                px = baseline if probe == "baseline" else checked(rec)[0]
                if px.shape != baseline.shape:
                    raise ValueError("ROI size drift")
                if probe != "baseline":
                    report["output_count"] += 1
                white_mask = baseline == 65535
                black_mask = baseline == 0
                detail = {
                    **rec,
                    "color_change": color_change(baseline, px),
                    "old_white_samples_moved_below_white_fraction": float(
                        np.mean(px[white_mask] < 65535)
                    )
                    if np.any(white_mask)
                    else None,
                    "old_black_samples_moved_above_black_fraction": float(
                        np.mean(px[black_mask] > 0)
                    )
                    if np.any(black_mask)
                    else None,
                }
                if probe != "baseline":
                    detail["difference"] = code_difference(px, baseline)
                    w = m["windows"][probe]
                    base = m["windows"]["baseline"]
                    if (
                        w["log_dense"] == base["log_dense"]
                        and w["log_thin"] == base["log_thin"]
                        and not np.array_equal(px, baseline)
                    ):
                        raise ValueError("Unchanged range caused a render drift")
                sample["results"][probe + "__" + mode] = detail
        report["cases"].append(sample)
        accepted.append((c, m))
        print(c["id"], "verified", flush=True)
    expected = {str(p.resolve()) for p in PRIMARY.glob("*.RW2")}
    if raw_paths != expected:
        raise ValueError("Original sample folder not completely covered")
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    for mode in modes:
        for page in range((len(accepted) + 5) // 6):
            subset = accepted[page * 6 : (page + 1) * 6]
            canvas = Image.new("RGB", (1040, 520 * len(subset)), "#202020")
            draw = ImageDraw.Draw(canvas)
            for row, (c, m) in enumerate(subset):
                for col, probe in enumerate(("baseline", "both")):
                    px, icc = checked(m["results"][probe + "__" + mode])
                    canvas.paste(
                        preview(px, icc, (520, 480)), (col * 520, row * 520 + 35)
                    )
                    draw.text(
                        (col * 520 + 7, row * 520 + 10),
                        c["id"] + " | " + mode + " | " + probe,
                        fill="white",
                    )
            canvas.save(folder / f"{mode}-page-{page + 1:02d}.png", icc_profile=srgb)
    report.update(
        state="complete",
        raw_paths=sorted(raw_paths),
        raw_count=len(raw_paths),
        case_count=len(accepted),
    )
    (folder / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return str(folder)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    print(summarize(p.parse_args().folder))
