"""Summarize completed, unchanged 2x4 matrices across multiple captures."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw

from .engines import ROOT
from .density_matrix import RENDERER_CHOICES
from .native import NEGPY_SHA, sha256
from .study import preview, read_rgb


def code_difference(a, b):
    """Full-resolution encoded-sample difference; never a color accuracy score."""
    if a.shape != b.shape or a.dtype != np.uint16 or b.dtype != np.uint16:
        raise ValueError("Expected matching RGB16 samples")
    absolute = np.zeros(3, dtype=np.float64)
    signed = np.zeros(3, dtype=np.float64)
    maximum = 0
    for y in range(0, a.shape[0], 128):
        delta = a[y : y + 128].astype(np.int32) - b[y : y + 128].astype(np.int32)
        absolute += np.abs(delta).sum(axis=(0, 1), dtype=np.float64)
        signed += delta.sum(axis=(0, 1), dtype=np.float64)
        maximum = max(maximum, int(np.abs(delta).max()))
    count = a.shape[0] * a.shape[1]
    return {
        "mean_absolute_percent_full_scale": float(
            absolute.mean() / count / 65535 * 100
        ),
        "mean_absolute_rgb_percent_full_scale": (
            absolute / count / 65535 * 100
        ).tolist(),
        "mean_signed_rgb_percent_full_scale": (signed / count / 65535 * 100).tolist(),
        "maximum_absolute_code_delta": maximum,
    }


def summarize(batch):
    batch = Path(batch).resolve()
    index = json.loads((batch / "index.json").read_text())
    if index["state"] != "complete" or any(
        not f.get("resolved_by") for f in index["failures"]
    ):
        raise ValueError(
            "Require a completed batch; do not silently drop failed samples"
        )
    summary = {
        "state": "running",
        "domain": "Full-resolution ROI Adobe RGB 16-bit encoded sample differences; not DeltaE, physical density error or accuracy",
        "sampling": "All sequential triplets in the supplied rgb2 directory; original NegPy channel/scene classification",
        "independent_film_stock_count": None,
        "factory_resources_loaded": False,
        "automatic_detection_failures": index["failures"],
        "samples": [],
    }
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    sheet = Image.new("RGB", (2000, 300 * len(index["matrices"])), "#202020")
    overlays = Image.new(
        "RGB", (1200, 300 * ((len(index["matrices"]) + 2) // 3)), "#202020"
    )
    draw, overlay_draw = ImageDraw.Draw(sheet), ImageDraw.Draw(overlays)
    seen = set()
    common_icc = None
    ordered = sorted(
        index["matrices"],
        key=lambda name: json.loads(
            (ROOT / "outputs" / name / "manifest.json").read_text()
        )["inputs"][0]["path"],
    )
    input_paths = set()
    for row, name in enumerate(ordered):
        folder = ROOT / "outputs" / name
        m = json.loads((folder / "manifest.json").read_text())
        if m["state"] != "complete" or m["negpy_commit"] != NEGPY_SHA:
            raise ValueError("Incomplete matrix or mismatched baseline")
        signatures = tuple(r["sha256"] for r in m["inputs"])
        if signatures in seen:
            raise ValueError("Repeated capture in batch")
        seen.add(signatures)
        for record in m["inputs"]:
            input_paths.add(str(Path(record["path"]).resolve()))
            if sha256(record["path"]) != record["sha256"]:
                raise ValueError("RAW hash changed")
        control_path = Path(m["native_control"]["path"])
        if sha256(control_path) != m["native_control"]["sha256"]:
            raise ValueError("Native control file hash mismatch")
        native, icc = read_rgb(control_path)
        if common_icc is None:
            common_icc = icc
        if icc != common_icc:
            raise ValueError("Batch ICC differs")
        x0, y0, x1, y1 = m["picture_detection"]["analysis_roi"]
        native_crop = native[y0:y1, x0:x1]
        label = f"{row + 1:02d}: " + Path(m["inputs"][0]["path"]).stem
        shot = {
            "label": label,
            "matrix": str(folder),
            "inputs": m["inputs"],
            "roi": [x0, y0, x1, y1],
            "picture_detection": m["picture_detection"],
            "density_spans": {
                key: (np.array(v["log_thin"]) - v["log_dense"]).tolist()
                for key, v in m["density_windows"].items()
            },
            "renderers": {},
        }
        overlay = preview(native, icc, (400, 266))
        ow, oh = overlay.size
        ImageDraw.Draw(overlay).rectangle(
            (
                x0 * ow / native.shape[1],
                y0 * oh / native.shape[0],
                x1 * ow / native.shape[1],
                y1 * oh / native.shape[0],
            ),
            outline="#00ff60",
            width=2,
        )
        ox, oy = (row % 3) * 400, (row // 3) * 300
        overlays.paste(overlay, (ox, oy + 30))
        overlay_draw.text((ox + 5, oy + 7), label, fill="white")
        sheet.paste(preview(native_crop, icc, (400, 266)), (0, row * 300 + 30))
        draw.text((5, row * 300 + 7), label + " | Native control", fill="white")
        for col, renderer in enumerate(RENDERER_CHOICES):
            crops = []
            entry = {}
            for provider in ("percentile-0.5-99.5", "negpy-0.68"):
                record = m["results"][provider + "__" + renderer]
                path = Path(record["path"])
                if sha256(path) != record["sha256"]:
                    raise ValueError("Matrix TIFF hash mismatch")
                full, output_icc = read_rgb(path)
                if output_icc != common_icc or full.shape != native.shape:
                    raise ValueError("TIFF output contract mismatch")
                crop = full[y0:y1, x0:x1].copy()
                crops.append(crop)
                entry[provider] = {
                    "versus_native_control": code_difference(crop, native_crop),
                    "black_clip_fraction": record["black_clip_fraction"],
                    "white_clip_fraction": record["white_clip_fraction"],
                }
            entry["density_change"] = code_difference(crops[1], crops[0])
            shot["renderers"][renderer] = entry
            sheet.paste(
                preview(crops[1], icc, (400, 266)), ((col + 1) * 400, row * 300 + 30)
            )
            draw.text(
                ((col + 1) * 400 + 5, row * 300 + 7),
                "NegPy density | " + renderer,
                fill="white",
            )
        summary["samples"].append(shot)
        print(label, "verified", flush=True)
    expected = {str(p.resolve()) for p in Path(index["source_directory"]).glob("*.RW2")}
    if input_paths != expected:
        raise ValueError("Batch does not cover all supplied RW2 files exactly")
    sheet.save(batch / "all-samples.png", icc_profile=srgb)
    overlays.save(batch / "all-regions.png", icc_profile=srgb)
    summary.update(
        state="complete",
        source_files_verified=len(input_paths),
        icc_sha256=hashlib.sha256(common_icc).hexdigest(),
    )
    (batch / "measurements.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    return str(batch)


def compare_regions(initial, interior):
    """Change only metering regions; always score both outputs on the interior ROI."""
    initial, interior = Path(initial).resolve(), Path(interior).resolve()
    indices = [json.loads((b / "index.json").read_text()) for b in (initial, interior)]
    if any(i["state"] != "complete" for i in indices):
        raise ValueError("Both batches must be complete")
    maps = []
    for index in indices:
        captures = {}
        for folder in index["matrices"]:
            m = json.loads((ROOT / "outputs" / folder / "manifest.json").read_text())
            captures[tuple(r["sha256"] for r in m["inputs"])] = m
        maps.append(captures)
    if maps[0].keys() != maps[1].keys():
        raise ValueError("Region study requires identical capture sets")
    results = {
        "domain": "Encoded-sample differences, not color accuracy; both members scored on the same conservative interior ROI",
        "initial_batch": str(initial),
        "interior_batch": str(interior),
        "samples": [],
    }

    def checked(record):
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("Region-study file hash mismatch")
        return read_rgb(record["path"])

    for key, new in sorted(
        maps[1].items(), key=lambda pair: pair[1]["inputs"][0]["path"]
    ):
        old = maps[0][key]
        x0, y0, x1, y1 = new["picture_detection"]["analysis_roi"]
        a, ia = checked(old["native_control"])
        b, ib = checked(new["native_control"])
        if ia != ib or not np.array_equal(a, b):
            raise ValueError(
                "Unchanged Native control drifted between region experiments"
            )
        sample = {
            "raw": new["inputs"][0]["path"],
            "evaluation_roi": [x0, y0, x1, y1],
            "native_control_identical": True,
            "region_change": {},
        }
        for renderer in ("cool-density-linear", "flex-research-codes"):
            for provider in ("percentile-0.5-99.5", "negpy-0.68"):
                variant = provider + "__" + renderer
                a, ia = checked(old["results"][variant])
                b, ib = checked(new["results"][variant])
                if ia != ib or a.shape != b.shape:
                    raise ValueError("Region-study contracts differ")
                sample["region_change"][variant] = code_difference(
                    b[y0:y1, x0:x1], a[y0:y1, x0:x1]
                )
        results["samples"].append(sample)
        print(Path(sample["raw"]).name, "region comparison verified", flush=True)
    results["state"] = "complete"
    (interior / "region-comparison.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    )
    return str(interior / "region-comparison.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch")
    parser.add_argument("--compare-initial")
    args = parser.parse_args()
    print(summarize(args.batch))
    if args.compare_initial:
        print(compare_regions(args.compare_initial, args.batch))
