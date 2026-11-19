"""Verify all expanded three-mode outputs and compare historical controls."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw

from .native import sha256
from .sample_study import code_difference
from .study import preview, read_rgb


def summarize(batch):
    batch = Path(batch).resolve()
    index = json.loads((batch / "index.json").read_text())
    if index["state"] == "running":
        raise ValueError("Wait for the complete batch, including failures")
    root = Path(__file__).resolve().parents[1]
    old_tone = json.loads(
        (root / "outputs/tone-study-d266ad6b6cd8/manifest.json").read_text()
    )
    old_rgb = json.loads(
        (root / "outputs/rgb-curve-study-3b35cffc65cd/manifest.json").read_text()
    )
    old_flex = json.loads(
        (root / "outputs/endpoint-study-9d6bb2e937c9/manifest.json").read_text()
    )
    report = {
        "state": "running",
        "domain": "Encoded RGB and declared linear-Y diagnostics; not color accuracy. Per-image tone fitting uses Native common-ROI reference.",
        "cases": [],
        "failures": [],
        "regressions": [],
    }
    rotations = {
        f"new-{i:02d}": (90 if i in (6, 7, 9) else 180 if i >= 12 else 0)
        for i in range(1, 23)
    }
    report["preview_orientation"] = {
        "method": "Manual scene review; rotation applies to contact-sheet previews only, all three modes equally. TIFF pixels/ROI/capture orientation unchanged.",
        "counterclockwise_degrees": rotations,
    }
    icc_common = None
    new_raw = set()
    all_raw = set()
    modes = ("native", "cool-shared-rgb", "flex-endpoint")
    for cohort in ("1111-new", "rgb2-regression"):
        cases = [c for c in index["cases"] if c["cohort"] == cohort]
        for page in range((len(cases) + 5) // 6):
            subset = cases[page * 6 : (page + 1) * 6]
            canvas = Image.new("RGB", (1560, len(subset) * 520), "#202020")
            draw = ImageDraw.Draw(canvas)
            for row, c in enumerate(subset):
                if c["state"] != "complete":
                    draw.text(
                        (8, row * 520 + 15),
                        c["id"] + " FAILED: see index.json",
                        fill="white",
                    )
                    report["failures"].append(c)
                    continue
                m = json.loads((Path(c["directory"]) / "manifest.json").read_text())
                if m["state"] != "complete":
                    raise ValueError("Case status disagreement")
                for rec in m["inputs"]:
                    if sha256(rec["path"]) != rec["sha256"]:
                        raise ValueError("RAW hash changed")
                    all_raw.add(rec["path"])
                    if cohort == "1111-new":
                        new_raw.add(rec["path"])
                result = {
                    "id": c["id"],
                    "cohort": cohort,
                    "inputs": m["inputs"],
                    "roi": m["picture_region"]["analysis_roi"],
                    "modes": m["results"],
                    "preview_rotation_degrees": rotations.get(c["id"], 0),
                }
                draw.rectangle((0, row * 520, 1560, row * 520 + 30), fill="#202020")
                for col, mode in enumerate(modes):
                    draw.text(
                        (col * 520 + 6, row * 520 + 8),
                        c["id"] + " | " + mode,
                        fill="white",
                    )
                for mode in modes:
                    rec = m["results"][mode]
                    if sha256(rec["path"]) != rec["sha256"]:
                        raise ValueError("Output hash changed")
                    px, icc = read_rgb(rec["path"])
                    if px.dtype != np.uint16 or list(px.shape) != rec["shape"]:
                        raise ValueError("Invalid RGB16 output")
                    if icc_common is None:
                        icc_common = icc
                    if icc != icc_common:
                        raise ValueError("ICC drift between modes/cases")
                    thumb = preview(px, icc, (900, 900))
                    angle = rotations.get(c["id"], 0)
                    if angle:
                        thumb = thumb.rotate(angle, expand=True)
                    thumb.thumbnail((520, 480))
                    col = modes.index(mode)
                    canvas.paste(thumb, (col * 520, row * 520 + 35))
                    if cohort == "rgb2-regression":
                        i = c["old_index"]
                        if mode == "native":
                            prev = old_tone["samples"][i]["reference"]
                            previous, pi = read_rgb(prev["path"])
                            x0, y0, x1, y1 = c["roi"]
                            previous = previous[y0:y1, x0:x1]
                        else:
                            prev = (
                                old_rgb["samples"][i]["engines"]["cool-density-linear"]
                                if mode == "cool-shared-rgb"
                                else old_flex["samples"][i]
                            )
                            previous, pi = read_rgb(prev["path"])
                        if sha256(prev["path"]) != prev["sha256"]:
                            raise ValueError("Historical output hash changed")
                        exact = icc == pi and np.array_equal(px, previous)
                        report["regressions"].append(
                            {
                                "id": c["id"],
                                "mode": mode,
                                "pixel_and_icc_exact": exact,
                                "difference": code_difference(px, previous),
                            }
                        )
                report["cases"].append(result)
                print(c["id"], "all outputs verified", flush=True)
            profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
            canvas.save(
                batch / f"{cohort}-page-{page + 1:02d}.png", icc_profile=profile
            )
    expected = {str(p.resolve()) for p in Path(index["new_cohort_root"]).rglob("*.RW2")}
    if not report["failures"] and new_raw != expected:
        raise ValueError("New RAW coverage mismatch")
    report.update(
        state="complete" if not report["failures"] else "complete-with-failures",
        new_raw_verified=len(new_raw),
        all_raw_verified=len(all_raw),
        tiffs_verified=len(report["cases"]) * 3,
        regressions_exact=all(r["pixel_and_icc_exact"] for r in report["regressions"]),
    )
    (batch / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return str(batch)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("batch")
    print(summarize(p.parse_args().batch))
