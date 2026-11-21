"""Presentation-only lift candidates from the user's 191→255 curve screenshot."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw
import tifffile

from .native import sha256
from .preview_data import DEFAULT_STUDY, load_samples, verified_pixels
from .study import preview, read_rgb
from .workspace import write_json

GAIN = 255 / 191
PIVOT = 0.25  # encoded code position, not 18% scene reflectance
SOFT_GAIN = GAIN * (1 - PIVOT) / (1 - GAIN * PIVOT)


def lift_codes(pixels, soft=False):
    x = np.asarray(pixels, dtype=np.float64) / 65535
    y = SOFT_GAIN * x / (1 + (SOFT_GAIN - 1) * x) if soft else np.minimum(GAIN * x, 1)
    return np.rint(y * 65535).astype(np.uint16)


def run(output, study=DEFAULT_STUDY):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    samples = load_samples(study)
    report = {
        "state": "running",
        "scope": "Encoded RGB presentation only; not source density recovery, not validated Nikon 9000 ED reproduction",
        "user_reference": "191 to 255 reference curve",
        "gain": GAIN,
        "soft_gain": SOFT_GAIN,
        "soft_pivot": PIVOT,
        "cases": [],
    }
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    pages = []
    for start in range(0, len(samples), 6):
        subset = samples[start : start + 6]
        sheet = Image.new("RGB", (1560, len(subset) * 520), "#202020")
        draw = ImageDraw.Draw(sheet)
        for row, sample in enumerate(subset):
            source = sample.record("cool-shared-rgb", "baseline")
            pixels, icc = verified_pixels(source)
            case = {
                "id": sample.id,
                "source": source,
                "results": {},
                "upstream_clipping_not_recovered": True,
            }
            with tifffile.TiffFile(source["path"]) as tif:
                pg = tif.pages[0]
                dpi = {
                    "resolution": (pg.tags[282].value, pg.tags[283].value),
                    "resolutionunit": int(pg.tags[296].value),
                }
            for col, (name, px) in enumerate(
                [
                    ("baseline", pixels),
                    ("whitepoint-191", lift_codes(pixels)),
                    ("soft-shoulder", lift_codes(pixels, True)),
                ]
            ):
                if col:
                    path = output / (sample.id + "-" + name + ".tiff")
                    tifffile.imwrite(
                        path,
                        px,
                        photometric="rgb",
                        compression="deflate",
                        metadata=None,
                        extratags=[
                            (34675, 7, len(icc), icc, False),
                            (274, 3, 1, 1, False),
                        ],
                        **dpi,
                    )
                    actual, ai = read_rgb(path)
                    if not np.array_equal(actual, px) or ai != icc:
                        raise ValueError("Curve TIFF RGB16/ICC mismatch")
                    record = {"path": str(path), "sha256": sha256(path)}
                else:
                    record = source
                case["results"][name] = {
                    **record,
                    "icc_sha256": hashlib.sha256(icc).hexdigest(),
                    "white_sample_fraction": float(np.mean(px == 65535)),
                    "new_white_sample_fraction": float(
                        np.mean((px == 65535) & (pixels < 65535))
                    ),
                }
                sheet.paste(preview(px, icc, (520, 480)), (col * 520, row * 520 + 35))
                draw.text(
                    (col * 520 + 8, row * 520 + 10),
                    sample.id + " | " + name,
                    fill="white",
                )
            report["cases"].append(case)
            if sha256(source["path"]) != source["sha256"]:
                raise ValueError("Baseline output changed")
        path = output / f"comparison-page-{start // 6 + 1:02d}.png"
        sheet.save(path, icc_profile=srgb)
        pages.append(str(path))
    report.update(
        state="complete",
        case_count=len(samples),
        output_count=2 * len(samples),
        pages=pages,
    )
    write_json(output / "verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {k: result[k] for k in ("state", "case_count", "output_count")},
            ensure_ascii=False,
        )
    )
