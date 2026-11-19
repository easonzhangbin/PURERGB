"""Controlled region/transfer study from completed comparison manifests."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw
import tifffile

from .native import sha256


def read_rgb(path):
    with tifffile.TiffFile(path) as tif:
        return tif.asarray(), bytes(tif.pages[0].tags[34675].value)


def preview(pixels, icc, size=(520, 500)):
    step = max(1, max(pixels.shape[:2]) // 1200)
    image = Image.fromarray(np.round(pixels[::step, ::step] / 257).astype(np.uint8))
    image = ImageCms.profileToProfile(
        image,
        ImageCms.ImageCmsProfile(io.BytesIO(icc)),
        ImageCms.createProfile("sRGB"),
        outputMode="RGB",
    )
    image.thumbnail(size)
    return image


def study(old_path, region_path, corrected_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifests = [
        json.loads((Path(p) / "manifest.json").read_text())
        for p in (old_path, region_path, corrected_path)
    ]
    old, region, corrected = manifests
    if not all(
        m["state"] == "complete" and m["inputs"] == old["inputs"] for m in manifests
    ):
        raise ValueError("Study requires completed comparisons of identical inputs")
    if region["analysis"] != corrected["analysis"]:
        raise ValueError(
            "Region and transfer comparisons must use identical frozen analysis"
        )
    for record in old["inputs"]:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("RAW hash changed")
    x0, y0, x1, y1 = corrected["analysis"]["roi"]
    summary = {
        "inputs": old["inputs"],
        "common_roi": (x0, y0, x1, y1),
        "results": {},
        "endpoint_change_percent": {
            key: (
                100 * (np.array(region["analysis"][key]) / old["analysis"][key] - 1)
            ).tolist()
            for key in ("base", "dark")
        },
        "statistics_domain": "16-bit Adobe RGB encoded sample values, not color accuracy or physical density",
    }
    labels = (
        "A: previous manual ROI + extra Flex OETF",
        "B: detected picture ROI + extra Flex OETF",
        "C: detected picture ROI + direct Flex codes",
    )
    canvas = Image.new("RGB", (1560, 1350), "#202020")
    draw = ImageDraw.Draw(canvas)
    for column, name in enumerate(("native", "cool-colors-auto", "flexcolor-research")):
        frames = []
        for row, manifest in enumerate(manifests):
            record = manifest["results"][name]
            if sha256(record["path"]) != record["sha256"]:
                raise ValueError("Comparison TIFF hash mismatch")
            full, icc = read_rgb(record["path"])
            frame = full[y0:y1, x0:x1]
            if row == 2:
                cropped, cropped_icc = read_rgb(record["picture_path"])
                if not np.array_equal(frame, cropped) or icc != cropped_icc:
                    raise ValueError(
                        "Picture TIFF differs from integer source crop or ICC"
                    )
            thumb = preview(frame, icc, size=(520, 410))
            canvas.paste(thumb, (column * 520, row * 450 + 40))
            draw.text(
                (column * 520 + 7, row * 450 + 5),
                f"{name} | {labels[row]}",
                fill="white",
            )
            frames.append(frame)
        summary["results"][name] = {
            "region_only_mean_absolute_code_delta": float(
                np.mean(np.abs(frames[1].astype(np.int32) - frames[0]))
            ),
            "transfer_only_mean_absolute_code_delta": float(
                np.mean(np.abs(frames[2].astype(np.int32) - frames[1]))
            ),
            "region_only_max_absolute_code_delta": int(
                np.max(np.abs(frames[1].astype(np.int32) - frames[0]))
            ),
            "transfer_only_max_absolute_code_delta": int(
                np.max(np.abs(frames[2].astype(np.int32) - frames[1]))
            ),
            "encoded_mean_rgb_by_row": [
                np.mean(f.astype(np.float64), axis=(0, 1)).tolist() for f in frames
            ],
        }
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    canvas.save(output / "controlled-comparison.png", icc_profile=srgb)
    full, icc = read_rgb(corrected["results"]["native"]["path"])
    overlay = preview(full, icc, size=(1200, 1000))
    draw = ImageDraw.Draw(overlay)
    sx, sy = overlay.width / full.shape[1], overlay.height / full.shape[0]
    boxes = (
        (old["analysis"]["roi"], "#00d5ff", "previous ROI"),
        (
            corrected["picture_detection"]["detected_picture_roi"],
            "#ffb000",
            "detected picture",
        ),
        (corrected["analysis"]["roi"], "#00ff60", "inset analysis/crop"),
    )
    for i, (box, color, label) in enumerate(boxes):
        a, b, c, d = box
        draw.rectangle(
            (round(a * sx), round(b * sy), round(c * sx), round(d * sy)),
            outline=color,
            width=3,
        )
        draw.rectangle((0, i * 24, 210, i * 24 + 23), fill="#202020")
        draw.text((5, i * 24 + 5), label, fill=color)
    overlay.save(output / "region-overlay.png", icc_profile=srgb)
    (output / "measurements.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("old")
    parser.add_argument("region")
    parser.add_argument("corrected")
    parser.add_argument("output")
    args = parser.parse_args()
    summary = study(args.old, args.region, args.corrected, args.output)
    print(json.dumps(summary, ensure_ascii=False))
