"""Verify real-frame controls and illustrate explicit photometric contracts."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw

from .native import sha256
from .study import preview, read_rgb


def verify_and_present(previous, scene, base, gamma1, output):
    roots = [Path(p) for p in (previous, scene, base, gamma1)]
    manifests = [json.loads((p / "manifest.json").read_text()) for p in roots]
    old, current, anchored, gamma = manifests
    if any(
        m["state"] != "complete"
        or m["inputs"] != old["inputs"]
        or m["analysis"] != old["analysis"]
        for m in manifests
    ):
        raise ValueError("Controls must have identical RAW inputs and scene endpoints")
    if (
        current["film_base_reference"] != anchored["film_base_reference"]
        or current["film_base_reference"] != gamma["film_base_reference"]
    ):
        raise ValueError(
            "Base/gamma controls require the same frozen film-base candidate"
        )
    for r in old["inputs"]:
        if sha256(r["path"]) != r["sha256"]:
            raise ValueError("RAW hash mismatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "inputs": old["inputs"],
        "analysis": current["analysis"],
        "base_reference": current["film_base_reference"],
        "density_contract": current["density_contract"],
        "controls": {},
        "sample_statistics": {},
        "statistics_domain": "encoded uint16 code values, not color accuracy",
    }
    x0, y0, x1, y1 = current["analysis"]["roi"]

    def load(which, name):
        record = manifests[which]["results"][name]
        path = record.get("picture_path", record["path"])
        expected = record.get("picture_sha256", record["sha256"])
        if sha256(path) != expected:
            raise ValueError("Result TIFF hash mismatch")
        pixels, icc = read_rgb(path)
        if not record.get("picture_path"):
            pixels = pixels[y0:y1, x0:x1]
        if pixels.dtype != np.uint16 or pixels.shape != (y1 - y0, x1 - x0, 3):
            raise ValueError("Incorrect pixel geometry or bit depth")
        return pixels, icc

    reference, common_icc = load(0, "native")
    report["controls"]["native_bit_exact"] = all(
        np.array_equal(reference, load(i, "native")[0]) for i in (1, 2, 3)
    )
    report["controls"]["scene_density_reference_cancels"] = np.array_equal(
        load(0, "cool-colors-auto")[0], load(1, "cool-colors-auto")[0]
    )
    report["controls"]["scene_flex_unchanged_by_unused_reference"] = np.array_equal(
        load(0, "flexcolor-research")[0], load(1, "flexcolor-research")[0]
    )
    report["controls"]["changing_flex_gamma_does_not_change_cool"] = np.array_equal(
        load(2, "cool-colors-auto")[0], load(3, "cool-colors-auto")[0]
    )
    code2, _ = load(2, "flexcolor-research")
    code1, _ = load(3, "flexcolor-research")
    residual = np.abs(code1.astype(np.float64) - code2.astype(np.float64) ** 2 / 65535)
    report["controls"]["flex_gamma1_vs_gamma2_squared_max_lsb"] = float(
        np.max(residual)
    )
    if (
        not all(
            v
            for k, v in report["controls"].items()
            if k.endswith(("exact", "cancels", "reference", "cool"))
        )
        or float(np.max(residual)) > 2
    ):
        raise ValueError("A photometric control failed")
    del residual, code1, code2, reference
    panels = [
        (1, "native", "Native unchanged"),
        (1, "cool-colors-auto", "cool: scene-relative density"),
        (2, "cool-colors-auto", "cool: rebate-anchored density"),
        (1, "flexcolor-research", "flex: scene-range, Gamma=2 codes"),
        (2, "flexcolor-research", "flex: signal/base, Gamma=2 codes"),
        (3, "flexcolor-research", "flex: signal/base, Gamma=1 codes"),
    ]
    canvas = Image.new("RGB", (1560, 940), "#202020")
    draw = ImageDraw.Draw(canvas)
    for index, (which, name, label) in enumerate(panels):
        pixels, icc = load(which, name)
        if icc != common_icc:
            raise ValueError("ICC differs between controls")
        image = preview(pixels, icc, (520, 430))
        col, row = index % 3, index // 3
        canvas.paste(image, (col * 520, row * 470 + 36))
        draw.text((col * 520 + 8, row * 470 + 10), label, fill="white")
        report["sample_statistics"][label] = {
            "mean_rgb_code": np.mean(pixels, axis=(0, 1)).tolist(),
            "black_clip_fraction": float(np.mean(pixels == 0)),
            "white_clip_fraction": float(np.mean(pixels == 65535)),
        }
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    canvas.save(output / "contracts-vs-native.png", icc_profile=srgb)
    full, icc = read_rgb(current["results"]["native"]["path"])
    overlay = preview(full, icc, (1200, 1000))
    sx, sy = overlay.width / full.shape[1], overlay.height / full.shape[0]
    draw = ImageDraw.Draw(overlay)
    for box, color in (
        (current["analysis"]["roi"], "#00ff60"),
        (current["film_base_reference"]["roi"], "#ff5050"),
    ):
        a, b, c, d = box
        draw.rectangle(
            (round(a * sx), round(b * sy), round(c * sx), round(d * sy)),
            outline=color,
            width=3,
        )
    draw.rectangle((0, 0, 360, 45), fill="#202020")
    draw.text((8, 6), "green: scene analysis / red: rebate candidate", fill="white")
    draw.text((8, 25), "candidate is not an independently measured base", fill="white")
    overlay.save(output / "base-and-picture-regions.png", icc_profile=srgb)
    (output / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return report["controls"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("previous", "scene", "base", "gamma1", "output"):
        parser.add_argument(name)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_and_present(
                args.previous, args.scene, args.base, args.gamma1, args.output
            )
        )
    )
