"""Reproducible three-mode RAW test with unchanged candidate algorithms."""

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

from .engines import check_reference_sources
from .cli import inspect_tiff
from .density_matrix import fixed_process, negpy_window, render_variant, study_region
from .endpoint_study import apply_endpoint_curve, fit_endpoint_curve
from .native import NEGPY_SHA, native_export, open_triplet, sha256
from .rgb_curve_study import apply_shared_curve, fit_shared_curve
from .study import preview, read_rgb
from .tone_study import GAMMA, ICC_SHA256, brightness_statistics


def run_case(raw, destination, roi):
    out = Path(destination).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["NEGPY_USER_DIR"] = str(out / "negpy-state")
    report = {
        "state": "running",
        "negpy_commit": NEGPY_SHA,
        "scope": "Native common-ROI reference, cool shared RGB candidate, Flex endpoint candidate; not factory engines",
        "results": {},
    }
    try:
        check_reference_sources()
        processor, config, source, records = open_triplet(raw)
        region = study_region(source, roi)
        roi = region["analysis_roi"]
        x0, y0, x1, y1 = roi
        window = negpy_window(source, config, roi)
        frozen = replace(
            config, process=fixed_process(config.process, window, source.shape)
        )
        native_payload, export = native_export(processor, frozen, records)
        pixels, icc = inspect_tiff(native_payload)
        if hashlib.sha256(icc).hexdigest() != ICC_SHA256:
            raise ValueError("Unexpected working ICC")
        reference = pixels[y0:y1, x0:x1].copy()
        with tifffile.TiffFile(io.BytesIO(native_payload)) as tif:
            pg = tif.pages[0]
            dpi = {
                "resolution": (pg.tags[282].value, pg.tags[283].value),
                "resolutionunit": int(pg.tags[296].value),
            }
        del pixels, native_payload
        linear_reference = (reference.astype(float) / 65535) ** GAMMA
        report.update(
            inputs=records,
            picture_region=region,
            density_window=asdict(window),
            native_config=frozen.to_dict(),
            source_shape=list(source.shape),
            reference_brightness=brightness_statistics(linear_reference),
        )
        canvas = Image.new("RGB", (1560, 520), "#202020")
        draw = ImageDraw.Draw(canvas)

        def save(mode, px, col, meta):
            path = out / (mode + ".tiff")
            tifffile.imwrite(
                path,
                px,
                photometric="rgb",
                compression="deflate",
                metadata=None,
                extratags=[(34675, 7, len(icc), icc, False), (274, 3, 1, 1, False)],
                **dpi,
            )
            actual, actual_icc = read_rgb(path)
            if actual_icc != icc or not np.array_equal(actual, px):
                raise ValueError("RGB16/ICC roundtrip mismatch")
            report["results"][mode] = {
                "path": str(path),
                "sha256": sha256(path),
                "icc_sha256": hashlib.sha256(icc).hexdigest(),
                "shape": list(px.shape),
                "brightness": brightness_statistics(
                    (px.astype(float) / 65535) ** GAMMA
                ),
                **meta,
            }
            thumb = preview(px, icc, (520, 480))
            canvas.paste(thumb, (col * 520, 35))
            draw.text((col * 520 + 8, 10), mode, fill="white")

        save(
            "native",
            reference,
            0,
            {"kind": "Original Native renderer, frozen common NegPy density window"},
        )
        for renderer, mode, col in (
            ("cool-density-linear", "cool-shared-rgb", 1),
            ("flex-research-codes", "flex-endpoint", 2),
        ):
            payload, meta = render_variant(
                renderer, processor, config, records, source, window, export
            )
            px, other_icc = inspect_tiff(payload)
            if other_icc != icc:
                raise ValueError("Intermediate ICC mismatch")
            px = px[y0:y1, x0:x1].copy()
            linear = (px.astype(float) / 65535) ** GAMMA
            del payload
            if mode == "cool-shared-rgb":
                fit = fit_shared_curve(linear, linear_reference)
                transformed, diagnostics = apply_shared_curve(linear, fit)
            else:
                fit = fit_endpoint_curve(linear, linear_reference)
                transformed = apply_endpoint_curve(linear, fit)
                diagnostics = {
                    "curve": "sigmoid(slope*logit(x)+offset), same R/G/B; exact f(0)=0 and f(1)=1"
                }
            encoded = np.rint(np.minimum(transformed, 1) ** (1 / GAMMA) * 65535).astype(
                np.uint16
            )
            diagnostics.update(
                fit=asdict(fit),
                source_white_sample_fraction=float(np.mean(px == 65535)),
                source_black_sample_fraction=float(np.mean(px == 0)),
                additional_white_sample_fraction=float(
                    np.mean((encoded == 65535) & (px < 65535))
                ),
                additional_black_sample_fraction=float(
                    np.mean((encoded == 0) & (px > 0))
                ),
                source_white_endpoint_outputs=np.unique(encoded[px == 65535]).tolist(),
                source_black_endpoint_outputs=np.unique(encoded[px == 0]).tolist(),
                intermediate_metadata=meta,
            )
            save(mode, encoded, col, diagnostics)
            del px, linear, transformed, encoded
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", nargs=3, required=True)
    parser.add_argument("--roi", nargs=4, type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(run_case(args.raw, args.output, args.roi))
