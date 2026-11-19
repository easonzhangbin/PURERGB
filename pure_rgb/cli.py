"""Batch comparison runner. No desktop UI or user database changes."""

import argparse
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np
from PIL import Image, ImageCms, ImageDraw
import tifffile

from .engines import (
    CAPABILITIES,
    COOL_SHA,
    FLEX_SHA,
    ROOT,
    analyze,
    check_reference_sources,
    density_auto,
    density_contract,
    flexcolor_research,
    sample_film_base,
)
from .native import NEGPY_SHA, encode_experiment, native_export, open_triplet, sha256
from .picture import detect_picture


def inspect_tiff(data):
    from .diagnostics import span

    with span("inspect_tiff_decode", payload_bytes=len(data)):
        with tifffile.TiffFile(io.BytesIO(data)) as tif:
            page = tif.pages[0]
            pixels = page.asarray()
            icc = page.tags.get(34675)
            if (
                len(tif.pages) != 1
                or pixels.dtype != np.uint16
                or pixels.ndim != 3
                or pixels.shape[2] != 3
                or not icc
            ):
                raise ValueError("Output must be single-page RGB16 with ICC")
            return pixels, bytes(icc.value)


def comparison(args):
    out = Path(args.output).resolve() / ("compare-" + uuid.uuid4().hex[:12])
    out.mkdir(parents=True, exist_ok=False)
    os.environ["NEGPY_USER_DIR"] = str(out / "negpy-state")
    manifest = {
        "schema": 1,
        "state": "running",
        "negpy_commit": NEGPY_SHA,
        "cool_colors_commit": COOL_SHA,
        "flexcolor_commit": FLEX_SHA,
        "cpu_only": True,
        "classic_exact": False,
        "source_files_modified": False,
        "output_space": "Adobe RGB (1998)",
        "experimental_color_basis": "Assumed NegPy working primaries for diagnostic rendering; not a measured colorimetric mapping",
        "results": {},
    }
    try:
        check_reference_sources()
        processor, config, source, records = open_triplet(
            args.raw,
            raw_import=getattr(args, "raw_import", "legacy-v1"),
            **(
                {"registration_check": args.registration_check}
                if getattr(args, "registration_check", None)
                else {}
            ),
        )
        from .raw_import import frozen_import_receipt

        raw_receipt = frozen_import_receipt(processor)
        if raw_receipt is not None:
            manifest["raw_import_receipt"] = raw_receipt
        if getattr(args, "registration_check", None):
            manifest["registration_check"] = args.registration_check
        manifest["inputs"] = records
        manifest["native_config"] = config.to_dict()
        region = (
            detect_picture(source, args.picture_inset) if args.auto_picture else None
        )
        if region:
            manifest["picture_detection"] = region
        roi = (
            region["analysis_roi"] if region else tuple(args.roi) if args.roi else None
        )
        analysis = analyze(np.maximum(source, 0), roi=roi)
        manifest["analysis"] = asdict(analysis)
        reference = (
            sample_film_base(np.maximum(source, 0), tuple(args.base_roi))
            if args.base_roi
            else None
        )
        manifest["film_base_reference"] = asdict(reference) if reference else None
        manifest["density_contract"] = density_contract(
            np.maximum(source, 0), analysis, reference
        )
        manifest["density_normalization"] = args.density_normalization
        manifest["negative_input"] = {
            "domain": "linear selected sensor bands, Native source corrections",
            "shape": list(source.shape),
            "negative_samples_floored": int(np.count_nonzero(source < 0)),
        }
        data, export = native_export(processor, config, records)
        manifest["export_config"] = {
            k: getattr(v, "value", v) for k, v in asdict(export).items()
        }
        encoded = {"native": (data, {"kind": "NegPy Native unchanged"})}
        encoded["cool-colors-auto"] = (
            encode_experiment(
                processor,
                density_auto(
                    np.maximum(source, 0),
                    analysis,
                    reference,
                    args.density_normalization,
                ),
                export,
            ),
            {
                "kind": "cool-colors auto density adaptation",
                "normalization": args.density_normalization,
                "output_contract": "linear positive density fraction -> one Adobe RGB 563/256 OETF -> uint16",
                "limitations": [
                    "not Nikon exact",
                    "no Nikon builder or CML4 CMS",
                    "no factory resource loaded",
                ],
            },
        )
        flex, info = flexcolor_research(
            np.maximum(source, 0),
            analysis,
            args.flex_profile,
            args.flex_gamma,
            reference,
            args.flex_input,
        )
        info["output_transfer_policy"] = args.flex_transfer
        info["output_transfer_basis"] = (
            "Reference CLI stores post-neg_gamma values directly; assume working-encoded codes for compatibility, not a factory TRC claim"
            if args.flex_transfer == "upstream-codes"
            else "Diagnostic legacy policy: interpret post-neg_gamma as linear and add working OETF"
        )
        encoded["flexcolor-research"] = (
            encode_experiment(
                processor,
                flex,
                export,
                transfer="working-encoded"
                if args.flex_transfer == "upstream-codes"
                else "linear",
            ),
            info,
        )
        thumbs = []
        for name, (payload, metadata) in encoded.items():
            pixels, icc = inspect_tiff(payload)
            target = out / (name + ".tiff")
            with target.open("xb") as stream:
                stream.write(payload)
            manifest["results"][name] = {
                "path": str(target),
                "sha256": sha256(target),
                "icc_sha256": __import__("hashlib").sha256(icc).hexdigest(),
                "shape": list(pixels.shape),
                "dtype": str(pixels.dtype),
                **metadata,
            }
            if region:
                from negpy.services.export.encoders import encode_tiff

                x0, y0, x1, y1 = region["analysis_roi"]
                pixels = pixels[y0:y1, x0:x1]
                picture_path = out / (name + "-picture.tiff")
                # Crop integer samples after the single CMS step; preserve ICC and DPI.
                from negpy.features.metadata.resolution import Resolution

                with tifffile.TiffFile(io.BytesIO(payload)) as tif:
                    page = tif.pages[0]
                    resolution = Resolution(
                        page.tags["XResolution"].value,
                        page.tags["YResolution"].value,
                        int(page.tags["ResolutionUnit"].value),
                    )
                picture_path.write_bytes(
                    encode_tiff(pixels, icc=icc, resolution=resolution)
                )
                manifest["results"][name]["picture_path"] = str(picture_path)
                manifest["results"][name]["picture_sha256"] = sha256(picture_path)
                manifest["results"][name]["picture_shape"] = list(pixels.shape)
            step = max(1, max(pixels.shape[:2]) // 1000)
            small = Image.fromarray(
                (pixels[::step, ::step] / 257).round().astype(np.uint8)
            )
            small = ImageCms.profileToProfile(
                small,
                ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
            )
            small.thumbnail((650, 900))
            small.save(
                out / (name + "-preview.png"),
                icc_profile=ImageCms.ImageCmsProfile(
                    ImageCms.createProfile("sRGB")
                ).tobytes(),
            )
            thumbs.append((name, small))
        canvas = Image.new(
            "RGB",
            (650 * len(thumbs), max(thumb.height for _, thumb in thumbs) + 36),
            "#202020",
        )
        draw = ImageDraw.Draw(canvas)
        for i, (name, thumb) in enumerate(thumbs):
            canvas.paste(thumb, (i * 650, 36))
            draw.text((i * 650 + 8, 10), name, fill="white")
        canvas.save(
            out / "comparison.png",
            icc_profile=ImageCms.ImageCmsProfile(
                ImageCms.createProfile("sRGB")
            ).tobytes(),
        )
        if any(sha256(r["path"]) != r["sha256"] for r in records):
            raise RuntimeError("RAW changed during comparison")
        manifest["state"] = "complete"
    except BaseException as exc:
        manifest.update(state="failed", error=str(exc))
        raise
    finally:
        (out / "manifest.json").write_text(
            json.dumps(
                manifest, ensure_ascii=False, indent=2, default=str, allow_nan=False
            )
            + "\n"
        )
    print(
        json.dumps(
            {"output": str(out), "manifest": str(out / "manifest.json")},
            ensure_ascii=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="PURE RGB research comparison; factory equivalence is not claimed"
    )
    region_options = parser.add_mutually_exclusive_group()
    region_options.add_argument(
        "--auto-picture",
        action="store_true",
        help="Detect picture area and exclude borders from research analysis and comparison",
    )
    parser.add_argument(
        "--picture-inset",
        type=float,
        default=0.02,
        help="Per-side safety inset inside detected picture (default 0.02)",
    )
    parser.add_argument(
        "--raw", nargs=3, help="Three same-frame RW2 files, in any order"
    )
    parser.add_argument(
        "--raw-import",
        choices=("legacy-v1", "standard-crop-v1", "standard-cfa-dng-v1"),
        default="legacy-v1",
        help="Versioned input geometry; standard crop uses crop-local ROIs and strict registration",
    )
    parser.add_argument("--list-engines", action="store_true")
    parser.add_argument(
        "--registration-check",
        choices=("translation-consistency-v1",),
        help="Opt-in spatial consistency proof for standard RAW translation alignment",
    )
    parser.add_argument("--output", default=str(ROOT / "outputs"))
    region_options.add_argument(
        "--roi",
        nargs=4,
        type=int,
        help="Analysis ROI: x0 y0 x1 y1, in final linear source coordinates",
    )
    parser.add_argument(
        "--flex-profile",
        help="User-supplied film plist; defaults to an explicitly synthetic neutral profile",
    )
    parser.add_argument(
        "--flex-transfer",
        choices=("upstream-codes", "linear"),
        default="upstream-codes",
        help="FlexColor output contract: direct upstream code values (default), or legacy linear interpretation for diagnostics",
    )
    parser.add_argument(
        "--base-roi",
        nargs=4,
        type=int,
        help="Explicit clean rebate candidate ROI, separate from picture metering",
    )
    parser.add_argument(
        "--density-normalization", choices=("scene", "base-anchored"), default="scene"
    )
    parser.add_argument(
        "--flex-input",
        choices=("scene-range", "base-relative", "raw-linear"),
        default="scene-range",
    )
    parser.add_argument(
        "--flex-gamma",
        type=float,
        help="Reconstructed neg-gamma parameter, distinct from output transfer",
    )
    args = parser.parse_args()
    if args.list_engines:
        print(json.dumps(CAPABILITIES, ensure_ascii=False, indent=2))
        return 0
    if not args.raw:
        parser.error("--raw requires three same-frame RW2 files")
    try:
        comparison(args)
    except Exception as exc:
        print(f"PURE RGB: {exc}", file=sys.stderr)
        return 1
    return 0
