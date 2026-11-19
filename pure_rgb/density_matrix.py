"""Independent density-window selection and renderer cross-comparison."""

import argparse
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import uuid

import numpy as np
from PIL import Image, ImageCms, ImageDraw

from .engines import (
    Analysis,
    ROOT,
    analyze,
    check_reference_sources,
    density_auto,
    flexcolor_research,
)
from .native import NEGPY_SHA, encode_experiment, native_export, open_triplet, sha256
from .picture import detect_picture
from .cli import inspect_tiff
from .study import preview

DENSITY_CHOICES = ("percentile", "negpy")
RENDERER_CHOICES = (
    "native-renderer",
    "native-kodak-endura",
    "cool-density-linear",
    "flex-research-codes",
)


def check_density_domain(process, window=None, allow_unmix=False):
    if process.crosstalk_strength != 0:
        from .film_profiles import film_signature

        if not allow_unmix and (
            window is None
            or window.metadata.get("negative_unmix") != film_signature(process)
        ):
            raise ValueError(
                "Shared window bridge requires unmixed source channels; dye-unmix needs an explicit channel-domain adapter"
            )
    trims = (
        process.white_point_offset,
        process.black_point_offset,
        process.white_point_trim_red,
        process.white_point_trim_green,
        process.white_point_trim_blue,
        process.black_point_trim_red,
        process.black_point_trim_green,
        process.black_point_trim_blue,
    )
    if any(v != 0 for v in trims):
        raise ValueError(
            "Shared window bridge currently requires neutral point trims to avoid applying a trim twice"
        )


def select_options(densities=None, renderers=None):
    densities = list(DENSITY_CHOICES if densities is None else densities)
    renderers = list(RENDERER_CHOICES if renderers is None else renderers)
    if (
        not densities
        or not renderers
        or len(set(densities)) != len(densities)
        or len(set(renderers)) != len(renderers)
    ):
        raise ValueError("Select nonempty, unique density and renderer identifiers")
    if set(densities) - set(DENSITY_CHOICES) or set(renderers) - set(RENDERER_CHOICES):
        raise ValueError(
            "Unavailable density/renderer; factory data routes are not enabled"
        )
    return densities, renderers


def render_variant(renderer, processor, config, records, source, window, export):
    frozen = replace(
        config, process=fixed_process(config.process, window, source.shape)
    )
    if renderer in ("native-renderer", "native-kodak-endura"):
        if renderer == "native-kodak-endura":
            frozen = replace(
                frozen, exposure=replace(frozen.exposure, paper_profile="kodak_endura")
            )
        payload, _ = native_export(processor, frozen, records)
        return payload, {
            "native_config": frozen.to_dict(),
            "dataset": "Original bundled "
            + frozen.exposure.paper_profile
            + " paper model; not a stock-specific negative film model",
        }
    if renderer == "cool-density-linear":
        return encode_experiment(
            processor, density_auto(np.maximum(source, 0), window.analysis), export
        ), {"transfer": "linear positive density fraction -> one working OETF"}
    if renderer == "flex-research-codes":
        samples, metadata = flexcolor_research(np.maximum(source, 0), window.analysis)
        return encode_experiment(
            processor, samples, export, transfer="working-encoded"
        ), metadata
    raise ValueError("Unavailable renderer")


@dataclass(frozen=True)
class DensityWindow:
    provider: str
    analysis: Analysis
    log_dense: tuple[float, float, float]
    log_thin: tuple[float, float, float]
    metadata: dict


def fixed_process(process, window, shape):
    check_density_domain(process, window=window)
    x0, y0, x1, y1 = window.analysis.roi
    h, w = shape[:2]
    return replace(
        process,
        analysis_rect=(x0 / w, y0 / h, x1 / w, y1 / h),
        analysis_buffer=0.0,
        # JSON replay restores Python floats. Use the same scalar arithmetic
        # domain in the first render, including NumPy-derived percentile bounds.
        local_floors=tuple(float(value) for value in window.log_dense),
        local_ceils=tuple(float(value) for value in window.log_thin),
        locked_floors=(0.0, 0.0, 0.0),
        locked_ceils=(0.0, 0.0, 0.0),
        use_luma_average=False,
        use_color_average=False,
        manual_dmin=None,
        manual_dmax=None,
    )


def negpy_window(image, config, roi, *, allow_unmix=False):
    from negpy.domain.interfaces import PipelineContext
    from negpy.features.exposure.processor import NormalizationProcessor
    from negpy.kernel.system.config import render_scale_factor

    check_density_domain(config.process, allow_unmix=allow_unmix)
    h, w = image.shape[:2]
    x0, y0, x1, y1 = roi
    process = replace(
        config.process,
        analysis_rect=(x0 / w, y0 / h, x1 / w, y1 / h),
        analysis_buffer=0.0,
        local_floors=(0.0, 0.0, 0.0),
        local_ceils=(0.0, 0.0, 0.0),
        locked_floors=(0.0, 0.0, 0.0),
        locked_ceils=(0.0, 0.0, 0.0),
        use_luma_average=False,
        use_color_average=False,
        manual_dmin=None,
        manual_dmax=None,
    )
    context = PipelineContext(
        original_size=(h, w),
        scale_factor=render_scale_factor(image.shape),
        process_mode=process.process_mode,
        wants_uv_grid=False,
        cache_stages=False,
    )
    NormalizationProcessor(process).process(image, context)
    bounds = context.metrics["final_bounds"]
    dense = tuple(float(v) for v in bounds.floors)
    thin = tuple(float(v) for v in bounds.ceils)
    if np.any(np.asarray(thin) <= np.asarray(dense)):
        raise ValueError("NegPy did not return a valid negative density window")
    a = Analysis(
        tuple(10 ** np.array(dense)),
        tuple(10 ** np.array(thin)),
        tuple(roi),
        None,
    )
    result = DensityWindow(
        "negpy-0.68",
        a,
        dense,
        thin,
        {
            "source": "NormalizationProcessor.process; original Rust/block-median/same-pixel bound analysis",
            "input_log_sign": "log10(signal)=-D, not positive optical density",
            "unclamped_negative_coordinate": "q=(log10(signal)-log_dense)/(log_thin-log_dense)",
            "positive_density_coordinate": "u=1-q; clip only in the selected renderer",
            "physical_base_calibration": False,
            "luma_range_clip": process.luma_range_clip,
            "color_range_clip": process.color_range_clip,
            "norm_density_range": context.metrics["norm_density_range"],
        },
    )
    if config.process.crosstalk_strength != 0:
        from .film_profiles import film_signature

        result.metadata["negative_unmix"] = film_signature(config.process)
    return result


def percentile_window(image, roi):
    a = analyze(image, roi)
    return DensityWindow(
        "percentile-0.5-99.5",
        a,
        tuple(np.log10(a.dark)),
        tuple(np.log10(a.base)),
        {
            "source": "independent picture percentiles",
            "physical_base_calibration": False,
            "positive_density_coordinate": "log10(high/signal)/log10(high/low)",
        },
    )


def study_region(source, roi=None):
    if roi is None:
        return detect_picture(source)
    h, w = source.shape[:2]
    if len(roi) != 4 or any(not isinstance(v, (int, np.integer)) for v in roi):
        raise ValueError("Explicit study ROI must have four integer pixel coordinates")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Explicit study ROI lies outside the merged source")
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise ValueError("Explicit study ROI is too small")
    return {
        "method": "Explicit study ROI supplied by caller",
        "coordinate_space": "EXIF-oriented merged linear source; half-open xyxy",
        "detected_picture_roi": None,
        "analysis_roi": tuple(roi),
        "inset_per_side": None,
        "confidence": None,
        "limitations": "Manually specified axis-aligned region; no automatic detection or semantic mask",
    }


def run(raw, destination, densities=None, renderers=None, roi=None):
    densities, renderers = select_options(densities, renderers)
    out = Path(destination).resolve() / ("density-matrix-" + uuid.uuid4().hex[:12])
    out.mkdir(parents=True, exist_ok=False)
    os.environ["NEGPY_USER_DIR"] = str(out / "negpy-state")
    manifest = {
        "schema": 1,
        "state": "running",
        "negpy_commit": NEGPY_SHA,
        "density_windows": {},
        "results": {},
        "scope": "Independent selected density providers x renderers; no Nikon/Hasselblad factory resource loaded",
        "selected_densities": densities,
        "selected_renderers": renderers,
        "native_dataset_scope": "Bundled NegPy paper/dye models under its original license; not a complete stock-specific negative forward model",
        "factory_resources": {"nikon_cml4": False, "hasselblad_profile": False},
        "experimental_color_basis": "Assumed working primaries; not device colorimetric calibration",
    }
    try:
        check_reference_sources()
        processor, config, source, records = open_triplet(raw)
        region = study_region(source, roi)
        roi = region["analysis_roi"]
        manifest.update(
            inputs=records, picture_detection=region, native_config=config.to_dict()
        )
        windows = [
            percentile_window(np.maximum(source, 0), roi)
            if name == "percentile"
            else negpy_window(source, config, roi)
            for name in densities
        ]
        native_bytes, export = native_export(processor, config, records)
        (out / "native-control.tiff").write_bytes(native_bytes)
        manifest["native_control"] = {
            "path": str(out / "native-control.tiff"),
            "sha256": sha256(out / "native-control.tiff"),
        }
        _, common_icc = inspect_tiff(native_bytes)
        canvas = Image.new("RGB", (520 * len(renderers), 470 * len(windows)), "#202020")
        draw = ImageDraw.Draw(canvas)
        for row, window in enumerate(windows):
            manifest["density_windows"][window.provider] = asdict(window)
            for col, renderer in enumerate(renderers):
                payload, meta = render_variant(
                    renderer, processor, config, records, source, window, export
                )
                name = window.provider + "__" + renderer
                path = out / (name + ".tiff")
                path.write_bytes(payload)
                pixels, icc = inspect_tiff(payload)
                if icc != common_icc or pixels.shape != source.shape:
                    raise ValueError("Output contract mismatch")
                x0, y0, x1, y1 = roi
                crop = pixels[y0:y1, x0:x1]
                thumb = preview(crop, icc, (520, 430))
                canvas.paste(thumb, (520 * col, 470 * row + 36))
                draw.text(
                    (520 * col + 8, 470 * row + 10),
                    window.provider + " / " + renderer,
                    fill="white",
                )
                thumb.save(
                    out / (name + "-preview.png"),
                    icc_profile=ImageCms.ImageCmsProfile(
                        ImageCms.createProfile("sRGB")
                    ).tobytes(),
                )
                manifest["results"][name] = {
                    "path": str(path),
                    "sha256": sha256(path),
                    "density_provider": window.provider,
                    "renderer": renderer,
                    "mean_rgb_code": np.mean(crop, axis=(0, 1)).tolist(),
                    "black_clip_fraction": float(np.mean(crop == 0)),
                    "white_clip_fraction": float(np.mean(crop == 65535)),
                    "icc_sha256": __import__("hashlib").sha256(icc).hexdigest(),
                    **meta,
                }
        canvas.save(
            out / "matrix.png",
            icc_profile=ImageCms.ImageCmsProfile(
                ImageCms.createProfile("sRGB")
            ).tobytes(),
        )
        if any(sha256(r["path"]) != r["sha256"] for r in records):
            raise ValueError("RAW hash changed")
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
    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", nargs=3, required=True)
    parser.add_argument("--output", default=str(ROOT / "outputs"))
    parser.add_argument("--roi", nargs=4, type=int)
    parser.add_argument(
        "--densities", nargs="+", choices=DENSITY_CHOICES, default=list(DENSITY_CHOICES)
    )
    parser.add_argument(
        "--renderers",
        nargs="+",
        choices=RENDERER_CHOICES,
        default=list(RENDERER_CHOICES),
    )
    args = parser.parse_args()
    print(run(args.raw, args.output, args.densities, args.renderers, args.roi))
