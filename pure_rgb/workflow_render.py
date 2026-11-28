"""Subprocess RAW renderer using the existing tested Native/cool/Flex adapters."""

from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import ImageCms
import tifffile

from .cli import inspect_tiff
from .common_positive import (
    apply_common_positive,
    checked_common_positive_controls,
)
from .cool_whitepoint_study import lift_codes, SOFT_GAIN, PIVOT
from .density_matrix import (
    fixed_process,
    negpy_window,
    percentile_window,
    render_variant,
    study_region,
)
from .endpoint_study import FIT_REFERENCE_STRIDE, fit_endpoint_curve
from .engines import check_reference_sources
from .editing_render import experimental_positive
from .flatfield import attach_calibration
from .diagnostics import event, span
from .decoded_cache import ImportDecodeCache, render_decode_context
from .positive_post import post_codes
from .film_profiles import native_film_context
from .native import native_export, native_export_settings, open_triplet, sha256
from .native_headroom import (
    native_headroom_active, native_headroom_engine, native_headroom_strength, native_local_highlight,
    print_exposure_receipt,
)
from .picture import NoReliablePictureRectangle, detect_picture_from_locators
from .preview_data import MODES
from .professional_controls import native_config, native_print_ev, output_codes
from .rgb_curve_study import fit_shared_curve
from .study import preview, read_rgb
from .tone_study import GAMMA
from .workspace import checked_recipe, write_json, output_rectangle
from .raw_import import frozen_import_receipt, is_standard_import
from .native_float_stream import render_native_float_stream


def emit(**data):
    event("render_event", **data)
    print(json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)


def exposure_codes(pixels, ev):
    if ev == 0:
        return pixels
    linear = (pixels.astype(np.float64) / 65535) ** GAMMA
    return np.rint(np.clip(linear * 2**ev, 0, 1) ** (1 / GAMMA) * 65535).astype(
        np.uint16
    )


def _study_picture_region(source, roi, locators, *, automatic_import=False, frame_roi=None):
    if roi is None and frame_roi is not None:
        from .picture import AUTO_PICTURE_INSET, metering_region_from_frame

        return dict(method="Confirmed output frame",
            analysis_roi=metering_region_from_frame(frame_roi, source.shape),
            detected_picture_roi=list(frame_roi), inset_per_side=AUTO_PICTURE_INSET,
            coordinate_space="EXIF-oriented merged linear source; half-open xyxy",
            confidence="confirmed frame", limitations="Confirmed frame used for measurement")
    try:
        return study_region(source, roi)
    except NoReliablePictureRectangle:
        if roi is None and locators is not None:
            region = detect_picture_from_locators(locators, source.shape)
            if region is not None:
                return region
        if automatic_import and roi is None:
            from .picture import normalized_box

            _, inner = normalized_box((0., 0., 1., 1.), source.shape, .1)
            return dict(method="Automatic import: central source metering fallback",
                analysis_roi=inner, detected_picture_roi=None, inset_per_side=.1,
                coordinate_space="EXIF-oriented merged linear source; half-open xyxy",
                confidence=None, measurement_only_fallback=True,
                limitations="Film box not detected; central measurement region, output remains uncropped")
        raise


def _native_float_export(processor, config, records, export, document):
    """Real Native PRINT SDK float, then original SDK final encode.

    Bind the same upstream source/geometry token as native_export. The point
    recipe and configuration are separately bound in the stream receipt.
    Old and neutral recipes never call this helper.
    """
    if hasattr(processor, "pure_raw_import_token"):
        processor._load_source_f32(records[0]["path"], config)
    identity = hashlib.sha256(
        (
            "".join(record["sha256"] for record in records)
            + getattr(processor, "pure_raw_import_token", "")
        ).encode()
    ).hexdigest()
    return render_native_float_stream(
        processor,
        records[0]["path"],
        config,
        export,
        identity,
        document,
        prefer_gpu=False,
    )


def _export_source_key(processor, config, records):
    if hasattr(processor, "pure_raw_import_token"):
        processor._load_source_f32(records[0]["path"], config)
    return hashlib.sha256(
        (
            "".join(record["sha256"] for record in records)
            + getattr(processor, "pure_raw_import_token", "")
        ).encode()
    ).hexdigest()


def _native_highlight_export(
    processor, config, records, source, export, document, point=None, *, film=None, print_ev=0.0,
):
    from .highlight_workflow import render_highlight_workflow

    adapter = native_headroom_engine(processor, strength=0, print_ev=print_ev) if print_ev else nullcontext()
    with adapter:
        result = render_highlight_workflow(
            processor,
            records[0]["path"],
            source,
            config,
            export,
            _export_source_key(processor, config, records),
            document,
            native_float_controls=point,
            film=film,
        )
    if result is None:
        raise ValueError("活动高光路由不可收到关闭参数")
    return result


def _native_headroom_export(processor, config, records, export, point=None, *, strength=1.0, print_ev=0.0):
    with native_headroom_engine(processor, strength=strength, print_ev=print_ev) as mapping:
        if point:
            payload, float_info = _native_float_export(
                processor, config, records, export or native_export_settings(config), point,
            )
        else:
            payload, _ = native_export(processor, config, records)
            float_info = None
    if mapping.receipt is None:
        raise ValueError("负片动态范围映射未取得印相前的密度信号")
    return payload, mapping.receipt, float_info


def _cool_headroom_export(
    processor, config, records, source, analysis, fit, export, document, geometry=None
):
    from .cool_headroom_workflow import encode_cool_headroom_workflow
    from .cool_headroom import SIGNAL_DOMAIN

    return encode_cool_headroom_workflow(
        processor,
        source,
        analysis,
        fit,
        export,
        document,
        source_key=_export_source_key(processor, config, records),
        domain=SIGNAL_DOMAIN,
        source_geometry=geometry,
        prepare_source_nonnegative=True,
    )


def _flex_research_export(
    processor, config, records, source, analysis, icc, document, geometry=None,
    experimental_film=None,
):
    from .flex_research_workflow import render_flex_research_workflow

    result = render_flex_research_workflow(
        source,
        analysis,
        document,
        working_icc=icc,
        source_key=_export_source_key(processor, config, records),
        source_geometry=geometry,
        experimental_film=experimental_film,
    )
    if result is None:
        raise ValueError("活动 Flex 研究路由不可收到关闭参数")
    return result


def _nikon_reference_guard(config):
    from .native_float import _config_guard
    from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
    from negpy.features.process.models import ProcessMode

    _config_guard(config, config.export)
    if config.process.process_mode != ProcessMode.C41:
        raise ValueError(
            "Nikon 研究链当前仅支持彩负 C41 正像参考，黑白与正片需独立路径"
        )
    if config.geometry != DEFAULT_WORKSPACE_CONFIG.geometry:
        raise ValueError("Nikon 参考正像暂不支持 Native SDK 几何变化")


def _nikon_reference_config(manifest):
    """Select an actual pre-edit reference, never infer it from a final TIFF."""
    for key in ("nikon_research_reference_config", "native_reference_config"):
        if key in manifest:
            if not isinstance(manifest[key], dict) or not manifest[key]:
                raise ValueError("Nikon 参考配置证明必须为非空对象")
            return manifest[key]
    if any(
        manifest["recipe"].get(key)
        for key in (
            "native_controls",
            "native_film",
            "native_float_controls",
            "highlight_controls",
        )
    ):
        raise ValueError(
            "旧任务缺少未施 Native 专属编辑的参考配置，不能启用 Nikon 研究链"
        )
    return manifest["native_config"]


def _nikon_research_export(
    processor, config, records, pixels, icc, document, geometry=None,
    experimental_film=None,
):
    from .nikon_research_workflow import render_nikon_research_workflow

    _nikon_reference_guard(config)
    result = render_nikon_research_workflow(
        pixels,
        document,
        working_icc=icc,
        source_key=_export_source_key(processor, config, records),
        reference_config=config.to_dict(),
        source_geometry=geometry,
        experimental_film=experimental_film,
    )
    if result is None:
        raise ValueError("活动 Nikon 研究路由不可收到关闭参数")
    return result


def render(request, *, emit_callback=None, prepared_source=None):
    emit_event = emit_callback or emit
    recipe = checked_recipe(request["recipe"])
    if prepared_source is not None and recipe.get("flatfield"):
        raise ValueError("Prepared full-grid fallback requires a separate flat-field receipt")
    only_mode = request.get("only_mode")
    if only_mode is not None and only_mode not in MODES:
        raise ValueError("未知草稿预览呈现模式")
    if recipe.get("source_mode") == "bw-single-v1":
        from .bw_workflow import render_bw

        if only_mode not in (None, "native"):
            raise ValueError("黑白草稿仅支持原生预览")
        return render_bw(request, emit_callback=emit_callback)
    if recipe.get("nikon_research_controls"):
        from .nikon_research_workflow import checked_nikon_resources

        checked_nikon_resources()
    out = Path(request["directory"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["NEGPY_USER_DIR"] = str(out / "negpy-state")
    report = {
        "state": "running",
        "recipe": recipe,
        "inputs": request["inputs"],
        "results": {},
        "scope": "Rendered RGB16, existing experimental algorithms; not factory CMS or physical color calibration",
    }
    processor = None
    preconfigured_flatfield = None
    decode_context = (
        render_decode_context(request, recipe, request.get("decode_anchor"))
        if prepared_source is None
        else None
    )
    import_cache = None
    if (decode_context is None and prepared_source is None
            and request.get("auto_import") is True and not request.get("transient_preview")):
        import_cache = ImportDecodeCache(request, recipe)
        decode_context = import_cache

    def configure_standard_source(candidate, initial_config, automatic_records):
        nonlocal preconfigured_flatfield
        # User-confirmed roles can differ from classification. Preserve the
        # existing reorder/redecode path for that case rather than correcting
        # the classifier's different channel assignment.
        if automatic_records != request["inputs"]:
            return initial_config
        calibration = request["calibration"]
        if Path(calibration["path"]).resolve() != Path(recipe["flatfield"]).resolve():
            raise ValueError("平场配方与冻结源不一致")
        emit_event(type="progress", fraction=0.15, message="读取／生成平场增益")
        initial_config, preconfigured_flatfield = attach_calibration(
            initial_config, calibration, out / "calibration", recipe["raw"],
            strength=recipe.get("flatfield_strength", 1.0),
        )
        candidate.pure_flatfield_reference_shape = (
            (preconfigured_flatfield.get("provenance") or {})
            .get("capture", {}).get("sensor_shape")
        )
        return initial_config

    try:
        if [r["path"] for r in request["inputs"]] != recipe["raw"] or [
            r["role"] for r in request["inputs"]
        ] != list("RGB"):
            raise ValueError("冻结配方和输入通道记录不一致")
        for r in request["inputs"]:
            if sha256(r["path"]) != r["sha256"]:
                raise ValueError("RAW 自入队后发生变化，请重新导入")
        emit_event(type="progress", fraction=0.05, message="核对通道、解码与配准")
        check_reference_sources()
        with span("raw_decode_align_merge"):
            if prepared_source is None:
                # Standard-crop flatfields attach through the hook, so the
                # open_triplet decode is final; legacy and CFA-DNG flatfields
                # finalize below after the calibration attach instead.
                open_cache = (
                    decode_context
                    if decode_context is not None
                    and (
                        not recipe["flatfield"]
                        or recipe.get("raw_import") in ("standard-crop-v1", "standard-mono-v1")
                    )
                    else None
                )
                processor, config, source, automatic = open_triplet(
                    recipe["raw"], raw_import=recipe.get("raw_import", "legacy-v1"),
                    **({"lens_correction": True} if recipe.get("lens_correction") else {}),
                    **({"chromatic_aberration": True} if recipe.get("chromatic_aberration") else {}),
                    **({"lens_profile": recipe["lens_profile"]} if recipe.get("lens_profile") else {}),
                    **({"registration_check": recipe["registration_check"]}
                       if recipe.get("registration_check") else {}),
                    **({"configure_source": configure_standard_source}
                       if recipe["flatfield"]
                       and recipe.get("raw_import") in ("standard-crop-v1", "standard-mono-v1") else {}),
                    **({"decode_cache": open_cache} if open_cache is not None else {}),
                )
            else:
                from .immutable_source_v44 import certificate_for

                processor, config, source, automatic = prepared_source
                certificate = certificate_for(source)
                if certificate is None:
                    raise ValueError("Prepared preview source needs its immutable authenticated lifetime")
                certificate.validate()
                from .live_worker import preview_checkpoint

                certificate.checkpoint = preview_checkpoint()
                if automatic != request["inputs"]:
                    raise ValueError("Prepared preview source records differ from the frozen RAW inputs")
                report["prepared_source_reused"] = True
        records = request["inputs"]
        if [r["path"] for r in automatic] != recipe["raw"]:
            config = replace(
                config,
                rgbscan=replace(
                    config.rgbscan,
                    green_path=recipe["raw"][1],
                    blue_path=recipe["raw"][2],
                ),
            )
            source, _, _ = processor._load_source_f32(recipe["raw"][0], config)
        report["automatic_roles"] = automatic
        report["role_assignment"] = (
            "User-confirmed RGB order; original scene/classification checks retained"
        )
        if recipe["flatfield"] and prepared_source is None:
            calibration = request["calibration"]
            if (
                Path(calibration["path"]).resolve()
                != Path(recipe["flatfield"]).resolve()
            ):
                raise ValueError("平场配方与冻结源不一致")
            if preconfigured_flatfield is None:
                emit_event(type="progress", fraction=0.15, message="读取／生成平场增益")
                # The pre-attach document keys the decode cache; the attached
                # profile identifier is process-local and must stay out of it.
                config_document = config.to_dict()
                config, applied = attach_calibration(
                    config, calibration, out / "calibration", recipe["raw"],
                    strength=recipe.get("flatfield_strength", 1.0),
                )
                before = source[::64, ::64].copy()
                if is_standard_import(recipe.get("raw_import")):
                    processor.pure_flatfield_reference_shape = (
                        (applied.get("provenance") or {})
                        .get("capture", {})
                        .get("sensor_shape")
                    )
                restored = (
                    decode_context.restore(
                        processor, config_document, recipe["raw"][0], config
                    )
                    if decode_context is not None
                    else None
                )
                if restored is not None:
                    source = restored
                else:
                    source, ir, color_space = processor._load_source_f32(
                        recipe["raw"][0], config
                    )
                    if decode_context is not None:
                        decode_context.store(
                            processor, config_document, recipe["raw"][0],
                            source, ir, color_space,
                        )
                sampled_change = float(np.mean(np.abs(source[::64, ::64] - before)))
            else:
                applied = preconfigured_flatfield
                sampled_change = processor.pure_flatfield_sampled_source_mean_absolute_change
                if sampled_change is None or not np.isfinite(sampled_change):
                    raise ValueError("平场单次导入缺少源像素变化诊断")
            report["calibration"] = {
                **applied,
                "sampled_source_mean_absolute_change": sampled_change,
            }
        raw_receipt = frozen_import_receipt(processor)
        if raw_receipt is not None:
            report["raw_import_receipt"] = raw_receipt
        locators = None
        if not request.get("transient_preview"):
            # Persist both locators before automatic detection can fail.
            from .workflow_geometry_preview import write_geometry_previews

            locators = write_geometry_previews(source, out)
        region = _study_picture_region(source, recipe["roi"], locators,
                                       automatic_import=bool(request.get("auto_import")),
                                       frame_roi=recipe.get("output_roi") if recipe.get("crop") else None)
        roi = region["analysis_roi"]
        x0, y0, x1, y1 = roi
        output_roi = output_rectangle(recipe, roi, source.shape)
        emit_event(type="progress", fraction=0.22, message="测量画面内部密度")
        original_metering = recipe.get("measurement_method") == "negpy"
        config = replace(
            config, exposure=replace(config.exposure, paper_profile=recipe["paper"])
        )
        method = "negpy-original" if original_metering else recipe["density"]
        with span("density_analysis", roi=roi, method=method):
            if original_metering:
                from .measurement_method import original_config, original_window

                frozen = original_config(config, roi, source.shape)
                window = original_window(source, frozen, roi)
            else:
                window = (
                    negpy_window(source, config, roi)
                    if recipe["density"] == "negpy"
                    else percentile_window(np.maximum(source, 0), roi)
                )
                frozen = replace(
                    config, process=fixed_process(config.process, window, source.shape)
                )
        native_overrides = recipe.get("native_controls")
        float_controls = recipe.get("native_float_controls")
        headroom = native_headroom_active(recipe)
        print_ev = native_print_ev(recipe)
        highlight_controls = native_local_highlight(recipe)
        spatial_controls = recipe.get("native_spatial_controls")
        # These Native-only renders produce their own RGB16 and export tags.
        # Legacy highlight needs the same export settings, while only the
        # other engines consume the neutral reference pixels.
        ordinary_native_edits = native_overrides and not any((
            float_controls, highlight_controls, recipe.get("native_film"),
        ))
        skip_neutral_native = (
            only_mode == "native"
            and (headroom or print_ev or spatial_controls or highlight_controls or ordinary_native_edits)
            and not recipe.get("density_metering")
            and not recipe.get("nikon_research_controls")
        )
        if skip_neutral_native:
            payload = export = native = icc = None
            if highlight_controls:
                export = native_export_settings(frozen)
        elif recipe.get("density_metering"):
            from .shared_density_metering_v43 import render_shared_measurement

            report["shared_measurement_input_density_window"] = asdict(window)
            with span("shared_density_measurement", backend="cpu"):
                payload, export, shared_info, shared_snapshot, window = render_shared_measurement(
                    processor, source, frozen, records, roi, recipe["density_metering"],
                    source_key=_export_source_key(processor, frozen, records),
                )
            report["shared_measurement_processing"] = shared_info
            report["shared_measurement_snapshot"] = shared_snapshot
        else:
            with span("native_reference", backend="cpu"):
                payload, export = native_export(processor, frozen, records)
        if payload is not None:
            native, icc = inspect_tiff(payload)
        nikon_controls = recipe.get("nikon_research_controls")
        nikon_reference = None
        if nikon_controls:
            _nikon_reference_guard(frozen)
            if native.shape != source.shape:
                raise ValueError("Nikon 参考正像与完整来源几何不一致")
            nikon_reference = native.view()
            nikon_reference.setflags(write=False)
            report["nikon_research_reference_config"] = frozen.to_dict()
        if payload is not None:
            with tifffile.TiffFile(io.BytesIO(payload)) as tif:
                page = tif.pages[0]
                dpi = {
                    "resolution": (page.tags[282].value, page.tags[283].value),
                    "resolutionunit": int(page.tags[296].value),
                }
        # The fitters use only this exact ROI-origin grid. Convert the selected
        # RGB16 samples rather than retaining a full float64 reference ROI.
        # Creative Native edits below remain independent of this neutral grid.
        fit_stride = FIT_REFERENCE_STRIDE
        reference = (
            None
            if only_mode == "native"
            else (native[y0:y1:fit_stride, x0:x1:fit_stride].astype(float) / 65535) ** GAMMA
        )
        report.update(
            roi=roi,
            output_roi=output_roi,
            picture_region=region,
            source_shape=list(source.shape),
            density_window=asdict(window),
            native_config=frozen.to_dict(),
            **({"icc_sha256": hashlib.sha256(icc).hexdigest()} if icc is not None else {}),
        )
        # Keep neutral reference and all experimental fitting independent of
        # Native-only creative controls. The original export above is immutable.
        float_info = None
        highlight_info = None
        headroom_info = None
        with span("native_film_context"):
            film_config, film_receipt = native_film_context(
                frozen, source, roi, recipe["density"], recipe.get("native_film")
            )
        if recipe.get("native_film"):
            report["native_film"] = film_receipt
        if only_mode in (None, "native") and (
            native_overrides
            or not film_receipt["identity"]
            or float_controls
            or highlight_controls
            or spatial_controls
        ):
            adjusted = native_config(
                film_config, native_overrides or {"version": 1, "values": {}}
            )
            if original_metering:
                adjusted = original_config(adjusted, roi, source.shape)
                if adjusted.process == frozen.process:
                    actual_window = window
                else:
                    with span("native_density_reanalysis", roi=roi):
                        actual_window = original_window(source, adjusted, roi)
                report["native_metering_window"] = asdict(actual_window)
                if only_mode == "native":
                    report["density_window"] = asdict(actual_window)
            if spatial_controls:
                from .spatial_workflow_v31 import encode_spatial_workflow

                payload, _, spatial_info, spatial_snapshot = encode_spatial_workflow(
                    processor,
                    source,
                    adjusted,
                    records,
                    roi,
                    spatial_controls,
                    source_key=_export_source_key(processor, adjusted, records),
                    highlight=highlight_controls,
                    point=float_controls,
                    film=recipe.get("native_film"),
                    print_ev=print_ev,
                )
                report["native_spatial_processing"] = spatial_info
                report["native_spatial_snapshot"] = spatial_snapshot
            elif headroom or (print_ev and not highlight_controls):
                emit_event(type="progress", fraction=0.35, message="正在映射负片动态范围并生成正像…")
                payload, headroom_info, float_info = _native_headroom_export(
                    processor, adjusted, records, export, float_controls,
                    strength=native_headroom_strength(recipe) if headroom else 0,
                    print_ev=print_ev,
                )
                if headroom:
                    report["native_headroom_processing"] = headroom_info
                else:
                    headroom_info = None
                if float_info is not None:
                    report["native_float_processing"] = float_info
            elif highlight_controls:
                emit_event(type="progress", fraction=0.35, message="正在渲染与合成高光找回…")
                payload, highlight_info = _native_highlight_export(
                    processor,
                    adjusted,
                    records,
                    source,
                    export,
                    highlight_controls,
                    float_controls,
                    film=recipe.get("native_film"),
                    print_ev=print_ev,
                )
                float_info = (
                    highlight_info.get("native_float_processing")
                    if float_controls
                    else None
                )
                report["highlight_processing"] = highlight_info
            elif float_controls:
                payload, float_info = _native_float_export(
                    processor, adjusted, records, export, float_controls
                )
                report["native_float_processing"] = float_info
            else:
                payload, _ = native_export(processor, adjusted, records)
            native, adjusted_icc = inspect_tiff(payload)
            if icc is None:
                from .positive_float import working_icc_bytes

                if adjusted_icc != working_icc_bytes():
                    raise ValueError("Native 输出 ICC 与共同工作色彩空间不一致")
                icc = adjusted_icc
                report["icc_sha256"] = hashlib.sha256(icc).hexdigest()
                with tifffile.TiffFile(io.BytesIO(payload)) as tif:
                    page = tif.pages[0]
                    dpi = {
                        "resolution": (page.tags[282].value, page.tags[283].value),
                        "resolutionunit": int(page.tags[296].value),
                    }
            elif adjusted_icc != icc:
                raise ValueError("Native 控件渲染的 ICC 与参考不一致")
            report["native_reference_config"] = report["native_config"]
            report["native_config"] = adjusted.to_dict()
            if print_ev:
                report["native_print_exposure_processing"] = print_exposure_receipt(print_ev)
            report["reference_policy"] = (
                "v4: Native film, creative, float and highlight controls excluded from experimental fitting"
                if highlight_controls
                else "v3: Native film, creative and float controls excluded from experimental fitting"
                if float_controls
                else "v2: Native film calibration and creative overrides excluded from experimental fitting"
            )
        if recipe.get("measurement_method") == "negpy" and not any((
            float_controls, highlight_controls, spatial_controls, recipe.get("density_metering"),
        )):
            report["native_print"] = getattr(processor, "pure_native_print", None)
        srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

        def save(mode, px, meta):
            ox0, oy0, ox1, oy1 = output_roi
            px = px[oy0:oy1, ox0:ox1]
            common = checked_common_positive_controls(
                recipe.get("common_positive_controls")
            )
            if common:
                with span("common_positive_controls", mode=mode):
                    px = apply_common_positive(px, common)
                meta = {**meta, "common_positive_controls": common}
            controls = recipe.get("output_controls", {}).get(mode)
            if controls:
                px, diagnostics = output_codes(px, controls, recipe["exposure_ev"])
                meta = {
                    **meta,
                    "output_controls": controls,
                    "output_diagnostics": diagnostics,
                }
            else:
                # Keep the exact historical exposure/rounding path for old recipes.
                px = exposure_codes(px, recipe["exposure_ev"])
            if mode == "cool-shared-rgb" and not nikon_controls:
                if recipe["cool_tone"] == "soft-shoulder":
                    px = lift_codes(px, soft=True)
                meta = {
                    **meta,
                    "cool_tone": recipe["cool_tone"],
                    "presentation_version": 1,
                    "soft_shoulder_parameters": {
                        "gain": SOFT_GAIN,
                        "encoded_pivot": PIVOT,
                    }
                    if recipe["cool_tone"] == "soft-shoulder"
                    else None,
                }
            path = out / (mode + ".tiff")
            post = recipe.get("post_controls", {}).get(mode)
            if post:
                with span("common_positive_post", mode=mode):
                    px, post_info = post_codes(px, post)
                meta = {**meta, "post_controls": post, "post_processing": post_info}
            if recipe.get("output_orientation"):
                from .output_orientation import orient_codes

                px = orient_codes(px, recipe["output_orientation"])
                meta = {**meta, "output_orientation": recipe["output_orientation"]}
            with span("render_save_tiff", mode=mode):
                tifffile.imwrite(
                    path,
                    px,
                    photometric="rgb",
                    compression="deflate",
                    metadata=None,
                    extratags=[(34675, 7, len(icc), icc, False), (274, 3, 1, 1, False)],
                    **dpi,
                )
                actual, ai = read_rgb(path)
                if ai != icc or not np.array_equal(actual, px):
                    raise ValueError("RGB16/ICC 输出核验失败")
                del actual, ai
            report["results"][mode] = {
                "path": str(path),
                "sha256": sha256(path),
                "shape": list(px.shape),
                "icc_sha256": report["icc_sha256"],
                **meta,
            }
            if request.get("repair_plan") is not None:
                from .repair_pipeline import render_repair

                with span("confirmed_repair_replay", mode=mode):
                    render_repair(
                        request, report, report["results"][mode], mode,
                        directory=out / (mode + "-repair"),
                        on_progress=lambda progress: emit_event(**progress),
                    )
            preview(px, icc, (520, 440)).save(
                out / (mode + "-preview.png"), icc_profile=srgb
            )

        emit_event(type="progress", fraction=0.4, message="生成 Native 正像")
        if only_mode in (None, "native"):
            save(
                "native",
                native,
                {
                    "kind": "NegPy PRINT with density-driven highlight roll-off"
                    if headroom else "NegPy PRINT with luminance-only print exposure"
                    if print_ev else "Native PRINT with explicit positive float point controls"
                    if float_controls
                    else "NegPy unchanged renderer",
                    "native_controls": native_overrides,
                    **({"native_headroom_processing": headroom_info} if headroom_info else {}),
                    **({"native_print_exposure_processing": report["native_print_exposure_processing"]}
                       if print_ev else {}),
                    **(
                        {
                            "native_float_controls": float_controls,
                            "native_float_processing": float_info,
                        }
                        if float_controls
                        else {}
                    ),
                    **(
                        {
                            "highlight_controls": highlight_controls,
                            "highlight_processing": highlight_info,
                        }
                        if highlight_controls
                        else {}
                    ),
                },
            )
        del native, payload
        for renderer, mode, fraction in [
            ("cool-density-linear", "cool-shared-rgb", 0.6),
            ("flex-research-codes", "flex-endpoint", 0.8),
        ]:
            if only_mode is not None and mode != only_mode:
                continue
            emit_event(
                type="progress", fraction=fraction, message="生成 " + MODES[mode]
            )
            payload, meta = render_variant(
                renderer, processor, config, records, source, window, export
            )
            px, other = inspect_tiff(payload)
            if other != icc:
                raise ValueError("中间 ICC 不一致")
            # Active research adapters consume the neutral Native codes or
            # source transmission directly, never this legacy full-grid linear
            # buffer. Preserve their off-replay fit on the identical sample grid.
            research_adapter = (
                mode == "cool-shared-rgb" and nikon_controls
            ) or (
                mode == "flex-endpoint" and recipe.get("flex_research_controls")
            )
            linear = None if research_adapter else (px.astype(float) / 65535) ** GAMMA
            fit_samples = (
                (px[y0:y1:fit_stride, x0:x1:fit_stride].astype(float) / 65535) ** GAMMA
                if research_adapter else linear[y0:y1:fit_stride, x0:x1:fit_stride]
            )
            with span("fit_reference_curve", mode=mode):
                fit = replace(
                    (
                        fit_shared_curve(fit_samples, reference, stride=1)
                        if mode == "cool-shared-rgb"
                        else fit_endpoint_curve(fit_samples, reference, stride=1)
                    ),
                    grid_stride=fit_stride,
                )
            del fit_samples
            with span("experimental_engine", mode=mode, backend="cpu"):
                nikon_research = nikon_controls if mode == "cool-shared-rgb" else None
                flex_research = (
                    recipe.get("flex_research_controls")
                    if mode == "flex-endpoint"
                    else None
                )
                headroom = (
                    recipe.get("cool_headroom_controls")
                    if mode == "cool-shared-rgb"
                    else None
                )
                if nikon_research:
                    encoded, ni_icc, ni_receipt = _nikon_research_export(
                        processor, frozen, records, nikon_reference, icc, nikon_research,
                        experimental_film=recipe.get("experimental_film_controls"),
                    )
                    if ni_icc != icc:
                        raise ValueError("Nikon 研究链输出必须实际转换到共同工作 ICC")
                    adjustment = {
                        "nikon_research_controls": nikon_research,
                        "nikon_research_processing": ni_receipt,
                        "fit_usage": "legacy fit retained only for explicit off replay; not applied to active Nikon CMS chain",
                    }
                    nikon_reference = None
                elif flex_research:
                    encoded, flex_icc, flex_receipt = _flex_research_export(
                        processor,
                        config,
                        records,
                        source,
                        window.analysis,
                        icc,
                        flex_research,
                        experimental_film=recipe.get("experimental_film_controls"),
                    )
                    if flex_icc != icc:
                        raise ValueError("Flex 研究链输出必须实际转换到共同工作 ICC")
                    adjustment = {
                        "flex_research_controls": flex_research,
                        "flex_research_processing": flex_receipt,
                        "fit_usage": "legacy fit retained only for explicit off replay; not applied to active research chain",
                    }
                elif headroom:
                    encoded, head_icc, head_receipt = _cool_headroom_export(
                        processor,
                        config,
                        records,
                        source,
                        window.analysis,
                        fit,
                        export,
                        headroom,
                    )
                    if head_icc != icc:
                        raise ValueError("Cool 保留余量输出 ICC 与参考不一致")
                    adjustment = {
                        "cool_headroom_controls": headroom,
                        "cool_headroom_processing": head_receipt,
                    }
                else:
                    encoded, adjustment = experimental_positive(
                        mode,
                        linear,
                        fit,
                        recipe.get("engine_controls", {}).get(mode),
                        source,
                        window.analysis,
                        processor,
                        export,
                        profile=recipe.get("flex_profile")
                        if mode == "flex-endpoint"
                        else None,
                        expected_icc=icc,
                    )
            save(mode, encoded, {"fit": asdict(fit), "adapter": meta, **adjustment})
            del px, payload, linear, encoded
        if any(sha256(r["path"]) != r["sha256"] for r in records):
            raise ValueError("RAW 处理期间发生变化")
        if (
            recipe["flatfield"]
            and sha256(request["calibration"]["path"])
            != request["calibration"]["sha256"]
        ):
            raise ValueError("平场源文件处理期间发生变化")
        report.update(state="complete", raw_hashes_unchanged=True)
    except BaseException as error:
        report.update(state="failed", error=str(error))
        from .raw_registration import RegistrationModelCheckError

        if isinstance(error, RegistrationModelCheckError):
            report["registration_check_diagnostic"] = error.diagnostic
        raise
    finally:
        write_json(out / "manifest.json", report)
        if (request.get("transient_preview") is True and processor is not None
                and prepared_source is None):
            processor.cleanup()
            processor.destroy_all()
    if import_cache is not None:
        import_cache.commit(report, source)
    emit_event(type="result", directory=str(out))
    return report


if __name__ == "__main__":
    try:
        render(json.loads(Path(sys.argv[1]).read_text()))
    except Exception as error:
        emit(type="error", message=str(error))
        raise SystemExit(1)
