"""Display samples from full-grid NegPy measurement and color conversion."""

from dataclasses import replace

import numpy as np
from negpy.services.rendering.engine import DarkroomEngine

from .diagnostics import span


class _SampleForDisplay(DarkroomEngine):
    """Keep full support through the last source-dependent stage."""

    def __init__(self, rectangle, step, stage):
        super().__init__()
        self.rectangle = rectangle
        self.step = step
        self.stage = stage
        self.sampled = False
        self.bounds = None

    def _run_stage(self, img, config, cache_field, processor_fn, context, pipeline_changed):
        args = img, config, cache_field, processor_fn, context, pipeline_changed
        if cache_field == "base":
            with span("native_preview_full_grid_measure", source_shape=list(img.shape)):
                result, changed = super()._run_stage(*args)
            self.bounds = context.metrics["final_bounds"]
            # Inspection-only raster. The print stages receive it as their
            # image input and need only the scalar metrics from the context.
            context.metrics.pop("normalized_log", None)
        elif cache_field == self.stage:
            with span("native_preview_full_grid_lab"):
                result, changed = super()._run_stage(*args)
        else:
            return super()._run_stage(*args)
        from .live_worker import preview_checkpoint

        preview_checkpoint()()
        if cache_field != self.stage:
            return result, changed
        if context.cache_stages or self.sampled:
            raise ValueError("原生测量预览需要独立的完整来源统计")
        x0, y0, x1, y1 = self.rectangle
        sample = np.array(result[y0:y1:self.step, x0:x1:self.step], order="C", copy=True)
        self.sampled = True
        return sample, changed


def _render_sample(cache, config, export, rectangle, step, stage):
    """Keep the display grid in the original SDK working transfer function."""
    from .native_print_metrics import collect_print_values

    processor = cache.session.processor
    path = cache.owner.records[0]["path"]
    loaded, _, _ = processor._load_source_f32(path, config)
    cache.certificate.validate(loaded)
    engine = _SampleForDisplay(rectangle, step, stage)
    original_engine = processor.engine_cpu
    processor.engine_cpu = engine
    try:
        with span("native_preview_screen_render", step=step), collect_print_values(processor):
            image, target = processor.render_export(path, config, export, cache.source_key, prefer_gpu=False)
    finally:
        processor.engine_cpu = original_engine
    if image is None:
        raise RuntimeError(target)
    x0, y0, x1, y1 = rectangle
    expected = ((y1 - y0 + step - 1) // step, (x1 - x0 + step - 1) // step, 3)
    if not engine.sampled or image.shape != expected:
        raise ValueError("原生测量预览没有保留约定的源像素网格")
    image.setflags(write=False)
    return dict(encoded=image, target=target, bounds=dict(log_dense=list(engine.bounds.floors),
                log_thin=list(engine.bounds.ceils)), native_print=processor.pure_native_print, export=export)


def _encode_sample(processor, image, export, target):
    from .cli import inspect_tiff

    with span("native_preview_screen_encode"):
        payload, status = processor.encode_export(image, export, target)
        if payload is None:
            raise RuntimeError(status)
        pixels, icc = inspect_tiff(payload)
    pixels.setflags(write=False)
    return pixels, icc


def _highlight_sample(cache, config, cached, highlight, point, rectangle, step, stage):
    from .density_preview_v44 import _key
    from .highlight_sample_v46 import compose_native_sample
    from .highlight_workflow import highlight_reference_config
    from .positive_float import working_icc_bytes

    source = cache.session.source
    roi = highlight["reference_roi"]
    reference = cached.get("highlight_reference")
    reference_hit = reference is not None and reference["roi"] == roi
    if not reference_hit:
        cache._progress("highlight_reference", .55, "正在测量高光参考并生成参考预览…")
        with span("native_preview_highlight_reference"):
            settings, window = highlight_reference_config(source, config, roi)
            reference = _render_sample(cache, settings, cached["export"], rectangle, step, stage)
        if (reference["target"] != cached["target"]
                or reference["encoded"].dtype != cached["encoded"].dtype):
            raise ValueError("高光参考的色彩空间或像素类型与原生正像不一致")
        reference.update(roi=roi, window=dict(log_dense=list(window.log_dense), log_thin=list(window.log_thin)))
        cached["highlight_reference"] = reference
        cache.reference_measurement_calls += 1
    key = _key([highlight, point])
    result = reference.get("result")
    if result is None or result["key"] != key:
        cache._progress("sampled_color", .86, "正在合成高光找回预览…")
        masks = {
            name: cache._highlight_mask(cache.source_key, source.shape, name, highlight.get(name), rectangle, step)
            for name in ("protection", "effect_area")
        }
        with span("native_preview_highlight_join", pixels=int(cached["encoded"].shape[0] * cached["encoded"].shape[1])):
            image, _ = compose_native_sample(
                cached["encoded"], icc=working_icc_bytes(), reference_encoded=reference["encoded"],
                highlight=highlight, point=point,
                protection_mask=masks["protection"].mask if masks["protection"] else None,
                effect_mask=masks["effect_area"].mask if masks["effect_area"] else None,
            )
            pixels, icc = _encode_sample(cache.session.processor, image, cached["export"], cached["target"])
        result = reference["result"] = dict(key=key, pixels=pixels, icc=icc)
    return result["pixels"], result["icc"], reference_hit, reference["window"]


def _nikon_sample(owner, recipe, session, config, cached, phase, step):
    """Convert the neutral Native RGB16 sample with the actual Nikon CMS."""
    from .density_preview_v44 import _key
    from .nikon_research_workflow import checked_nikon_resources
    from .workflow_render import _nikon_research_export

    controls = recipe["nikon_research_controls"]
    resources = checked_nikon_resources()["binding_sha256"]
    key = _key([controls, resources])
    result = cached.get("nikon_cms")
    if result is None or result["key"] != key:
        sample = cached["pixels"]
        with span("nikon_preview_screen_cms", pixels=int(sample.shape[0] * sample.shape[1])):
            converted, other, _ = _nikon_research_export(
                session.processor, config, owner.records, sample, cached["icc"], controls,
                geometry=dict(full_source_shape=list(session.source.shape), view_roi=phase, sample_step=step),
            )
        if other != cached["icc"] or converted.shape != sample.shape:
            raise ValueError("Nikon 预览的输出网格或工作 ICC 不一致")
        converted.setflags(write=False)
        result = cached["nikon_cms"] = {"key": key, "pixels": converted}
        return converted, False
    return result["pixels"], True


def try_original_density_screen(owner, request, recipe, mode):
    """Sample before encoding and CMS. Other output contracts keep the full renderer."""
    nikon = mode == "cool-shared-rgb" and bool(recipe.get("nikon_research_controls"))
    flex = mode == "flex-endpoint" and not any(
        recipe.get(key) for key in ("flex_research_controls", "flex_profile", "experimental_film_controls")
    )
    if ((mode != "native" and not nikon and not flex) or recipe.get("measurement_method") != "negpy"
            or recipe.get("source_mode")
            or request.get("full_resolution") or request.get("repair_plan")
            or request.get("crop_locator") or not recipe.get("roi")):
        return None
    film = recipe.get("native_film")
    highlight = recipe.get("highlight_controls") if mode == "native" else None
    point = recipe.get("native_float_controls") if mode == "native" else None
    if mode == "native" and ((film and film["strength"] != 0)
                              or (point and not highlight)):
        return None
    if flex and set(recipe.get("engine_controls", {}).get(mode, {}).get("values", {})) - {
        "curve_contrast", "curve_offset"
    }:
        return None
    if nikon:
        from .nikon_draft_preview_v40 import NikonDraftPreview

        if recipe.get("experimental_film_controls"):
            return None
        if owner.nikon_preview is None:
            owner.nikon_preview = NikonDraftPreview(owner.manifest, owner.digest)
        if owner.nikon_preview.eligible(recipe):
            return None

    from negpy.domain.models import AspectRatio, ExportResolutionMode
    from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG

    from .density_preview_v44 import _key, point_c41_supported
    from .draft_preview import _anchor_recipe
    from .flex_draft_preview_v40 import oriented_source_sample_roi
    from .measurement_method import original_config
    from .native import native_export_settings
    from .output_orientation import orient_codes
    from .preview_grid import post_requires_full_grid
    from .professional_controls import native_config
    from .viewport_preview_v43 import (
        checked_viewport, output_size_for_recipe, source_rectangle_for_viewport,
    )
    from .workspace import output_rectangle

    if post_requires_full_grid(recipe.get("post_controls", {}).get(mode)):
        return None
    cache = owner.density_preview
    if (cache.certificate is None and mode in owner.manifest["results"]
            and _anchor_recipe(recipe, mode) == _anchor_recipe(owner.recipe, mode)):
        return None
    shape = owner.manifest["source_shape"]
    roi = recipe["roi"]
    if not (0 <= roi[0] <= roi[2] - 16 < roi[2] <= shape[1]
            and 0 <= roi[1] <= roi[3] - 16 < roi[3] <= shape[0]):
        return None
    # Filter before preparing a resident full source, so an unsupported
    # spatial/look recipe does not pay for an extra source on fallback.
    # Nikon always starts from the neutral producer, including when the
    # active Native engine has its own creative controls.
    controls = (recipe.get("native_controls") if mode == "native" else None) or {"version": 1, "values": {}}
    wanted = native_config(DEFAULT_WORKSPACE_CONFIG, controls)
    point_lab = replace(wanted.lab, clahe_strength=0, chroma_denoise=0,
                        sharpen=0, glow_amount=0, halation_strength=0)
    if (not point_c41_supported(replace(wanted, lab=point_lab))
            or wanted.export.export_resolution_mode != ExportResolutionMode.ORIGINAL.value
            or wanted.export.paper_aspect_ratio != AspectRatio.ORIGINAL.value
            or wanted.export.icc_input_path or wanted.export.icc_output_path):
        return None
    if wanted.lab.sharpen > 0 and wanted.lab.sharpen_auto_masking:
        return None
    stage = "base" if point_c41_supported(wanted) else "lab"

    size = output_size_for_recipe(recipe, shape, roi, mode)
    rectangle = output_rectangle(recipe, roi, shape)
    direction = recipe.get("output_orientation")
    viewport = None
    if "viewport" in request:
        viewport = checked_viewport(request["viewport"], size)
        rectangle = source_rectangle_for_viewport(rectangle, direction, viewport)
        step = viewport["sample_step"]
    else:
        step = max(1, (max(size) + 1399) // 1400)
    phase = oriented_source_sample_roi(rectangle, direction, step)

    with span("native_preview_source"):
        session = cache.prepare_source()
    base = cache.pristine_config
    config = original_config(native_config(
        replace(base, exposure=replace(base.exposure, paper_profile=recipe["paper"])),
        controls), roi, shape)
    export = native_export_settings(config)
    if highlight:
        from .highlight_workflow import _configuration

        _configuration(config, export, bounds_override=None)
    # Legacy Flex needs exactly the same ROI-origin fit grid as saving.
    native_rect, native_step = phase, step
    if flex:
        from .endpoint_study import FIT_REFERENCE_STRIDE

        native_rect, native_step = roi, FIT_REFERENCE_STRIDE
    key = _key(["native-original", cache.certificate.sha256, config.to_dict(), native_rect, native_step])
    cached = cache.samples.get(key)
    hit = cached is not None
    if cached is None:
        cache._progress("density_support", .28, "正在按完整来源测量 NegPy 密度…")
        cached = _render_sample(cache, config, export, native_rect, native_step, stage)
        cache.samples[key] = cached
        while len(cache.samples) > 3:
            cache.samples.popitem(last=False)
        cache.measurement_calls += 1
    else:
        cache.samples.move_to_end(key)
    if highlight:
        pixels, icc, highlight_hit, highlight_window = _highlight_sample(
            cache, config, cached, highlight, point, phase, step, stage)
    else:
        if "pixels" not in cached:
            cached["pixels"], cached["icc"] = _encode_sample(
                session.processor, cached["encoded"], cached["export"], cached["target"])
        pixels, icc = cached["pixels"], cached["icc"]
    color_hit = False
    if nikon:
        cache._progress("sampled_color", .86, "正在转换当前预览的 Nikon CMS 色彩…")
        pixels, color_hit = _nikon_sample(owner, recipe, session, config, cached, phase, step)
    elif flex:
        from .flex_legacy_screen import flex_legacy_sample

        cache._progress("sampled_color", .78, "正在拟合曲线并转换当前 FlexColor 预览…")
        pixels, color_hit = flex_legacy_sample(owner, recipe, session, cached, phase, step)
    pixels = orient_codes(pixels, direction)
    reference = cache._reference(recipe, mode, phase, step, direction, pixels, icc)
    current = owner._finish(pixels, recipe, mode)
    cache.certificate.validate()
    owner._validate(request)
    owner.output_size, owner.sample_step, owner.viewport_result = size, step, viewport
    if mode == "native" and not highlight:
        owner.native_print = cached["native_print"]
    owner.metering_receipt = dict(roi=list(roi), metering=None, automatic=None, report={})
    if highlight:
        owner.metering_receipt["highlight_reference_roi"] = list(highlight["reference_roi"])
    owner.preview_quality = {
        "mode": "nikon-original-screen" if nikon else "flex-original-screen" if flex else "native-original-screen",
        "approximate": False,
    }
    cache.last_receipt = dict(
        route="native-original-full-meter-then-screen", full_support_measured=True,
        measurement_cache_hit=hit, measurement_calls=cache.measurement_calls,
        source_prepare_calls=cache.source_prepare_calls, full_Native_exports=0,
        sample_pixels=int(pixels.shape[0] * pixels.shape[1]), full_source_shape=list(shape),
        sampled_after=stage, measurement_window=cached["bounds"],
        retained_source_bytes=int(session.source.nbytes),
    )
    if nikon:
        cache.last_receipt.update(nikon_CMS_pixels=0 if color_hit else int(pixels.shape[0] * pixels.shape[1]),
                                  nikon_CMS_cache_hit=color_hit, legacy_fit_calls=0)
    if flex:
        cache.last_receipt.update(flex_screen_cache_hit=color_hit, flex_fit_stride=native_step)
    if highlight:
        cache.last_receipt.update(highlight_reference_cache_hit=highlight_hit,
                                  highlight_reference_window=highlight_window,
                                  highlight_reference_measurement_calls=cache.reference_measurement_calls)
    return np.array(current, copy=True), reference, icc
