"""Display samples from NegPy measurement and color conversion."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
from negpy.kernel.caching.logic import calculate_config_hash
from negpy.services.rendering.engine import DarkroomEngine

from .diagnostics import event, span


_METER_PROXY_MAX_EDGE = 768
_HIGHLIGHT_DRAG_MAX_EDGE = 704


class _SampleForDisplay(DarkroomEngine):
    """Keep full support through the last source-dependent stage."""

    def __init__(self, rectangle, step, stage, preview_cache, *, process=None, finer_key=None,
                 companion_grid=None, point_print_lab=None, headroom=None):
        super().__init__()
        self.rectangle = rectangle
        self.step = step
        self.stage = stage
        self.preview_cache = preview_cache
        self.sampled = False
        self.bounds = None
        self.base_cache_hit = False
        self.base_sample_key = None
        self.meter_process = process
        self.finer_key = finer_key
        self.companion_grid = companion_grid
        self.companion = None
        self.point_print_lab = point_print_lab
        self.lab_schedule = None
        self.print_tile = None
        self.normalize_tile = False
        self.headroom = headroom

    def _normalize_source(self, img, context):
        import negpy_core
        from negpy.features.exposure.normalization import (
            effective_crosstalk_matrix, normalize_log_image, unmix_log_image,
        )

        logs = negpy_core.log_density_open(np.ascontiguousarray(img))
        unmix = effective_crosstalk_matrix(self.meter_process, context.process_mode)
        return normalize_log_image(unmix_log_image(logs, unmix), context.metrics["final_bounds"])

    def _reuse_meter(self, img, context, entry):
        metrics, context.active_roi = entry
        context.metrics.update(deepcopy(metrics))
        if self.lab_schedule is not None:
            # Keep authenticated source pixels until the tile has passed both
            # point stages. No intermediate full normalized raster is needed.
            self.normalize_tile = True
            return img
        if self.stage == "base":
            x0, y0, x1, y1 = self.rectangle
            img = img[y0:y1:self.step, x0:x1:self.step]
        return self._normalize_source(img, context)

    def _grids(self):
        grids = ((self.rectangle, self.step),)
        return grids if self.companion_grid is None else grids + (self.companion_grid,)

    def _run_stage(self, img, config, cache_field, processor_fn, context, pipeline_changed):
        from .live_worker import preview_checkpoint

        args = img, config, cache_field, processor_fn, context, pipeline_changed
        key = None
        meter = None
        if cache_field == "base":
            cert = self.preview_cache.certificate
            identity = cert.sha256, calculate_config_hash(config)
            if self.stage == "base" and not context.cache_stages:
                # Use the SDK's normalization key. It includes meter settings
                # and target revisions while excluding independent print edits.
                key = (*identity, tuple(self.rectangle), self.step)
                self.base_sample_key = key
                entry = self.preview_cache.original_base_samples.get(key)
                if entry is None and self.finer_key is not None and self.finer_key[:2] == key[:2]:
                    fine = self.preview_cache.original_base_samples.get(self.finer_key)
                    if fine is not None:
                        grid, metrics, roi = fine
                        fx, fy, _, _ = self.finer_key[-2]
                        fine_step = self.finer_key[-1]
                        x, y, _, _ = self.rectangle
                        stride = self.step // fine_step
                        preview_checkpoint()()
                        self.preview_cache.original_base_samples.move_to_end(self.finer_key)
                        sample = self.preview_cache.remember_original_base(
                            key, grid[(y - fy) // fine_step::stride, (x - fx) // fine_step::stride], metrics, roi)
                        entry = sample, metrics, roi
                if entry is not None:
                    preview_checkpoint()()
                    sample, metrics, context.active_roi = entry
                    context.metrics.update(deepcopy(metrics))
                    self.bounds = context.metrics["final_bounds"]
                    self.preview_cache.original_base_samples.move_to_end(key)
                    self.sampled = self.base_cache_hit = True
                    return sample, True
            # Only the untouched, authenticated source with identity geometry
            # can replay the SDK's pixel-local conversion on another output grid.
            meters = self.preview_cache.original_meter_values
            replay = self.meter_process is not None and not context.cache_stages and img is cert.source
            if replay and self.point_print_lab is not None:
                from .native_lab_scheduling import schedule_lab_grids

                self.lab_schedule = schedule_lab_grids(img.shape, self.point_print_lab, context, self._grids())
            meter = meters.get(identity) if replay else None
            if meter is None:
                before = context.metrics.copy() if replay else None
                with span("native_preview_full_grid_measure", source_shape=list(img.shape)):
                    result, changed = super()._run_stage(*args)
                # Inspection-only raster. The print stages need the scalar
                # measurements and receive normalized pixels as their input.
                context.metrics.pop("normalized_log", None)
                if before is not None:
                    base_metrics = {name: value for name, value in context.metrics.items()
                                    if name not in before or before[name] is not value}
                    meters[identity] = deepcopy(base_metrics), context.active_roi
                    while len(meters) > 3:
                        meters.popitem(last=False)
            else:
                cert.validate(img)
                with span("native_preview_replay_meter", sample_grid=self.stage == "base",
                          deferred_normalize=self.lab_schedule is not None):
                    result = self._reuse_meter(img, context, meter)
                changed = self.base_cache_hit = True
                meters.move_to_end(identity)
            self.bounds = context.metrics["final_bounds"]
        elif cache_field == "exposure":
            if self.headroom is None:
                render_print = processor_fn
            else:
                def render_print(pixels, ctx):
                    return self.headroom.render(pixels, ctx, processor_fn)

            if self.lab_schedule is not None:
                # This path admits only pointwise print with no intervening
                # CLAHE, so keep the input and run the original print callback
                # immediately before Lab on each haloed tile.
                self.print_tile = render_print
                return img, True
            with span("native_preview_grid_print", source_shape=list(img.shape)):
                return super()._run_stage(img, config, cache_field, render_print, context, pipeline_changed)
        elif cache_field == self.stage:
            from .native_lab_scheduling import sample_lab_grids

            check = preview_checkpoint()
            if not self.sampled:
                if self.print_tile is None:
                    samples = sample_lab_grids(img, config, processor_fn, context, self._grids(), check=check)
                else:
                    def print_and_lab(tile, ctx):
                        if self.normalize_tile:
                            tile = self._normalize_source(tile, ctx)
                        return processor_fn(self.print_tile(tile, ctx), ctx)

                    samples = self.lab_schedule.render(img, print_and_lab, context, check=check, fused=True)
                if samples is not None:
                    check()
                    self.sampled = True
                    if self.companion_grid is not None:
                        self.companion = samples[1]
                    return samples[0], True
            with span("native_preview_full_grid_lab"):
                result, changed = super()._run_stage(*args)
        else:
            return super()._run_stage(*args)
        preview_checkpoint()()
        if cache_field != self.stage:
            return result, changed
        if context.cache_stages or self.sampled:
            raise ValueError("原生测量预览需要独立的完整来源统计")
        if self.companion_grid is not None:
            (a, b, c, d), stride = self.companion_grid
            self.companion = np.array(result[b:d:stride, a:c:stride], order="C", copy=True)
        x0, y0, x1, y1 = self.rectangle
        visible = result if cache_field == "base" and meter is not None else result[y0:y1:self.step, x0:x1:self.step]
        sample = (
            self.preview_cache.remember_original_base(key, visible, context.metrics, context.active_roi)
            if key is not None
            else np.array(visible, order="C", copy=True)
        )
        self.sampled = True
        return sample, changed


def _render_sample(cache, config, export, rectangle, step, stage, *, finer_key=None,
                   companion_grid=None, headroom=None, print_ev=0.0):
    """Keep the display grid in the original SDK working transfer function."""
    from negpy.features.geometry.models import GeometryConfig

    from .density_preview_v44 import point_c41_supported
    from .native_print_metrics import collect_print_values

    processor = cache.session.processor
    path = cache.owner.records[0]["path"]
    loaded, _, _ = processor._load_source_f32(path, config)
    cache.certificate.validate(loaded)
    process = config.process if config.geometry == GeometryConfig() else None
    point_print_lab = None
    if stage == "lab":
        point_lab = replace(config.lab, chroma_denoise=0, sharpen=0, glow_amount=0, halation_strength=0)
        if point_c41_supported(replace(config, lab=point_lab)):
            point_print_lab = config.lab
    headroom_renderer = None
    if print_ev:
        from .native_headroom import NativePrintExposure

        headroom_renderer = NativePrintExposure(config, print_ev, strength=headroom or 0)
    elif headroom is not None:
        from .native_headroom import NativeHeadroom

        headroom_renderer = NativeHeadroom(headroom)
    engine = _SampleForDisplay(rectangle, step, stage, cache, process=process,
                              finer_key=finer_key, companion_grid=companion_grid,
                              point_print_lab=point_print_lab, headroom=headroom_renderer)
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
    sample = dict(encoded=image, target=target, bounds=dict(log_dense=list(engine.bounds.floors),
                  log_thin=list(engine.bounds.ceils)), native_print=processor.pure_native_print,
                  export=export, base_cache_hit=engine.base_cache_hit,
                  base_sample_key=engine.base_sample_key, tone_backend="cpu")
    if headroom_renderer is not None:
        sample["native_headroom"] = headroom_renderer.receipt
    if engine.companion is not None:
        from negpy.kernel.image.logic import working_oetf_encode

        # The alternate-engine caller proves the only active step after
        # Lab is the original working-space transfer function.
        (a, b, c, d), stride = companion_grid
        expected = len(range(b, d, stride)), len(range(a, c, stride)), 3
        if engine.companion.shape != expected:
            raise ValueError("原生相邻引擎预览没有保留约定的源像素网格")
        encoded = working_oetf_encode(engine.companion)
        encoded.setflags(write=False)
        sample["companion"] = {**sample, "encoded": encoded, "base_cache_hit": True}
    return sample


def _encode_sample(processor, image, export, target):
    from negpy.domain.models import ColorSpace, ExportFormat
    from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry, WORKING_COLOR_SPACE

    with span("native_preview_screen_encode"):
        if (export.export_fmt == ExportFormat.TIFF and export.export_bit_depth == 16
                and not export.icc_input_path and not export.icc_output_path
                and target != ColorSpace.GREYSCALE.value
                and ColorSpaceRegistry.get_icc_path(target) is not None):
            # Reuse the exact NegPy quantizer and ICC conversion. The display
            # needs their result without building and immediately unpacking a TIFF.
            pixels, icc = processor._export_pixels(image, 16, False, WORKING_COLOR_SPACE,
                                                   target, None, None)
            if pixels.dtype != np.uint16 or pixels.ndim != 3 or pixels.shape[2] != 3 or not icc:
                raise ValueError("NegPy 预览未生成带 ICC 的 RGB16 像素")
        else:
            from .cli import inspect_tiff

            payload, status = processor.encode_export(image, export, target)
            if payload is None:
                raise RuntimeError(status)
            pixels, icc = inspect_tiff(payload)
    pixels.setflags(write=False)
    return pixels, icc


def _meter_identity(cache, config):
    from negpy.features.exposure.models import TARGETS_REVISION

    return (cache.certificate.sha256, config.process, config.geometry,
            config.exposure.cast_removal_strength > 0.0, TARGETS_REVISION)


def _meter_drag_changes_process(cache, config):
    from .native_metering_controls import PROCESS_DEFAULTS

    baseline = (cache.original_screen_base[0][1] if cache.original_screen_base is not None
                else cache.pristine_config.process)
    names = (key.removeprefix("process.") for key in PROCESS_DEFAULTS)
    return any(getattr(config.process, name) != getattr(baseline, name) for name in names)


def _render_meter_drag(owner, request, recipe, cache, config, rectangle, size, edge):
    """Run the NegPy meter on a bounded area-resampled source during a meter drag."""
    import negpy_core
    from negpy.domain.interfaces import PipelineContext
    from negpy.kernel.system.config import render_scale_factor

    from .live_worker import preview_checkpoint
    from .measurement_method import original_config
    from .native import native_export_settings
    from .output_orientation import orient_codes

    checkpoint = preview_checkpoint()
    source = cache.certificate.source
    digest = cache.certificate.validate()
    height, width = source.shape[:2]
    step = (max(height, width) + edge - 1) // edge
    proxy_height, proxy_width = ((dimension + step - 1) // step for dimension in (height, width))

    def mapped(box):
        return [round(value * proxy / full) for value, proxy, full in
                zip(box, (proxy_width, proxy_height) * 2, (width, height) * 2, strict=True)]

    roi = mapped(recipe["roi"])
    x0, y0, x1, y1 = mapped(rectangle)
    if min(roi[2] - roi[0], roi[3] - roi[1]) < 8 or min(x1 - x0, y1 - y0) < 1:
        # Preserve separately drawn small analysis regions on the ordinary route.
        return None

    checkpoint()
    retained = cache.native_meter_proxy
    if step == 1:
        proxy = source
    elif retained is None or retained[:2] != (digest, step):
        with span("native_meter_proxy_source", source_shape=list(source.shape), sample_step=step):
            proxy = negpy_core.resize_area(source, (proxy_width, proxy_height))
            proxy = np.frombuffer(proxy.tobytes(), dtype=np.float32).reshape(proxy_height, proxy_width, 3)
        checkpoint()
        cache.native_meter_proxy = (digest, step, proxy)
    else:
        proxy = retained[2]

    processor = cache.session.processor
    cam_xyz, camera_wb = processor._cam_xyz_by_path.get(owner.records[0]["path"], (None, None))
    export = native_export_settings(config)
    proxy_config = original_config(replace(config, export=export), roi, proxy.shape)
    context = PipelineContext(
        original_size=proxy.shape[:2], scale_factor=render_scale_factor(proxy.shape),
        process_mode=config.process.process_mode, wants_uv_grid=False, cache_stages=False,
        cam_xyz=cam_xyz, camera_wb=camera_wb,
    )
    with span("native_meter_proxy_render", source_shape=list(proxy.shape)):
        image = DarkroomEngine().process(proxy, proxy_config, digest, context)
    checkpoint()
    if image.shape != proxy.shape:
        raise ValueError("NegPy 拖动预览的源像素网格发生变化")
    pixels, icc = _encode_sample(processor, image[y0:y1, x0:x1], export, export.export_color_space)
    pixels = orient_codes(pixels, recipe.get("output_orientation"))
    current = owner._finish(pixels, recipe, "native", fit=True)
    current = np.frombuffer(current.tobytes(), dtype=np.uint16).reshape(current.shape)

    cache.certificate.validate()
    owner._validate(request)
    owner.output_size, owner.sample_step, owner.viewport_result = size, step, None
    owner.reference_label, owner.metering_receipt = "", {}
    owner.preview_quality = dict(
        mode="native-original-meter-proxy", approximate=True, interactive_frame=True,
        label="拖动预览", full_support_measurement=False, sample_step=step, full_output_size=size,
    )
    return current, current, icc


def _render_metal_tone_sample(cache, config, export, rectangle, step, *, finer_key=None):
    from .live_worker import preview_checkpoint
    from .native_metal_preview import NativeMetalTone

    checkpoint = preview_checkpoint()
    meter_identity = (*_meter_identity(cache, config), tuple(rectangle), step)
    slot = "interactive_metal_tone" if finer_key is not None else "native_metal_tone"
    metal = getattr(cache, slot)
    if metal is None:
        metal = NativeMetalTone()
        setattr(cache, slot, metal)
    checkpoint()
    result = metal.render(cache, config, export, meter_identity)
    checkpoint()
    if result is None:
        result = _render_sample(cache, config, export, rectangle, step, "base", finer_key=finer_key)
        checkpoint()
        metal.qualify(cache, config, rectangle, step, meter_identity, result)
        checkpoint()
    return result


def _matching_highlight_reference(cached, highlight):
    reference = cached.get("highlight_reference") if cached is not None else None
    return (reference is not None and reference["roi"] == highlight["reference_roi"]
            and reference.get("reference_density_offset", 0.0)
            == highlight.get("reference_density_offset", 0.0))


def _highlight_sample(cache, config, cached, highlight, point, rectangle, step, stage, *,
                      subgrid=None, film=None, print_ev=0.0):
    from negpy.features.exposure.models import TARGETS_REVISION

    from .density_preview_v44 import _key
    from .highlight_sample_v46 import compose_native_sample
    from .highlight_workflow import _highlight_reference_render_config, highlight_reference_config
    from .positive_float import working_icc_bytes

    source = cache.session.source
    roi = highlight["reference_roi"]
    offset = highlight.get("reference_density_offset", 0.0)
    reference = cached.get("highlight_reference")
    reference_hit = _matching_highlight_reference(cached, highlight)
    if not reference_hit:
        with span("native_preview_highlight_reference"):
            # Metering reads only RAW and negative process coordinates. Paper,
            # exposure and the displayed grid can change without re-reading them.
            meter_key = (cache.certificate.sha256, tuple(roi), calculate_config_hash(config.process), TARGETS_REVISION)
            measurements = cache.original_highlight_measurements
            measured = measurements.get(meter_key)
            if measured is None:
                cache._progress("highlight_reference", .55, "正在测量高光参考…")
                settings, window = highlight_reference_config(
                    source, config, roi, reference_density_offset=offset, film=film,
                )
                measurements[meter_key] = settings.process, window
                while len(measurements) > 3:
                    measurements.popitem(last=False)
                cache.reference_measurement_calls += 1
            else:
                measurements.move_to_end(meter_key)
                process, window = measured
                settings = _highlight_reference_render_config(config, process, offset)
            reference = _render_sample(cache, settings, cached["export"], rectangle, step, stage, print_ev=print_ev)
        if (reference["target"] != cached["target"]
                or reference["encoded"].dtype != cached["encoded"].dtype):
            raise ValueError("高光参考的色彩空间或像素类型与原生正像不一致")
        reference.update(roi=roi, reference_density_offset=offset,
                         window=dict(log_dense=list(window.log_dense), log_thin=list(window.log_thin)))
        cached["highlight_reference"] = reference
    key = _key([highlight, point, subgrid])
    slot = "interactive_result" if subgrid is not None else "result"
    result = reference.get(slot)
    if result is None or result["key"] != key:
        cache._progress("sampled_color", .86, "正在合成高光找回预览…")
        masks = {
            name: cache._highlight_mask(cache.source_key, source.shape, name, highlight.get(name), rectangle, step)
            for name in ("protection", "effect_area")
        }
        encoded, reference_encoded = cached["encoded"], reference["encoded"]
        protection = masks["protection"].mask if masks["protection"] else None
        effect = masks["effect_area"].mask if masks["effect_area"] else None
        if subgrid is not None:
            y, x, stride = subgrid
            region = slice(y, None, stride), slice(x, None, stride)
            encoded = np.ascontiguousarray(encoded[region])
            reference_encoded = np.ascontiguousarray(reference_encoded[region])
            if protection is not None:
                protection = np.ascontiguousarray(protection[region])
            if effect is not None:
                effect = np.ascontiguousarray(effect[region])
        with span("native_preview_highlight_join", pixels=int(encoded.shape[0] * encoded.shape[1])):
            image, _ = compose_native_sample(
                encoded, icc=working_icc_bytes(), reference_encoded=reference_encoded,
                highlight=highlight, point=point,
                protection_mask=protection, effect_mask=effect,
            )
            pixels, icc = _encode_sample(cache.session.processor, image, cached["export"], cached["target"])
        result = reference[slot] = dict(key=key, pixels=pixels, icc=icc)
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
    locator = bool(request.get("crop_locator"))
    crop_grid = locator and request.get("crop_source_grid") is True
    engine_proxy = request.get("engine_proxy") is True
    if ((mode != "native" and not nikon and not flex) or recipe.get("measurement_method") != "negpy"
            or recipe.get("source_mode")
            or request.get("full_resolution")
            or (request.get("repair_plan") is not None
                and ((locator and not crop_grid) or (engine_proxy and mode == "native")))
            or (locator and (mode != "native" or recipe["crop"] or "viewport" in request))
            or not recipe.get("roi")):
        return None
    film = recipe.get("native_film") if mode == "native" else None
    if film and film["strength"] == 0:
        film = None
    from .native_headroom import native_headroom_active, native_headroom_strength, native_local_highlight
    from .professional_controls import native_print_ev

    headroom = mode == "native" and native_headroom_active(recipe)
    print_ev = native_print_ev(recipe) if mode == "native" else 0.0
    headroom_strength = native_headroom_strength(recipe) if headroom else None
    highlight = native_local_highlight(recipe) if mode == "native" else None
    point = recipe.get("native_float_controls") if mode == "native" else None
    if (film and not (highlight or headroom or print_ev)) or (point and not highlight):
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
    from negpy.features.exposure import models as exposure_models
    from negpy.features.lab.models import SharpenMethod
    from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG

    from .density_preview_v44 import _key, point_c41_supported
    from .draft_preview import _anchor_recipe, _saved_color_valid
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
    if (not locator and cache.certificate is None and mode in owner.manifest["results"]
            and _anchor_recipe(recipe, mode) == _anchor_recipe(owner.recipe, mode)
            and _saved_color_valid(owner.manifest, mode)):
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
    native_tone = mode == "native" and request.get("tone_edit") is True and not engine_proxy
    edge = request.get("interactive_edge")
    fit_edge = request.get("fit_edge", 1400) if native_tone else 1400
    if engine_proxy:
        fit_edge = edge
    viewport = None
    if "viewport" in request:
        viewport = checked_viewport(request["viewport"], size)
        rectangle = source_rectangle_for_viewport(rectangle, direction, viewport)
        step = viewport["sample_step"]
    else:
        step = max(1, (max(size) + fit_edge - 1) // fit_edge)
    phase = oriented_source_sample_roi(rectangle, direction, step)
    lab = wanted.lab
    approximate = (
        (native_tone or engine_proxy or crop_grid) and stage == "lab"
        and viewport is None and step > 1
        and lab.clahe_strength == lab.glow_amount == lab.halation_strength == 0
        and (lab.sharpen == 0 or lab.sharpen_method == SharpenMethod.USM)
    )
    if engine_proxy and (highlight or stage == "lab" and not approximate):
        return None
    repair_proxy = (request.get("defer_repair") is True and native_tone and viewport is None
                    and edge is not None and (approximate or stage == "base"))

    with span("native_preview_source"):
        session = cache.prepare_source()
    base = cache.pristine_config
    config = native_config(
        replace(base, exposure=replace(base.exposure, paper_profile=recipe["paper"])),
        controls)
    if film is not None:
        from .film_profiles import film_config

        # SDK normalization applies this matrix to the original source exactly
        # once. The reference meter is bound to the same film configuration.
        config = film_config(config, film)
    config = original_config(config, roi, shape)
    finer_key = None
    interactive = engine_proxy
    if (edge and native_tone and viewport is None
            and not highlight and (approximate or stage == "base")
            and cache.original_screen_base is not None):
        identity, normal_key = cache.original_screen_base
        factor = (max(size) + edge * step - 1) // (edge * step)
        if (normal_key[-2:] == (tuple(phase), step)
                and normal_key in cache.original_base_samples and identity == _meter_identity(cache, config)):
            interactive = True
            if factor > 1:
                # Keep the source phase under rotations and reflections.
                finer_key = normal_key
                step *= factor
                phase = oriented_source_sample_roi(rectangle, direction, step)
    if repair_proxy and not interactive:
        # No settled frame is required to render the first repaired-photo
        # drag. Reuse the original meter and sample only the small display grid.
        step *= max(1, (max(size) + edge * step - 1) // (edge * step))
        phase = oriented_source_sample_roi(rectangle, direction, step)
        interactive = True
    if (edge and edge <= _METER_PROXY_MAX_EDGE and native_tone and viewport is None
            and not interactive and not highlight and not headroom and not print_ev
            and (request.get("metering_edit") is True or _meter_drag_changes_process(cache, config))):
        drag = _render_meter_drag(owner, request, recipe, cache, config, rectangle, size, edge)
        if drag is not None:
            return drag
    # Legacy Flex needs exactly the same ROI-origin fit grid as saving.
    from .endpoint_study import FIT_REFERENCE_STRIDE

    fit_grid = (roi, FIT_REFERENCE_STRIDE)
    screen_grid = (phase, step)
    native_rect, native_step = fit_grid if flex else screen_grid
    sample_identity = ["native-original", cache.certificate.sha256, exposure_models.TARGETS_REVISION]
    if headroom:
        sample_identity.extend(("native-headroom", headroom_strength))
    if print_ev:
        sample_identity.extend(("native-print-exposure", print_ev))
    key = _key([*sample_identity, native_rect, native_step, config.to_dict(), stage])
    # The quick engine frame must leave reusable exact renders available for
    # the following refinement.
    samples = cache.engine_proxy_samples if engine_proxy else cache.samples
    sample_limit = 2 if engine_proxy else 3
    # Reuse an exact sample when both sides of a requested highlight blend
    # are ready. A cached baseline alone would still require full-grid Lab
    # for the reference and give the two sides different neighbourhoods.
    exact = samples.get(key)
    if approximate and exact is not None:
        approximate = bool(highlight and not _matching_highlight_reference(exact, highlight))
    if approximate:
        # Keep SDK metering on the full source. Only the screen's chroma
        # denoise and fixed-mask USM operate at the reduced pixel scale.
        config = replace(config, lab=replace(config.lab,
            chroma_denoise=config.lab.chroma_denoise / native_step,
            sharpen_radius=config.lab.sharpen_radius / native_step))
        stage = "base"
        # The reference and baseline must share the CPU Lab and tone path.
        backend = ["highlight-cpu"] if highlight else []
        key = _key([*sample_identity, native_rect, native_step, config.to_dict(), stage, *backend])
    export = native_export_settings(config)
    if highlight:
        from .highlight_workflow import _configuration

        _configuration(config, export, bounds_override=None, film=film)
    prefer_metal = False
    if approximate and mode == "native" and not highlight and not headroom and not print_ev:
        from .native_metal_preview import metal_tone_allowed

        prefer_metal = metal_tone_allowed(config, native_rect, native_step, controls)
    cached = samples.get(key)
    if cached is not None and cached["tone_backend"] == "metal" and not prefer_metal:
        cached = None
    companion = cache.original_companion
    companion_hit = bool(cached is None and not engine_proxy and companion is not None and companion[0] == key)
    if companion_hit:
        cache.original_companion = None
        cached = samples[key] = companion[1]
        event("native_preview_companion_grid_hit", mode=mode, step=native_step)
    saved_flex_binding = None
    if cached is None and flex and viewport is None and not approximate:
        from .flex_legacy_screen import read_saved_native_fit, saved_native_fit_source

        binding = saved_native_fit_source(owner, recipe, cache.certificate, config)
        if binding is not None:
            record, identity, _, _ = binding
            saved_key = _key([key, record["sha256"], identity])
            saved = samples.get(saved_key)
            if saved is None:
                try:
                    cache._progress("density_support", .28, "正在读取已保存的原生拟合参考…")
                    saved = read_saved_native_fit(binding, export)
                except (OSError, ValueError, IndexError) as exc:
                    event("flex_preview_saved_fit_fallback", error=type(exc).__name__)
            if saved is not None:
                key, cached, saved_flex_binding = saved_key, saved, binding
                samples[key] = cached
                event("flex_preview_saved_fit_reused", stride=native_step)
    hit = cached is not None
    if cached is None:
        cache._progress("density_support", .28, "正在生成 NegPy 转换预览…")
        companion_grid = None
        if (stage == "lab" and (nikon or flex) and viewport is None and not edge
                and not engine_proxy and not headroom):
            from negpy.features.toning.models import ToningConfig
            from .output_orientation_cache import cache_budget_bytes

            other_rect, other_step = screen_grid if flex else fit_grid
            a, b, c, d = other_rect
            size_bytes = len(range(a, c, other_step)) * len(range(b, d, other_step)) * 3 * np.dtype(np.float32).itemsize
            # C41 with no finishing geometry is already proven above. The
            # following guard makes the Lab output pointwise through export.
            if (config.toning == ToningConfig()
                    and 0 < size_bytes <= min(16 * 1024**2, cache_budget_bytes())):
                companion_key = _key([*sample_identity, other_rect, other_step,
                                      config.to_dict(), stage])
                if companion_key != key and companion_key not in samples:
                    companion_grid = other_rect, other_step
        cached = (
            _render_metal_tone_sample(cache, config, export, native_rect, native_step, finer_key=finer_key)
            if prefer_metal
            else _render_sample(cache, config, export, native_rect, native_step, stage,
                                finer_key=finer_key, companion_grid=companion_grid,
                                headroom=headroom_strength, print_ev=print_ev)
        )
        companion_sample = cached.pop("companion", None)
        if companion_sample is not None:
            cache.original_companion = companion_key, companion_sample
        samples[key] = cached
        hit = cached["base_cache_hit"]
        cache.measurement_calls += int(not hit)
    else:
        samples.move_to_end(key)
    while len(samples) > sample_limit:
        samples.popitem(last=False)
    if (mode == "native" and not engine_proxy and finer_key is None
            and viewport is None and not highlight and stage == "base"):
        fit_base = cached["base_sample_key"]
        if fit_base in cache.original_base_samples:
            cache.original_screen_base = (_meter_identity(cache, config), fit_base)
    if highlight:
        highlight_subgrid = None
        highlight_phase, highlight_step = phase, step
        if (edge and native_tone and viewport is None and not interactive
                and _matching_highlight_reference(cached, highlight)):
            # Both float grids and both masks have the same original source
            # phase. Strength is per-pixel, so decimate all four before blend.
            target = min(edge, _HIGHLIGHT_DRAG_MAX_EDGE) * step
            factor = (max(size) + target - 1) // target
            step *= factor
            phase = oriented_source_sample_roi(rectangle, direction, step)
            highlight_subgrid = ((phase[1] - highlight_phase[1]) // highlight_step,
                                 (phase[0] - highlight_phase[0]) // highlight_step, factor)
            interactive = True
        pixels, icc, highlight_hit, highlight_window = _highlight_sample(
            cache, config, cached, highlight, point, highlight_phase, highlight_step, stage,
            subgrid=highlight_subgrid, film=film, print_ev=print_ev)
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
    if interactive or locator:
        owner.reference_label = ""
    else:
        reference = cache._reference(recipe, mode, phase, step, direction, pixels, icc,
                                     share=fit_edge > 1400 and viewport is None, bounded=viewport is not None)
    # Store just the last finish for each small RGB16 fit sample. An unchanged
    # release can publish the actual dragged pixels, while other recipes still
    # run the appropriate final colour controls.
    retain_finish = interactive and mode == "native" and not highlight
    drag_finish = cached.get("drag_finish") if mode == "native" and not highlight else None
    if retain_finish or drag_finish is not None:
        finish = _key([direction, recipe["exposure_ev"], recipe.get("common_positive_controls"),
                       recipe.get("output_controls", {}).get(mode), recipe.get("post_controls", {}).get(mode)])
    else:
        finish = None
    if drag_finish is not None and drag_finish[0] == finish:
        current = drag_finish[1]
    else:
        current = owner._finish(pixels, recipe, mode, fit=viewport is None)
        if retain_finish:
            current = np.frombuffer(current.tobytes(), dtype=np.uint16).reshape(current.shape)
            cached["drag_finish"] = (finish, current)
    cache.certificate.validate()
    owner._validate(request)
    if saved_flex_binding is not None:
        from .flex_draft_preview_v40 import _file_identity

        record, identity, _, _ = saved_flex_binding
        if _file_identity(record["path"]) != identity:
            raise ValueError("Flex 保存参考在草稿预览期间发生变化")
    owner.output_size, owner.sample_step, owner.viewport_result = size, step, viewport
    if mode == "native" and not highlight:
        owner.native_print = cached["native_print"]
    owner.metering_receipt = dict(roi=list(roi), metering=None, automatic=None, report={})
    if highlight:
        owner.metering_receipt["highlight_reference_roi"] = list(highlight["reference_roi"])
    owner.preview_quality = {
        "mode": "nikon-original-screen" if nikon else "flex-original-screen" if flex else "native-original-screen",
        "approximate": approximate,
    }
    if approximate:
        owner.preview_quality.update(
            label="交互预览 · 100%查看精细效果",
            full_support_measurement=True, sample_step=step, full_output_size=size,
            native_tone_backend=cached["tone_backend"],
        )
        event("native_preview_tone_backend", backend=cached["tone_backend"], sample_step=step,
              measurement_cache_hit=hit)
    if interactive:
        reference = current
    if (nikon or flex) and viewport is None:
        current, reference, icc = owner._alternate_fit_frame(request, mode, current, reference, icc)
        interactive = owner.preview_quality.get("interactive_frame", False)
    elif interactive:
        label = "引擎快速预览" if engine_proxy else "拖动预览"
        owner.preview_quality.update(approximate=True, interactive_frame=True,
                                     label=label, sample_step=step, full_output_size=size)
    cache.last_receipt = dict(
        route="native-original-full-meter-then-screen", full_support_measured=True,
        measurement_cache_hit=hit, measurement_calls=cache.measurement_calls,
        source_prepare_calls=cache.source_prepare_calls, full_Native_exports=0,
        sample_pixels=int(pixels.shape[0] * pixels.shape[1]), full_source_shape=list(shape),
        sampled_after=stage, measurement_window=cached["bounds"],
        retained_source_bytes=int(session.source.nbytes),
    )
    if approximate:
        cache.last_receipt["native_tone_backend"] = cached["tone_backend"]
    if headroom:
        cache.last_receipt["native_headroom"] = cached["native_headroom"]
    if print_ev:
        from .native_headroom import print_exposure_receipt

        cache.last_receipt["native_print_exposure"] = print_exposure_receipt(print_ev)
    if companion_hit:
        cache.last_receipt["native_companion_grid_hit"] = True
    if nikon:
        cache.last_receipt.update(nikon_CMS_pixels=0 if color_hit else int(pixels.shape[0] * pixels.shape[1]),
                                  nikon_CMS_cache_hit=color_hit, legacy_fit_calls=0)
    if flex:
        cache.last_receipt.update(flex_screen_cache_hit=color_hit, flex_fit_stride=native_step,
                                  flex_saved_native_fit=saved_flex_binding is not None)
    if highlight:
        cache.last_receipt.update(highlight_reference_cache_hit=highlight_hit,
                                  highlight_reference_window=highlight_window,
                                  highlight_reference_measurement_calls=cache.reference_measurement_calls)
    if interactive or locator:
        return current, current, icc
    return (current if drag_finish is not None and drag_finish[0] == finish
            else np.array(current, copy=True)), reference, icc
