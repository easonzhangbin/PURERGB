"""Explicit original NegPy and experimental zonal measurement choices."""

from copy import deepcopy
from dataclasses import replace
from .native_metering_controls import process_controls_active

NATIVE = "negpy"
ZONAL = "zonal"


def checked_method(recipe):
    method = recipe.get("measurement_method")
    if method is not None and method not in (NATIVE, ZONAL):
        raise ValueError("未知密度测量算法")
    if method == NATIVE and (recipe.get("native_spatial_controls") or recipe.get("density_metering")):
        raise ValueError("NegPy 原生测量不能同时启用实验分区统计")
    values = recipe.get("native_controls", {}).get("values", {})
    if (recipe.get("native_spatial_controls") or recipe.get("density_metering")) and process_controls_active(values):
        raise ValueError("实验分区测量暂不支持原生白点／黑点及范围截断，请复位后切换。")
    return method


def native_recipe(recipe):
    """Keep the ROI and creative edits while leaving experimental support dormant."""
    result = deepcopy(recipe)
    spatial = result.pop("native_spatial_controls", None)
    result.pop("density_metering", None)
    result.pop("bw_metering", None)
    if spatial and spatial.get("lab"):
        values = result.setdefault("native_controls", dict(version=1, values={}))["values"]
        for key, value in spatial["lab"].items():
            values["lab." + key] = deepcopy(value)
    result["measurement_method"] = NATIVE
    return result


def original_config(config, roi, shape):
    """Select the exact requested rectangle and let NegPy measure its own bounds."""
    h, w = shape[:2]
    a, b, c, d = roi
    return replace(config, process=replace(config.process,
        analysis_rect=(a / w, b / h, c / w, d / h), analysis_buffer=0.,
        local_floors=(0., 0., 0.), local_ceils=(0., 0., 0.),
    ))


def original_window(source, config, roi):
    """Read the bounds from the unchanged upstream normalization processor."""
    import numpy as np
    from negpy.domain.interfaces import PipelineContext
    from negpy.features.exposure.processor import NormalizationProcessor
    from negpy.kernel.system.config import render_scale_factor
    from .density_matrix import DensityWindow
    from .engines import Analysis

    config = original_config(config, roi, source.shape)
    context = PipelineContext(original_size=source.shape[:2],
        scale_factor=render_scale_factor(source.shape), process_mode=config.process.process_mode,
        wants_uv_grid=False, cache_stages=False)
    NormalizationProcessor(config.process).process(source, context)
    bounds = context.metrics["final_bounds"]
    dense, thin = tuple(map(float, bounds.floors)), tuple(map(float, bounds.ceils))
    analysis = Analysis(tuple(np.power(10., dense)), tuple(np.power(10., thin)), tuple(roi), None)
    return DensityWindow("negpy-original", analysis, dense, thin, dict(
        source="Unchanged NegPy NormalizationProcessor with active native process controls",
        luma_range_clip=config.process.luma_range_clip,
        color_range_clip=config.process.color_range_clip,
        norm_density_range=float(context.metrics["norm_density_range"]),
        original_same_pixel_bound_rule=True,
        auxiliary_meters="unchanged NegPy native export",
    ))
