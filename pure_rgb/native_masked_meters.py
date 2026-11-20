"""Opt-in CPU auxiliary meters on complete valid original analysis-grid cells.

Partial masks keep spatial topology for the activity gate. Only after that
gate may a selected population be passed to the original scalar aggregators.
No image is packed for a renderer, no source pixels are changed, and no legacy
provider/default calls this module.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json

import numpy as np

from .density_matrix import check_density_domain
from .metering_masks import validate_metering_mask
from .metering_regions import metering_process, select_metering_roi
from .metering_snapshot import (
    FrozenMeteringBundle,
    _check_source,
    restore_zonal_metering,
)

from negpy.domain.types import LUMA_B, LUMA_G, LUMA_R
from negpy.features.exposure import normalization as norm
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, RenderIntent
from negpy.features.geometry.models import GeometryConfig
from negpy.features.process.models import ProcessConfig, ProcessMode
from negpy.features.retouch.models import RetouchConfig


SCHEMA = "pure-rgb-native-masked-auxiliary-meters"
VERSION = 1
PROVIDER = "original-grid-masked-spatial-aux-v1"
BOUNDS_KEYS = ("log_bounds_base", "pre_trim_bounds", "final_bounds", "anchor_bounds")
SCALAR_KEYS = (
    "norm_density_range",
    "metered_anchor",
    "textural_range",
    "shadow_point",
    "highlight_point",
)
METRIC_KEYS = set(BOUNDS_KEYS + SCALAR_KEYS) | {
    "neutral_axis_refs",
    "scan_clip_fractions",
    "density_above_base",
}


@dataclass(frozen=True)
class NativeMaskedMeterResult:
    metrics: dict
    receipt: dict
    effective_config: object
    sha256: str


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _array_sha(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def _text(bundle):
    return bundle.to_json() if isinstance(bundle, FrozenMeteringBundle) else bundle


def process_config_sha256(process):
    """Canonical JSON of dataclasses.asdict(ProcessConfig), with no NumPy values."""
    return _digest(asdict(process))


def validate_native_masked_config(config):
    """Meter-domain gates; renderer adapters add their own post-spatial gates."""
    process = config.process
    check_density_domain(process)
    if process.process_mode != ProcessMode.C41 or process.positive_source:
        raise ValueError("Masked Native meters support CPU C41 negative print only")
    if config.exposure.render_intent != RenderIntent.PRINT:
        raise ValueError("Masked Native meters require print intent")
    if config.geometry != GeometryConfig() or config.retouch != RetouchConfig():
        raise ValueError("Masked Native meters require no-op geometry and retouch")
    if config.local.masks:
        raise ValueError("Masked Native meters do not support local masks")
    if config.exposure.contrast_mask_strength != 0 or config.exposure.diffusion != 0:
        raise ValueError("Masked Native meters do not support spatial print planes")
    if config.lab.clahe_strength != 0:
        raise ValueError("Masked Native meters do not support CLAHE")
    if process.sensor_matrix is not None:
        raise ValueError("Masked Native meters do not support sensor correction")
    default = ProcessConfig()
    if any(
        getattr(process, name) != getattr(default, name)
        for name in asdict(default)
        if name.startswith("fade_")
    ):
        raise ValueError("Masked Native meters do not support fade restoration")
    if (
        process.use_luma_average
        or process.use_color_average
        or process.use_cast_average
        or process.lock_bounds
        or process.locked_neutral_axis is not None
        or process.manual_dmin is not None
        or process.manual_dmax is not None
        or process.film_base is not None
    ):
        raise ValueError("Masked Native meters do not support pooled/manual base state")


def frozen_process_for_window(process, window, shape):
    """Freeze the supplied density window and exact source-pixel ROI edges."""
    selection = select_metering_roi(
        shape, window.analysis.roi, min_side=2, min_area_fraction=1.0
    )
    result = metering_process(process, window, selection)
    if not result.is_local_initialized:
        raise ValueError("Frozen density floors collide with Native's zero sentinel")
    return result


def native_masked_metrics_payload(metrics):
    """JSON-native metric representation; bounds are not retained by alias."""
    if not isinstance(metrics, dict) or set(metrics) != METRIC_KEYS:
        raise ValueError("Malformed Native masked metric fields")
    result = {}
    for key in BOUNDS_KEYS:
        value = metrics[key]
        if not isinstance(value, norm.LogNegativeBounds):
            raise ValueError("Expected original LogNegativeBounds")
        result[key] = {
            "floors": [float(v) for v in value.floors],
            "ceils": [float(v) for v in value.ceils],
        }
    for key in SCALAR_KEYS:
        if type(metrics[key]) is not float or not np.isfinite(metrics[key]):
            raise ValueError(
                "Native masked scalar metrics must be finite Python floats"
            )
        result[key] = metrics[key]
    axis = metrics["neutral_axis_refs"]
    if axis is not None:
        if not isinstance(axis, tuple) or len(axis) != 4:
            raise ValueError("Malformed neutral axis")
        axis = [
            list(axis[0]),
            list(axis[1]),
            None if axis[2] is None else list(axis[2]),
            axis[3],
        ]
    result["neutral_axis_refs"] = axis
    result["scan_clip_fractions"] = list(metrics["scan_clip_fractions"])
    result["density_above_base"] = metrics["density_above_base"]
    _canonical(result)
    return result


def _luma(grid, bounds=None):
    channels = []
    if bounds is None:
        channels = [grid[..., ch] for ch in range(3)]
    else:
        # The effective bounds contain Python floats, preserving _meter_norm's
        # original float32 operation domain rather than NumPy float64 scalars.
        for channel, (floor, floor_wide, span, span_wide) in enumerate(
            norm._meter_norm(bounds)
        ):
            values = grid[..., channel]
            difference = (
                values.astype(np.float64) - floor
                if floor_wide
                else values - np.float32(floor)
            )
            values = (
                (difference.astype(np.float64) / span).astype(np.float32)
                if (floor_wide or span_wide)
                else difference / np.float32(span)
            )
            channels.append(values)
    return (
        np.float32(LUMA_R) * channels[0]
        + np.float32(LUMA_G) * channels[1]
        + np.float32(LUMA_B) * channels[2]
    )


def _spatial_activity(luma, valid):
    """Original sector topology, with only completely valid sectors voting.

    Partial-mask policy is independently versioned. Its vote fraction uses
    complete valid sectors as denominator; too few active sectors falls back
    to all valid cells, never to excluded cells.
    """
    block, threshold, minimum = norm._activity_gate()
    side = 2 * block
    h, w = valid.shape
    sector_rows, sector_cols = h // side, w // side
    selected = np.zeros_like(valid)
    complete = active = 0
    for row in range(sector_rows):
        for column in range(sector_cols):
            y, x = row * side, column * side
            if not valid[y : y + side, x : x + side].all():
                continue
            complete += 1
            means = []
            for by, bx in ((0, 0), (0, block), (block, 0), (block, block)):
                square = luma[y + by : y + by + block, x + bx : x + bx + block]
                total = np.float32(0)
                for values in square:
                    total = np.float32(total + np.sum(values, dtype=np.float32))
                means.append(np.float32(total / np.float32(block * block)))
            if np.float32(max(means) - min(means)) > np.float32(threshold):
                active += 1
                selected[y : y + side, x : x + side] = True
    fraction = active / complete if complete else 0.0
    fallback = not complete or fraction < minimum
    if fallback:
        selected = valid.copy()
    return selected, {
        "activity_block": block,
        "sector_shape_cells": [side, side],
        "geometric_sector_count": sector_rows * sector_cols,
        "complete_valid_sector_count": complete,
        "active_sector_count": active,
        "active_fraction_of_complete_valid_sectors": fraction,
        "minimum_active_fraction": minimum,
        "threshold": threshold,
        "fallback_all_valid_cells": fallback,
        "fallback_reason": "no-complete-valid-sectors"
        if not complete
        else ("active-fraction-below-threshold" if fallback else None),
        "selected_cell_count": int(np.count_nonzero(selected)),
        "selected_cell_mask_sha256": _array_sha(selected),
    }


def _packet(grid, support):
    # _block_median_grid early-returns at b<=1 or width<b. At width=1 it
    # therefore never re-prefilters; Rust textured_cells has ws=0 and returns
    # every supplied cell. This is scalar aggregation AFTER the real 2D gate.
    return np.ascontiguousarray(grid[support].reshape(-1, 1, 3), dtype=np.float32)


def analyze_native_masked_meters(source, config, density_bundle_json, current_identity):
    """Measure auxiliary stats once against a V15 source-bound density window."""
    validate_native_masked_config(config)
    restored = restore_zonal_metering(_text(density_bundle_json), current_identity)
    _check_source(source, restored.source_identity)
    before = restored.source_identity["source_pixels_sha256"]
    try:
        process = frozen_process_for_window(
            config.process, restored.window, source.shape
        )
        effective_config = replace(config, process=process)
        roi = tuple(restored.config["roi"])
        x0, y0, x1, y1 = roi
        rh, rw = y1 - y0, x1 - x0
        support = np.ones((rh, rw), dtype=bool)
        if restored.selection_mask is not None:
            support &= validate_metering_mask(
                restored.selection_mask, source.shape, roi
            )
        for bx0, by0, bx1, by1 in restored.receipt["effective_excluded_rois"]:
            support[by0 - y0 : by1 - y0, bx0 - x0 : bx1 - x0] = False
        available = int(np.count_nonzero(support))
        requested_b = int(
            np.ceil(max(rh, rw) / int(EXPOSURE_CONSTANTS["analysis_grid"]))
        )
        b = (
            requested_b
            if requested_b > 1 and rh >= requested_b and rw >= requested_b
            else 1
        )
        gh, gw = rh // b, rw // b
        hb, wb = gh * b, gw * b
        valid = support[:hb, :wb].reshape(gh, b, gw, b).all(axis=(1, 3))
        retained_cells = int(np.count_nonzero(valid))
        retained = retained_cells * b * b
        if (
            not retained_cells
            or retained / (rh * rw) < restored.config["min_area_fraction"]
        ):
            raise ValueError("Insufficient complete valid Native analysis-grid support")
        from .immutable_source_v44 import certificate_for
        from .film_density_domain_v45 import film_domain_for

        certificate = certificate_for(source)
        domain = film_domain_for(source)
        sensor_source = domain.original if domain is not None else source
        if domain is not None:
            grid = norm._block_median_grid(source[y0:y1, x0:x1])
        else:
            grid = (certificate.prefilter_grid(list(roi)) if certificate is not None
                    else norm.prefilter_log_grid(source, (y0, y1, x0, x1), 0.0))
        assert grid.shape == (gh, gw, 3)
        bounds, base = norm.resolve_bounds_detailed(
            process,
            lambda: (_ for _ in ()).throw(
                AssertionError("Frozen bounds unexpectedly reanalyzed")
            ),
        )
        anchor_bounds = norm.luma_source_bounds(process, base)
        normalized_support, normalized_activity = _spatial_activity(
            _luma(grid, anchor_bounds), valid
        )
        raw_support, raw_activity = _spatial_activity(_luma(grid), valid)
        full_support = bool(support.all())
        axis_grid = grid if full_support else _packet(grid, valid)
        meter_grid = grid if full_support else _packet(grid, normalized_support)
        range_grid = grid if full_support else _packet(grid, raw_support)
        axis = norm.measure_neutral_axis_from_log(axis_grid, bounds, None, 0.0)
        metrics = {
            "log_bounds_base": base,
            "pre_trim_bounds": bounds,
            "final_bounds": bounds,
            "anchor_bounds": anchor_bounds,
            "norm_density_range": float(norm.luminance_density_range(bounds)),
            "neutral_axis_refs": axis,
            "metered_anchor": float(
                norm.measure_anchor_from_log(meter_grid, anchor_bounds, None, 0.0)
            ),
            "textural_range": float(
                norm.measure_textural_range_from_log(range_grid, None, 0.0)
            ),
            "shadow_point": float(
                norm.measure_shadow_point_from_log(meter_grid, anchor_bounds, None, 0.0)
            ),
            "highlight_point": float(
                norm.measure_highlight_point_from_log(
                    meter_grid, anchor_bounds, None, 0.0
                )
            ),
            "density_above_base": None,
        }
        if full_support:
            clip = norm.measure_clip_fractions(sensor_source, (y0, y1, x0, x1), 0.0)
        elif certificate is not None:
            # Exact integer reduction over the identical retained complete
            # cells, without packing a full-camera float RGB copy for a count.
            counts = np.zeros(3, dtype=np.int64)
            row_group = max(1, 512 // b)
            for first in range(0, gh, row_group):
                last = min(gh, first + row_group)
                selected = np.repeat(np.repeat(valid[first:last], b, axis=0), b, axis=1)
                pixels = sensor_source[y0 + first*b:y0 + last*b, x0:x0+wb]
                for channel in range(3):
                    clipped = pixels[..., channel] >= EXPOSURE_CONSTANTS["scan_clip_level"]
                    counts[channel] += np.count_nonzero(clipped & selected)
            clip = counts.astype(np.float64) / retained
        else:
            pixels = np.asarray(sensor_source[y0 : y0 + hb, x0 : x0 + wb], dtype=np.float32)
            pixel_support = np.repeat(np.repeat(valid, b, axis=0), b, axis=1)
            clip = np.mean(
                pixels[pixel_support] >= EXPOSURE_CONSTANTS["scan_clip_level"], axis=0
            )
        metrics["scan_clip_fractions"] = tuple(float(value) for value in clip)
        payload = native_masked_metrics_payload(metrics)
        receipt = {
            "schema": SCHEMA,
            "version": VERSION,
            "provider": PROVIDER,
            "density_bundle_sha256": restored.sha256,
            "source_identity": deepcopy(restored.source_identity),
            "source_pixels_sha256_before": before,
            "source_pixels_sha256_after": before,
            "input_process_config_sha256": process_config_sha256(config.process),
            "process_config": asdict(process),
            "process_config_sha256": process_config_sha256(process),
            "mask_sha256": None
            if restored.selection_mask is None
            else restored.selection_mask.mask_sha256,
            "confirmed_exclusions": deepcopy(
                restored.receipt["effective_excluded_rois"]
            ),
            "requested_roi": list(roi),
            "grid_shape": [gh, gw],
            "grid_block_size": b,
            "requested_grid_block_size": requested_b,
            "grid_origin_source_xy": [x0, y0],
            "grid_tail_pixel_area": rh * rw - hb * wb,
            "total_pixel_area": rh * rw,
            "available_pixel_area": available,
            "supported_pixel_area": retained,
            "available_fraction": available / (rh * rw),
            "retained_fraction": retained / (rh * rw),
            "valid_grid_cell_count": retained_cells,
            "original_pixel_area_per_grid_cell": b * b,
            "support_signature_sha256": _digest(
                {
                    "roi": list(roi),
                    "shape": list(source.shape),
                    "b": b,
                    "valid_sha256": _array_sha(valid),
                }
            ),
            "retained_grid_RGB_sha256": _array_sha(grid[valid]),
            "all_selected_original_helpers": full_support,
            "normalized_texture": normalized_activity,
            "raw_log_texture": raw_activity,
            "partial_mask_original_NegPy_equivalent": False,
            "full_support_auxiliary_oracle_requires_same_frozen_bounds": True,
            "scalar_packet_semantics": "Partial branch only: Nx1 preserves selected triplet population; width1 prevents secondary block-median and texture gate, after actual spatial gate",
            "auxiliary_log_policy": "Original float32 log10(clamp(signal,1e-6,1)); original ROI block median and tail truncation",
            "full_image_log_policy": "Renderer must separately use original log_density_open; >1 preserved",
            "scan_clip_population": "Original complete ROI including grid tail"
            if full_support
            else "Original pixels of retained complete grid cells only",
            "neutral_axis_unavailable_is_frozen_None_no_resampling": axis is None,
            "metrics_payload": payload,
            "metrics_sha256": _digest(payload),
            "source_values_changed": False,
            "RAW_decode_calls": 0,
            "renderer_calls": 0,
            "GPU_supported": False,
        }
    finally:
        _check_source(source, restored.source_identity)
    return NativeMaskedMeterResult(metrics, receipt, effective_config, _digest(receipt))


def validate_native_masked_result(
    result, source, config, density_bundle_json, current_identity
):
    """Validate bytes/metadata only; never repeat a meter or density analysis."""
    if not isinstance(result, NativeMaskedMeterResult):
        raise ValueError("Expected NativeMaskedMeterResult")
    validate_native_masked_config(config)
    restored = restore_zonal_metering(_text(density_bundle_json), current_identity)
    _check_source(source, restored.source_identity)
    receipt = result.receipt
    expected_process = frozen_process_for_window(
        config.process, restored.window, source.shape
    )
    if (
        _digest(receipt) != result.sha256
        or receipt.get("schema") != SCHEMA
        or receipt.get("version") != VERSION
        or receipt.get("density_bundle_sha256") != restored.sha256
        or receipt.get("source_identity") != restored.source_identity
        or receipt.get("input_process_config_sha256")
        != process_config_sha256(config.process)
        or receipt.get("process_config_sha256")
        != process_config_sha256(expected_process)
        or result.effective_config != replace(config, process=expected_process)
        or native_masked_metrics_payload(result.metrics)
        != receipt.get("metrics_payload")
        or _digest(native_masked_metrics_payload(result.metrics))
        != receipt.get("metrics_sha256")
    ):
        raise ValueError("Native masked source/config/metric receipt binding changed")
    return True
