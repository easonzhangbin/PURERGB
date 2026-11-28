"""Explicit local-reference highlight workflow, separate from automatic detection.

Two matched Native CPU PRINT renders feed the existing guarded linear operator.
Only row-sized linear arrays are retained; the engine render, protection raster
and final SDK encoder still use full-frame buffers. Disabled controls return a
sentinel so callers execute their unchanged old route without another render.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import math
import re

import numpy as np

from .density_matrix import check_density_domain, negpy_window
from .highlight_protection import build_highlight_protection
from .highlight_reference_guarded import (
    apply_highlight_reference_guarded,
    checked_highlight_reference_guarded,
)
from .metering_masks import MAX_SHAPES, _shape
from .metering_regions import metering_process, select_metering_roi
from .native_float import (
    _config_guard,
    _digest,
    _geometry,
    _require,
    checked_native_float,
)
from .native_float_stream import (
    DEFAULT_BLOCK_ROWS,
    _check_layout,
    _header,
    _layout,
    _merge_point,
    _pixels,
    _target,
)
from .positive_float import (
    GAMMA,
    ICC_SHA256,
    decode_working_oetf_pixels,
    apply_positive_float,
    positive_domain,
    working_icc_bytes,
)


SCHEMA = "pure-rgb-native-highlight-workflow"
MAX_REFERENCE_DENSITY_OFFSET = 2.0


def _neutral_document():
    return {
        "version": 1,
        "reference_roi": None,
        "controls": checked_highlight_reference_guarded(),
        "protection": None,
    }


def _roi(value, shape=None):
    _require(
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(type(v) is int for v in value),
        "Highlight reference_roi requires four integer source coordinates",
    )
    x0, y0, x1, y1 = value
    _require(
        x0 >= 0 and y0 >= 0 and x1 - x0 >= 2 and y1 - y0 >= 2,
        "Highlight reference ROI must contain at least2x2 source pixels",
    )
    if shape is not None:
        _require(
            x1 <= shape[1] and y1 <= shape[0],
            "Highlight reference ROI outside full source",
        )
    return [x0, y0, x1, y1]


def _protection(document):
    if document is None:
        return None
    _require(
        isinstance(document, dict)
        and not set(document) - {"include", "feather_pixels"},
        "Highlight protection expects explicit include regions and optional feather_pixels",
    )
    include = document.get("include")
    _require(
        isinstance(include, (list, tuple)) and 1 <= len(include) <= MAX_SHAPES,
        "Highlight protection requires1..64 explicit vector regions",
    )
    feather = document.get("feather_pixels", 64.0)
    _require(
        type(feather) in (int, float), "Protection feather must be finite in[0,512]"
    )
    try:
        feather = float(feather)
        # Normalize vector syntax now; real full-source extent is checked when
        # the source is available, before any engine or mask allocation.
        normalized = [_shape(entry, (2**31, 2**31, 3)) for entry in include]
    except (OverflowError, TypeError) as error:
        raise ValueError(
            "Protection coordinates and feather must be finite JSON numbers"
        ) from error
    _require(
        math.isfinite(feather) and 0 <= feather <= 512,
        "Protection feather must be finite in[0,512]",
    )
    return {"include": normalized, "feather_pixels": feather}


def _checked_reference_density_offset(value):
    _require(
        type(value) in (int, float) and 0 <= value <= MAX_REFERENCE_DENSITY_OFFSET,
        "Highlight reference density offset must be finite in [0,2]",
    )
    return float(value)


def checked_highlight_workflow(document=None):
    """Canonical version1 wrapper around guarded-v2 controls and source regions.

    Zero strength or a zero effect budget canonicalizes to the same disabled
    document. Neutral recipes carry no stale source-coordinate selection.
    """
    if document is None:
        return _neutral_document()
    _require(
        isinstance(document, dict)
        and not set(document) - {
            "version", "reference_roi", "controls", "protection", "effect_area",
            "automatic_reference", "reference_density_offset",
        },
        "Unexpected highlight workflow fields",
    )
    version = document.get("version", 1)
    _require(
        type(version) is int and version == 1, "Unsupported highlight workflow version"
    )
    controls = checked_highlight_reference_guarded(document.get("controls"))
    region = document.get("reference_roi")
    if region is not None:
        region = _roi(region)
    protection = _protection(document.get("protection"))
    effect_area = _protection(document.get("effect_area"))
    offset = _checked_reference_density_offset(document.get("reference_density_offset", 0.0))
    automatic = document.get("automatic_reference", False)
    _require(type(automatic) is bool, "automatic_reference must be boolean")
    if (
        controls["strength"] == 0
        or controls["max_burn_ev"] == 0
        or controls["min_log_slope"] == 1
    ):
        return _neutral_document()
    _require(
        region is not None,
        "Active highlight workflow requires an explicit reference_roi",
    )
    return {
        "version": 1,
        "reference_roi": region,
        **({"automatic_reference": True} if automatic else {}),
        "controls": controls,
        "protection": protection,
        **({"effect_area": effect_area} if effect_area is not None else {}),
        **({"reference_density_offset": offset} if offset else {}),
    }


def highlight_support(recipe, source_shape, additional_exclusions=()):
    """Exact pixel-centre selection support, without opening or altering RAWs."""
    from .metering_masks import build_metering_mask

    roi = recipe["roi"]
    meter = recipe.get("native_spatial_controls", {}).get("metering", {})
    mask = build_metering_mask(
        source_shape,
        roi,
        include=meter.get("include", []),
        exclude=meter.get("exclude", []),
    ).mask.copy()
    x0, y0, x1, y1 = roi
    padding = meter.get("padding", 0)
    for a, b, c, d in list(meter.get("excluded_rois", [])) + list(
        additional_exclusions
    ):
        a, b, c, d = (
            max(a - padding, x0),
            max(b - padding, y0),
            min(c + padding, x1),
            min(d + padding, y1),
        )
        if a < c and b < d:
            mask[b - y0 : d - y0, a - x0 : c - x0] = False
    return mask


def resolve_spatial_highlight_reference(highlight, restored):
    """Resolve an automatic reference using the current frozen zonal support."""
    if not highlight.get("automatic_reference"):
        return highlight
    x0, y0, x1, y1 = restored.config["roi"]
    selection = restored.selection_mask
    allowed = (selection.mask.copy() if selection is not None
               else np.ones((y1 - y0, x1 - x0), dtype=bool))
    for a, b, c, d in restored.receipt["effective_excluded_rois"]:
        allowed[b - y0:d - y0, a - x0:c - x0] = False
    recipe = dict(roi=[x0, y0, x1, y1], native_spatial_controls=True)
    return {**highlight, "reference_roi": select_highlight_reference(recipe, allowed)}


def align_highlight_reference(recipe, rectangle):
    """Trim only the reference's far edges to complete Native prefilter cells."""
    if (
        not recipe.get("native_spatial_controls")
        or recipe.get("source_mode") == "bw-single-v1"
    ):
        return list(rectangle)
    from .native_masked_meters import EXPOSURE_CONSTANTS

    x0, y0, x1, y1 = rectangle
    while True:
        width, height = x1 - x0, y1 - y0
        requested = math.ceil(
            max(width, height) / int(EXPOSURE_CONSTANTS["analysis_grid"])
        )
        block = requested if requested > 1 and min(width, height) >= requested else 1
        w, h = width // block * block, height // block * block
        if min(w, h) < 4:
            raise ValueError("高光参考区过小，请重新绘制")
        if (w, h) == (width, height):
            return [x0, y0, x1, y1]
        x1, y1 = x0 + w, y0 + h


def select_highlight_reference(recipe, allowed):
    """Select a rectangle wholly inside the supplied pixel support."""
    roi = recipe["roi"]
    x0, y0, x1, y1 = roi
    if min(x1 - x0, y1 - y0) < 4:
        raise ValueError("高光参考至少需要 4 × 4 源像素")
    if allowed.all():
        return align_highlight_reference(recipe, roi)
    rows, cols = min(16, (y1 - y0) // 4), min(24, (x1 - x0) // 4)
    ys = np.linspace(0, y1 - y0, rows + 1, dtype=int)
    xs = np.linspace(0, x1 - x0, cols + 1, dtype=int)
    heights = [0] * cols
    best, best_area = None, 0
    for row in range(rows):
        for col in range(cols):
            clean = allowed[ys[row] : ys[row + 1], xs[col] : xs[col + 1]].all()
            heights[col] = heights[col] + 1 if clean else 0
        stack = []
        for col in range(cols + 1):
            height = heights[col] if col < cols else 0
            start = col
            while stack and stack[-1][1] > height:
                left, count = stack.pop()
                top = row - count + 1
                area = int((xs[col] - xs[left]) * (ys[row + 1] - ys[top]))
                if area > best_area:
                    best_area = area
                    best = [
                        x0 + int(xs[left]),
                        y0 + int(ys[top]),
                        x0 + int(xs[col]),
                        y0 + int(ys[row + 1]),
                    ]
                start = left
            if height and (not stack or stack[-1][1] < height):
                stack.append((start, height))
    if best is None:
        raise ValueError("测量支持内没有足够完整的高光参考区，请调整选区或排除范围")
    return align_highlight_reference(recipe, best)


_POINT_TRIM_FIELDS = (
    "white_point_offset", "black_point_offset",
    "white_point_trim_red", "white_point_trim_green", "white_point_trim_blue",
    "black_point_trim_red", "black_point_trim_green", "black_point_trim_blue",
)


def _reference_meter_process(config, film):
    process = config.process
    allow_unmix = process.crosstalk_strength != 0
    if allow_unmix or film is not None:
        from negpy.features.process.models import ProcessMode
        from .film_profiles import film_config, film_signature

        selected = film_config(config, film).process if film is not None else None
        selected_active = selected is not None and selected.crosstalk_strength != 0
        if allow_unmix or selected_active:
            _require(
                selected_active and process.process_mode == ProcessMode.C41
                and not process.positive_source and film_signature(process) == film_signature(selected),
                "高光参考与当前胶片密度矩阵不一致，请重新选择胶片配置",
            )
    trims = {key: getattr(process, key) for key in _POINT_TRIM_FIELDS if getattr(process, key) != 0}
    meter = replace(process, meter_rect=None, **dict.fromkeys(trims, 0.0))
    check_density_domain(meter, allow_unmix=allow_unmix)
    return meter, trims, allow_unmix


def _configuration(config, export, bounds_override, *, film=None):
    from negpy.domain.models import (
        AspectRatio,
        ColorSpace,
        ExportFormat,
        ExportResolutionMode,
    )

    _config_guard(config, export)
    _require(
        export.export_fmt == ExportFormat.TIFF,
        "Highlight workflow requires original SDK TIFF output",
    )
    if export.export_color_space != ColorSpace.SAME_AS_SOURCE.value:
        _target(export, export.export_color_space)
    _require(
        bounds_override is None,
        "Highlight reference requires its own measured bounds; external bounds_override unsupported",
    )
    _require(
        export.export_resolution_mode == ExportResolutionMode.ORIGINAL
        and export.paper_aspect_ratio == AspectRatio.ORIGINAL,
        "Highlight workflow requires original full-source export geometry",
    )
    geom = config.geometry
    transformed = any(
        (
            geom.rotation,
            geom.fine_rotation,
            geom.flip_horizontal,
            geom.flip_vertical,
            geom.converge_v,
            geom.converge_h,
            geom.distortion_k1,
            geom.crop_to_valid,
            geom.autocrop_offset,
            geom.crop_rect,
            geom.crop_from_auto,
            geom.autocrop_square,
            geom.lens_distortion_from_metadata,
            geom.lens_ca_from_metadata,
        )
    )
    _require(
        not transformed,
        "Highlight source-coordinate regions require identity SDK geometry",
    )
    _reference_meter_process(config, film)


def _unchanged(image, layout, sha, label):
    _check_layout(image, layout, label + " layout changed")
    _require(_pixels(image) == sha, label + " pixels changed")


def _verified_loader(processor, file_path, config, source, source_layout, source_sha):
    loader = getattr(processor, "_load_source_f32", None)
    _require(
        callable(loader), "Highlight workflow requires source producer verification"
    )
    loaded, _, label = loader(file_path, config)
    _check_layout(loaded, source_layout, "SDK source producer geometry/domain differs")
    _require(
        _pixels(loaded) == source_sha,
        "SDK source producer pixels differ from supplied full source",
    )
    _unchanged(source, source_layout, source_sha, "Caller full source")
    return str(label)


def _bounded_render_buffer(buffer, source_shape, block_rows, checkpoint):
    _header(buffer, block_rows)
    _require(
        buffer.shape == source_shape,
        "SDK highlight render geometry differs from full source",
    )
    for start in range(0, len(buffer), block_rows):
        checkpoint()
        part = buffer[start : start + block_rows]
        _require(
            np.isfinite(part).all() and np.all(part >= 0) and np.all(part <= 1),
            "Guarded highlight input requires bounded finite Native working-OETF buffer",
        )


def highlight_reference_config(source, config, region, *, reference_density_offset=0.0, film=None):
    """Measure the exact reference ROI and bind its window to the full source."""
    offset = _checked_reference_density_offset(reference_density_offset)
    region = _roi(region, source.shape)
    x0, y0, x1, y1 = region
    # Crop by integer edges before metering to avoid normalized-edge round trips.
    # metering_process fixes the normalized auxiliary-meter pixel edges.
    selected = np.ascontiguousarray(source[y0:y1, x0:x1])
    meter, trims, allow_unmix = _reference_meter_process(config, film)
    window = negpy_window(
        selected, replace(config, process=meter), (0, 0, x1 - x0, y1 - y0),
        allow_unmix=allow_unmix,
    )
    window = replace(window, analysis=replace(window.analysis, roi=tuple(region)))
    selection = select_metering_roi(
        source.shape, region, min_side=2, min_area_fraction=1.0
    )
    # Freeze the untrimmed bounds. The SDK applies the original point trims
    # to these bounds when normalizing the original, still-mixed RAW source.
    process = replace(metering_process(meter, window, selection), **trims)
    if trims:
        window.metadata["native_point_trims"] = trims
    return _highlight_reference_render_config(config, process, offset), window


def _highlight_reference_render_config(config, process, offset):
    exposure = (replace(config.exposure, density=config.exposure.density + offset)
                if offset else config.exposure)
    return replace(config, process=process, exposure=exposure)


def _merge_highlight(current, incoming):
    if current is None:
        return deepcopy(incoming)
    old_count = current["pixel_count"]
    for key in (
        "base_input_range",
        "reference_input_range",
        "output_range",
        "protection_mask_range",
    ):
        if current[key] is not None:
            current[key]["minimum"] = min(
                current[key]["minimum"], incoming[key]["minimum"]
            )
            current[key]["maximum"] = max(
                current[key]["maximum"], incoming[key]["maximum"]
            )
    if "effect_mask_range" in current:
        current["effect_mask_range"]["minimum"] = min(
            current["effect_mask_range"]["minimum"], incoming["effect_mask_range"]["minimum"]
        )
        current["effect_mask_range"]["maximum"] = max(
            current["effect_mask_range"]["maximum"], incoming["effect_mask_range"]["maximum"]
        )
    for key in ("affected_pixel_count", "fully_protected_pixel_count"):
        current[key] = current.get(key, 0) + incoming.get(key, 0)
    count = old_count + incoming["pixel_count"]
    burn, next_burn = current["applied_burn_ev"], incoming["applied_burn_ev"]
    burn["mean"] = (
        burn["mean"] * old_count + next_burn["mean"] * incoming["pixel_count"]
    ) / count
    burn["minimum"] = min(burn["minimum"], next_burn["minimum"])
    burn["maximum"] = max(burn["maximum"], next_burn["maximum"])
    current["gain_range"] = [
        min(current["gain_range"][0], incoming["gain_range"][0]),
        max(current["gain_range"][1], incoming["gain_range"][1]),
    ]
    current["pixel_count"] = count
    current["affected_pixel_fraction"] = current["affected_pixel_count"] / count
    current["identity"] = current["affected_pixel_count"] == 0
    return current


def render_highlight_workflow(
    processor,
    file_path,
    source,
    config,
    export,
    source_key,
    document=None,
    *,
    native_float_controls=None,
    block_rows=DEFAULT_BLOCK_ROWS,
    metrics=None,
    bounds_override=None,
    film=None,
):
    """Return ``None`` when disabled, else ``(SDK TIFF bytes, bound receipt)``.

    The source is the caller's already prepared full oriented transmission
    array. Both actual SDK loaders must reproduce its bytes and layout. Their
    source cache requests are not a blanket promise of zero RAW decoding; the
    surrounding workflow controls and tests the producer/cache policy.
    """
    recipe = checked_highlight_workflow(document)
    if recipe["controls"]["strength"] == 0:
        return None
    from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
    from negpy.features.process.models import ProcessMode
    from negpy.kernel.image.logic import working_oetf_encode

    from .diagnostics import span
    from .live_worker import preview_checkpoint

    checkpoint = preview_checkpoint()
    checkpoint()
    _configuration(config, export, bounds_override, film=film)
    point = checked_native_float(native_float_controls)
    _header(source, block_rows)
    _require(
        isinstance(source_key, str)
        and re.fullmatch("[0-9a-f]{64}", source_key) is not None,
        "Highlight workflow source_key requires explicit caller SHA256 identity",
    )
    _require(
        isinstance(file_path, str) and bool(file_path),
        "Explicit highlight source path required",
    )
    region = _roi(recipe["reference_roi"], source.shape)
    for start in range(0, len(source), block_rows):
        checkpoint()
        _require(
            np.isfinite(source[start : start + block_rows]).all(),
            "Highlight full source must be finite",
        )
    source_layout, source_sha = _layout(source), _pixels(source)
    original_config_sha, export_sha = _digest(config.to_dict()), _digest(asdict(export))
    if recipe["protection"]:
        for entry in recipe["protection"]["include"]:
            _shape(
                entry, source.shape
            )  # All source geometry fails before engine calls.
    from .highlight_protection import automatic_highlight_feather

    checkpoint()
    automatic_feather = automatic_highlight_feather(source.shape)
    protection = (
        build_highlight_protection(
            source.shape, source_key, dtype="float64", **recipe["protection"],
            automatic_feather_pixels=automatic_feather,
        )
        if recipe["protection"]
        else None
    )
    protection_layout = _layout(protection.mask) if protection else None
    checkpoint()
    effect_area = (
        build_highlight_protection(
            source.shape, source_key, dtype="float64", **recipe["effect_area"],
            automatic_feather_pixels=automatic_feather,
        )
        if recipe.get("effect_area")
        else None
    )
    effect_layout = _layout(effect_area.mask) if effect_area else None
    # Frozen black-and-white reference configurations retain their recorded
    # print settings. The darker alternate is currently for ordinary C41.
    offset = (recipe.get("reference_density_offset", 0.0)
              if config.process.process_mode == ProcessMode.C41 else 0.0)
    checkpoint()
    with span("highlight_reference_meter", roi=region):
        reference_config, window = highlight_reference_config(
            source, config, region, reference_density_offset=offset, film=film,
        )
    reference_config_sha = _digest(reference_config.to_dict())
    _require(
        reference_config.exposure == replace(
            config.exposure, density=config.exposure.density + offset,
        )
        and asdict(config.lab) == asdict(reference_config.lab),
        "Highlight paper/creative controls must match apart from the reference density offset",
    )
    checkpoint()
    source_labels = []
    source_labels.append(
        _verified_loader(
            processor, file_path, config, source, source_layout, source_sha
        )
    )
    checkpoint()
    with span("highlight_native_base", backend="cpu"):
        base_buffer, base_label = processor.render_export(
            file_path,
            config,
            export,
            source_key,
            metrics=metrics,
            prefer_gpu=False,
            bounds_override=None,
        )
    checkpoint()
    if base_buffer is None:
        raise RuntimeError(base_label)
    _bounded_render_buffer(base_buffer, source.shape, block_rows, checkpoint)
    # A second SDK render may reuse its own first output/cache. Preserve only
    # this complete encoded snapshot, never a complete decoded float64 image.
    base = np.array(base_buffer, copy=True, order="C")
    base.setflags(write=False)
    del base_buffer
    checkpoint()
    source_labels.append(
        _verified_loader(
            processor, file_path, reference_config, source, source_layout, source_sha
        )
    )
    checkpoint()
    with span("highlight_native_reference", backend="cpu"):
        reference, reference_label = processor.render_export(
            file_path,
            reference_config,
            export,
            source_key,
            metrics=metrics,
            prefer_gpu=False,
            bounds_override=None,
        )
    checkpoint()
    if reference is None:
        raise RuntimeError(reference_label)
    _bounded_render_buffer(reference, source.shape, block_rows, checkpoint)
    _require(
        base_label == reference_label and base.dtype == reference.dtype,
        "Matched highlight SDK destination label and dtype required",
    )
    target = _target(export, base_label)
    reference_layout, reference_sha = _layout(reference), _pixels(reference)
    base_sha = _pixels(base)
    icc = working_icc_bytes()
    encoded_domain, linear_domain = (
        positive_domain("working-oetf", icc),
        positive_domain("linear", icc),
    )
    encoded = np.empty(source.shape, dtype=np.float32)
    encoded_layout = _layout(encoded)
    high_receipt = point_receipt = None
    base_linear_sha = hashlib.sha256()
    ref_linear_sha = hashlib.sha256()
    high_linear_sha = hashlib.sha256()
    final_linear_sha = hashlib.sha256()
    lower = np.zeros(3, dtype=np.int64)
    upper = np.zeros(3, dtype=np.int64)
    below = np.full(3, np.inf)
    above = np.full(3, -np.inf)
    preserved = 0
    with span("highlight_join", source_shape=list(source.shape), backend="cpu"):
        for start in range(0, len(source), block_rows):
            checkpoint()
            base_part = base[start : start + block_rows]
            ref_part = reference[start : start + block_rows]
            base_linear = decode_working_oetf_pixels(base_part, icc_bytes=icc)
            ref_linear = decode_working_oetf_pixels(ref_part, icc_bytes=icc)
            mask = protection.mask[start : start + len(base_part)] if protection else None
            high, observation = apply_highlight_reference_guarded(
                base_linear,
                ref_linear,
                recipe["controls"],
                domain=linear_domain,
                protection_mask=mask,
                effect_mask=effect_area.mask[start : start + len(base_part)] if effect_area else None,
                recover_clipped=bool(offset),
            )
            high_receipt = _merge_highlight(high_receipt, observation)
            base_linear_sha.update(memoryview(base_linear).cast("B"))
            ref_linear_sha.update(memoryview(ref_linear).cast("B"))
            high_linear_sha.update(memoryview(high).cast("B"))
            if point["values"]:
                post = apply_positive_float(
                    high, point, domain=linear_domain, icc_bytes=icc
                )
                final = post.image
                point_receipt = _merge_point(point_receipt, post.receipt)
            else:
                final = high
            final_linear_sha.update(memoryview(final).cast("B"))
            lower += (final < 0).sum((0, 1))
            upper += (final > 1).sum((0, 1))
            below, above = (
                np.minimum(below, final.min((0, 1))),
                np.maximum(above, final.max((0, 1))),
            )
            final_encoded = working_oetf_encode(np.clip(final, 0, 1))
            if not point["values"]:
                identical = np.all(final == base_linear, axis=2)
                final_encoded[identical] = base_part[identical]
                preserved += int(identical.sum())
            encoded[start : start + len(base_part)] = final_encoded
            del base_linear, ref_linear, high, final, final_encoded
            if point["values"]:
                del post
    checkpoint()
    high_receipt["shape"] = list(source.shape)
    if point_receipt:
        point_receipt["input_pixels_sha256"] = high_linear_sha.hexdigest()
        point_receipt["output_pixels_sha256"] = final_linear_sha.hexdigest()
    _unchanged(source, source_layout, source_sha, "Caller full source")
    _unchanged(reference, reference_layout, reference_sha, "SDK reference buffer")
    _check_layout(encoded, encoded_layout, "Highlight encoded output layout changed")
    if protection:
        _unchanged(
            protection.mask,
            protection_layout,
            protection.receipt["mask_pixels_sha256"],
            "Highlight protection",
        )
    if effect_area:
        _unchanged(
            effect_area.mask, effect_layout, effect_area.receipt["mask_pixels_sha256"],
            "Highlight effect area",
        )
    encoded_sha = _pixels(encoded)
    checkpoint()
    with span("highlight_encode_export", backend="cpu"):
        payload, status = processor.encode_export(
            encoded, export, target, WORKING_COLOR_SPACE
        )
    checkpoint()
    if payload is None:
        raise RuntimeError(status)
    _unchanged(source, source_layout, source_sha, "Caller full source")
    _unchanged(reference, reference_layout, reference_sha, "SDK reference buffer")
    _unchanged(encoded, encoded_layout, encoded_sha, "Highlight encoded output")
    _require(
        _digest(config.to_dict()) == original_config_sha
        and _digest(reference_config.to_dict()) == reference_config_sha
        and _digest(asdict(export)) == export_sha,
        "Highlight source/config/export binding changed during rendering",
    )
    checkpoint()
    receipt = {
        "schema": SCHEMA,
        "version": 1,
        "highlight_controls": recipe,
        "native_float_controls": point,
        "source_key": source_key,
        "source_pixels_sha256": source_sha,
        "source_layout": source_layout,
        "source_loader_verification": {
            "requests": 2,
            "SDK_source_labels": source_labels,
            "pixels_and_layout_equal_supplied_source": True,
            "no_RAW_decode_guaranteed_by_this_helper": False,
        },
        "base_config_sha256": original_config_sha,
        "reference_config_sha256": reference_config_sha,
        "base_config": config.to_dict(),
        "reference_config": reference_config.to_dict(),
        "render_export_geometry": _geometry(export),
        "export_sha256": export_sha,
        "reference_density_window": asdict(window),
        "reference_roi": region,
        "reference_density_exact_ROI_crop_then_restored_source_coordinates": True,
        "reference_auxiliary_pixel_edges_roundtrip_exact": True,
        "reference_meter_rect_cleared": True,
        "paper_and_creative_controls_matched": not bool(offset),
        **({"reference_density_offset": offset,
            "reference_tone_policy": "preclip-density-offset"} if offset else {}),
        "SDK_working_OETF_domain": encoded_domain,
        "linear_domain": linear_domain,
        "working_ICC_sha256": ICC_SHA256,
        "SDK_destination_color_space": base_label,
        "base_encoded_sha256": base_sha,
        "reference_encoded_sha256": reference_sha,
        "base_linear_sha256": base_linear_sha.hexdigest(),
        "reference_linear_sha256": ref_linear_sha.hexdigest(),
        "highlight_linear_sha256": high_linear_sha.hexdigest(),
        "final_linear_sha256": final_linear_sha.hexdigest(),
        "highlight_processing": high_receipt,
        "native_float_processing": point_receipt,
        "protection": protection.receipt if protection else None,
        **({"effect_area": effect_area.receipt} if effect_area else {}),
        "final_encoding_boundary": {
            "linear_clip_bounds": [0, 1],
            "lower_truncated_count_rgb": lower.tolist(),
            "upper_truncated_count_rgb": upper.tolist(),
            "preclip_minimum_rgb": below.tolist(),
            "preclip_maximum_rgb": above.tolist(),
            "working_gamma": GAMMA,
            "original_Native_OETF_then_SDK_encode": True,
            "unchanged_highlight_pixels_preserve_original_encoded_samples_when_point_neutral": True,
            "original_encoded_preserved_pixel_count": preserved,
            "preclip_float_retained": False,
            "current_block_float_until_explicit_final_clip": True,
        },
        "final_encoded_sha256": encoded_sha,
        "output_bytes_sha256": hashlib.sha256(payload).hexdigest(),
        "SDK_format": status,
        "target_color_space": target,
        "actual_SDK_render_calls": 2,
        "block_rows": block_rows,
        "full_intermediate_encoded_buffer_dtype": encoded.dtype.str,
        "retained_full_linear_master": False,
        "SDK_render_and_final_encoder_row_bounded": False,
        "source_unchanged": True,
        "source_geometry_external_authenticity_certified": False,
        "automatic_reference_or_person_detection": False,
        "upstream_native_clipping_recovered": False,
        "neutral_fit_reference_modified": False,
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return payload, receipt
