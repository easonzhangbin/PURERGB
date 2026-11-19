"""Active-only row-bounded Cool density-headroom workflow adapter.

The original Analysis and shared fit are frozen inputs. Source preparation
matches the existing workflow's explicit nonnegative-signal policy, then the
V22 density-derived rendering mathematics runs in rows. Only the original
working OETF and SDK final RGB16 encoder follow it. There is no intermediate
RGB16 boundary, no fitting, no Nikon CMS and no Flex/Native point processing.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import numpy as np

from .cli import inspect_tiff
from .cool_headroom import (
    SIGNAL_DOMAIN,
    _analysis as _core_analysis,
    _fit as _core_fit,
    checked_cool_headroom,
    render_cool_headroom,
)
from .positive_float import ICC_SHA256, working_icc_bytes
from .engines import Analysis
from .rgb_curve_study import SharedCurveFit


SCHEMA = "pure-rgb-cool-headroom-workflow"
VERSION = 1
DEFAULT_BLOCK_ROWS = 64
_VECTOR_COUNTS = (
    "source_at_or_below_floor_count_rgb",
    "negative_u_black_count_rgb",
    "previous_upper_density_clip_candidate_count_rgb",
    "mapped_density_exact_one_due_to_rounding_count_rgb",
    "output_exact_one_due_to_rounding_count_rgb",
    "positive_float_cast_to_zero_count_rgb",
)
_PIXEL_COUNTS = (
    "density_mapping_changed_pixel_count",
    "maximum_shoulder_pixel_count",
    "post_fit_or_dtype_changed_pixel_count",
)


def checked_cool_headroom_controls(document):
    """Return an independent active document, or None for a neutral alias."""
    if not isinstance(document, dict):
        raise ValueError("Cool headroom controls require a versioned object or {}")
    controls = checked_cool_headroom(document)
    return controls if controls["strength"] else None


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Cool workflow bindings require finite JSON") from error


def _layout(image):
    return {
        "shape": list(image.shape),
        "dtype": image.dtype.str,
        "strides": list(image.strides),
        "C_contiguous": bool(image.flags.c_contiguous),
    }


def _same_layout(image, layout, message):
    _require(isinstance(image, np.ndarray) and _layout(image) == layout, message)


def _rows(source, block_rows):
    for y0 in range(0, source.shape[0], block_rows):
        yield y0, np.ascontiguousarray(source[y0 : y0 + block_rows])


def _source_sha(source, block_rows):
    digest = hashlib.sha256()
    for _, block in _rows(source, block_rows):
        digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


def _pixels_sha(pixels):
    return hashlib.sha256(memoryview(pixels).cast("B")).hexdigest()


def _source_header(source, block_rows):
    _require(
        isinstance(source, np.ndarray)
        and source.dtype in (np.dtype("float32"), np.dtype("float64"))
        and source.ndim == 3
        and source.shape[-1] == 3
        and source.size > 0
        and all(stride > 0 for stride in source.strides)
        and source.strides[-1] == source.dtype.itemsize,
        "Cool workflow requires float32/64 HWC RGB with positive spatial strides and contiguous channels",
    )
    _require(
        type(block_rows) is int and 1 <= block_rows <= 4096,
        "block_rows must be an integer in [1,4096]",
    )


def _geometry(document, shape):
    if document is None:
        h, w = shape[:2]
        document = {
            "full_source_shape": list(shape),
            "view_roi": [0, 0, w, h],
            "sample_step": 1,
        }
    _require(
        isinstance(document, dict)
        and set(document) == {"full_source_shape", "view_roi", "sample_step"},
        "Explicit full_source_shape/view_roi/sample_step geometry required",
    )
    full, roi, step = (
        document["full_source_shape"],
        document["view_roi"],
        document["sample_step"],
    )
    _require(
        isinstance(full, (tuple, list))
        and len(full) == 3
        and all(type(v) is int and v > 0 for v in full)
        and full[2] == 3,
        "Invalid full source HWC shape",
    )
    _require(
        isinstance(roi, (tuple, list))
        and len(roi) == 4
        and all(type(v) is int for v in roi)
        and type(step) is int
        and step >= 1,
        "Invalid source view ROI or sample step",
    )
    x0, y0, x1, y1 = roi
    _require(
        0 <= x0 < x1 <= full[1] and 0 <= y0 < y1 <= full[0],
        "Source view ROI outside full source",
    )
    expected = [(y1 - y0 + step - 1) // step, (x1 - x0 + step - 1) // step, 3]
    _require(
        list(shape) == expected, "Source view shape disagrees with declared crop/stride"
    )
    return {"full_source_shape": list(full), "view_roi": list(roi), "sample_step": step}


def _export_guard(export):
    from negpy.domain.models import ExportFormat
    from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE

    _require(
        export.export_fmt == ExportFormat.TIFF
        and type(export.export_bit_depth) is int
        and export.export_bit_depth == 16,
        "Cool headroom workflow requires the original TIFF RGB16 export",
    )
    _require(
        export.export_color_space == WORKING_COLOR_SPACE,
        "Cool headroom workflow exports the pinned working ICC only",
    )
    _require(
        not export.icc_input_path and not export.icc_output_path,
        "Custom ICC overrides are outside the Cool workflow adapter",
    )
    return WORKING_COLOR_SPACE


def _reduce_core(current, incoming):
    if current is None:
        return deepcopy(incoming)
    special = {
        "shape",
        "pixel_count",
        "frozen_analysis",
        "output_range",
        "post_fit_maximum_before_output_shoulder",
        *_VECTOR_COUNTS,
        *_PIXEL_COUNTS,
    }
    _require(
        {k: v for k, v in current.items() if k not in special}
        == {k: v for k, v in incoming.items() if k not in special},
        "Cool core semantics changed between row blocks",
    )
    for key in _VECTOR_COUNTS:
        current[key] = [a + b for a, b in zip(current[key], incoming[key], strict=True)]
    for key in (*_PIXEL_COUNTS, "pixel_count"):
        current[key] += incoming[key]
    current["post_fit_maximum_before_output_shoulder"] = max(
        current["post_fit_maximum_before_output_shoulder"],
        incoming["post_fit_maximum_before_output_shoulder"],
    )
    current["output_range"]["minimum_rgb"] = np.minimum(
        current["output_range"]["minimum_rgb"], incoming["output_range"]["minimum_rgb"]
    ).tolist()
    current["output_range"]["maximum_rgb"] = np.maximum(
        current["output_range"]["maximum_rgb"], incoming["output_range"]["maximum_rgb"]
    ).tolist()
    return current


def encode_cool_headroom_workflow(
    processor,
    source,
    analysis,
    fit,
    export,
    document=None,
    *,
    source_key,
    domain,
    source_geometry=None,
    block_rows=DEFAULT_BLOCK_ROWS,
    prepare_source_nonnegative=True,
):
    """Return (RGB16 pixels, working ICC bytes, finite receipt), active only.

    A cropped/strided source carries its original full-source geometry and
    Analysis coordinates. Temporary row ROIs only satisfy the core's shape
    contract; no signal, window, gray balance or shared fit is remeasured.
    """
    from negpy.kernel.image.logic import working_oetf_encode

    controls = checked_cool_headroom_controls(document)
    _require(
        controls is not None,
        "Disabled Cool headroom must bypass the complete old route, including intermediate RGB16",
    )
    _require(
        isinstance(domain, str) and domain == SIGNAL_DOMAIN,
        "Explicit selected-band-linear-signal domain required",
    )
    _require(
        isinstance(source_key, str) and bool(source_key), "Explicit source key required"
    )
    _require(
        type(prepare_source_nonnegative) is bool,
        "prepare_source_nonnegative must be an explicit bool",
    )
    _source_header(source, block_rows)
    geometry = _geometry(source_geometry, source.shape)
    _, _, frozen_analysis = _core_analysis(analysis, geometry["full_source_shape"])
    frozen_fit = _core_fit(fit)
    target = _export_guard(export)
    icc = working_icc_bytes()
    source_layout = _layout(source)
    before_source_sha = _source_sha(source, block_rows)
    export_binding = _canonical(vars(export))
    analysis_binding = _canonical(frozen_analysis)
    fit_binding = _canonical(frozen_fit)
    # The row mathematics consumes these independent snapshots. Dataclasses
    # can still contain mutable endpoint lists supplied by an external caller.
    math_analysis = Analysis(
        tuple(frozen_analysis["dark_rgb"]),
        tuple(frozen_analysis["base_rgb"]),
        tuple(frozen_analysis["roi_xyxy"]),
        None
        if frozen_analysis["percentiles"] is None
        else tuple(frozen_analysis["percentiles"]),
    )
    math_fit = SharedCurveFit(**frozen_fit)
    encoded = np.empty(source.shape, dtype=np.float32)
    encoded_layout = _layout(encoded)
    negative = np.zeros(3, np.int64)
    original_minimum, original_maximum = np.full(3, np.inf), np.full(3, -np.inf)
    prepared_minimum, prepared_maximum = np.full(3, np.inf), np.full(3, -np.inf)
    core_receipt = None
    count = 0
    for y0, block in _rows(source, block_rows):
        _require(np.isfinite(block).all(), "Cool source contains nonfinite samples")
        block_negative = (block < 0).sum((0, 1))
        _require(
            prepare_source_nonnegative or not np.any(block_negative),
            "Negative source signal requires explicit old-workflow nonnegative preparation",
        )
        negative += block_negative
        original_minimum = np.minimum(original_minimum, block.min((0, 1)))
        original_maximum = np.maximum(original_maximum, block.max((0, 1)))
        prepared = np.maximum(block, 0) if prepare_source_nonnegative else block
        prepared_minimum = np.minimum(prepared_minimum, prepared.min((0, 1)))
        prepared_maximum = np.maximum(prepared_maximum, prepared.max((0, 1)))
        row_analysis = replace(
            math_analysis, roi=(0, 0, source.shape[1], block.shape[0])
        )
        rendered, row_receipt = render_cool_headroom(
            prepared, row_analysis, math_fit, controls, domain=SIGNAL_DOMAIN
        )
        _require(
            rendered.shape == block.shape and rendered.dtype == source.dtype,
            "Cool row output geometry or dtype changed",
        )
        encoded[y0 : y0 + block.shape[0]] = working_oetf_encode(rendered)
        core_receipt = _reduce_core(core_receipt, row_receipt)
        count += 1
        del rendered, row_receipt, prepared, block
    core_receipt["shape"] = list(source.shape)
    core_receipt["frozen_analysis"] = frozen_analysis
    _same_layout(
        source, source_layout, "Cool source layout changed during point processing"
    )
    _same_layout(
        encoded, encoded_layout, "Cool encoded layout changed during point processing"
    )
    _require(
        _source_sha(source, block_rows) == before_source_sha,
        "Cool source changed during point processing",
    )
    encoded_sha = _pixels_sha(encoded)
    payload, status = processor.encode_export(encoded, export, target)
    if payload is None:
        raise RuntimeError(status)
    _same_layout(
        source, source_layout, "Cool source layout changed during SDK encoding"
    )
    _same_layout(
        encoded, encoded_layout, "Cool encoded layout changed during SDK encoding"
    )
    _require(
        _source_sha(source, block_rows) == before_source_sha,
        "Cool source changed during SDK encoding",
    )
    _require(_pixels_sha(encoded) == encoded_sha, "SDK changed the Cool encoded buffer")
    _require(_canonical(vars(export)) == export_binding, "Cool export settings changed")
    _, _, after_analysis = _core_analysis(analysis, geometry["full_source_shape"])
    _require(
        _canonical(after_analysis) == analysis_binding
        and _canonical(_core_fit(fit)) == fit_binding,
        "Frozen Cool Analysis or shared fit changed during processing",
    )
    del encoded
    pixels, actual_icc = inspect_tiff(payload)
    _require(
        list(pixels.shape) == list(source.shape), "SDK changed the Cool output geometry"
    )
    _require(
        actual_icc == icc and hashlib.sha256(actual_icc).hexdigest() == ICC_SHA256,
        "SDK Cool output ICC differs from the pinned working profile",
    )
    payload_sha = hashlib.sha256(payload).hexdigest()
    del payload
    pixels.setflags(write=False)
    receipt = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "active-experimental",
        "controls": controls,
        "source_key": source_key,
        "source_key_is_caller_binding_not_RAW_authentication": True,
        "source_domain": SIGNAL_DOMAIN,
        "source_geometry": geometry,
        "source_geometry_certified": False,
        "source_layout": source_layout,
        "source_view_pixels_sha256": before_source_sha,
        "source_preparation": {
            "prepare_source_nonnegative": prepare_source_nonnegative,
            "policy": "Per-row max(signal,0) in original dtype, matching the historical Cool workflow"
            if prepare_source_nonnegative
            else "Strict nonnegative source; negative samples rejected",
            "original_negative_sample_count_rgb": negative.tolist(),
            "original_range": {
                "minimum_rgb": original_minimum.tolist(),
                "maximum_rgb": original_maximum.tolist(),
            },
            "prepared_range": {
                "minimum_rgb": prepared_minimum.tolist(),
                "maximum_rgb": prepared_maximum.tolist(),
            },
        },
        "frozen_analysis": frozen_analysis,
        "frozen_shared_fit": frozen_fit,
        "core_processing": core_receipt,
        "row_analysis_roi_policy": "Original full-source Analysis coordinates retained above; each temporary core row uses its own full row rectangle only for shape validation. Endpoints/fit remain unchanged; no measurement performed.",
        "block_rows": block_rows,
        "block_count": count,
        "maximum_block_shape": [min(block_rows, source.shape[0]), source.shape[1], 3],
        "point_processing_row_bounded": True,
        "SDK_encoder_row_bounded": False,
        "full_float64_intermediate_retained": False,
        "full_encoded_float32_buffer_bytes": int(np.prod(source.shape)) * 4,
        "encoded_buffer_sha256": encoded_sha,
        "working_ICC_sha256": ICC_SHA256,
        "SDK_target": target,
        "SDK_format": status,
        "SDK_payload_sha256": payload_sha,
        "output_RGB16_pixels_sha256": _pixels_sha(pixels),
        "SDK_payload_is_downstream_saved_file": False,
        "intermediate_RGB16_before_shared_fit": False,
        "final_SDK_RGB16_encode_performed": True,
        "OETF_count": 1,
        "new_density_or_shared_fit_performed": False,
        "source_and_layout_unchanged": True,
        "controls_are_not_recovered_EV": True,
        "default_compatibility": "Caller neutral route must remain the complete historical implementation",
        "colour_basis": "Density-derived rendering values assigned to working primaries for presentation, not measured positive scene radiometry",
    }
    receipt["processing_binding_sha256"] = hashlib.sha256(
        _canonical(receipt).encode()
    ).hexdigest()
    return pixels, actual_icc, receipt
