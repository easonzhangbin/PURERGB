"""Bounded-row Native float point export with the original final SDK encoder.

The SDK still renders a complete working-OETF buffer, and its final encoder may
allocate complete output/CMS arrays. Only the intervening positive point stage
is row bounded. No full linear master is retained by this synchronous export
entry point; callers needing that object use :mod:`native_float` instead.
"""

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from .native_float import (
    _canonical,
    _config_guard,
    _digest,
    _geometry,
    _require,
    checked_native_float,
)
from .positive_float import (
    GAMMA,
    ICC_SHA256,
    apply_positive_float,
    positive_domain,
    working_icc_bytes,
)


SCHEMA = "pure-rgb-native-positive-float-stream"
DEFAULT_BLOCK_ROWS = 64


def _pixels(image):
    # A buffer view, not ndarray.tobytes(), avoids a full-image bytes copy.
    return hashlib.sha256(memoryview(image).cast("B")).hexdigest()


def _layout(image):
    return {
        "shape": list(image.shape),
        "dtype": image.dtype.str,
        "strides": list(image.strides),
        "C_contiguous": bool(image.flags.c_contiguous),
    }


def _check_layout(image, expected, message):
    # Bytes alone do not bind their interpretation: reshaping/retyping a view
    # can preserve every byte while changing the declared source geometry.
    _require(isinstance(image, np.ndarray) and _layout(image) == expected, message)


def _header(buffer, block_rows):
    _require(
        isinstance(buffer, np.ndarray)
        and buffer.dtype in (np.dtype("float32"), np.dtype("float64"))
        and buffer.ndim == 3
        and buffer.shape[-1] == 3
        and buffer.size > 0
        and buffer.flags.c_contiguous,
        "Native stream requires C-order float32/float64 HWC RGB",
    )
    _require(
        type(block_rows) is int and 1 <= block_rows <= 4096,
        "Native stream block_rows must be an integer in[1,4096]",
    )


def _target(export, color_space):
    from negpy.domain.models import ColorSpace
    from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry

    target = export.export_color_space
    if target == ColorSpace.SAME_AS_SOURCE.value:
        target = color_space
    _require(isinstance(target, str) and bool(target), "Explicit target ICC required")
    if export.icc_output_path:
        _require(
            Path(export.icc_output_path).is_file(), "Final output ICC file unavailable"
        )
    else:
        _require(
            ColorSpaceRegistry.get_icc_path(target) is not None,
            "Unknown target ICC; no silent working-profile fallback",
        )
    return target


def _merge_range(current, incoming):
    """Reduce point-range receipts without keeping a list per row block."""
    _require(
        current.keys() == incoming.keys(), "Point range schema changed between blocks"
    )
    if "dtype" in current:
        _require(current["dtype"] == incoming["dtype"], "Point block dtype changed")
    for key in current:
        if key == "minimum_rgb":
            current[key] = np.minimum(current[key], incoming[key]).tolist()
        elif key == "maximum_rgb":
            current[key] = np.maximum(current[key], incoming[key]).tolist()
        elif key != "dtype":
            current[key] = [
                a + b for a, b in zip(current[key], incoming[key], strict=True)
            ]


def _merge_stage(current, incoming):
    for key in ("input_range", "output_range"):
        _merge_range(current[key], incoming[key])
    _require(
        {k: v for k, v in current.items() if k not in ("input_range", "output_range")}
        == {
            k: v
            for k, v in incoming.items()
            if k not in ("input_range", "output_range")
        },
        "Point operation semantics changed between blocks",
    )


def _merge_point(current, incoming):
    if current is None:
        return deepcopy(incoming)
    for key in ("input_range", "output_range"):
        _merge_range(current[key], incoming[key])
    for key in ("stages",):
        _require(len(current[key]) == len(incoming[key]), "Point stage count changed")
        for old, new in zip(current[key], incoming[key], strict=True):
            _merge_stage(old, new)
    old, new = current["adaptation"], incoming["adaptation"]
    _require(len(old["stages"]) == len(new["stages"]), "Adaptation stage count changed")
    for first, second in zip(old["stages"], new["stages"], strict=True):
        _merge_stage(first, second)
    return current


def _native_range(point_range):
    return {
        "minimum_rgb": point_range["minimum_rgb"],
        "maximum_rgb": point_range["maximum_rgb"],
        "below_zero_count_rgb": point_range["below_zero_sample_count_rgb"],
        "above_one_count_rgb": point_range["above_one_sample_count_rgb"],
    }


def encode_native_float_stream(
    processor,
    buffer,
    config,
    export,
    document=None,
    *,
    source_key,
    color_space,
    block_rows=DEFAULT_BLOCK_ROWS,
    embed_plan=None,
):
    """Apply unchanged float64 point math in rows, then original final encoding.

    The input is borrowed synchronously, never mutated, and checked by SHA both
    before the SDK encoder and after it returns. This API does not snapshot a
    mutable producer or return a retained full linear frame. There are no
    intermediate disk files; publishing/cancellation remains the caller's
    existing workflow responsibility.
    """
    from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
    from negpy.kernel.image.logic import working_oetf_encode

    _config_guard(config, export)
    controls = checked_native_float(document)
    _header(buffer, block_rows)
    _require(
        isinstance(source_key, str)
        and bool(source_key)
        and isinstance(color_space, str)
        and bool(color_space),
        "Explicit source key and SDK destination label required",
    )
    target = _target(export, color_space)
    config_sha = _digest(config.to_dict())
    geometry = _geometry(export)
    export_binding = _canonical(vars(export))
    source_layout = _layout(buffer)
    source_sha = _pixels(buffer)
    icc = working_icc_bytes()
    original_domain = positive_domain("working-oetf", icc)
    active = bool(controls["values"])
    encoded = np.empty(buffer.shape, dtype=np.float32) if active else buffer
    encoded_layout = _layout(encoded)
    point_receipt = None
    linear_hasher = hashlib.sha256()
    source_min = np.full(3, np.inf)
    source_max = np.full(3, -np.inf)
    source_below = np.zeros(3, dtype=np.int64)
    source_above = np.zeros(3, dtype=np.int64)
    natural_zero = np.zeros(3, dtype=np.int64)
    natural_one = np.zeros(3, dtype=np.int64)
    h, w = buffer.shape[:2]
    for y0 in range(0, h, block_rows):
        block = buffer[y0 : y0 + block_rows]
        _require(
            np.isfinite(block).all(), "Native stream buffer contains nonfinite values"
        )
        source_min = np.minimum(source_min, block.min((0, 1)))
        source_max = np.maximum(source_max, block.max((0, 1)))
        source_below += (block < 0).sum((0, 1))
        source_above += (block > 1).sum((0, 1))
        if active:
            post = apply_positive_float(
                block, controls, domain=original_domain, icc_bytes=icc
            )
            point_receipt = _merge_point(point_receipt, post.receipt)
            linear_hasher.update(memoryview(post.image).cast("B"))
            natural_zero += (post.image == 0).sum((0, 1))
            natural_one += (post.image == 1).sum((0, 1))
            # Same ordering/dtypes as encode_native_float: float64 point data,
            # explicit float64 clip, Native OETF's own float32 conversion.
            encoded[y0 : y0 + len(block)] = working_oetf_encode(
                np.clip(post.image, 0, 1)
            )
            del post
    _check_layout(
        buffer, source_layout, "Caller Native source layout changed during point export"
    )
    _check_layout(
        encoded,
        encoded_layout,
        "Native encoded buffer layout changed during point export",
    )
    _require(
        _pixels(buffer) == source_sha,
        "Caller Native source changed during point export",
    )
    original_range = {
        "minimum_rgb": source_min.tolist(),
        "maximum_rgb": source_max.tolist(),
        "below_zero_count_rgb": source_below.tolist(),
        "above_one_count_rgb": source_above.tolist(),
    }
    if active:
        point_receipt["input_pixels_sha256"] = source_sha
        point_receipt["output_pixels_sha256"] = linear_hasher.hexdigest()
        linear_range = _native_range(point_receipt["output_range"])
        final = {
            "policy": "clip-working-linear-to-unit; pinned-Native-OETF; original-SDK-quantize-and-CMS",
            "linear_clip_bounds": [0, 1],
            "lower_truncated_count_rgb": linear_range["below_zero_count_rgb"],
            "upper_truncated_count_rgb": linear_range["above_one_count_rgb"],
            "lower_truncated_fraction_rgb": (
                np.asarray(linear_range["below_zero_count_rgb"]) / (h * w)
            ).tolist(),
            "upper_truncated_fraction_rgb": (
                np.asarray(linear_range["above_one_count_rgb"]) / (h * w)
            ).tolist(),
            "natural_zero_count_rgb": natural_zero.tolist(),
            "natural_one_count_rgb": natural_one.tolist(),
            "preclip_range": linear_range,
            "preclip_float_retained": False,
            "preclip_float_retained_until_each_block_final_clip": True,
            "preclip_retention_scope": "current row block only; no retained full linear master",
            "OETF": "original working_oetf_encode after explicit unit clip; float32 native power",
            "working_gamma": GAMMA,
        }
    else:
        linear_range = None
        final = {
            "policy": "original-SDK-encoded-buffer-pass-through",
            "native_buffer_passed_without_decode_reencode": True,
            "SDK_source_code_clip_bounds": [0, 1],
            "source_code_below_zero_count_rgb": original_range["below_zero_count_rgb"],
            "source_code_above_one_count_rgb": original_range["above_one_count_rgb"],
            "linear_clip_performed_by_bridge": False,
        }
    receipt = {
        "schema": SCHEMA,
        "version": 1,
        "native_float_controls": controls,
        "source_key": source_key,
        "config_sha256": config_sha,
        "render_export_geometry": geometry,
        "SDK_destination_color_space": color_space,
        "original_buffer_domain": original_domain,
        "linear_domain": positive_domain("linear", icc) if active else None,
        "point_processing": point_receipt,
        "original_buffer_range": original_range,
        "linear_pre_encode_range": linear_range,
        "shape": list(buffer.shape),
        "original_buffer_layout": source_layout,
        "final_encoded_buffer_layout": encoded_layout,
        "original_encoded_sha256": source_sha,
        "linear_sha256": linear_hasher.hexdigest() if active else None,
        "final_encoded_buffer_sha256": _pixels(encoded),
        "neutral_encoder_original_buffer_bypass": not active,
        "SDK_buffer_was_already_working_OETF": True,
        "source_key_is_caller_binding_not_RAW_authentication": True,
        "upstream_PhotoLab_or_OETF_clipping_recovered": False,
        "source_density_reinterpreted_as_positive": False,
        "retained_full_float_frame": False,
        "point_processing_row_bounded": True,
        "SDK_render_and_final_encoder_row_bounded": False,
        "block_rows": block_rows,
        "maximum_block_shape": [min(block_rows, h), w, 3],
        "block_count": (h + block_rows - 1) // block_rows,
        "full_intermediate_encoded_buffer_dtype": encoded.dtype.str,
        "full_intermediate_encoded_buffer_bytes": int(encoded.nbytes) if active else 0,
        "source_unchanged_before_SDK_encode": True,
    }
    receipt["processing_binding_sha256"] = _digest(receipt)
    encoded_sha = receipt["final_encoded_buffer_sha256"]
    payload, status = processor.encode_export(
        encoded, export, target, WORKING_COLOR_SPACE, embed_plan=embed_plan
    )
    if payload is None:
        raise RuntimeError(status)
    _check_layout(
        buffer, source_layout, "Caller Native source layout changed during SDK encoding"
    )
    _check_layout(
        encoded, encoded_layout, "SDK encoder changed the Native encoded buffer layout"
    )
    _require(
        _pixels(buffer) == source_sha,
        "Caller Native source changed during SDK encoding",
    )
    _require(
        _pixels(encoded) == encoded_sha, "SDK encoder changed the Native encoded buffer"
    )
    _require(
        _digest(config.to_dict()) == config_sha
        and _canonical(vars(export)) == export_binding,
        "Native config/export changed during stream encoding",
    )
    encoding = {
        "schema": "pure-rgb-native-float-stream-final-encode",
        "version": 1,
        "processing_binding_sha256": receipt["processing_binding_sha256"],
        "native_float_controls": controls,
        "final_encoding_boundary": final,
        "working_ICC_sha256": ICC_SHA256,
        "target_color_space": target,
        "SDK_format": status,
        "output_bytes_sha256": hashlib.sha256(payload).hexdigest(),
        "target_ICC_gamut_clipping_measured": False,
        "upstream_native_clipping_recovered": False,
        "retained_full_float_frame": False,
        "borrowed_source_buffer_unchanged": True,
    }
    # The outer manifest keys remain compatible; `frame` is an explicitly
    # different stream-receipt schema, never a serialized NativeFloatFrame.
    return payload, {"frame": receipt, "encoding": encoding}


def render_native_float_stream(
    processor,
    file_path,
    config,
    export,
    source_key,
    document=None,
    *,
    prefer_gpu=False,
    metrics=None,
    bounds_override=None,
    block_rows=DEFAULT_BLOCK_ROWS,
    embed_plan=None,
):
    """One actual CPU SDK render, followed by bounded-row point export."""
    _config_guard(config, export)
    checked_native_float(document)
    _require(prefer_gpu is False, "Native float stream is CPU-only; no GPU fallback")
    _require(
        isinstance(source_key, str) and bool(source_key), "Explicit source key required"
    )
    _require(type(block_rows) is int and 1 <= block_rows <= 4096, "Invalid block_rows")
    # An unknown SAME_AS_SOURCE label can only be checked after SDK rendering.
    from negpy.domain.models import ColorSpace

    if export.export_color_space != ColorSpace.SAME_AS_SOURCE.value:
        _target(export, export.export_color_space)
    buffer, color_space = processor.render_export(
        file_path,
        config,
        export,
        source_key,
        metrics=metrics,
        prefer_gpu=False,
        bounds_override=bounds_override,
    )
    if buffer is None:
        raise RuntimeError(color_space)
    return encode_native_float_stream(
        processor,
        buffer,
        config,
        export,
        document,
        source_key=source_key,
        color_space=color_space,
        block_rows=block_rows,
        embed_plan=embed_plan,
    )
