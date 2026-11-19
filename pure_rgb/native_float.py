"""Opt-in Native PRINT SDK buffer -> positive float controls -> final encode.

render_export's colour-space string names the *destination*, while its buffer
is still working-OETF RGB. Neutral controls pass those original samples to the
unchanged encoder. Active controls retain linear signed/headroom values until
an explicit final clip/OETF/SDK-encode boundary. This cannot recover a value
already clipped by the Native engine's PhotoLab or working OETF stages.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .positive_float import (
    GAMMA,
    ICC_SHA256,
    apply_positive_float,
    checked_float_recipe,
    positive_domain,
    working_icc_bytes,
)


SCHEMA = "pure-rgb-native-positive-float-frame"
GEOMETRY_EXPORT_FIELDS = (
    "export_resolution_mode",
    "export_target_long_edge_px",
    "export_dpi",
    "export_print_size",
    "paper_aspect_ratio",
)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Native float binding must be finite JSON") from error


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _pixels(image):
    return hashlib.sha256(memoryview(image).cast("B")).hexdigest()


def _range(image):
    return {
        "minimum_rgb": image.min((0, 1)).tolist(),
        "maximum_rgb": image.max((0, 1)).tolist(),
        "below_zero_count_rgb": (image < 0).sum((0, 1)).tolist(),
        "above_one_count_rgb": (image > 1).sum((0, 1)).tolist(),
    }


def checked_native_float(document=None):
    """Root recipe field native_float_controls; neutral may use the old route."""
    return checked_float_recipe(
        {"version": 1, "values": {}} if document is None else document
    )


def _config_guard(config, export):
    from negpy.features.exposure.models import RenderIntent

    _require(
        config.exposure.render_intent == RenderIntent.PRINT,
        "Native float bridge requires PRINT positive output; flat master/density is not positive RGB",
    )
    _require(
        not export.icc_input_path,
        "Custom input ICC changes source primaries; this working-buffer bridge does not assume its domain",
    )
    _require(
        type(export.export_bit_depth) is int and export.export_bit_depth in (8, 16),
        "Only explicit8/16-bit final SDK exports supported",
    )


def _geometry(export):
    document = asdict(export)
    return {name: document[name] for name in GEOMETRY_EXPORT_FIELDS}


def _input(buffer):
    _require(
        isinstance(buffer, np.ndarray)
        and buffer.dtype in (np.dtype("float32"), np.dtype("float64"))
        and buffer.ndim == 3
        and buffer.shape[-1] == 3
        and buffer.size > 0
        and buffer.flags.c_contiguous
        and np.isfinite(buffer).all(),
        "Native render buffer must be finite C-order floating HWC RGB; never integer or density codes",
    )


@dataclass(frozen=True)
class NativeFloatFrame:
    original_encoded: np.ndarray
    linear: np.ndarray
    canonical_receipt: str

    @property
    def receipt(self):
        return json.loads(self.canonical_receipt)

    @property
    def recipe(self):
        return self.receipt["native_float_controls"]

    @property
    def domain(self):
        return self.receipt["linear_domain"]


def _frame(original, linear, receipt):
    original.setflags(write=False)
    linear.setflags(write=False)
    receipt = json.loads(_canonical(receipt))
    receipt.update(
        original_encoded_sha256=_pixels(original),
        linear_sha256=_pixels(linear),
        shape=list(linear.shape),
    )
    receipt.pop("frame_binding_sha256", None)
    receipt["frame_binding_sha256"] = _digest(receipt)
    return NativeFloatFrame(original, linear, _canonical(receipt))


def _validate(frame):
    _require(isinstance(frame, NativeFloatFrame), "NativeFloatFrame required")
    receipt = frame.receipt
    checksum = receipt.pop("frame_binding_sha256", None)
    _require(
        receipt.get("schema") == SCHEMA
        and type(receipt.get("version")) is int
        and receipt["version"] == 1
        and checksum == _digest(receipt),
        "Native float frame version/checksum differs",
    )
    working = working_icc_bytes()
    _require(
        receipt["original_buffer_domain"] == positive_domain("working-oetf", working)
        and receipt["linear_domain"] == positive_domain("linear", working),
        "Native float domain or ICC differs",
    )
    _require(
        receipt["native_float_controls"]
        == checked_native_float(receipt["native_float_controls"]),
        "Noncanonical Native float controls",
    )
    bypass = receipt.get("neutral_encoder_original_buffer_bypass")
    _require(
        type(bypass) is bool
        and bypass is (not bool(receipt["native_float_controls"]["values"])),
        "Native neutral encoder bypass contradicts canonical controls",
    )
    for image, key in (
        (frame.original_encoded, "original_encoded_sha256"),
        (frame.linear, "linear_sha256"),
    ):
        _input(image)
        _require(
            not image.flags.writeable
            and list(image.shape) == receipt["shape"]
            and _pixels(image) == receipt[key],
            "Native float pixels, geometry or readonly identity changed",
        )
    receipt["frame_binding_sha256"] = checksum
    return receipt


def post_native_float_buffer(
    buffer, config, export, document=None, *, source_key, color_space
):
    """Post a caller-verified Native PRINT working-OETF buffer, before encoding.

    source_key is an opaque caller provenance binding, not an authenticated RAW
    SHA. A changed producer (e.g. an opt-in recovery engine) must bind its own
    recipe/domain/source in that key. This function does not relabel a flat or
    negative-density producer, nor certify upstream numerical headroom.
    """
    _config_guard(config, export)
    controls = checked_native_float(document)
    _input(buffer)
    _require(
        isinstance(source_key, str)
        and bool(source_key)
        and isinstance(color_space, str)
        and bool(color_space),
        "Explicit source key and SDK destination label required",
    )
    # Copy the SDK buffer: a subsequent render/cache reuse cannot alter it.
    original = np.array(buffer, copy=True, order="C")
    original.setflags(write=False)
    icc = working_icc_bytes()
    original_domain = positive_domain("working-oetf", icc)
    post = apply_positive_float(
        original, controls, domain=original_domain, icc_bytes=icc
    )
    linear = np.array(post.image, copy=True, order="C")
    receipt = {
        "schema": SCHEMA,
        "version": 1,
        "native_float_controls": controls,
        "source_key": source_key,
        "config_sha256": _digest(config.to_dict()),
        "render_export_geometry": _geometry(export),
        "SDK_destination_color_space": color_space,
        "original_buffer_domain": original_domain,
        "linear_domain": post.domain,
        "point_processing": post.receipt,
        "original_buffer_range": _range(original),
        "linear_pre_encode_range": _range(linear),
        "crop_history": [],
        "neutral_encoder_original_buffer_bypass": not bool(controls["values"]),
        "SDK_buffer_was_already_working_OETF": True,
        "source_key_is_caller_binding_not_RAW_authentication": True,
        "upstream_PhotoLab_or_OETF_clipping_recovered": False,
        "source_density_reinterpreted_as_positive": False,
        "encode_or_final_clip_performed": False,
    }
    return _frame(original, linear, receipt)


def render_native_float(
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
):
    """Actual SDK render_export call, then the explicit post-buffer bridge."""
    _config_guard(config, export)
    _require(
        prefer_gpu is False, "Native float v1 is CPU-only; no implicit GPU fallback"
    )
    _require(
        isinstance(source_key, str) and bool(source_key),
        "Explicit source key required before SDK rendering",
    )
    checked_native_float(document)  # Reject invalid controls before any rendering.
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
    return post_native_float_buffer(
        buffer, config, export, document, source_key=source_key, color_space=color_space
    )


def crop_native_float(frame, roi):
    """Same half-open crop for original and post float; no meter/render rerun."""
    receipt = _validate(frame)
    h, w = frame.linear.shape[:2]
    _require(
        isinstance(roi, (list, tuple))
        and len(roi) == 4
        and all(type(value) is int for value in roi),
        "Explicit integer output ROI required",
    )
    x0, y0, x1, y1 = roi
    _require(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, "Output ROI outside Native frame")
    if list(roi) == [0, 0, w, h]:
        return frame
    original = np.array(frame.original_encoded[y0:y1, x0:x1], copy=True, order="C")
    linear = np.array(frame.linear[y0:y1, x0:x1], copy=True, order="C")
    receipt["crop_history"].append(
        {
            "parent_shape": [h, w, 3],
            "roi_xyxy": list(roi),
            "parent_frame_binding_sha256": receipt["frame_binding_sha256"],
        }
    )
    receipt["linear_pre_encode_range"] = _range(linear)
    receipt["original_buffer_range"] = _range(original)
    receipt["point_processing_range_scope"] = (
        "Point receipt describes the parent processing; top-level ranges/hashes describe the current cropped frame"
    )
    return _frame(original, linear, receipt)


def encode_native_float(processor, frame, export, *, color_space=None, embed_plan=None):
    """Explicit final working-linear clip, original OETF and SDK quantize/CMS.

    Neutral frames skip a decode/re-encode and use the original SDK float buffer
    exactly. Working-space clip counts do not measure later target-ICC gamut
    clipping. No float/master data is replaced by the encoded result.
    """
    from negpy.domain.models import ColorSpace
    from negpy.infrastructure.display.color_spaces import (
        WORKING_COLOR_SPACE,
        ColorSpaceRegistry,
    )
    from negpy.kernel.image.logic import working_oetf_encode

    receipt = _validate(frame)
    _require(
        not export.icc_input_path, "Custom input ICC unsupported at this float boundary"
    )
    _require(
        _geometry(export) == receipt["render_export_geometry"],
        "Export render geometry changed; render/crop a new frame",
    )
    _require(
        type(export.export_bit_depth) is int and export.export_bit_depth in (8, 16),
        "Explicit8/16-bit final export required",
    )
    target = color_space or export.export_color_space
    if target == ColorSpace.SAME_AS_SOURCE.value:
        target = receipt["SDK_destination_color_space"]
    _require(
        isinstance(target, str) and bool(target), "Explicit final target space required"
    )
    if export.icc_output_path:
        _require(
            Path(export.icc_output_path).is_file(), "Final output ICC file unavailable"
        )
    else:
        _require(
            ColorSpaceRegistry.get_icc_path(target) is not None,
            "Unknown target ICC; no silent working-profile fallback",
        )
    neutral = receipt["neutral_encoder_original_buffer_bypass"]
    if neutral:
        encoded = frame.original_encoded
        final = {
            "policy": "original-SDK-encoded-buffer-pass-through",
            "native_buffer_passed_without_decode_reencode": True,
            "SDK_source_code_clip_bounds": [0, 1],
            "source_code_below_zero_count_rgb": (encoded < 0).sum((0, 1)).tolist(),
            "source_code_above_one_count_rgb": (encoded > 1).sum((0, 1)).tolist(),
            "linear_clip_performed_by_bridge": False,
        }
    else:
        linear = frame.linear
        count = linear.shape[0] * linear.shape[1]
        final = {
            "policy": "clip-working-linear-to-unit; pinned-Native-OETF; original-SDK-quantize-and-CMS",
            "linear_clip_bounds": [0, 1],
            "lower_truncated_count_rgb": (linear < 0).sum((0, 1)).tolist(),
            "upper_truncated_count_rgb": (linear > 1).sum((0, 1)).tolist(),
            "lower_truncated_fraction_rgb": ((linear < 0).sum((0, 1)) / count).tolist(),
            "upper_truncated_fraction_rgb": ((linear > 1).sum((0, 1)) / count).tolist(),
            "natural_zero_count_rgb": (linear == 0).sum((0, 1)).tolist(),
            "natural_one_count_rgb": (linear == 1).sum((0, 1)).tolist(),
            "preclip_range": _range(linear),
            "preclip_float_retained": True,
            "OETF": "original working_oetf_encode after explicit unit clip; float32 native power",
            "working_gamma": GAMMA,
        }
        encoded = working_oetf_encode(np.clip(linear, 0, 1))
    payload, status = processor.encode_export(
        encoded, export, target, WORKING_COLOR_SPACE, embed_plan=embed_plan
    )
    if payload is None:
        raise RuntimeError(status)
    _validate(frame)
    return payload, {
        "schema": "pure-rgb-native-float-final-encode",
        "version": 1,
        "frame_binding_sha256": receipt["frame_binding_sha256"],
        "native_float_controls": receipt["native_float_controls"],
        "final_encoding_boundary": final,
        "working_ICC_sha256": ICC_SHA256,
        "target_color_space": target,
        "SDK_format": status,
        "output_bytes_sha256": hashlib.sha256(payload).hexdigest(),
        "target_ICC_gamut_clipping_measured": False,
        "upstream_native_clipping_recovered": False,
        "retained_float_frame_unchanged": True,
    }
