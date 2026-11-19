"""Explicit, opt-in Flex scalar/color/ICC workflow with bounded row buffers.

This joins verified numerical stages under declared research assumptions. It
does not identify a camera capture with a scanner, infer application globals,
remeasure a density window, or replace the historical Flex endpoint adapter.
"""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import plistlib
from xml.parsers.expat import ExpatError

import numpy as np

from .engine_resources import checked_resource
from .engines import Analysis
from .flex_color_adjustments import color_map14, color_map_parameters
from .flex_color_management import _profile, scanner_rgb14_icc_bridge
from .flex_compensation import density_carrier, encode_density_carrier
from .flex_lightness import lightness_curve_parameters, lightness_lut
from .flex_pipeline import (
    CurvePreferences,
    apply_profile_curves,
    compile_profile_curves,
)
from .tone_study import ICC_SHA256

SCHEMA = "pure-rgb-flex-research-workflow"
VERSION = 1
BLOCK_PIXELS = 262_144
ICC_CHUNK_PIXELS = 262_144
LOG_FLOOR = 1e-6
_ACTIVE_KEYS = {
    "version",
    "enabled",
    "profile",
    "input_icc",
    "output_icc_sha256",
    "preferences",
    "gray_suppressed",
    "density",
    "curve_overrides",
    "color_overrides",
    "negative_common_active",
    "negative_channels_active",
    "lightness_anchor_preference",
    "assumption",
    "stage_policy",
}
_CURVE_KEYS = {
    "ApplyCurves",
    "ApplySliders",
    "ApplyHistogram",
    "Gamma",
    "EnhancedShadow",
    "Gray",
    "Shadow",
    "Highlight",
    "DotColor",
    "EndPoints",
    "Contrast",
    "Brightness",
}
_COLOR_KEYS = {"ColorCorr", "Saturation", "ApplyCC", "Lightness"}
_PROFILE_KEYS = (
    _CURVE_KEYS
    | _COLOR_KEYS
    | {
        "FilmType",
        "Mode",
        "Gradations",
        "ApplyDust",
        "ApplyUSM",
        "Convert",
        "AutoHighlight",
        "AutoShadow",
        "USMAmount",
        "USMRadius",
        "USMDarkLimit",
        "USMNoiseLimit",
        "USMColFactor",
        "InputProfile",
        "RGBProfile",
        "GrayProfile",
        "CMYKProfile",
        "ColorCor",
        "DustLevel",
        "EmbedProfile",
        "RemoveCastHighlight",
        "RemoveCastShadow",
        "SoftProof",
        "Threshold",
        "streamableVersion",
    }
)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("Flex workflow requires finite JSON values") from error


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(value):
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        "Explicit SHA256 required",
    )
    return value


def _number(value, low, high, label):
    _require(type(value) in (int, float), label + " requires a finite number")
    try:
        valid = math.isfinite(float(value)) and low <= value <= high
    except (TypeError, ValueError, OverflowError):
        valid = False
    _require(valid, label + " outside supported finite range")
    return float(value)


def _integer(value, low, high, label):
    _require(
        type(value) is int and low <= value <= high,
        label + " requires an integer in range",
    )
    return value


def _boolean(value, label):
    _require(type(value) is bool, label + " requires an explicit boolean")
    return value


def _text(value, label):
    _require(
        isinstance(value, str) and bool(value.strip()) and len(value) <= 4096,
        label + " requires nonempty provenance/assumption text",
    )
    return value


def _payload(resource, label):
    resource = checked_resource(resource)
    path = Path(resource["path"])
    try:
        _require(
            0 < path.stat().st_size <= 20_000_000,
            label + " resource size outside bounds",
        )
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(label + " resource unavailable") from error
    _require(
        hashlib.sha256(data).hexdigest() == resource["sha256"],
        label + " resource hash changed",
    )
    return resource, data


def _int_list(value, count, low, high, label):
    _require(
        isinstance(value, list) and len(value) == count, label + " has wrong shape"
    )
    return [_integer(v, low, high, label) for v in value]


def _resource_descriptor(document, label):
    """Validate/normalize JSON path+SHA syntax without consulting the filesystem."""
    _require(
        isinstance(document, dict) and set(document) == {"path", "sha256"},
        label + " requires exactly path and sha256",
    )
    path = document["path"]
    _require(
        isinstance(path, str)
        and bool(path.strip())
        and len(path) <= 4096
        and "\x00" not in path,
        label + " requires a valid bounded local path string",
    )
    return {"path": os.path.abspath(path), "sha256": _sha(document["sha256"])}


def _parameter_fields(fields, *, partial=False):
    """Shared scalar/list bounds for explicit overrides and full effective fields.

    Partial validation never infers a missing profile's flags, defaults, curves
    or cross-field state. Those remain mandatory in the active resource check.
    """
    for flag in (
        "ApplyCurves",
        "ApplySliders",
        "ApplyHistogram",
        "EnhancedShadow",
        "ApplyCC",
    ):
        if not partial or flag in fields:
            _boolean(fields.get(flag), flag)
    if not partial or "Gamma" in fields:
        _number(fields.get("Gamma"), 0.8, 8.0, "Gamma")
    for key in ("Brightness", "Contrast"):
        if not partial or key in fields:
            _integer(fields.get(key), -128, 127, key)
    for key, n, maximum in (
        ("Gray", 4, 255),
        ("Shadow", 4, 16383),
        ("Highlight", 4, 16383),
        ("DotColor", 14, 255),
        ("EndPoints", 2, 2),
    ):
        if not partial or key in fields:
            _int_list(fields.get(key), n, 0, maximum, key)
    if not partial or "ColorCorr" in fields:
        _int_list(fields.get("ColorCorr"), 36, -32768, 32767, "ColorCorr")
    for key in ("Saturation", "Lightness"):
        if not partial or key in fields:
            _integer(fields.get(key), -32768, 32767, key)


def _effective_fields(path, payload, curve, color, anchor):
    try:
        data = plistlib.loads(payload)
        fields = data["ImageSetting"]["ImageCorrection"]
    except (
        ValueError,
        TypeError,
        KeyError,
        plistlib.InvalidFileException,
        ExpatError,
    ) as error:
        raise ValueError("Invalid original Flex profile structure") from error
    _require(
        isinstance(fields, dict) and not set(fields) - _PROFILE_KEYS,
        "Unknown original profile fields/stages are not silently ignored",
    )
    _require(
        type(fields.get("FilmType")) is int and fields["FilmType"] == 1,
        "Joined Flex workflow supports color-negative FilmType1 only; B&W is separate",
    )
    _require(
        type(fields.get("Mode")) is int and fields["Mode"] == 0,
        "Joined Flex workflow requires explicit RGB Mode0",
    )
    for flag in (
        "ApplyDust",
        "ApplyUSM",
        "Convert",
        "RemoveCastHighlight",
        "RemoveCastShadow",
        "SoftProof",
        "EmbedProfile",
    ):
        if flag in fields:
            _boolean(fields[flag], flag)
    for flag in ("ApplyDust", "SoftProof"):
        _require(
            not fields.get(flag, False), "Unsupported active original stage: " + flag
        )
    for field in ("AutoHighlight", "AutoShadow"):
        if field in fields:
            _number(fields[field], -1e12, 1e12, "Original automatic histogram metadata")
    _require(
        not fields.get("ColorCor"), "Unimplemented ColorCor alias must not be active"
    )
    gradations = fields.get("Gradations")
    _require(
        isinstance(gradations, list) and len(gradations) == 7,
        "Original profile must contain exactly seven Gradations",
    )
    effective = deepcopy(fields)
    effective.update(curve)
    effective.update(color)
    _parameter_fields(effective)
    for item in gradations:
        _require(
            isinstance(item, dict) and set(item) == {"Points"},
            "Unknown Gradation structure",
        )
        points = item["Points"]
        _require(
            isinstance(points, list) and 2 <= len(points) <= 1024,
            "Gradation point budget exceeded",
        )
        last = -1.0
        for point in points:
            _require(
                isinstance(point, dict)
                and set(point) <= {"X", "Y", "DY"}
                and {"X", "Y"} <= set(point),
                "Unknown curve point fields",
            )
            x = _number(point["X"], 0, 255, "curve X")
            _require(x > last, "Curve X must increase")
            last = x
            _number(point["Y"], 0, 255, "curve Y")
            _number(point.get("DY", 1), -1e6, 1e6, "curve DY")
    # These are parameter gates only: no curve/image/CMM operator runs here.
    color_map_parameters(
        color_corr=effective["ColorCorr"],
        saturation=effective["Saturation"],
        apply_cc=effective["ApplyCC"],
    )
    lightness_curve_parameters(
        effective["Lightness"],
        anchor_preference=anchor,
        enabled=effective["ApplySliders"],
    )
    return effective


def checked_flex_research_controls(document, *, verify_resources=True):
    """Canonical JSON, or None for the two exact disabled contracts.

    The default also verifies resource bytes/profile-dependent effective fields.
    Explicit verify_resources=False is solely a historical metadata read: it
    validates all declared JSON values and resource descriptors without opening
    those resources. Its return value does not attest resource availability,
    integrity or executable effective settings. Rendering always uses True.
    """
    _boolean(verify_resources, "verify_resources")
    if document is None:
        return None
    _require(
        isinstance(document, dict), "Flex research controls require a versioned object"
    )
    _require(
        type(document.get("version")) is int and document["version"] == VERSION,
        "Unsupported Flex research controls version",
    )
    _boolean(document.get("enabled"), "enabled")
    if not document["enabled"]:
        _require(
            set(document) == {"version", "enabled"},
            "Disabled Flex recipe cannot hide active fields",
        )
        return None
    _require(
        set(document) == _ACTIVE_KEYS,
        "Active Flex research recipe has missing/unknown fields",
    )
    _require(
        len(_canonical(document).encode()) <= 65_536,
        "Flex research JSON exceeds size bound",
    )
    result = deepcopy(document)
    _require(
        result["stage_policy"] == "explicit-color-stages-only",
        "Explicit color-only stage selection required; this is not a full preset",
    )
    _require(
        _sha(result["output_icc_sha256"]) == ICC_SHA256,
        "Flex target must be the actual supported working ICC SHA",
    )
    _text(result["assumption"], "Scanner-domain assumption")
    _boolean(result["gray_suppressed"], "gray_suppressed")
    _boolean(result["negative_common_active"], "negative_common_active")
    channels = result["negative_channels_active"]
    _require(
        isinstance(channels, list)
        and len(channels) == 3
        and all(type(v) is bool for v in channels),
        "Explicit three negative-channel booleans required",
    )
    anchor = result["lightness_anchor_preference"]
    if anchor is not None:
        _integer(anchor, 0, 65535, "Lightness anchor")
    prefs = result["preferences"]
    _require(
        isinstance(prefs, dict)
        and set(prefs)
        == {"neg_var_gamma", "stretch_neg_gamma", "no_neg_curve", "provenance"},
        "All external CurvePreferences fields must be explicit",
    )
    _boolean(prefs["no_neg_curve"], "NoNegCurve")
    _text(prefs["provenance"], "Preference provenance")
    for key in ("neg_var_gamma", "stretch_neg_gamma"):
        if prefs[key] is not None:
            prefs[key] = _number(prefs[key], 1e-12, 1e6, key)
    density = result["density"]
    _require(
        isinstance(density, dict), "Explicit density normalization object required"
    )
    normalization = density.get("normalization")
    _require(
        normalization in ("scene-channel-span", "shared-span"),
        "Unknown density normalization",
    )
    keys = (
        {"normalization", "shared_span"}
        if normalization == "shared-span"
        else {"normalization"}
    )
    _require(
        set(density) == keys, "Shared span must be explicit and is not silently ignored"
    )
    if normalization == "shared-span":
        density["shared_span"] = _number(
            density["shared_span"], 1e-12, 1e12, "shared_span"
        )
    for key, allowed in (
        ("curve_overrides", _CURVE_KEYS),
        ("color_overrides", _COLOR_KEYS),
    ):
        _require(
            isinstance(result[key], dict) and not set(result[key]) - allowed,
            "Unknown Flex curve/color override field",
        )
    _parameter_fields(
        {**result["curve_overrides"], **result["color_overrides"]}, partial=True
    )
    for key, label in (("profile", "Flex profile"), ("input_icc", "Input ICC")):
        result[key] = _resource_descriptor(result[key], label)
    if verify_resources:
        result["profile"], payload = _payload(result["profile"], "Flex profile")
        result["input_icc"], icc = _payload(result["input_icc"], "Input ICC")
        _profile(icc, result["input_icc"]["sha256"], "explicit assumed scanner input")
        _effective_fields(
            result["profile"]["path"],
            payload,
            result["curve_overrides"],
            result["color_overrides"],
            anchor,
        )
    return result


def _geometry(document, shape):
    if document is None:
        document = {
            "full_source_shape": list(shape),
            "view_roi": [0, 0, shape[1], shape[0]],
            "sample_step": 1,
        }
    _require(
        isinstance(document, dict)
        and set(document) == {"full_source_shape", "view_roi", "sample_step"},
        "Explicit full source/view ROI/sample step geometry required",
    )
    full, roi, step = (
        document["full_source_shape"],
        document["view_roi"],
        document["sample_step"],
    )
    _require(
        isinstance(full, (list, tuple))
        and len(full) == 3
        and all(type(v) is int and v > 0 for v in full)
        and full[2] == 3,
        "Invalid full source geometry",
    )
    _require(
        isinstance(roi, (list, tuple))
        and len(roi) == 4
        and all(type(v) is int for v in roi)
        and type(step) is int
        and step >= 1,
        "Invalid ROI/sample step",
    )
    x0, y0, x1, y1 = roi
    _require(
        0 <= x0 < x1 <= full[1] and 0 <= y0 < y1 <= full[0],
        "View ROI outside full source",
    )
    _require(
        list(shape) == [(y1 - y0 + step - 1) // step, (x1 - x0 + step - 1) // step, 3],
        "View geometry disagrees with source shape",
    )
    return {"full_source_shape": list(full), "view_roi": list(roi), "sample_step": step}


def _source_layout(source):
    _require(
        isinstance(source, np.ndarray)
        and source.dtype in (np.dtype("float32"), np.dtype("float64"))
        and source.ndim == 3
        and source.shape[-1] == 3
        and source.size
        and all(s > 0 for s in source.strides)
        and source.strides[-1] == source.dtype.itemsize,
        "Flex source requires floating HWC RGB with positive spatial strides and contiguous channels",
    )
    return {
        "shape": list(source.shape),
        "dtype": source.dtype.str,
        "strides": list(source.strides),
        "writeable": bool(source.flags.writeable),
    }


def _source_hash(source, rows):
    sha = hashlib.sha256()
    for y in range(0, source.shape[0], rows):
        block = np.ascontiguousarray(source[y : y + rows])
        _require(
            np.isfinite(block).all(),
            "Nonfinite source rejected before Flex/ICC processing",
        )
        sha.update(memoryview(block).cast("B"))
    return sha.hexdigest()


def _frozen_lightness(codes, table):
    """Same verified ApplyYGrad arithmetic using one precompiled immutable LUT."""
    flat = codes.reshape(-1, 3)
    clamped, wrapped = np.zeros(3, np.int64), np.zeros(3, np.int64)
    for start in range(0, len(flat), 32768):
        rgb = flat[start : start + 32768].astype(np.int32)
        y = (333 * rgb.sum(1, dtype=np.int64) // 1000).astype(np.int32)
        delta = ((table[y].astype(np.int32) - y + 32768) & 65535) - 32768
        values = rgb + delta[:, None]
        clamped += (values >= 16384).sum(0)
        wrapped += (values < 0).sum(0)
        flat[start : start + len(rgb)] = np.where(
            values >= 16384, 16383, values & 65535
        ).astype(np.uint16)
    _require(
        codes.max() <= 16383,
        "Lightness wrapped outside RGB14; refuse ICC instead of clipping",
    )
    return clamped, wrapped


def render_flex_research_workflow(
    source, analysis, document, *, working_icc, source_key, source_geometry=None,
    experimental_film=None,
):
    """Return actual working-profile RGB16/ICC/receipt, or None when disabled.

    ``experimental_film`` is an optional explicitly-uncalibrated experimental
    film document; when its Flex sub-document is active, the verified RGB14
    Gradation tables are recompiled with GammaNeg/channel Negative gates
    enabled. Production defaults and the frozen resource recipe are unchanged
    when it is absent or inert.
    """
    controls = checked_flex_research_controls(document)
    if controls is None:
        return None
    experimental_flex = None
    if experimental_film is not None:
        from .experimental_film import checked_experimental_film_controls

        experimental_doc = checked_experimental_film_controls(experimental_film)
        if experimental_doc is not None:
            experimental_flex = experimental_doc.get("flex")
    _require(
        isinstance(working_icc, bytes)
        and hashlib.sha256(working_icc).hexdigest() == ICC_SHA256,
        "Actual working ICC bytes must match the fixed working SHA",
    )
    _profile(working_icc, ICC_SHA256, "actual working output")
    _sha(source_key)
    layout = _source_layout(source)
    geometry = _geometry(source_geometry, source.shape)
    _require(
        isinstance(analysis, Analysis),
        "Frozen Analysis required; no automatic metering here",
    )
    analysis_doc = {
        "dark": list(analysis.dark),
        "base": list(analysis.base),
        "roi": list(analysis.roi),
        "percentiles": list(analysis.percentiles)
        if analysis.percentiles is not None
        else None,
    }
    _canonical(analysis_doc)
    high, low = (
        np.asarray(analysis.base, np.float64),
        np.asarray(analysis.dark, np.float64),
    )
    _require(
        high.shape == low.shape == (3,)
        and np.isfinite([high, low]).all()
        and np.all(low >= 0)
        and np.all(high > np.maximum(low, LOG_FLOOR)),
        "Invalid frozen density endpoints",
    )
    roi = analysis_doc["roi"]
    full = geometry["full_source_shape"]
    _require(
        len(roi) == 4
        and all(type(v) is int for v in roi)
        and 0 <= roi[0] < roi[2] <= full[1]
        and 0 <= roi[1] < roi[3] <= full[0],
        "Analysis ROI outside declared full source geometry",
    )
    rows = max(1, BLOCK_PIXELS // source.shape[1])
    source_sha = _source_hash(source, rows)
    profile, profile_payload = _payload(controls["profile"], "Flex profile")
    input_profile, input_icc = _payload(controls["input_icc"], "Input ICC")
    fields = _effective_fields(
        profile["path"],
        profile_payload,
        controls["curve_overrides"],
        controls["color_overrides"],
        controls["lightness_anchor_preference"],
    )
    tables, compiled = compile_profile_curves(
        profile["path"],
        expected_sha256=profile["sha256"],
        preferences=CurvePreferences(**controls["preferences"]),
        gray_suppressed=controls["gray_suppressed"],
        field_overrides=controls["curve_overrides"],
        negative_gamma_active=False,
        negative_common_active=controls["negative_common_active"],
        negative_channels_active=tuple(controls["negative_channels_active"]),
    )
    experimental_compiled = None
    applied_gamma_neg = False
    if experimental_flex is not None:
        from .experimental_film import compile_flex_film_tables

        tables, experimental_compiled = compile_flex_film_tables(
            profile["path"],
            profile["sha256"],
            CurvePreferences(**controls["preferences"]),
            controls["gray_suppressed"],
            experimental_flex,
        )
        applied_gamma_neg = bool(experimental_flex["gamma_neg"])
    color = color_map_parameters(
        color_corr=fields["ColorCorr"],
        saturation=fields["Saturation"],
        apply_cc=fields["ApplyCC"],
    )
    lightness = lightness_curve_parameters(
        fields["Lightness"],
        anchor_preference=controls["lightness_anchor_preference"],
        enabled=fields["ApplySliders"],
    )
    light_table = None
    if lightness["stage_active"]:
        light_table, lightness = lightness_lut(
            fields["Lightness"],
            anchor_preference=controls["lightness_anchor_preference"],
            enabled=fields["ApplySliders"],
        )
    output = np.empty(source.shape, np.uint16)
    counts = {
        k: np.zeros(3, np.int64)
        for k in (
            "negative_source",
            "source_below_log_floor",
            "density_below_zero",
            "density_above_one",
            "lightness_upper_clamped",
            "lightness_negative_wrapped",
        )
    }
    minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
    density_recipe = None
    blocks = 0
    icc_calls = 0
    icc_receipt = None
    for y in range(0, source.shape[0], rows):
        block = source[y : y + rows]
        counts["negative_source"] += (block < 0).sum((0, 1))
        prepared = np.maximum(block, 0)
        counts["source_below_log_floor"] += (prepared < LOG_FLOOR).sum((0, 1))
        density, density_info = density_carrier(
            prepared,
            analysis,
            anchor="scene-high",
            normalization=controls["density"]["normalization"],
            shared_span=controls["density"].get("shared_span"),
            log_floor=LOG_FLOOR,
        )
        counts["density_below_zero"] += (density < 0).sum((0, 1))
        counts["density_above_one"] += (density > 1).sum((0, 1))
        minimum = np.minimum(minimum, density.min((0, 1)))
        maximum = np.maximum(maximum, density.max((0, 1)))
        if density_recipe is None:
            density_recipe = {
                key: density_info[key]
                for key in (
                    "formula",
                    "anchor",
                    "normalization",
                    "reference_rgb",
                    "scene_high_rgb",
                    "scene_low_rgb",
                    "scene_span_rgb",
                    "denominator_rgb",
                    "log_floor",
                )
            }
        codes, _ = encode_density_carrier(density)
        del density, prepared
        adjusted = apply_profile_curves(codes, tables)
        color_map14(
            adjusted,
            color_corr=fields["ColorCorr"],
            saturation=fields["Saturation"],
            apply_cc=fields["ApplyCC"],
            input_domain="flex-post-gradation-rgb14",
            out=adjusted,
        )
        if light_table is not None:
            clamped, wrapped = _frozen_lightness(adjusted, light_table)
            counts["lightness_upper_clamped"] += clamped
            counts["lightness_negative_wrapped"] += wrapped
        converted = scanner_rgb14_icc_bridge(
            adjusted,
            input_icc,
            working_icc,
            source_sha256=input_profile["sha256"],
            output_sha256=ICC_SHA256,
            input_domain="assumed-flextight-scanner-rgb14",
            assumption=controls["assumption"],
            chunk_pixels=ICC_CHUNK_PIXELS,
        )
        _require(
            converted.icc == working_icc
            and converted.pixels.dtype == np.uint16
            and converted.pixels.shape == adjusted.shape,
            "Actual Flex output pixels/ICC violate working-space contract",
        )
        output[y : y + len(block)] = converted.pixels
        if icc_receipt is None:
            icc_receipt = {
                k: converted.receipt[k]
                for k in (
                    "engine",
                    "intent",
                    "flags",
                    "input_domain",
                    "provenance",
                    "source_profile",
                    "output_profile",
                    "cmm_input_adapter",
                    "output_adapter",
                    "extra_gamma_or_oetf_applied",
                )
            }
        blocks += 1
        icc_calls += (
            adjusted.shape[0] * adjusted.shape[1] + ICC_CHUNK_PIXELS - 1
        ) // ICC_CHUNK_PIXELS
    _require(
        _source_layout(source) == layout and _source_hash(source, rows) == source_sha,
        "Source changed during Flex research processing",
    )
    _require(
        _digest(analysis_doc)
        == _digest(
            {
                "dark": list(analysis.dark),
                "base": list(analysis.base),
                "roi": list(analysis.roi),
                "percentiles": list(analysis.percentiles)
                if analysis.percentiles is not None
                else None,
            }
        ),
        "Frozen Analysis changed",
    )
    for resource in (profile, input_profile):
        _payload(resource, "Bound Flex")
    binding = {
        "controls": controls,
        "source_key": source_key,
        "source_pixels_sha256": source_sha,
        "source_layout": layout,
        "source_geometry": geometry,
        "analysis": analysis_doc,
        "working_icc_sha256": ICC_SHA256,
    }
    receipt = {
        "schema": SCHEMA,
        "version": VERSION,
        "binding": binding,
        "binding_sha256": _digest(binding),
        "order": (
            "explicit scene-high density -> one RGB14 quantization -> experimental "
            "GammaNeg/channel Negative enabled -> ColorMap14 -> ApplyYGrad -> assumed "
            "scanner ICC -> actual working RGB16"
            if experimental_compiled is not None
            else "explicit scene-high density -> one RGB14 quantization -> frozen original final LUTs (GammaNeg bypass) -> ColorMap14 -> ApplyYGrad -> assumed scanner ICC -> actual working RGB16"
        ),
        "curve_pipeline": compiled,
        "experimental_curve_pipeline": experimental_compiled,
        "stage_policy": controls["stage_policy"],
        "excluded_original_stages": {
            "selection": "USM, dust, cast removal, soft proof and automatic measurement are intentionally not executed by the explicitly selected color-only workflow",
            "original_requested_values": {
                key: deepcopy(compiled["original_fields"][key])
                for key in (
                    "ApplyUSM",
                    "ApplyDust",
                    "DustLevel",
                    "RemoveCastHighlight",
                    "RemoveCastShadow",
                    "SoftProof",
                    "AutoHighlight",
                    "AutoShadow",
                    "USMAmount",
                    "USMRadius",
                    "USMDarkLimit",
                    "USMNoiseLimit",
                    "USMColFactor",
                    "Threshold",
                )
                if key in compiled["original_fields"]
            },
            "Convert_replaced_by_explicit_bound_ICC": True,
            "original_Convert": compiled["original_fields"].get("Convert"),
            "original_profile_names_are_metadata_only": True,
            "all_original_fields_retained_in_curve_pipeline": True,
        },
        "ColorMap": color,
        "Lightness": lightness,
        "density": density_recipe,
        "density_prequantization_minimum_rgb": minimum.tolist(),
        "density_prequantization_maximum_rgb": maximum.tolist(),
        "counts_rgb": {k: v.tolist() for k, v in counts.items()},
        "pixel_count": source.shape[0] * source.shape[1],
        "source_preparation": "maximum(source,0) per bounded row block; frozen scene endpoints unchanged",
        "scene_high_is_measured_film_base": False,
        "negative_gamma_active": applied_gamma_neg,
        "experimental_film_active": experimental_compiled is not None,
        "experimental_film_calibration": (
            experimental_compiled["experimental_film"]["calibration"]
            if experimental_compiled is not None
            else None
        ),
        "curve_compile_calls": 1,
        "lightness_table_compile_calls": int(light_table is not None),
        "blocks": blocks,
        "maximum_block_pixels": min(source.shape[0], rows) * source.shape[1],
        "ICC": icc_receipt,
        "ICC_transform_calls": icc_calls,
        "ICC_output_sha256": ICC_SHA256,
        "output_pixels_sha256": hashlib.sha256(
            memoryview(output).cast("B")
        ).hexdigest(),
        "source_and_resources_unchanged": True,
        "source_access_contract": {
            "writeable_at_entry": layout["writeable"],
            "borrowed_source_is_not_made_readonly": True,
            "caller_must_not_write_during_call": True,
            "source_layout_and_full_view_SHA_checked_before_and_after": True,
            "checks_are_not_a_concurrent_writer_lock_or_absolute_immutability_proof": True,
        },
        "density_or_gray_remeasurement_calls": 0,
        "legacy_endpoint_fit_applied": False,
        "extra_OETF_or_working_gamma_applied": False,
        "camera_to_scanner_calibration_verified": False,
        "original_ColorSync_equivalence_verified": False,
        "full_factory_preset_executed": False,
        "production_defaults_changed": False,
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    return output, working_icc, receipt
