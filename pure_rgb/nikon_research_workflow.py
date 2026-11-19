"""Opt-in Native positive RGB16 -> assumed Nikon CMS -> actual working ICC.

The input is already rendered and quantized. Its working RGB values are an
explicit uncalibrated device-code hypothesis, not an LS-5000 acquisition. The
original CML4/fixed arithmetic is reused unchanged, then Nikon's own output ICC
is actually transformed to the common working profile before shared post work.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import stat
from threading import Lock

import numpy as np

from .flex_color_management import _profile, rgb16_icc_transform
from .nikon_cms_research import evaluate_assumed_codes
from .nikon_resource_probe import probe_resources
from .tone_study import ICC_SHA256


SCHEMA = "pure-rgb-nikon-research-workflow"
VERSION = 1
SOURCE = "neutral-native-working-positive"
INPUT_HYPOTHESIS = "working-rgb-positive-rgb16"
NIKON_ICC_SHA256 = "a8d0d753bd6129357cc2647435ce675e8637a679eb526fa180fba460874ce1d3"
ICC_CHUNK_PIXELS = 262_144
_RESOURCE_BINDING = None
_RESOURCE_LOCK = Lock()
_ACTIVE_KEYS = {
    "version",
    "enabled",
    "source",
    "input_hypothesis",
    "fixed_strength",
    "threads",
    "output_icc_sha256",
    "assumption",
}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Nikon workflow requires finite JSON metadata") from error


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(value):
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        "Nikon workflow requires an explicit SHA256 source key",
    )
    return value


def _resource_identity(path):
    try:
        info = path.stat()
    except OSError as error:
        raise ValueError("Bound Nikon resource unavailable") from error
    _require(stat.S_ISREG(info.st_mode), "Bound Nikon resource is not a regular file")
    return [
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    ]


def checked_nikon_resources():
    """Verify pinned resources once, then guard this process's file identities.

    No reference code is imported or evaluated here. A changed resource fails
    closed rather than reloading a previously cached model/result. Root callers
    must invoke this before using a hot Nikon result as well as cold rendering.
    File identities are a local lifetime guard, not an authenticated filesystem
    or concurrent-writer lock. A new process starts a fresh full SHA validation.
    """
    global _RESOURCE_BINDING
    with _RESOURCE_LOCK:
        if _RESOURCE_BINDING is None:
            probe = probe_resources()
            _require(
                probe["capabilities"]["public_downstream_resource_integrity_verified"]
                is True,
                "Nikon CMS 本机研究资源缺失或校验未通过。可继续使用 NegPy 和 FlexColor；"
                "Nikon 资源要求见操作手册。",
            )
            records = [
                *probe["public_source_files"].values(),
                *probe["public_cml4_assets"].values(),
            ]
            _require(bool(records), "Missing pinned Nikon resource records")
            resources = []
            for record in records:
                _require(record.get("verified") is True, "Unverified Nikon resource")
                _sha(record.get("sha256"))
                path = Path(record["path"])
                identity = _resource_identity(path)
                _require(
                    identity[2] == record["bytes"] and 0 < identity[2] <= 2_000_000,
                    "Bound Nikon resource size changed",
                )
                try:
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                except OSError as error:
                    raise ValueError("Bound Nikon resource unreadable") from error
                _require(
                    digest == record["sha256"] and _resource_identity(path) == identity,
                    "Nikon resource changed during initial SHA validation",
                )
                resources.append(
                    {"path": str(path), "sha256": digest, "file_identity": identity}
                )
            _RESOURCE_BINDING = {
                "schema": "pure-rgb-nikon-resource-lifetime",
                "version": 1,
                "public_reference_commit": probe["public_reference_commit"],
                "resources": resources,
                "initial_SHA_validation": True,
                "subsequent_guard": "dev/inode/size/mtime_ns/ctime_ns; change rejects without reload",
                "reference_code_or_CML4_CMM_evaluated": False,
            }
            _RESOURCE_BINDING["binding_sha256"] = _digest(_RESOURCE_BINDING)
        for resource in _RESOURCE_BINDING["resources"]:
            _require(
                _resource_identity(Path(resource["path"])) == resource["file_identity"],
                "Nikon resource changed after validation; restart required, no cache reload",
            )
        return deepcopy(_RESOURCE_BINDING)


def checked_nikon_research_controls(document):
    """Return canonical active controls, or None for the exact disabled forms.

    An active fixed_strength=0 is CMS-only, not a disabled workflow. This is
    deliberately distinct from the optional fixed-curve strength inside CMS.
    """
    if document is None:
        return None
    _require(isinstance(document, dict), "Nikon controls require a versioned object")
    _require(
        type(document.get("version")) is int and document["version"] == VERSION,
        "Unsupported Nikon workflow version",
    )
    _require(type(document.get("enabled")) is bool, "Nikon enabled must be boolean")
    if not document["enabled"]:
        _require(
            set(document) == {"version", "enabled"},
            "Disabled Nikon controls cannot hide active fields",
        )
        return None
    _require(set(document) == _ACTIVE_KEYS, "Missing or unknown active Nikon fields")
    _require(document["source"] == SOURCE, "Explicit neutral Native positive required")
    _require(
        document["input_hypothesis"] == INPUT_HYPOTHESIS,
        "Explicit uncalibrated positive-working Nikon hypothesis required",
    )
    strength = document["fixed_strength"]
    _require(type(strength) in (int, float), "Nikon fixed strength must be numeric")
    try:
        strength = float(strength)
    except OverflowError as error:
        raise ValueError("Nikon fixed strength outside finite range") from error
    _require(
        math.isfinite(strength) and 0 <= strength <= 1,
        "Nikon fixed strength must be finite in[0,1]",
    )
    _require(
        type(document["threads"]) is int and 1 <= document["threads"] <= 16,
        "Nikon CMS threads must be an integer in[1,16]",
    )
    _require(
        document["output_icc_sha256"] == ICC_SHA256,
        "Nikon output target must be the pinned actual working ICC",
    )
    assumption = document["assumption"]
    _require(
        isinstance(assumption, str)
        and bool(assumption.strip())
        and len(assumption) <= 4096,
        "Nikon workflow requires an explicit bounded assumption string",
    )
    result = deepcopy(document)
    result["fixed_strength"] = strength
    _canonical(result)
    return result


def _layout(pixels, *, readonly=False):
    _require(
        isinstance(pixels, np.ndarray)
        and pixels.dtype == np.uint16
        and pixels.ndim == 3
        and pixels.shape[-1] == 3
        and pixels.size > 0
        and pixels.flags.c_contiguous,
        "Nikon positive workflow requires contiguous nonempty HWC RGB16",
    )
    _require(
        not readonly or not pixels.flags.writeable,
        "Nikon positive source must already be readonly",
    )
    return {
        "shape": list(pixels.shape),
        "dtype": pixels.dtype.str,
        "strides": list(pixels.strides),
        "writeable": bool(pixels.flags.writeable),
    }


def _pixels_sha(pixels):
    return hashlib.sha256(memoryview(pixels).cast("B")).hexdigest()


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
        "Nikon geometry requires full source shape, view ROI and sample step",
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
        "Invalid Nikon full-source geometry",
    )
    _require(
        isinstance(roi, (list, tuple))
        and len(roi) == 4
        and all(type(v) is int for v in roi)
        and type(step) is int
        and step >= 1,
        "Invalid Nikon view ROI/sample step",
    )
    x0, y0, x1, y1 = roi
    _require(
        0 <= x0 < x1 <= full[1] and 0 <= y0 < y1 <= full[0],
        "Nikon view ROI outside full source",
    )
    _require(
        list(shape) == [(y1 - y0 + step - 1) // step, (x1 - x0 + step - 1) // step, 3],
        "Nikon pixel shape disagrees with declared view geometry",
    )
    return {"full_source_shape": list(full), "view_roi": list(roi), "sample_step": step}


def render_nikon_research_workflow(
    pixels,
    document,
    *,
    working_icc,
    source_key,
    reference_config,
    source_geometry=None,
    experimental_film=None,
):
    """Return actual working RGB16/ICC/receipt, or None without work when off.

    source_key/config bind the caller's neutral Native producer. This helper
    checks the supplied RGB16 and metadata, not the historical RAW render or
    its neutrality. Shared output/post controls belong after this function.

    ``experimental_film`` optionally selects the explicitly-uncalibrated
    parameterized analyzer/LUT/fixed/CMS join instead of the CMS-only assumed
    route. Production behavior is unchanged when it is absent or inert.
    """
    controls = checked_nikon_research_controls(document)
    if controls is None:
        return None
    experimental_nikon = None
    experimental_doc = None
    if experimental_film is not None:
        from .experimental_film import checked_experimental_film_controls

        experimental_doc = checked_experimental_film_controls(experimental_film)
        if experimental_doc is not None:
            experimental_nikon = experimental_doc.get("nikon")
    parameterized = (
        experimental_nikon is not None
        and experimental_nikon["mode"] == "parameterized"
    )
    _require(
        isinstance(working_icc, bytes)
        and hashlib.sha256(working_icc).hexdigest() == ICC_SHA256,
        "Nikon workflow requires the actual pinned working ICC bytes",
    )
    working_profile = _profile(working_icc, ICC_SHA256, "actual working source/output")
    _sha(source_key)
    _require(
        isinstance(reference_config, dict) and bool(reference_config),
        "Nikon workflow requires the explicit neutral Native reference config",
    )
    config_json = _canonical(reference_config)
    _require(len(config_json.encode()) <= 262_144, "Native reference config too large")
    config = deepcopy(reference_config)
    original_document_json = _canonical(document)
    original_geometry_json = _canonical(source_geometry)
    layout = _layout(pixels, readonly=True)
    geometry = _geometry(source_geometry, pixels.shape)
    source_sha = _pixels_sha(pixels)
    resources = checked_nikon_resources()

    def unchanged_source():
        _require(
            _layout(pixels, readonly=True) == layout
            and _pixels_sha(pixels) == source_sha,
            "Nikon workflow source layout/pixels changed",
        )
        _require(
            _canonical(reference_config) == config_json
            and _canonical(document) == original_document_json
            and _canonical(source_geometry) == original_geometry_json,
            "Nikon workflow source recipe/config/geometry changed",
        )

    if parameterized:
        from .experimental_film import build_nikon_parameter_result
        from .nikon_parameterized import evaluate_parameterized_codes

        parameters = build_nikon_parameter_result(experimental_nikon)
        nikon = evaluate_parameterized_codes(
            pixels,
            parameters,
            input_hypothesis=experimental_nikon["input_hypothesis"],
            fixed_strength=experimental_nikon["fixed_strength"],
            threads=experimental_nikon["threads"],
        )
        unchanged_source()
        _require(
            checked_nikon_resources() == resources,
            "Nikon resource binding changed during parameterized evaluation",
        )
        nikon_layout = _layout(nikon.pixels)
        _require(nikon.pixels.shape == pixels.shape, "Nikon parameterized output geometry changed")
        nikon_sha = _pixels_sha(nikon.pixels)
        nikon_profile = _profile(nikon.icc, NIKON_ICC_SHA256, "actual Nikon CMS output")
        _require(
            isinstance(nikon.receipt, dict)
            and nikon.receipt.get("schema") == "pure-rgb-nikon-parameterized-research"
            and nikon.receipt.get("source_pixels_sha256") == source_sha
            and nikon.receipt.get("output_pixels_sha256") == nikon_sha
            and nikon.receipt.get("input_hypothesis")
            == experimental_nikon["input_hypothesis"]
            and nikon.receipt.get("fixed_strength")
            == experimental_nikon["fixed_strength"]
            and nikon.receipt.get("film_kind") == "color-negative"
            and nikon.receipt.get("source_unchanged") is True
            and nikon.receipt.get("acquisition_binding_proven") is False,
            "Nikon parameterized receipt disagrees with actual source/output/controls",
        )
        experimental_tag = deepcopy(experimental_nikon)
    else:
        nikon = evaluate_assumed_codes(
            pixels,
            input_hypothesis=controls["input_hypothesis"],
            fixed_strength=controls["fixed_strength"],
            threads=controls["threads"],
        )
        unchanged_source()
        _require(
            checked_nikon_resources() == resources,
            "Nikon resource binding changed during CMS evaluation",
        )
        nikon_layout = _layout(nikon.pixels)
        _require(nikon.pixels.shape == pixels.shape, "Nikon CMS output geometry changed")
        nikon_sha = _pixels_sha(nikon.pixels)
        nikon_profile = _profile(nikon.icc, NIKON_ICC_SHA256, "actual Nikon CMS output")
        _require(
            isinstance(nikon.receipt, dict)
            and nikon.receipt.get("source_pixels_sha256") == source_sha
            and nikon.receipt.get("output_pixels_sha256") == nikon_sha
            and nikon.receipt.get("output_icc_sha256") == NIKON_ICC_SHA256
            and nikon.receipt.get("input_hypothesis", {}).get("id") == INPUT_HYPOTHESIS
            and nikon.receipt.get("fixed_strength") == controls["fixed_strength"]
            and nikon.receipt.get("fixed_bypassed") is (controls["fixed_strength"] == 0)
            and nikon.receipt.get("source_unchanged") is True
            and nikon.receipt.get("factory_equivalent") is False,
            "Nikon CMS receipt disagrees with actual source/output/controls",
        )
        experimental_tag = None
    nikon_receipt = deepcopy(nikon.receipt)
    nikon_receipt_json = _canonical(nikon_receipt)
    converted = rgb16_icc_transform(
        nikon.pixels,
        nikon.icc,
        working_icc,
        source_sha256=NIKON_ICC_SHA256,
        output_sha256=ICC_SHA256,
        input_domain="icc-encoded-rgb16",
        provenance=controls["assumption"],
        intent=1,
        chunk_pixels=ICC_CHUNK_PIXELS,
    )
    unchanged_source()
    _require(
        checked_nikon_resources() == resources,
        "Nikon resource binding changed during working conversion",
    )
    _require(
        _layout(nikon.pixels) == nikon_layout
        and _pixels_sha(nikon.pixels) == nikon_sha
        and _canonical(nikon.receipt) == nikon_receipt_json,
        "Nikon intermediate RGB16/receipt changed during working ICC conversion",
    )
    _layout(converted.pixels)
    _require(
        converted.pixels.shape == pixels.shape and converted.icc == working_icc,
        "Nikon working conversion output pixels/ICC disagree with declared target",
    )
    output_sha = _pixels_sha(converted.pixels)
    _require(
        converted.receipt.get("input_pixels_sha256") == nikon_sha
        and converted.receipt.get("output_pixels_sha256") == output_sha
        and converted.receipt.get("source_profile", {}).get("sha256")
        == NIKON_ICC_SHA256
        and converted.receipt.get("output_profile", {}).get("sha256") == ICC_SHA256
        and converted.receipt.get("extra_gamma_or_oetf_applied") is False,
        "Actual Nikon-to-working ICC receipt disagrees with pixels/profiles",
    )
    converted.pixels.setflags(write=False)
    binding = {
        "controls": controls,
        "source_key": source_key,
        "source_pixels_sha256": source_sha,
        "source_layout": layout,
        "source_geometry": geometry,
        "reference_config": config,
        "reference_config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "working_icc_sha256": ICC_SHA256,
        "Nikon_output_icc_sha256": NIKON_ICC_SHA256,
        "resources": resources,
    }
    receipt = {
        "schema": SCHEMA,
        "version": VERSION,
        "binding": binding,
        "binding_sha256": _digest(binding),
        "order": (
            "neutral Native working-positive RGB16 -> explicit unbound analyzer LUT inversion -> fixed mix -> captured Nikon CML4 -> paired Nikon output ICC -> actual working ICC conversion"
            if parameterized
            else "neutral Native working-positive RGB16 -> optional fixed mix -> assumed Nikon CML4 -> paired Nikon output ICC -> actual working ICC conversion -> shared output/post outside helper"
        ),
        "Nikon_chain_mode": "parameterized" if parameterized else "cms-only",
        "fixed_strength": (
            experimental_nikon["fixed_strength"]
            if parameterized
            else controls["fixed_strength"]
        ),
        "fixed_bypassed": (
            experimental_nikon["fixed_strength"] == 0
            if parameterized
            else controls["fixed_strength"] == 0
        ),
        "CMS_only_active": not parameterized and controls["fixed_strength"] == 0,
        "experimental_film_active": parameterized,
        "experimental_film": experimental_tag,
        "experimental_calibration": (
            experimental_doc["calibration"]
            if parameterized and experimental_doc is not None
            else None
        ),
        "Nikon_CMS": nikon_receipt,
        "Nikon_output_profile": nikon_profile,
        "Nikon_output_pixels_sha256": nikon_sha,
        "working_profile": working_profile,
        "working_ICC_conversion": deepcopy(converted.receipt),
        "output_pixels_sha256": output_sha,
        "output_icc_sha256": ICC_SHA256,
        "Nikon_CMS_evaluate_calls": 1,
        "ICC_transform_calls": (
            pixels.shape[0] * pixels.shape[1] + ICC_CHUNK_PIXELS - 1
        )
        // ICC_CHUNK_PIXELS,
        "source_and_recipe_unchanged": True,
        "source_was_quantized_RGB16": True,
        "preclip_float_master_retained": False,
        "intermediate_Nikon_pixels_retained_in_return": False,
        "extra_OETF_or_working_gamma_applied": False,
        "reference_config_neutrality_independently_verified": False,
        "source_key_is_caller_provenance_not_authenticated_RAW_identity": True,
        "camera_to_scanner_calibration_verified": False,
        "original_full_scan_chain_executed": False,
        "original_ColorSync_equivalence_verified": False,
        "production_defaults_changed": False,
        "resource_arithmetic_modified": False,
        "working_conversion_buffers_are_bounded_but_original_Nikon_CMS_is_not_streamed": True,
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    return converted.pixels, working_icc, receipt
