"""Durable, source-bound JSON replay of the explicit Native CPU auxiliaries.

Only small scalar/bounds/axis data and their provenance are serialized. The V15
RGB-shared mask/window bundle is supplied separately and must match exactly.
Restore checks the current prepared source and never measures auxiliaries or
reads RAW. Checksums detect changed records; they are not authenticity signatures.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import negpy_core

from .frozen_native_cpu import _config_document, _supported_config
from .metering_snapshot import _check_source, _json_values, restore_zonal_metering
from . import native_masked_meters as meters

SCHEMA = "pure-rgb-native-auxiliary-snapshot"
VERSION = 1
SCOPE = "source-bound-c41-print-cpu-auxiliary-replay-v1"
MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class FrozenNativeAuxiliarySnapshot:
    canonical_json: str
    sha256: str

    def to_json(self):
        return self.canonical_json


@dataclass(frozen=True)
class RestoredNativeAuxiliarySnapshot:
    result: meters.NativeMaskedMeterResult
    source_identity: dict
    density_bundle_sha256: str
    sha256: str


def _require(value, message):
    if not value:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _keys(value, keys, name):
    _require(
        type(value) is dict and set(value) == set(keys), f"Malformed {name} fields"
    )


def _code_binding():
    """Pin the small adapter/provider sources and the actual compiled kernel."""
    from negpy.features.exposure import normalization

    here = Path(__file__).resolve().parent
    paths = {
        name: here / name
        for name in (
            "metering_aux_snapshot.py",
            "frozen_native_cpu.py",
            "native_masked_meters.py",
            "metering_snapshot.py",
            "metering_masks.py",
            "metering_regions.py",
            "zonal_metering.py",
            "density_matrix.py",
        )
    }
    paths.update(
        normalization=Path(normalization.__file__), kernel=Path(negpy_core.__file__)
    )
    return {
        "files": {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in paths.items()
        },
        "numpy": np.__version__,
        "python": list(sys.version_info[:2]),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "provider": meters.PROVIDER,
        "provider_version": meters.VERSION,
    }


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _no_duplicates(items):
    result = {}
    for key, value in items:
        _require(key not in result, "Duplicate auxiliary snapshot JSON key")
        result[key] = value
    return result


def _load(text):
    wrapper_sha = None
    if isinstance(text, FrozenNativeAuxiliarySnapshot):
        wrapper_sha = text.sha256
        text = text.to_json()
    _require(
        type(text) is str and len(text) <= MAX_SNAPSHOT_BYTES,
        "Auxiliary snapshot must be bounded JSON text",
    )
    _require(
        len(text.encode("utf-8")) <= MAX_SNAPSHOT_BYTES,
        "Auxiliary snapshot JSON too large",
    )
    try:
        document = json.loads(
            text, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant
        )
        _json_values(document)
    except (TypeError, ValueError, RecursionError, OverflowError) as error:
        raise ValueError("Invalid auxiliary snapshot JSON") from error
    if wrapper_sha is not None:
        _require(
            type(document) is dict and wrapper_sha == document.get("payload_sha256"),
            "Auxiliary snapshot wrapper checksum mismatch",
        )
    return document


def _float(value, name):
    _require(
        type(value) is float and np.isfinite(value), f"Invalid finite float {name}"
    )
    return value


def _triple(value, name):
    _require(type(value) is list and len(value) == 3, f"Invalid RGB triple {name}")
    return tuple(_float(x, name) for x in value)


def _restore_metrics(payload):
    _keys(payload, meters.METRIC_KEYS, "auxiliary metrics")
    result = {}
    for key in meters.BOUNDS_KEYS:
        _keys(payload[key], {"floors", "ceils"}, key)
        floors = _triple(payload[key]["floors"], key)
        ceils = _triple(payload[key]["ceils"], key)
        _require(
            all(a < b for a, b in zip(floors, ceils)), "Invalid auxiliary bounds span"
        )
        result[key] = meters.norm.LogNegativeBounds(floors, ceils)
    for key in meters.SCALAR_KEYS:
        result[key] = _float(payload[key], key)
    axis = payload["neutral_axis_refs"]
    if axis is not None:
        _require(type(axis) is list and len(axis) == 4, "Invalid neutral axis shape")
        axis = (
            _triple(axis[0], "neutral mid"),
            _triple(axis[1], "neutral high"),
            None if axis[2] is None else _triple(axis[2], "neutral low"),
            _float(axis[3], "neutral weight"),
        )
    result["neutral_axis_refs"] = axis
    clip = _triple(payload["scan_clip_fractions"], "scan clipping")
    _require(all(0 <= x <= 1 for x in clip), "Invalid clip fractions")
    result["scan_clip_fractions"] = clip
    _require(
        payload["density_above_base"] is None,
        "Unsupported auxiliary density diagnostic",
    )
    result["density_above_base"] = None
    return result


def freeze_native_auxiliary(
    source, config, density_bundle_json, current_identity, *, measured=None
):
    """Measure once, or persist an explicitly supplied validated earlier result.

    No renderer executes here. The full workspace config is bound deliberately;
    changing even a supported post setting requires a new explicit snapshot.
    """
    _supported_config(config)
    density = restore_zonal_metering(density_bundle_json, current_identity)
    _check_source(source, density.source_identity)
    before = _config_document(config)
    runtime = _code_binding()
    if measured is None:
        measured = meters.analyze_native_masked_meters(
            source, config, density_bundle_json, current_identity
        )
    meters.validate_native_masked_result(
        measured, source, config, density_bundle_json, current_identity
    )
    _require(
        _config_document(config) == before,
        "Configuration changed while freezing auxiliary snapshot",
    )
    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "scope": SCOPE,
        "source_identity": deepcopy(density.source_identity),
        "density_bundle_sha256": density.sha256,
        "input_config": before,
        "input_config_sha256": _digest(before),
        "effective_config": _config_document(measured.effective_config),
        "algorithm_binding": runtime,
        "native_result": {
            "metrics": meters.native_masked_metrics_payload(measured.metrics),
            "receipt": deepcopy(measured.receipt),
            "sha256": measured.sha256,
        },
    }
    payload["effective_config_sha256"] = _digest(payload["effective_config"])
    payload["payload_sha256"] = _digest(payload)
    text = _canonical(payload)
    # Use the same semantic checks for construction and disk restoration, without
    # invoking a meter for validation or trusting caller-supplied result objects.
    restore_native_auxiliary(
        text, source, config, density_bundle_json, current_identity
    )
    return FrozenNativeAuxiliarySnapshot(text, payload["payload_sha256"])


def restore_native_auxiliary(
    snapshot_json,
    source,
    config,
    density_bundle_json,
    current_identity,
    *,
    prefer_gpu=False,
):
    """Validate and reconstruct CPU scalar state, with zero auxiliary remeasurement."""
    _require(
        prefer_gpu is False, "Auxiliary snapshot supports CPU only; no GPU fallback"
    )
    _supported_config(config)
    document = _load(snapshot_json)
    _keys(
        document,
        {
            "schema",
            "version",
            "scope",
            "source_identity",
            "density_bundle_sha256",
            "input_config",
            "input_config_sha256",
            "effective_config",
            "effective_config_sha256",
            "algorithm_binding",
            "native_result",
            "payload_sha256",
        },
        "auxiliary snapshot",
    )
    checksum = document["payload_sha256"]
    payload = {k: v for k, v in document.items() if k != "payload_sha256"}
    _require(
        document["schema"] == SCHEMA
        and type(document["version"]) is int
        and document["version"] == VERSION
        and document["scope"] == SCOPE,
        "Unsupported auxiliary snapshot schema/version/scope",
    )
    _require(
        type(checksum) is str and checksum == _digest(payload),
        "Auxiliary snapshot checksum mismatch",
    )
    _require(
        document["algorithm_binding"] == _code_binding(),
        "Auxiliary algorithm/runtime binding changed",
    )
    density = restore_zonal_metering(density_bundle_json, current_identity)
    _require(
        document["source_identity"] == density.source_identity
        and document["density_bundle_sha256"] == density.sha256,
        "Auxiliary snapshot belongs to different source/geometry/mask/window",
    )
    _check_source(source, density.source_identity)
    current = _config_document(config)
    _require(
        document["input_config"] == current
        and document["input_config_sha256"] == _digest(current),
        "Auxiliary snapshot input configuration changed",
    )
    effective = replace(
        config,
        process=meters.frozen_process_for_window(
            config.process, density.window, source.shape
        ),
    )
    expected = _config_document(effective)
    _require(
        document["effective_config"] == expected
        and document["effective_config_sha256"] == _digest(expected),
        "Auxiliary snapshot effective configuration changed",
    )
    native = document["native_result"]
    _keys(native, {"metrics", "receipt", "sha256"}, "native auxiliary result")
    metrics = _restore_metrics(native["metrics"])
    receipt = native["receipt"]
    _require(
        type(receipt) is dict
        and receipt.get("provider") == meters.PROVIDER
        and receipt.get("mask_sha256")
        == (
            None
            if density.selection_mask is None
            else density.selection_mask.mask_sha256
        )
        and receipt.get("confirmed_exclusions")
        == density.receipt["effective_excluded_rois"]
        and receipt.get("requested_roi") == density.config["roi"]
        and receipt.get("source_pixels_sha256_before")
        == density.source_identity["source_pixels_sha256"]
        and receipt.get("source_pixels_sha256_after")
        == density.source_identity["source_pixels_sha256"]
        and receipt.get("GPU_supported") is False
        and receipt.get("source_values_changed") is False
        and type(receipt.get("RAW_decode_calls")) is int
        and receipt["RAW_decode_calls"] == 0
        and type(receipt.get("renderer_calls")) is int
        and receipt["renderer_calls"] == 0,
        "Auxiliary receipt support/domain changed",
    )
    result = meters.NativeMaskedMeterResult(
        metrics, deepcopy(receipt), effective, native["sha256"]
    )
    meters.validate_native_masked_result(
        result, source, config, density_bundle_json, current_identity
    )
    return RestoredNativeAuxiliarySnapshot(
        result, deepcopy(density.source_identity), density.sha256, checksum
    )
