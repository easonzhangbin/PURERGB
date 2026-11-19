"""Single original RAW on the unchanged NegPy B&W decode path.

Receipts bind the decoded oriented source before any density or print edits.
No classification, registration, triplet assembly, forced neutral WB or DNG
conversion is introduced. The source is an owned readonly float32 copy, so the
SDK can reuse or modify its own decode cache independently.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .bw_recipe import BW_SOURCE_MODE, BW_SUFFIXES
from .native import NEGPY_SHA, check_baseline, sha256
from .raw_import import file_identity


SCHEMA = "pure-rgb-single-bw-source"
_RECEIPT_KEYS = {
    "schema",
    "version",
    "source_mode",
    "raw_import",
    "inputs",
    "raw_identity",
    "negpy_commit",
    "loader",
    "loader_contract",
    "source_shape",
    "source_pixels_sha256",
    "source_color_space",
    "geometry",
    "receipt_sha256",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _pixels(source, *, readonly=True):
    _require(
        isinstance(source, np.ndarray)
        and source.dtype == np.float32
        and source.ndim == 3
        and source.shape[-1] == 3
        and min(source.shape[:2]) >= 2
        and source.flags.c_contiguous
        and (not readonly or not source.flags.writeable),
        "B&W decoded source requires contiguous readonly HWC float32 RGB",
    )
    digest = hashlib.sha256()
    for start in range(0, source.shape[0], 128):
        block = source[start : start + 128]
        _require(np.isfinite(block).all(), "B&W source contains nonfinite values")
        digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


def _loader_contract(config):
    from negpy.features.process.models import ProcessMode
    from negpy.features.process.logic import (
        effective_highlight_reconstruction,
        effective_linear_raw,
        highlight_reconstruction_bakes_wb,
        demosaic_token,
    )
    from negpy.features.process.path import positive_decode
    from negpy.features.process.sensor import sensor_token
    from negpy.features.rgbscan.logic import rgbscan_token
    from negpy.features.stitch.models import stitch_token
    from negpy.features.hdr.models import hdr_token
    from negpy.features.flatfield.logic import flatfield_token
    from negpy.services.rendering.lens import (
        lens_decode_token,
        metadata_lens_corrections,
    )
    from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG

    _require(
        config.process.process_mode == ProcessMode.BW
        and config.process.source_color is False
        and not config.rgbscan.enabled
        and not config.rgbscan.green_path
        and not config.rgbscan.blue_path,
        "B&W source requires the explicit single-file NegPy B&W configuration",
    )
    _require(
        config.hdr == DEFAULT_WORKSPACE_CONFIG.hdr
        and config.stitch == DEFAULT_WORKSPACE_CONFIG.stitch
        and config.flatfield == DEFAULT_WORKSPACE_CONFIG.flatfield
        and config.geometry == DEFAULT_WORKSPACE_CONFIG.geometry
        and not config.retouch.ir_dust_remove,
        "B&W source has unsupported assembly, calibration or geometry",
    )
    linear = effective_linear_raw(config.process)
    bake = highlight_reconstruction_bakes_wb(config.process)
    return {
        "process_mode": str(config.process.process_mode),
        "source_color": False,
        "linear_raw": config.process.linear_raw,
        "effective_linear_raw": linear,
        "white_balance_policy": "NegPy camera WB"
        if bake or not linear
        else "NegPy neutral WB",
        "gamma": [1, 1],
        "positive_decode": positive_decode(config.process),
        "effective_highlight_reconstruction": effective_highlight_reconstruction(
            config.process
        ),
        "highlight_reconstruction_bakes_wb": bake,
        "rgbscan_token": rgbscan_token(config.rgbscan),
        "stitch_token": stitch_token(config.stitch),
        "hdr_token": hdr_token(config.hdr),
        "flatfield_token": flatfield_token(config.flatfield),
        "sensor_token": sensor_token(config.process),
        "demosaic_export_token": demosaic_token(config.process.demosaic_export),
        "lens_decode_token": lens_decode_token(
            metadata_lens_corrections(config), config.flatfield
        ),
        "fast_decode": False,
        "unmix": True,
        "ir_dust_remove": False,
    }


def bw_source_config():
    from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
    from negpy.features.process.models import ProcessMode, with_process_mode
    from negpy.features.rgbscan.models import RgbScanConfig
    from negpy.features.retouch.models import RetouchConfig

    config = with_process_mode(DEFAULT_WORKSPACE_CONFIG, ProcessMode.BW)
    return replace(
        config,
        process=replace(config.process, source_color=False),
        rgbscan=RgbScanConfig(enabled=False, green_path="", blue_path="", align=False),
        retouch=RetouchConfig(),
        exposure=replace(config.exposure, paper_profile="neutral"),
    )


def _records(records):
    _require(
        type(records) is list and len(records) == 1, "B&W requires one source record"
    )
    record = records[0]
    _require(
        type(record) is dict
        and set(record) == {"role", "path", "sha256"}
        and record["role"] == "BW"
        and type(record["path"]) is str
        and str(Path(record["path"]).resolve()) == record["path"]
        and Path(record["path"]).suffix.lower() in BW_SUFFIXES
        and type(record["sha256"]) is str
        and len(record["sha256"]) == 64
        and all(c in "0123456789abcdef" for c in record["sha256"]),
        "Invalid single B&W source record",
    )
    return record


def checked_bw_source_receipt(
    processor, config, source, records, *, source_color_space=None
):
    """Freeze actual source pixels/configuration after the original decoder ran."""
    record = _records(records)
    before = file_identity(record["path"])
    _require(sha256(record["path"]) == record["sha256"], "B&W RAW content changed")
    color_space = source_color_space
    if color_space is None:
        color_space = getattr(processor, "pure_bw_source_color_space", None)
    _require(
        type(color_space) is str and bool(color_space),
        "Actual B&W loader color space required",
    )
    pixels_sha = _pixels(source)
    receipt = {
        "schema": SCHEMA,
        "version": 1,
        "source_mode": BW_SOURCE_MODE,
        "raw_import": "legacy-v1",
        "inputs": deepcopy(records),
        "raw_identity": list(before),
        "negpy_commit": NEGPY_SHA,
        "loader": "unchanged NegPy ImageProcessor._load_source_f32",
        "loader_contract": _loader_contract(config),
        "source_shape": list(source.shape),
        "source_pixels_sha256": pixels_sha,
        "source_color_space": color_space,
        "geometry": {
            "coordinate_space": "full EXIF-oriented NegPy decoded source pixels",
            "additional_crop_or_registration": "none",
            "shape": list(source.shape),
        },
    }
    _require(
        file_identity(record["path"]) == before, "B&W RAW changed during source freeze"
    )
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt


def open_bw_source(path):
    """Decode one original RW2/DNG; no scene classifier or RGB sibling access."""
    from negpy.services.rendering.image_processor import ImageProcessor

    _require(type(path) is str, "B&W RAW path must be a string")
    path = str(Path(path).expanduser().resolve(strict=True))
    _require(Path(path).suffix.lower() in BW_SUFFIXES, "B&W source must be RW2 or DNG")
    check_baseline()
    before = file_identity(path)
    records = [{"role": "BW", "path": path, "sha256": sha256(path)}]
    config = bw_source_config()
    processor = ImageProcessor(use_gpu=False)
    try:
        source, _, color_space = processor._load_source_f32(path, config)
        _require(
            isinstance(source, np.ndarray) and source.dtype == np.float32,
            "Original NegPy B&W loader did not return float32 pixels",
        )
        # Never mark the SDK's own cached ndarray readonly.
        source = np.array(source, copy=True, order="C")
        source.setflags(write=False)
        _require(
            type(color_space) is str and bool(color_space),
            "Actual B&W loader color space required",
        )
        processor.pure_bw_source_color_space = color_space
        _require(file_identity(path) == before, "B&W RAW changed during decoding")
        processor.pure_bw_source_receipt = checked_bw_source_receipt(
            processor, config, source, records
        )
    except BaseException:
        processor.cleanup()
        processor.destroy_all()
        raise
    return processor, config, source, records


def validate_bw_source(manifest, config, source, records, *, source_color_space):
    """Recheck source pixels/geometry and decode settings before replaying bounds."""
    expected = manifest.get("bw_source_receipt")
    _require(
        type(expected) is dict and set(expected) == _RECEIPT_KEYS,
        "Frozen B&W source receipt missing or has unknown fields",
    )
    payload = {key: value for key, value in expected.items() if key != "receipt_sha256"}
    _require(
        expected.get("receipt_sha256") == _digest(payload),
        "Frozen B&W source receipt changed",
    )
    _require(
        expected.get("schema") == SCHEMA
        and type(expected.get("version")) is int
        and expected["version"] == 1
        and expected.get("source_mode") == BW_SOURCE_MODE
        and expected.get("raw_import") == "legacy-v1"
        and expected.get("negpy_commit") == NEGPY_SHA,
        "Unsupported B&W source receipt",
    )
    _require(
        expected.get("inputs") == records == manifest.get("inputs"),
        "B&W input identity differs",
    )
    record = _records(records)
    before = file_identity(record["path"])
    _require(
        expected.get("raw_identity") == list(before)
        and sha256(record["path"]) == record["sha256"],
        "B&W source file identity changed",
    )
    _require(
        list(source.shape)
        == expected.get("source_shape")
        == manifest.get("source_shape")
        and expected.get("geometry")
        == {
            "coordinate_space": "full EXIF-oriented NegPy decoded source pixels",
            "additional_crop_or_registration": "none",
            "shape": list(source.shape),
        }
        and _pixels(source) == expected.get("source_pixels_sha256"),
        "B&W source pixels or oriented geometry changed",
    )
    _require(
        _loader_contract(config) == expected.get("loader_contract")
        and source_color_space == expected.get("source_color_space")
        and expected["loader"] == "unchanged NegPy ImageProcessor._load_source_f32",
        "B&W decode configuration or source color space changed",
    )
    _require(
        file_identity(record["path"]) == before,
        "B&W RAW changed during replay validation",
    )
    return deepcopy(expected)
