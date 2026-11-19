"""Adapter to the unchanged, pinned NegPy source tree."""

from dataclasses import replace
from pathlib import Path
import hashlib
import subprocess

import numpy as np

from .engines import ROOT

NEGPY_SHA = "d8123a9b2a18fdaab5941347a91e402fa936d3dc"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_baseline():
    source = ROOT / "vendor/negpy"
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True
    )
    if actual != NEGPY_SHA or dirty:
        raise ValueError("NegPy must match the clean, pinned 0.68 baseline")


def open_triplet(
    paths, *, raw_import="legacy-v1", registration_check=None, configure_source=None,
    decode_cache=None, lens_correction=False,
):
    from negpy.features.rgbscan.logic import place_triplet
    from negpy.features.rgbscan.models import RgbScanConfig
    from negpy.features.retouch.models import RetouchConfig
    from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
    from .raw_import import (
        create_processor,
        probe_for_policy,
        STANDARD_DNG_IMPORT,
        IMPORT_POLICIES,
        checked_registration_check,
    )

    check_baseline()
    if raw_import not in IMPORT_POLICIES:
        raise ValueError("Unknown RAW import policy")
    registration_check = checked_registration_check(raw_import, registration_check)
    paths = [Path(p).resolve(strict=True) for p in paths]
    if (
        len(paths) != 3
        or len(set(paths)) != 3
        or any(
            p.suffix.lower()
            != (".dng" if raw_import == STANDARD_DNG_IMPORT else ".rw2")
            for p in paths
        )
    ):
        raise ValueError(
            "Select three distinct files with the explicit import policy's RAW suffix"
        )
    probe = probe_for_policy(raw_import)
    triplet, reason = place_triplet({str(p): probe(str(p)) for p in paths})
    if triplet is None:
        raise ValueError(f"Triplet classification failed: {reason}")
    records = [
        {"role": role, "path": p, "sha256": sha256(p)}
        for role, p in zip("RGB", (triplet.red, triplet.green, triplet.blue))
    ]
    config = replace(
        DEFAULT_WORKSPACE_CONFIG,
        process=replace(
            DEFAULT_WORKSPACE_CONFIG.process,
            linear_raw=True,
            highlight_reconstruction=0,
        ),
        rgbscan=RgbScanConfig(
            enabled=True, green_path=triplet.green, blue_path=triplet.blue
        ),
        retouch=RetouchConfig(),
    )
    processor = create_processor(
        raw_import,
        **({"lens_correction": True} if lens_correction else {}),
        **({"registration_check": registration_check} if registration_check else {}),
    )
    try:
        # The pre-attach document keys the render-mode decode cache; the
        # attached profile identifier is process-local and must stay out of it.
        document = config.to_dict() if decode_cache is not None else None
        if configure_source is not None:
            config = configure_source(processor, config, records)
        if decode_cache is not None and decode_cache.records == records:
            image = decode_cache.restore(processor, document, triplet.red, config)
            if image is not None:
                return processor, config, image, records
        image, ir, color_space = processor._load_source_f32(triplet.red, config)
        if decode_cache is not None and decode_cache.records == records:
            decode_cache.store(
                processor, document, triplet.red, image, ir, color_space
            )
    except BaseException:
        processor.cleanup()
        processor.destroy_all()
        raise
    return processor, config, image, records


def native_export(processor, config, records, *, prefer_gpu=False):
    from negpy.domain.models import ColorSpace, ExportFormat, TiffCompression

    export = replace(
        config.export,
        export_fmt=ExportFormat.TIFF,
        export_bit_depth=16,
        export_color_space=ColorSpace.ADOBE_RGB.value,
        tiff_compression=TiffCompression.ZIP,
    )
    if hasattr(processor, "pure_raw_import_token"):
        # Validate/replay source identity before binding the export cache token.
        processor._load_source_f32(records[0]["path"], config)
    identity = hashlib.sha256(
        (
            "".join(r["sha256"] for r in records)
            + getattr(processor, "pure_raw_import_token", "")
        ).encode()
    ).hexdigest()
    from .diagnostics import span

    with span("native_export_encode"):
        data, error = processor.process_export(
            records[0]["path"], config, export, identity, prefer_gpu=prefer_gpu
        )
    if data is None:
        raise RuntimeError(error)
    return data, export


def encode_experiment(processor, samples, export, transfer="linear"):
    from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
    from negpy.kernel.image.logic import working_oetf_encode

    if transfer not in ("linear", "working-encoded"):
        raise ValueError("Unknown experimental transfer contract")
    pixels = working_oetf_encode(samples) if transfer == "linear" else samples
    data, error = processor.encode_export(
        np.asarray(pixels), export, WORKING_COLOR_SPACE
    )
    if data is None:
        raise RuntimeError(error)
    return data
