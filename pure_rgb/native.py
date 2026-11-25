"""Adapter to the unchanged, pinned NegPy source tree."""

from dataclasses import replace
from pathlib import Path
import hashlib
import subprocess

import numpy as np

from . import BUNDLED_VENDOR, VENDOR_ROOT
from .engines import check_bundled_revision

NEGPY_SHA = "d8123a9b2a18fdaab5941347a91e402fa936d3dc"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_baseline():
    if BUNDLED_VENDOR:
        check_bundled_revision("negpy", NEGPY_SHA)
        return
    source = VENDOR_ROOT / "negpy"
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
    decode_cache=None, lens_correction=False, chromatic_aberration=False, lens_profile=None,
    frozen_roles=None,
):
    from negpy.features.rgbscan.logic import place_triplet
    from negpy.features.rgbscan.models import RgbScanConfig
    from negpy.features.retouch.models import RetouchConfig
    from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
    from .raw_import import (
        create_processor,
        probe_for_policy,
        STANDARD_DNG_IMPORT,
        STANDARD_MONO_IMPORT,
        IMPORT_POLICIES,
        checked_registration_check,
    )

    check_baseline()
    if raw_import not in IMPORT_POLICIES:
        raise ValueError("Unknown RAW import policy")
    if raw_import == STANDARD_MONO_IMPORT and lens_correction:
        raise ValueError("单色 RGB 暂不支持 RAW 内嵌镜头矫正")
    registration_check = checked_registration_check(raw_import, registration_check)
    paths = [Path(p).resolve(strict=True) for p in paths]
    if (
        len(paths) != 3
        or len(set(paths)) != 3
        or (raw_import != STANDARD_MONO_IMPORT and any(
            p.suffix.lower()
            != (".dng" if raw_import == STANDARD_DNG_IMPORT else ".rw2")
            for p in paths
        ))
    ):
        raise ValueError(
            "Select three distinct files with the explicit import policy's RAW suffix"
        )
    if frozen_roles is not None or raw_import == STANDARD_MONO_IMPORT:
        red, green, blue = map(str, paths)
    else:
        probe = probe_for_policy(raw_import)
        triplet, reason = place_triplet({str(p): probe(str(p)) for p in paths})
        if triplet is None:
            raise ValueError(f"Triplet classification failed: {reason}")
        red, green, blue = triplet.red, triplet.green, triplet.blue
    records = [
        {"role": role, "path": p, "sha256": sha256(p)}
        for role, p in zip("RGB", (red, green, blue))
    ]
    if frozen_roles is not None and records != frozen_roles:
        raise ValueError("已导入的通道记录与当前 RAW 不一致，请重新导入")
    config = replace(
        DEFAULT_WORKSPACE_CONFIG,
        process=replace(
            DEFAULT_WORKSPACE_CONFIG.process,
            linear_raw=True,
            highlight_reconstruction=0,
        ),
        rgbscan=RgbScanConfig(
            enabled=True, green_path=green, blue_path=blue
        ),
        retouch=RetouchConfig(),
    )
    processor = create_processor(
        raw_import,
        **({"lens_correction": True} if lens_correction else {}),
        **({"chromatic_aberration": True} if chromatic_aberration else {}),
        **({"lens_profile": lens_profile} if lens_profile else {}),
        **({"registration_check": registration_check} if registration_check else {}),
    )
    try:
        # The pre-attach document keys the render-mode decode cache; the
        # attached profile identifier is process-local and must stay out of it.
        document = config.to_dict() if decode_cache is not None else None
        if configure_source is not None:
            config = configure_source(processor, config, records)
        if decode_cache is not None and decode_cache.records == records:
            image = decode_cache.restore(processor, document, red, config)
            if image is not None:
                return processor, config, image, records
        image, ir, color_space = processor._load_source_f32(red, config)
        if decode_cache is not None and decode_cache.records == records:
            decode_cache.store(
                processor, document, red, image, ir, color_space
            )
    except BaseException:
        processor.cleanup()
        processor.destroy_all()
        raise
    return processor, config, image, records


def native_export_settings(config):
    from negpy.domain.models import ColorSpace, ExportFormat, TiffCompression

    return replace(
        config.export,
        export_fmt=ExportFormat.TIFF,
        export_bit_depth=16,
        export_color_space=ColorSpace.ADOBE_RGB.value,
        tiff_compression=TiffCompression.ZIP,
    )


def native_export(processor, config, records, *, prefer_gpu=False):
    export = native_export_settings(config)
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
    from .native_print_metrics import collect_print_values

    with span("native_export_encode"), collect_print_values(processor):
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
