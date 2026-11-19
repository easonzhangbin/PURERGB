"""Bounded-memory Flex post-stages with one frozen curve/color/ICC recipe.

This independent research executor never meters, compiles curves, estimates a
film base, decodes a RAW, or certifies camera-to-scanner calibration per tile.
"""

import hashlib
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .flex_color_adjustments import color_map14, color_map_parameters
from .flex_color_management import ICCBridgeResult, scanner_rgb14_icc_bridge
from .flex_pipeline import apply_profile_curves
from .flex_lightness import apply_lightness14, lightness_curve_parameters


def evaluate_post_stages(
    codes,
    tables,
    *,
    color_corr,
    saturation,
    apply_cc,
    input_icc,
    output_icc,
    input_icc_sha256,
    output_icc_sha256,
    assumption,
    input_domain,
    out=None,
    chunk_pixels=262_144,
    intent=1,
    workers=1,
    lightness=0,
    lightness_anchor_preference=None,
    apply_sliders=True,
    lightness_curve_present=True,
):
    """Final LUTs -> ColorMap14 -> optional ApplyYGrad -> explicit ICC.

    Independent ``out`` may be an on-disk uint16 memmap. Source codes always
    remain untouched. ``tables`` must already be frozen full RGB14 tables;
    all tiles share exactly those tables, ColorMap controls and ICC bytes.
    Optional workers use independent ICC calls and at most ``workers`` pending
    tiles. Destination writes and digest updates commit in source order.
    """
    if input_domain != "explicit-prepared-rgb14-research-carrier":
        raise ValueError("Explicit prepared research-carrier domain required")
    if (
        not isinstance(codes, np.ndarray)
        or codes.dtype != np.uint16
        or codes.ndim != 3
        or codes.shape[2] != 3
        or not codes.size
        or not codes.flags.c_contiguous
    ):
        raise ValueError("Nonempty C-contiguous uint16 HWC RGB14 required")
    if type(chunk_pixels) is not int or not 1 <= chunk_pixels <= 4_194_304:
        raise ValueError("Tile must contain 1..4194304 pixels")
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Explicit worker count 1..4 required")
    parameters = color_map_parameters(
        color_corr=color_corr, saturation=saturation, apply_cc=apply_cc
    )
    lightness_recipe = lightness_curve_parameters(
        lightness,
        anchor_preference=lightness_anchor_preference,
        enabled=apply_sliders,
        curve_present=lightness_curve_present,
    )
    if out is None:
        out = np.empty_like(codes)
    if (
        not isinstance(out, np.ndarray)
        or out.dtype != np.uint16
        or out.shape != codes.shape
        or not out.flags.c_contiguous
        or not out.flags.writeable
        or np.shares_memory(codes, out)
    ):
        raise ValueError("Separate writable C-contiguous uint16 output required")
    # Validate tables/ICC and the explicit domain before any destination writes.
    probe = apply_profile_curves(np.zeros((1, 1, 3), np.uint16), tables)
    if lightness_recipe["stage_active"]:
        probe, probe_lightness = apply_lightness14(
            probe,
            lightness=lightness,
            anchor_preference=lightness_anchor_preference,
            enabled=apply_sliders,
            curve_present=lightness_curve_present,
            input_domain="flex-post-colormap-rgb14",
        )
        lightness_recipe = {
            **lightness_recipe,
            "table_sha256": probe_lightness["table_sha256"],
            "table_entries": probe_lightness["table_entries"],
            "preflight_parameters_validated": True,
            "per_image_clipping_statistics_collected": False,
        }
    scanner_rgb14_icc_bridge(
        probe,
        input_icc,
        output_icc,
        source_sha256=input_icc_sha256,
        output_sha256=output_icc_sha256,
        input_domain="assumed-flextight-scanner-rgb14",
        assumption=assumption,
        intent=intent,
        chunk_pixels=chunk_pixels,
    )
    source, destination = codes.reshape(-1, 3), out.reshape(-1, 3)
    source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    times = {key: 0.0 for key in ("final_luts", "ColorMap14", "Lightness", "ICC")}
    tiles, max_in_flight = 0, 1

    def evaluate_tile(offset):
        local = {}
        block = source[offset : offset + chunk_pixels].reshape(1, -1, 3)
        stage = time.monotonic()
        adjusted = apply_profile_curves(block, tables)
        local["final_luts"] = time.monotonic() - stage
        stage = time.monotonic()
        color_map14(
            adjusted,
            color_corr=color_corr,
            saturation=saturation,
            apply_cc=apply_cc,
            input_domain="flex-post-gradation-rgb14",
            out=adjusted,
            chunk_pixels=chunk_pixels,
        )
        local["ColorMap14"] = time.monotonic() - stage
        stage = time.monotonic()
        if lightness_recipe["stage_active"]:
            adjusted, _ = apply_lightness14(
                adjusted,
                lightness=lightness,
                anchor_preference=lightness_anchor_preference,
                enabled=apply_sliders,
                curve_present=lightness_curve_present,
                input_domain="flex-post-colormap-rgb14",
                out=adjusted,
            )
        local["Lightness"] = (
            time.monotonic() - stage if lightness_recipe["stage_active"] else 0.0
        )
        stage = time.monotonic()
        converted = scanner_rgb14_icc_bridge(
            adjusted,
            input_icc,
            output_icc,
            source_sha256=input_icc_sha256,
            output_sha256=output_icc_sha256,
            input_domain="assumed-flextight-scanner-rgb14",
            assumption=assumption,
            intent=intent,
            chunk_pixels=chunk_pixels,
        )
        local["ICC"] = time.monotonic() - stage
        return offset, converted.pixels, local

    def commit(offset, pixels, local):
        nonlocal tiles
        count = pixels.shape[1]
        source_hash.update(memoryview(source[offset : offset + count]).cast("B"))
        destination[offset : offset + count] = pixels.reshape(-1, 3)
        output_hash.update(memoryview(pixels).cast("B"))
        for key in times:
            times[key] += local[key]
        tiles += 1

    started = time.monotonic()
    if workers == 1:
        for offset in range(0, len(source), chunk_pixels):
            commit(*evaluate_tile(offset))
    else:
        offsets = iter(range(0, len(source), chunk_pixels))
        pending = deque()
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="flex-icc"
        ) as pool:
            for offset in offsets:
                pending.append(pool.submit(evaluate_tile, offset))
                if len(pending) == workers:
                    break
            max_in_flight = len(pending)
            while pending:
                # Retrieve source order even if a later tile completes first.
                future = pending.popleft()
                commit(*future.result())
                del future
                offset = next(offsets, None)
                if offset is not None:
                    pending.append(pool.submit(evaluate_tile, offset))
    return ICCBridgeResult(
        out,
        output_icc,
        {
            "order": "frozen final RGB14 LUTs -> ColorMap14 -> ApplyYGrad (explicit Lightness or bypass) -> assumed scanner ICC -> actual output RGB16",
            "input_domain": input_domain,
            "assumption": assumption,
            "source_rgb14_pixels_sha256": source_hash.hexdigest(),
            "output_rgb16_pixels_sha256": output_hash.hexdigest(),
            "input_icc_sha256": input_icc_sha256,
            "output_icc_sha256": output_icc_sha256,
            "shape": list(codes.shape),
            "dtype": "uint16",
            "table_sha256_rgb": [
                hashlib.sha256(t.tobytes()).hexdigest() for t in tables
            ],
            "ColorMap": parameters,
            "Lightness": lightness_recipe,
            "chunk_pixels": chunk_pixels,
            "tiles": tiles,
            "maximum_tile_pixels": min(chunk_pixels, len(source)),
            "wall_seconds": time.monotonic() - started,
            "stage_seconds": times,
            "stage_seconds_scope": "Sum of per-tile wall times; may overlap across workers, not additive chain wall time",
            "workers": workers,
            "maximum_in_flight_tiles": max_in_flight,
            "ICC_worker_state": "Independent scanner_rgb14_icc_bridge/cms_transform call per tile; no shared CMM handle",
            "output_commit_order": "Source pixel order; single ordered writer, non-overlapping slices and ordered checksums",
            "meter_or_refit_calls": 0,
            "extra_gamma_or_oetf_applied": False,
            "camera_to_scanner_calibration_verified": False,
            "original_colorsync_equivalence_verified": False,
        },
    )
