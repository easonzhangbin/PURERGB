"""Exact reduced-grid preview of the legacy Flex endpoint renderer."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json

import numpy as np
import tifffile

from .cli import inspect_tiff
from .diagnostics import span
from .editing_render import experimental_positive
from .endpoint_study import FIT_REFERENCE_STRIDE, fit_endpoint_curve
from .engines import Analysis, check_reference_sources, flexcolor_research
from .native import encode_experiment
from .tone_study import GAMMA


MODE = "flex-endpoint"


def saved_native_fit_source(owner, recipe, certificate, config):
    """A saved neutral Native image can supply the exact legacy fit grid."""
    from .flex_draft_preview_v40 import _file_identity
    from .shared_density_metering_v43 import window_from_document
    from .tone_study import ICC_SHA256
    from .workspace import output_rectangle

    manifest, saved = owner.manifest, owner.recipe
    record = manifest.get("results", {}).get("native")
    roi, shape = recipe["roi"], manifest["source_shape"]
    saved_config = manifest.get("native_config")
    if (
        not record or record.get("kind") != "NegPy unchanged renderer"
        or record.get("icc_sha256") != ICC_SHA256 or manifest.get("icc_sha256") != ICC_SHA256
        or saved.get("measurement_method") != "negpy" or saved.get("roi") != roi
        or manifest.get("roi") != roi or saved.get("paper") != recipe["paper"]
        or saved["exposure_ev"] != 0 or saved.get("output_orientation")
        or any(saved.get(key) for key in (
            "native_controls", "native_float_controls", "native_spatial_controls",
            "native_film", "highlight_controls", "density_metering", "flatfield",
            "common_positive_controls",
        ))
        or any(saved.get(key, {}).get("native") for key in (
            "output_controls", "post_controls",
        ))
        or any(record.get(key) for key in (
            "native_controls", "post_controls", "common_positive_controls",
        ))
        or not isinstance(saved_config, dict)
        or (manifest.get("raw_import_receipt") or {}).get("source_pixels_sha256") != certificate.sha256
        or list(certificate.source.shape) != shape
    ):
        return None

    # The import and live worker have separate scratch directories. Only the
    # two filesystem destinations are irrelevant to rendering pixels in memory.
    paths = {"userDir", "export_path"}
    active_config = json.loads(json.dumps(config.to_dict()))
    if {key: value for key, value in saved_config.items() if key not in paths} != {
        key: value for key, value in active_config.items() if key not in paths
    }:
        return None

    a, b, c, d = canvas = output_rectangle(saved, roi, shape)
    x0, y0, x1, y1 = roi
    if (manifest.get("output_roi") != canvas or record.get("shape") != [d - b, c - a, 3]
            or not (a <= x0 < x1 <= c and b <= y0 < y1 <= d)):
        return None
    try:
        window = window_from_document(manifest.get("density_window"))
        if window.provider != "negpy-original" or list(window.analysis.roi) != roi:
            return None
        identity = _file_identity(record["path"])
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return None
    return record, identity, (x0 - a, y0 - b, x1 - a, y1 - b), dict(
        log_dense=list(window.log_dense), log_thin=list(window.log_thin),
    )


def read_saved_native_fit(binding, export):
    """Verify the saved TIFF and decode only strips intersecting fit samples."""
    from .flex_draft_preview_v40 import _file_identity, verified_tiff_icc
    from .live_worker import preview_checkpoint

    record, identity, (x0, y0, x1, y1), bounds = binding
    checkpoint = preview_checkpoint()
    with span("flex_preview_saved_native_fit", stride=FIT_REFERENCE_STRIDE):
        icc, verified = verified_tiff_icc(record)
        if identity != verified:
            raise ValueError("Flex 保存参考在认证期间发生变化")
        checkpoint()
        with tifffile.TiffFile(record["path"]) as tif:
            page = tif.pages[0]
            rows_per_strip = page.rowsperstrip
            # Writer layout in workflow_render.save. Other TIFF layouts use
            # the established renderer instead of a partial read guess.
            if (len(tif.pages) != 1 or page.is_tiled or page.planarconfig != 1
                    or page.photometric != tifffile.PHOTOMETRIC.RGB
                    or page.compression != tifffile.COMPRESSION.DEFLATE
                    or page.tags.get(274) is None or page.tags[274].value != 1
                    or not 0 < rows_per_strip * page.imagewidth * 6 <= 16 * 1024**2):
                raise ValueError("Flex 保存参考不是支持的 RGB16 条带布局")
            step = FIT_REFERENCE_STRIDE
            rows = np.arange(y0, y1, step)
            strips = list(map(int, np.unique(rows // rows_per_strip)))
            pixels = np.empty((rows.size, len(range(x0, x1, step)), 3), dtype=np.uint16)
            offsets = [page.dataoffsets[i] for i in strips]
            lengths = [page.databytecounts[i] for i in strips]
            batches = tif.filehandle.read_segments(offsets, lengths, indices=strips,
                                                      buffersize=16 * 1024**2, flat=False)

            decode_segment = page.decode

            def decode(segment):
                return decode_segment(*segment)

            decoded_rows = 0
            with ThreadPoolExecutor(max_workers=4) as workers:
                for batch in batches:
                    checkpoint()
                    for data, position, _ in workers.map(decode, batch):
                        row = position[2]
                        if data is None or data.ndim != 4 or data.shape[0] != 1 or data.shape[2:] != (page.imagewidth, 3):
                            raise ValueError("Flex 保存参考条带缺少原始网格")
                        start, stop = np.searchsorted(rows, [row, row + data.shape[1]])
                        pixels[start:stop] = data[0, rows[start:stop] - row, x0:x1:step]
                        decoded_rows += stop - start
            if decoded_rows != rows.size:
                raise ValueError("Flex 保存参考条带未覆盖完整拟合网格")
        if _file_identity(record["path"]) != identity:
            raise ValueError("Flex 保存参考在采样期间发生变化")
    checkpoint()
    pixels.setflags(write=False)
    return dict(pixels=pixels, icc=icc, export=export, bounds=bounds, tone_backend="cpu",
                saved_native_identity=identity)


def _grid(source, rectangle, step):
    x0, y0, x1, y1 = rectangle
    return source[y0:y1:step, x0:x1:step]


def _baseline_codes(processor, source, analysis, export, icc):
    # Preserve the original working-encoded float-to-RGB16 boundary before
    # fitting or editing. Only the selected pixels enter the Flex conversion.
    samples, _ = flexcolor_research(np.maximum(source, 0), analysis)
    payload = encode_experiment(processor, samples, export, transfer="working-encoded")
    pixels, actual_icc = inspect_tiff(payload)
    if pixels.shape != source.shape or actual_icc != icc:
        raise ValueError("Flex 预览工作 ICC 或采样网格不一致")
    return pixels


def _linear(pixels):
    return (pixels.astype(float) / 65535) ** GAMMA


def flex_legacy_sample(owner, recipe, session, cached, phase, step):
    """Fit with the original ROI-origin samples, then render only the display grid."""
    from .live_worker import preview_checkpoint

    if recipe.get("flex_research_controls") or recipe.get("flex_profile"):
        raise ValueError("Flex 原生密度快路仅支持旧版端点配平")
    state = cached.setdefault("flex_legacy", {})
    document = recipe.get("engine_controls", {}).get(MODE)
    key = (*phase, step, document)
    if state.get("key") == key:
        return state["pixels"], True

    checkpoint = preview_checkpoint()
    roi = recipe["roi"]
    bounds = cached["bounds"]
    analysis = Analysis(
        tuple(np.power(10., bounds["log_dense"])),
        tuple(np.power(10., bounds["log_thin"])),
        tuple(roi),
        None,
    )
    export, icc = cached["export"], cached["icc"]
    if "fit" not in state:
        check_reference_sources()
        with span("flex_preview_endpoint_fit", stride=FIT_REFERENCE_STRIDE):
            raw_fit = _grid(session.source, roi, FIT_REFERENCE_STRIDE)
            fit_codes = _baseline_codes(session.processor, raw_fit, analysis, export, icc)
            state["fit"] = fit_endpoint_curve(_linear(fit_codes), _linear(cached["pixels"]), stride=1)
        checkpoint()

    with span("flex_preview_screen_render", step=step):
        sample = _grid(session.source, phase, step)
        codes = _baseline_codes(session.processor, sample, analysis, export, icc)
        pixels, _ = experimental_positive(
            MODE, _linear(codes), state["fit"], document, sample, analysis,
            session.processor, export, expected_icc=icc,
        )
    checkpoint()
    pixels.setflags(write=False)
    state.update(key=deepcopy(key), pixels=pixels)
    return pixels, False
