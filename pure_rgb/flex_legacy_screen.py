"""Exact reduced-grid preview of the legacy Flex endpoint renderer."""

from copy import deepcopy

import numpy as np

from .cli import inspect_tiff
from .diagnostics import span
from .editing_render import experimental_positive
from .endpoint_study import FIT_REFERENCE_STRIDE, fit_endpoint_curve
from .engines import Analysis, check_reference_sources, flexcolor_research
from .native import encode_experiment
from .tone_study import GAMMA


MODE = "flex-endpoint"


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
