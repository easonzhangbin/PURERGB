"""Explicitly unvalidated Nikon CMS input experiments, separate from production.

This API does not declare camera/working RGB to be calibrated LS-5000 data.
It exists for comparing user-authorized hypotheses and optional pre-CMS fixed
tone strength. The default fixed strength is zero; no existing mode changes.
"""

import hashlib
import math
from numbers import Real

import numpy as np

from .engines import COOL_SHA
from .nikon_cms import CMSResult, _model

FIXED_CURVE_LABEL = "Nikon Scan4固定色调曲线（官方链路复现）"
FIXED_GENERATOR_TABLE_SHA256 = (
    "8729cae5a7aa551ae35926b80d097d73d92ecb6bde471130d6344c6c10ecbe7a"
)
BLOCK_PIXELS = 262144
INPUT_HYPOTHESES = {
    "camera-linear-transmission-rgb16": {
        "description": "相机三色线性透射值直接量化为RGB16，尚未反演或标定到scanner响应",
        "encoding": "clip(camera linear selected sensor bands,0,1)*65535; integer round",
    },
    "scene-normalized-transmission-rgb16": {
        "description": "场景窗口归一化的相机三色透射值，尚未标定到scanner响应",
        "encoding": "clip((signal-dark)/(high-dark),0,1)*65535; integer round",
    },
    "generic-density-positive-linear-rgb16": {
        "description": "通用密度正像直接线性编码，未经过scanner采集校准",
        "encoding": "positive density fraction*65535; no working OETF",
    },
    "generic-density-positive-working-rgb16": {
        "description": "通用密度正像经过工作TRC编码，假设可作为CMS设备码观察",
        "encoding": "positive density fraction after caller-declared working OETF",
    },
    "working-rgb-positive-rgb16": {
        "description": "已反演的工作RGB正像编码，假设可作为CMS设备码观察",
        "encoding": "caller-supplied positive working RGB encoded codes; no input ICC conversion inferred",
    },
}


def _pixels(rgb16):
    if (
        not isinstance(rgb16, np.ndarray)
        or rgb16.dtype != np.uint16
        or rgb16.ndim != 3
        or rgb16.shape[2] != 3
        or not rgb16.size
    ):
        raise ValueError("Nikon研究输入需非空H×W×3 RGB16整数编码")
    return rgb16


def _strength(strength):
    if (
        not isinstance(strength, Real)
        or isinstance(strength, (bool, np.bool_))
        or not math.isfinite(strength)
        or not 0 <= strength <= 1
    ):
        raise ValueError("fixed强度需0..1有限数值，默认0关闭")
    return float(strength)


def _hash(pixels):
    return hashlib.sha256(
        np.asarray(pixels, dtype="<u2", order="C").tobytes()
    ).hexdigest()


def _mix(rgb16, strength, table):
    if strength == 0:
        return rgb16
    if strength == 1:
        return table[rgb16]
    out = np.empty(rgb16.shape, dtype=np.uint16)
    rows = max(1, BLOCK_PIXELS // rgb16.shape[1])
    for start in range(0, rgb16.shape[0], rows):
        block = rgb16[start : start + rows]
        out[start : start + rows] = np.rint(
            (1 - strength) * block.astype(np.float64)
            + strength * table[block].astype(np.float64)
        ).astype(np.uint16)
    return out


def apply_fixed_strength(rgb16, strength):
    """Mix the pinned fixed table before CMS, in the supplied encoded domain.

    Zero returns the original array without lookup/allocation. One is exactly
    the original table. Partial strength is a PURE RGB research adaptation,
    using round-to-even integer quantization, not an original Nikon slider.
    The input is never changed by this operation.
    """
    pixels, value = _pixels(rgb16), _strength(strength)
    if value == 0:
        return pixels
    return _mix(pixels, value, _model(1).fixed_lut)


def _fixed_origin(table_sha256):
    """Attest this exact table under the independently probed x87 contract."""
    matched = table_sha256 == FIXED_GENERATOR_TABLE_SHA256
    return {
        "upstream": "rohanpandula/cool-colors",
        "commit": COOL_SHA,
        "extracted_upstream": True,
        "claimed_original_stage": "Nikon Scan4 LS-5000 fixed output-tone table",
        # Keep the legacy broad claim false: a local generator is not a
        # certification of the entire installer or running application.
        "original_installer_independently_verified": False,
        "original_generator_independently_verified": matched,
        "original_generator": {
            "module": "Nikon Scan 4.0.3 LS5000.md3",
            "module_sha256": "45afd6fb61a9517ff95d1896dc4257779c319310c2e8bbe75f3b4f3dada920af",
            "entry_va": "0x100111c0",
            "explicit_x87_control_word": "0x037f",
            "reference_table_sha256": FIXED_GENERATOR_TABLE_SHA256,
            "actual_table_sha256": table_sha256,
            "all_65536_entries_match": matched,
            "caller_order": "builder 0x100128f3 -> fixed 0x10012912",
            "runtime_FPU_captured": False,
            "all_runtime_precision_states_equivalent": False,
            "precision_boundary": "x87 0x027f differs by +1 at input 2982, 5139, 7296; runtime application CW not measured",
        },
        "official_general_purpose_lut": False,
    }


def evaluate_assumed_codes(
    rgb16,
    *,
    input_hypothesis,
    fixed_strength=0.0,
    threads=1,
):
    """Run user-authorized cross-domain hypotheses without changing strict APIs."""
    pixels, value = _pixels(rgb16), _strength(fixed_strength)
    if (
        not isinstance(input_hypothesis, str)
        or input_hypothesis not in INPUT_HYPOTHESES
    ):
        raise ValueError("需明确选择相机透射、通用密度正像或工作RGB正像研究假设")
    if type(threads) is not int or not 1 <= threads <= 16:
        raise ValueError("Nikon研究CMS线程数需1..16整数")
    before = _hash(pixels)
    model = _model(threads)
    device = _mix(pixels, value, model.fixed_lut)
    device_hash = _hash(device)
    result = model.evaluate(device)
    if _hash(pixels) != before or _hash(device) != device_hash:
        raise ValueError("Nikon研究CMS修改了输入编码")
    fixed_hash = hashlib.sha256(model.fixed_lut.tobytes()).hexdigest()
    return CMSResult(
        result,
        model.profile,
        {
            "schema": "pure-rgb-nikon-cms-assumed-domain",
            "version": 1,
            "input_domain": "explicitly-unvalidated-camera-or-working-rgb-hypothesis",
            "input_hypothesis": {
                "id": input_hypothesis,
                **INPUT_HYPOTHESES[input_hypothesis],
            },
            "source_pixels_sha256": before,
            "device_before_cms_pixels_sha256": device_hash,
            "output_pixels_sha256": _hash(result),
            "source_shape": list(pixels.shape),
            "source_dtype": "uint16",
            "source_unchanged": True,
            "fixed_curve_label": FIXED_CURVE_LABEL,
            "fixed_strength": value,
            "fixed_bypassed": value == 0,
            "fixed_lut_sha256": fixed_hash,
            "fixed_blend_formula": "rint((1-strength)*input + strength*fixed_lut[input]); before CMS in caller-supplied code domain",
            "fixed_strength_is_research_adaptation": True,
            "fixed_origin": _fixed_origin(fixed_hash),
            "stage_order": (
                ["assumed input codes", "fixed strength mix"]
                if value
                else ["assumed input codes", "fixed bypass"]
            )
            + ["captured CML4 stage1", "captured CML4 stage2"],
            "output_icc_sha256": hashlib.sha256(model.profile).hexdigest(),
            "output_extra_gamma_applied": False,
            "model": model.bundle_receipt(),
            "unvalidated_domain": True,
            "camera_to_scanner_calibration_verified": False,
            "acquisition_builder_executed": False,
            "factory_equivalent": False,
            "production_input_domain_guard_changed": False,
            "scope": "Explicit research hypothesis only; own Nikon output ICC is required. The exact fixed table has an independently matched original generator under explicit x87 0x037f, not a measured running-application FPU state; partial strength is a new adaptation.",
        },
    )
