"""Experimental, explicitly-uncalibrated film-style chains (FlexColor / Nikon).

This module composes the already verified original-math kernels
(``flex_pipeline`` RGB14 curve stages and ``nikon_parameterized`` analyzer/LUT/
fixed/CMS joins) into opt-in film curve styles that the production adapters
deliberately freeze or bypass.

Hard contract: every entry point requires and emits
``calibration == "uncalibrated-hypothesis"``. Nothing here claims scanner
acquisition binding, camera-to-scanner calibration, captured application
preferences, or factory-preset equivalence. Production defaults, frozen
recipes and existing render contracts are never mutated by this module;
callers opt in through a separate versioned recipe document.
"""

from copy import deepcopy
import json

import numpy as np

SCHEMA = "pure-rgb-experimental-film-controls"
VERSION = 1
CALIBRATION = "uncalibrated-hypothesis"
MAX_BYTES = 8192

FLEX_FILM_KINDS = ("color-negative",)
NIKON_FILM_KINDS = ("color-negative",)
NIKON_PARAMETERIZED_MODES = ("cms-only", "parameterized")
# Re-exported for callers; each kernel owns its authoritative hypothesis set.
from .nikon_parameterized import INPUT_HYPOTHESES as NIKON_INPUT_HYPOTHESES  # noqa: E402
from .nikon_cms_research import INPUT_HYPOTHESES as NIKON_CMS_INPUT_HYPOTHESES  # noqa: E402


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _finite_number(value, low, high, name):
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
        and low <= value <= high,
        f"{name} 必须为 [{low}, {high}] 内的有限数值",
    )
    return float(value)


def _boolean(value, name):
    _require(type(value) is bool, f"{name} 必须为明确布尔值")
    return value


def checked_flex_film(document):
    """Validate the FlexColor experimental film sub-document."""
    if document is None:
        return None
    _require(isinstance(document, dict), "Flex 实验胶片参数必须为对象")
    result = deepcopy(document)
    gamma_neg = _boolean(result.get("gamma_neg", False), "gamma_neg")
    channel_negative = result.get("channel_negative", [False, False, False])
    _require(
        isinstance(channel_negative, (list, tuple))
        and len(channel_negative) == 3
        and all(type(v) is bool for v in channel_negative),
        "channel_negative 必须为三个布尔值（R/G/B）",
    )
    film_kind = result.get("film_kind", "color-negative")
    _require(film_kind in FLEX_FILM_KINDS, f"Flex 实验链当前仅支持 {FLEX_FILM_KINDS}")
    var_gamma = result.get("var_gamma")
    if var_gamma is not None:
        # Mirror the original GenGamma14 admissible interval; the kernel itself
        # falls back to 0.2 outside it, but an experimental control must fail
        # loudly instead of silently substituting a different curve.
        _finite_number(var_gamma, 0.0991, 10.0999, "var_gamma")
    enhanced_shadow = _boolean(result.get("enhanced_shadow", False), "enhanced_shadow")
    if not (gamma_neg or any(channel_negative)):
        return None
    return {
        "gamma_neg": gamma_neg,
        "channel_negative": tuple(channel_negative),
        "film_kind": film_kind,
        "var_gamma": var_gamma,
        "enhanced_shadow": enhanced_shadow,
    }


def checked_nikon_film(document):
    """Validate the Nikon experimental film sub-document."""
    if document is None:
        return None
    _require(isinstance(document, dict), "Nikon 实验胶片参数必须为对象")
    result = deepcopy(document)
    mode = result.get("mode", "cms-only")
    _require(mode in NIKON_PARAMETERIZED_MODES, f"未知 Nikon 实验模式: {mode}")
    film_kind = result.get("film_kind", "color-negative")
    _require(film_kind in NIKON_FILM_KINDS, f"Nikon 实验链当前仅支持 {NIKON_FILM_KINDS}")
    fixed_strength = _finite_number(
        result.get("fixed_strength", 0.0), 0.0, 1.0, "fixed_strength"
    )
    input_hypothesis = result.get("input_hypothesis")
    allowed_hypotheses = (
        NIKON_CMS_INPUT_HYPOTHESES
        if mode == "cms-only"
        else NIKON_INPUT_HYPOTHESES
    )
    if input_hypothesis is None:
        input_hypothesis = (
            "working-rgb-positive-rgb16"
            if mode == "cms-only"
            else "synthetic-rgb16-probe"
        )
    _require(
        input_hypothesis in allowed_hypotheses,
        "Nikon 实验链需明确选择该模式允许的未绑定输入假设",
    )
    threads = result.get("threads", 1)
    _require(type(threads) is int and 1 <= threads <= 16, "threads 必须为 1..16 整数")
    if mode == "cms-only" and fixed_strength == 0.0:
        # Pure CMS with no fixed curve is the production default; an
        # experimental document that changes nothing has no effect to attest.
        return None
    normalized = {
        "mode": mode,
        "film_kind": film_kind,
        "fixed_strength": fixed_strength,
        "input_hypothesis": input_hypothesis,
        "threads": threads,
    }
    if mode == "parameterized":
        # The analyzer/LUT join needs explicit per-channel scalars and control
        # points. A deterministic synthetic default is offered for experiments,
        # never presented as a capture-derived or calibrated result.
        normalized["analyzer"] = _checked_analyzer(result.get("analyzer"))
    return normalized


# Deterministic synthetic analyzer used when a parameterized experiment does
# not declare its own scalars. Values mirror the isolated kernel test fixtures;
# they are not derived from any prescan, exposure tick or physical scanner.
DEFAULT_ANALYZER = {
    "scalars": [-0.5, -3.0, 1.9, -3.0, 1.0, 2.5, 0.3],
    "control_x": [round(float(v), 6) for v in np.linspace(0.0, 3.1, 32)],
    "control_y": [round(float(v), 6) for v in np.linspace(-3.0, 1.0, 32)],
}


def _checked_channel_analyzer(doc, label):
    _require(isinstance(doc, dict), f"{label} 分析器参数必须为对象")
    scalars = doc.get("scalars")
    _require(
        isinstance(scalars, (list, tuple)) and len(scalars) == 7
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
            for v in scalars
        ),
        f"{label} scalars 必须为 7 个有限数值（A,B,C,E,a,f,g）",
    )
    points = {}
    for axis in ("control_x", "control_y"):
        values = doc.get(axis)
        _require(
            isinstance(values, (list, tuple)) and len(values) == 32
            and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
                for v in values
            ),
            f"{label} {axis} 必须为 32 个有限数值",
        )
        points[axis] = [float(v) for v in values]
    _require(
        all(np.diff(points["control_x"]) >= 0),
        f"{label} control_x 必须单调非降",
    )
    return {
        "scalars": [float(v) for v in scalars],
        "control_x": points["control_x"],
        "control_y": points["control_y"],
    }


def _checked_analyzer(document):
    """Normalize an analyzer block to three validated channel documents."""
    if document is None:
        shared = _checked_channel_analyzer(DEFAULT_ANALYZER, "默认")
        return {"channels": [deepcopy(shared) for _ in range(3)],
                "provenance": "deterministic-synthetic-default-unbound"}
    _require(isinstance(document, dict), "analyzer 必须为对象")
    if "channels" in document:
        channels = document["channels"]
        _require(
            isinstance(channels, (list, tuple)) and len(channels) == 3,
            "analyzer.channels 必须包含 R/G/B 三个通道",
        )
        checked = [
            _checked_channel_analyzer(ch, f"通道{index}")
            for index, ch in enumerate(channels)
        ]
    else:
        shared = _checked_channel_analyzer(document, "共享")
        checked = [deepcopy(shared) for _ in range(3)]
    provenance = document.get("provenance")
    _require(
        provenance is None or (isinstance(provenance, str) and provenance.strip()),
        "analyzer provenance 必须为非空字符串",
    )
    return {
        "channels": checked,
        "provenance": provenance or "explicit-unbound-analyzer-parameters",
    }


def build_nikon_parameter_result(nikon_film):
    """Construct a ParameterResult from the validated experimental analyzer."""
    from .nikon_lut import BuilderParameters
    from .nikon_parameters import ParameterResult

    film = checked_nikon_film(nikon_film)
    _require(film is not None and film["mode"] == "parameterized",
             "parameterized ParameterResult 仅能由 parameterized 实验文档构造")
    channels_doc = film["analyzer"]["channels"]
    builder_channels, xs, ys = [], [], []
    for ch in channels_doc:
        builder_channels.append(BuilderParameters(*ch["scalars"]))
        xs.append(tuple(ch["control_x"]))
        ys.append(tuple(ch["control_y"]))
    return ParameterResult(
        tuple(builder_channels),
        tuple(xs),
        tuple(ys),
        {
            "status": "experimental-explicit-unbound-parameters",
            "film_kind": film["film_kind"],
            "analyzer_provenance": film["analyzer"]["provenance"],
            "acquisition_binding_proven": False,
            "camera_to_scanner_calibrated": False,
        },
    )


def checked_experimental_film_controls(document):
    """Validate the full versioned experimental film document.

    Returns ``None`` when the document is absent or every sub-chain is inert,
    so a disabled control stays byte-identical to the engine's own output.
    """
    if document is None:
        return None
    _require(isinstance(document, dict), "实验胶片参数必须为对象")
    _require(document.get("version") == VERSION, "实验胶片参数版本不支持")
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    _require(len(raw.encode()) <= MAX_BYTES, "实验胶片参数超出大小限制")
    _require(
        document.get("calibration") == CALIBRATION,
        "实验胶片链必须显式声明 uncalibrated-hypothesis，禁止暗示已标定",
    )
    enabled = _boolean(document.get("enabled", True), "enabled")
    if not enabled:
        return None
    flex = checked_flex_film(document.get("flex"))
    nikon = checked_nikon_film(document.get("nikon"))
    if flex is None and nikon is None:
        return None
    return {
        "version": VERSION,
        "calibration": CALIBRATION,
        "enabled": True,
        "flex": flex,
        "nikon": nikon,
    }


# --------------------------------------------------------------------------- #
# FlexColor RGB14 film curve chain
# --------------------------------------------------------------------------- #

def compile_flex_film_tables(
    profile_path,
    profile_sha256,
    preferences,
    gray_suppressed,
    flex_film,
):
    """Compile RGB14 Gradation tables with experimental negative stages enabled.

    ``preferences`` is the production ``CurvePreferences`` bound to the frozen
    resource. The experimental chain only flips the negative-stage gates and
    optionally the GammaNeg preference exponent; every other stage keeps the
    verified production settings.
    """
    from .flex_pipeline import CurvePreferences, compile_profile_curves

    film = checked_flex_film(flex_film)
    _require(film is not None, "Flex 实验胶片参数为空，无法编译胶片曲线")
    _require(isinstance(preferences, CurvePreferences), "需要已验证的 CurvePreferences")
    experimental_prefs = CurvePreferences(
        neg_var_gamma=film["var_gamma"]
        if film["gamma_neg"] and film["var_gamma"] is not None
        else preferences.neg_var_gamma,
        stretch_neg_gamma=preferences.stretch_neg_gamma,
        no_neg_curve=preferences.no_neg_curve,
        provenance=(
            "experimental uncalibrated film style; "
            "not captured FlexColor application preference"
        ),
    )
    tables, receipt = compile_profile_curves(
        profile_path,
        expected_sha256=profile_sha256,
        preferences=experimental_prefs,
        gray_suppressed=gray_suppressed,
        negative_gamma_active=film["gamma_neg"],
        negative_common_active=True,
        negative_channels_active=tuple(film["channel_negative"]),
    )
    receipt = deepcopy(receipt)
    receipt["experimental_film"] = {
        "calibration": CALIBRATION,
        "film_kind": film["film_kind"],
        "gamma_neg_active": film["gamma_neg"],
        "channel_negative_active": list(film["channel_negative"]),
        "enhanced_shadow": film["enhanced_shadow"],
        "var_gamma": film["var_gamma"],
        "factory_preset_equivalence_claimed": False,
        "acquisition_binding_proven": False,
    }
    return tables, receipt


def apply_flex_film_codes(codes_rgb14, tables):
    """Apply experimental RGB14 film tables through the verified bounded kernel."""
    from .flex_pipeline import apply_profile_curves

    data = np.asarray(codes_rgb14)
    _require(
        data.dtype.kind in "ui" and data.size and data.max() <= 16383 and data.min() >= 0,
        "Flex 实验链输入必须为 0..16383 的 RGB14 整数码",
    )
    return apply_profile_curves(data, tables)


# --------------------------------------------------------------------------- #
# Nikon RGB16 film curve chain
# --------------------------------------------------------------------------- #

def apply_nikon_film_cms(pixels, nikon_film):
    """CMS-only experimental style: captured CMS plus optional fixed curve.

    Uses the same assumed-code research entry as the production Nikon route,
    but permits an explicit unbound hypothesis and a full 0..1 fixed strength,
    with an experimental receipt rather than the neutral-adapter receipt.
    """
    from .nikon_cms_research import evaluate_assumed_codes

    film = checked_nikon_film(nikon_film)
    _require(film is not None, "Nikon 实验胶片参数为空")
    _require(film["mode"] == "cms-only", "cms-only 链收到 parameterized 参数")
    result = evaluate_assumed_codes(
        pixels,
        input_hypothesis=film["input_hypothesis"],
        fixed_strength=film["fixed_strength"],
        threads=film["threads"],
    )
    result.receipt["experimental_film"] = {
        "calibration": CALIBRATION,
        "mode": "cms-only",
        "film_kind": film["film_kind"],
        "fixed_strength": film["fixed_strength"],
        "input_hypothesis": film["input_hypothesis"],
        "factory_preset_equivalence_claimed": False,
        "acquisition_binding_proven": False,
    }
    return result


def apply_nikon_film_parameterized(pixels, parameters, nikon_film):
    """Full experimental analyzer/LUT/fixed/CMS join under an explicit hypothesis.

    ``parameters`` must be a ``ParameterResult`` whose receipt already attests
    ``film_kind == color-negative`` and unbound acquisition status.
    """
    from .nikon_parameterized import evaluate_parameterized_codes

    film = checked_nikon_film(nikon_film)
    _require(film is not None, "Nikon 实验胶片参数为空")
    _require(film["mode"] == "parameterized", "parameterized 链收到 cms-only 参数")
    result = evaluate_parameterized_codes(
        pixels,
        parameters,
        input_hypothesis=film["input_hypothesis"],
        fixed_strength=film["fixed_strength"],
        threads=film["threads"],
    )
    result.receipt["experimental_film"] = {
        "calibration": CALIBRATION,
        "mode": "parameterized",
        "film_kind": film["film_kind"],
        "fixed_strength": film["fixed_strength"],
        "input_hypothesis": film["input_hypothesis"],
        "factory_preset_equivalence_claimed": False,
        "acquisition_binding_proven": False,
        "camera_to_scanner_calibrated": False,
    }
    return result
