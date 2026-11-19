"""Hash-bound 4.8.13 profile fields -> verified scalar curve graph.

This is deliberately a curve-stage compiler, not a complete factory preset.
ColorCorr, saturation, device correction, sharpening, and CMS are separate
stages whose omission is retained in the receipt. It does not change the
existing production adapter or manufacture missing scanner preferences.
"""

from dataclasses import dataclass
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from .engine_resources import inventory_flex_profile
from .flex_chain import (
    compose_inner_luts,
    compose_outer_luts,
    gray_no_cms_parameters,
    single_point_lut,
)
from .flex_curves import point_curve_lut
from .flex_negative import PREPARE_NEGATIVE_PIVOTS, gamma_negative_lut, negative_lut


def implementation_sources():
    directory = Path(__file__).resolve().parent
    return {
        str(directory / name): hashlib.sha256(
            (directory / name).read_bytes()
        ).hexdigest()
        for name in (
            "flex_pipeline.py",
            "engine_resources.py",
            "flex_chain.py",
            "flex_routing.py",
            "flex_curves.py",
            "flex_negative.py",
            "flex_gamma.py",
            "flex_histogram.py",
            "flex_input.py",
            "flex_color_management.py",
            "flex_color_adjustments.py",
            "flex_sliders_mac.py",
            "flex_lightness.py",
        )
    }


@dataclass(frozen=True)
class CurvePreferences:
    """Explicit preference state; None requests the original fallback branch."""

    neg_var_gamma: float | None
    stretch_neg_gamma: float | None
    no_neg_curve: bool
    provenance: str

    def __post_init__(self):
        if (
            type(self.no_neg_curve) is not bool
            or not isinstance(self.provenance, str)
            or not self.provenance.strip()
        ):
            raise ValueError("Preferences require explicit NoNegCurve and provenance")


def _integer_list(value, count, low, high, label):
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(type(v) is not int or not low <= v <= high for v in value)
    ):
        raise ValueError(f"Invalid original {label} fields")
    return value


def compile_profile_curves(
    path,
    *,
    expected_sha256,
    preferences,
    gray_suppressed,
    field_overrides=None,
    negative_gamma_active=True,
    negative_common_active=True,
    negative_channels_active=(True, True, True),
):
    """Compile original R/G/B scalar stages with every external state explicit.

    Shadow/Highlight XML indices 1/2/3 map RGB; DotColor XML uses 0/1/2
    and 7/8/9 because its buffer starts one byte after getter enum base.
    EndPoints contains extension policies. ApplyHistogram
    controls inner Gamma, HighShadow and Gray; ApplyCurves controls seven point
    curves. The unlabelled suppression byte and preferences are not inferred
    from settings. Contrast/Brightness use the independently verified Mac
    signed-byte polynomial; their raw units are not EV or percentages.
    """
    if (
        not isinstance(preferences, CurvePreferences)
        or type(gray_suppressed) is not bool
        or type(negative_gamma_active) is not bool
        or type(negative_common_active) is not bool
        or not isinstance(negative_channels_active, tuple)
        or len(negative_channels_active) != 3
        or any(type(value) is not bool for value in negative_channels_active)
    ):
        raise ValueError("Require explicit preference state and Gray suppression flag")
    inventory = inventory_flex_profile(path)
    if inventory["resource"]["sha256"] != expected_sha256:
        raise ValueError("Original FlexColor profile changed; refusing stale binding")
    original_fields = inventory["fields"]
    fields = deepcopy(original_fields)
    allowed_overrides = {
        "ApplyCurves",
        "ApplySliders",
        "ApplyHistogram",
        "Gamma",
        "EnhancedShadow",
        "Gray",
        "Shadow",
        "Highlight",
        "DotColor",
        "EndPoints",
        "Contrast",
        "Brightness",
    }
    if field_overrides is None:
        field_overrides = {}
    if (
        not isinstance(field_overrides, dict)
        or set(field_overrides) - allowed_overrides
    ):
        raise ValueError("Unknown or out-of-scope curve field override")
    fields.update(deepcopy(field_overrides))
    if (
        type(fields.get("FilmType")) is not int
        or fields["FilmType"] not in (1, 2)
        or fields.get("Mode") != 0
    ):
        raise ValueError(
            "Curve compiler requires explicitly classified RGB color/B&W negative settings"
        )
    flags = {}
    for key in ("ApplyCurves", "ApplySliders", "ApplyHistogram"):
        if type(fields.get(key)) is not bool:
            raise ValueError(f"Explicit original {key} state required")
        flags[key] = fields[key]
    if type(fields.get("EnhancedShadow")) is not bool:
        raise ValueError("Explicit original EnhancedShadow state required")
    shadow = _integer_list(fields.get("Shadow"), 4, 0, 16383, "Shadow")
    highlight = _integer_list(fields.get("Highlight"), 4, 0, 16383, "Highlight")
    dots = _integer_list(fields.get("DotColor"), 14, 0, 255, "DotColor")
    endpoints = _integer_list(fields.get("EndPoints"), 2, 0, 2, "EndPoints")
    gray = _integer_list(fields.get("Gray"), 4, 0, 255, "Gray")
    if len(inventory["curves"]) != 7:
        raise ValueError("Original curve graph requires seven Gradations")
    from .flex_gamma import gamma_curve_lut
    from .flex_histogram import high_shadow_lut_mac
    from .flex_sliders_mac import contrast_brightness_lut_mac

    identity = np.frombuffer(np.arange(16384, dtype="<u2").tobytes(), dtype="<u2")
    film_type = fields["FilmType"]
    pivots = PREPARE_NEGATIVE_PIVOTS
    negative_common = (
        negative_lut(
            *pivots["common"], film_type=film_type, disabled=preferences.no_neg_curve
        )
        if negative_common_active
        else identity
    )
    negative_channels = [
        negative_lut(*pivots[c], film_type=film_type, disabled=preferences.no_neg_curve)
        if negative_channels_active[index]
        else identity
        for index, c in enumerate(("red", "green", "blue"))
    ]
    grays = (
        [single_point_lut(x, y) for x, y in gray_no_cms_parameters(gray)]
        if flags["ApplyHistogram"] and not gray_suppressed
        else [identity] * 3
    )
    inner = compose_inner_luts(
        gamma=gamma_curve_lut(fields["Gamma"], enabled=flags["ApplyHistogram"]),
        gradations=[
            point_curve_lut(item["points"], bits=14)
            if flags["ApplyCurves"]
            else identity
            for item in inventory["curves"]
        ],
        negative_common=negative_common,
        negative_channels=negative_channels,
        contrast=contrast_brightness_lut_mac(
            fields.get("Contrast"),
            fields.get("Brightness"),
            enabled=flags["ApplySliders"],
        ),
        gray=grays,
    )
    negative_gamma = gamma_negative_lut(
        var_gamma=preferences.neg_var_gamma,
        stretch=preferences.stretch_neg_gamma,
        enhanced_shadow=fields["EnhancedShadow"],
        film_type=film_type,
        enabled=negative_gamma_active,
    )
    wrapped = [
        high_shadow_lut_mac(
            inner[c],
            shadow=shadow[c + 1],
            highlight=highlight[c + 1],
            output_black=dots[c],
            output_white=dots[c + 7],
            endpoints=tuple(endpoints),
            apply_histogram=flags["ApplyHistogram"],
        )
        for c in range(3)
    ]
    result = compose_outer_luts(
        gamma_negative=[negative_gamma] * 3,
        high_shadow_wrapped_inner=wrapped,
        apply_rise_limiter=True,
    )
    return result, {
        "schema": "pure-rgb-original-flex-curve-profile",
        "implementation_sources_at_compile": implementation_sources(),
        "resource": inventory["resource"],
        "film_type": film_type,
        "category": "color-negative" if film_type == 1 else "monochrome-negative",
        "original_fields": original_fields,
        "effective_fields": fields,
        "PURE_research_overrides": field_overrides,
        "PURE_negative_gamma_bypass": not negative_gamma_active,
        "PURE_negative_common_bypass": not negative_common_active,
        "PURE_negative_channel_bypass_rgb": [
            not value for value in negative_channels_active
        ],
        "original_CalculateFinalGradations_enables_GammaNeg": True,
        "apply_flags": flags,
        "external_state": {
            "Gray_suppression_byte_5eb": gray_suppressed,
            "Gray_suppression_semantics_unresolved": True,
            "NegVarGamma": preferences.neg_var_gamma,
            "StretchNegGamma": preferences.stretch_neg_gamma,
            "NoNegCurve": preferences.no_neg_curve,
            "preference_provenance": preferences.provenance,
        },
        "curve_graph": (
            "GammaNeg"
            if negative_gamma_active
            else "GammaNeg(identity; explicit PURE positive-carrier bypass)"
        )
        + " -> HighShadow(Gamma -> G0 -> "
        + (
            "common Negative"
            if negative_common_active
            else "common Negative(identity; PURE bypass)"
        )
        + " -> channel Negative(active RGB="
        + str(negative_channels_active)
        + ") -> +G[1+c] -G[4+c] at saved channel reference -> Contrast -> Gray) -> rise limiter",
        "channel_binding": [
            {
                "channel": name,
                "shadow14": shadow[c + 1],
                "highlight14": highlight[c + 1],
                "dot_black8": dots[c],
                "dot_white8": dots[c + 7],
                "gray_control": gray[c + 1],
            }
            for c, name in enumerate(("R", "G", "B"))
        ],
        "neutral_contrast_explicit_identity": fields.get("Contrast") == 0
        and fields.get("Brightness") == 0,
        "contrast_stage": {
            "implementation": "Mac4.8.13 CContrastCurve, signed-byte controls",
            "contrast": fields.get("Contrast"),
            "brightness": fields.get("Brightness"),
            "enabled": flags["ApplySliders"],
        },
        "DotColor_XML_binding": "read buffer ic+0x1f3; getter base ic+0x1f2 with RGB enum1..3; XML RGB black0..2 / white7..9",
        "mac_scalar_arithmetic": True,
        "point_curve_kernel_provenance": "Windows 4.8.9.1 CPointCurve verified; default identity also replayed on Mac 4.8.13; nonidentity Mac point-loop replay not included",
        "all_seven_gradations_are_identity": all(
            all(x == y and dy == 1 for x, y, dy in item["points"])
            for item in inventory["curves"]
        ),
        "pow_import_substituted_with_host_math_pow": True,
        "original_Darwin_libm_bit_equivalence_verified": False,
        "table_sha256_rgb": [
            hashlib.sha256(table.tobytes()).hexdigest() for table in result
        ],
        "retained_but_not_executed_fields": {
            key: fields.get(key)
            for key in (
                "ApplyCC",
                "Saturation",
                "Lightness",
                "ColorCorr",
                "ColorCor",
                "ApplyUSM",
                "ApplyDust",
                "Convert",
                "InputProfile",
                "RGBProfile",
                "GrayProfile",
                "CMYKProfile",
            )
            if key in fields
        },
        "automatic_histogram_measurement_recreated": False,
        "device_correction_executed": False,
        "CMS_executed": False,
        "full_factory_preset_executed": False,
        "camera_to_scanner_calibration_verified": False,
        "production_defaults_changed": False,
    }


def apply_profile_curves(codes, tables):
    """Evaluate bounded 14-bit RGB codes without implicit clipping or ICC/OETF."""
    from .flex_routing import _table

    data = np.asarray(codes)
    if (
        data.ndim != 3
        or data.shape[-1] != 3
        or not data.size
        or data.dtype.kind not in "ui"
        or data.min() < 0
        or data.max() > 16383
        or not isinstance(tables, (list, tuple))
        or len(tables) != 3
    ):
        raise ValueError("Require nonempty RGB14 codes and three full tables")
    checked = [_table(t) for t in tables]
    if any(t.max() > 16383 for t in checked):
        raise ValueError("A curve contains an unhandled out-of-domain sentinel")
    return np.stack([checked[c][data[..., c]] for c in range(3)], axis=-1).astype(
        np.uint16
    )


def process_profile_codes(
    codes,
    path,
    *,
    expected_sha256,
    preferences,
    gray_suppressed,
    post_color_overrides=None,
    lightness_anchor_preference=None,
    trace=False,
    **curve_options,
):
    """Joined original curve -> ColorMap -> explicit Lightness research stages.

    Input must already be explicitly prepared RGB14. ICC remains a separate,
    explicit boundary. This API neither assumes a camera/scanner mapping nor
    claims omitted device correction or original automatic measurement.
    The caller may tile an image with one fixed compiled/metering recipe.
    """
    from .flex_color_adjustments import (
        color_map14,
        color_map_parameters,
    )
    from .flex_lightness import apply_lightness14, lightness_curve_parameters
    from .stage_trace import stage_statistics

    if type(trace) is not bool:
        raise ValueError("trace requires an explicit boolean")
    if post_color_overrides is None:
        post_color_overrides = {}
    if not isinstance(post_color_overrides, dict) or set(post_color_overrides) - {
        "ColorCorr",
        "Saturation",
        "ApplyCC",
        "Lightness",
    }:
        raise ValueError("Unknown post-gradation color override")
    tables, pipeline = compile_profile_curves(
        path,
        expected_sha256=expected_sha256,
        preferences=preferences,
        gray_suppressed=gray_suppressed,
        **curve_options,
    )
    fields = deepcopy(pipeline["effective_fields"])
    fields.update(deepcopy(post_color_overrides))
    if (
        "Lightness" not in fields
        or "ColorCorr" not in fields
        or "Saturation" not in fields
        or "ApplyCC" not in fields
    ):
        raise ValueError("Explicit original post-gradation fields required")
    lightness = lightness_curve_parameters(
        fields["Lightness"],
        enabled=fields["ApplySliders"],
        anchor_preference=lightness_anchor_preference,
    )
    color = color_map_parameters(
        color_corr=fields["ColorCorr"],
        saturation=fields["Saturation"],
        apply_cc=fields["ApplyCC"],
    )
    result = apply_profile_curves(codes, tables)
    stages = []
    if trace:
        stages.append(
            stage_statistics(
                result, domain="post-final-Gradation RGB14", bounds=(0, 16383)
            )
        )
    color_map14(
        result,
        color_corr=fields["ColorCorr"],
        saturation=fields["Saturation"],
        apply_cc=fields["ApplyCC"],
        input_domain="flex-post-gradation-rgb14",
        out=result,
    )
    if trace:
        stages.append(
            stage_statistics(
                result, domain="post-ColorMap14bit RGB14", bounds=(0, 16383)
            )
        )
    if lightness["stage_active"]:
        result, lightness = apply_lightness14(
            result,
            lightness=fields["Lightness"],
            anchor_preference=lightness_anchor_preference,
            enabled=fields["ApplySliders"],
            input_domain="flex-post-colormap-rgb14",
            out=result,
        )
        if result.max() > 16383:
            raise ValueError(
                "Original Lightness produced wrapped values outside RGB14; inspect the independent Lightness stage, do not silently clip before ICC"
            )
        if trace:
            stages.append(
                stage_statistics(
                    result, domain="post-ApplyYGrad RGB14", bounds=(0, 16383)
                )
            )
    return result, {
        "curve_pipeline": pipeline,
        "post_color_overrides": deepcopy(post_color_overrides),
        "ColorMap": color,
        "Lightness": lightness,
        "stages": stages,
        "order": "final Gradations -> ColorMap14bit -> ApplyYGrad (explicit Lightness or original bypass) -> caller explicit ICC",
        "ICC_executed": False,
        "input_domain": "explicit RGB14 research carrier",
        "full_factory_preset_executed": False,
        "production_defaults_changed": False,
    }
