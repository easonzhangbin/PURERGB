"""Shared, opt-in positive-domain processors reused from the pinned NegPy tree.

This is a post layer, not a replacement scanner/negative model. Empty documents
bypass it byte-for-byte. The linear API accepts values beyond display white;
enabled original Transfer/PhotoLab processors have bounded output. Their final
clamp inputs are not exposed, so range changes cannot certify how much was hard
clipped instead of compressed. The RGB16 adapter necessarily starts after the
existing engine's quantization.
"""

from dataclasses import replace
import numpy as np

from .professional_controls import Control, NATIVE_SPECS, MODE_KEYS, checked_controls
from .tone_study import GAMMA

TEMPERATURE_SPEC = Control(
    "temperature_kelvin",
    "色温（名义值，5500 K 为中性）",
    "后期 · 白平衡",
    5500.0,
    3000,
    12000,
    50,
)
HUE_SPEC = Control("hue_degrees", "色相（°）", TEMPERATURE_SPEC.group, 0.0, -180, 180, 1)


def _specs():
    exposure_fields = {
        "density",
        "grade",
        "shadow_density",
        "highlight_density",
        "toe",
        "toe_width",
        "shoulder",
        "shoulder_width",
        "dye_separation",
        "separation_damping",
        *(
            f"{zone}_{color}"
            for zone in ("wb", "shadow", "highlight")
            for color in ("cyan", "magenta", "yellow")
        ),
        *(
            f"{field}_trim_{channel}"
            for field in (
                "toe",
                "shoulder",
                "toe_width",
                "shoulder_width",
                "dye_separation",
            )
            for channel in ("red", "green", "blue")
        ),
    }
    result = []
    for spec in NATIVE_SPECS:
        section, field = spec.key.split(".")
        if section == "exposure" and field not in exposure_fields:
            continue
        if section not in ("exposure", "lab", "finish"):
            continue
        default = spec.default
        if spec.key in ("lab.chroma_denoise", "lab.sharpen", "lab.skin_protection"):
            default = 0.0
        if section == "lab":
            group = "后期 · 色度与细节"
        elif section == "finish":
            group = "后期 · 暗角"
        elif field.startswith(
            (
                "wb_",
                "shadow_c",
                "shadow_m",
                "shadow_y",
                "highlight_c",
                "highlight_m",
                "highlight_y",
            )
        ):
            group = "后期 · 白平衡"
        elif "trim_" in field:
            group = "后期 · RGB 曲线"
        else:
            group = "后期 · 色调与肩趾"
        result.append(replace(spec, group=group, default=default))
    result.extend((TEMPERATURE_SPEC, HUE_SPEC))
    for key, title, default, low, high, step, choices in (
        (
            "rotation",
            "旋转",
            0,
            0,
            270,
            90,
            ((0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")),
        ),
        ("flip_horizontal", "水平翻转", False, 0, 1, 1, ()),
        ("flip_vertical", "垂直翻转", False, 0, 1, 1, ()),
        ("fine_rotation", "精细旋转", 0.0, -45, 45, 0.05, ()),
        ("converge_v", "垂直透视（%）", 0.0, -15, 15, 0.1, ()),
        ("converge_h", "水平透视（%）", 0.0, -15, 15, 0.1, ()),
        ("distortion_k1", "径向畸变", 0.0, -0.1, 0.1, 0.001, ()),
        ("crop_to_valid", "裁掉变换空白角", False, 0, 1, 1, ()),
    ):
        result.append(
            Control(
                "geometry." + key,
                title,
                "后期 · 几何",
                default,
                low,
                high,
                step,
                choices,
            )
        )
    for key, title, high in (
        ("shadow_tint_hue", "阴影染色色相", 360),
        ("shadow_tint_strength", "阴影染色强度", 1),
        ("highlight_tint_hue", "高光染色色相", 360),
        ("highlight_tint_strength", "高光染色强度", 1),
    ):
        result.append(
            Control(
                "toning." + key,
                title,
                "后期 · 分离色调",
                0.0,
                0,
                high,
                1 if high == 360 else 0.01,
            )
        )
    return tuple(result)


POST_SPECS = _specs()


def checked_post(document):
    if not isinstance(document, dict) or set(document) - set(MODE_KEYS):
        raise ValueError("未知公共后期模式")
    return {
        mode: value
        for mode, doc in document.items()
        if (value := checked_controls(doc, POST_SPECS))["values"]
    }


def _range_receipt(image):
    """Observed linear values, not a claim that a tone curve clipped them."""
    return {
        "minimum_rgb": image.min(axis=(0, 1)).tolist(),
        "maximum_rgb": image.max(axis=(0, 1)).tolist(),
        "below_zero_fraction_rgb": (image < 0).mean(axis=(0, 1)).tolist(),
        "above_one_fraction_rgb": (image > 1).mean(axis=(0, 1)).tolist(),
    }


def _bounded_stage_receipt(name, before, after):
    return {
        "stage": name,
        "input_range": before,
        "output_range": _range_receipt(after),
        "original_processor_output_bounds": [0.0, 1.0],
        "pre_final_clamp_values_observed": False,
        "hard_clipped_fraction_rgb": None,
        "interpretation": "Input/output ranges only; the original processor can compress before its final clamp, so headroom disappearance is not a measured hard-clip count",
    }


def post_linear(linear, document, *, sample_step=1):
    values = checked_controls(document, POST_SPECS)["values"]
    if not values:
        return linear, {"identity": True}
    if (
        linear.ndim != 3
        or linear.shape[2] != 3
        or not linear.size
        or not np.isfinite(linear).all()
        or np.any(linear < 0)
    ):
        raise ValueError("公共后期需非负、有限的线性工作 RGB")
    if sample_step < 1:
        raise ValueError("无效预览步长")
    from negpy.domain.interfaces import PipelineContext
    from negpy.features.process.models import ProcessMode
    from negpy.features.exposure.models import ExposureConfig
    from negpy.features.exposure.normalization import LogNegativeBounds
    from negpy.features.transparency.logic import (
        TRANSFER_DENSITY_RANGE,
        transfer_bounds,
        slide_kelvin_to_wb,
    )
    from negpy.features.transparency.processor import TransferProcessor
    from negpy.features.geometry.models import GeometryConfig
    from negpy.features.geometry.processor import GeometryProcessor, CropProcessor
    from negpy.features.lab.models import LabConfig
    from negpy.features.lab.processor import PhotoLabProcessor
    from negpy.features.lab.logic import apply_clahe
    from negpy.features.toning.models import ToningConfig
    from negpy.features.toning.processor import ToningProcessor
    from negpy.features.finish.models import FinishConfig
    from negpy.features.finish.processor import FinishProcessor

    def section(name):
        return {
            k.split(".", 1)[1]: v for k, v in values.items() if k.startswith(name + ".")
        }

    context = PipelineContext(
        original_size=(linear.shape[0] * sample_step, linear.shape[1] * sample_step),
        scale_factor=1 / sample_step,
        process_mode=ProcessMode.C41,
        wants_uv_grid=False,
        cache_stages=False,
        readback_metrics=False,
    )
    out = linear
    stages = []
    bounded_stages = []
    skipped_groups = []
    geometry = section("geometry")
    if geometry:
        out = np.ascontiguousarray(out, dtype=np.float32)
        if "rotation" in geometry:
            geometry["rotation"] = int(geometry["rotation"] // 90)
        config = GeometryConfig(**geometry)
        out = GeometryProcessor(config).process(out, context)
        out = CropProcessor(config).process(out, context)
        stages.append("NegPy.GeometryProcessor+CropProcessor")
    exposure = section("exposure")
    if "temperature_kelvin" in values:
        cyan, yellow = slide_kelvin_to_wb(values["temperature_kelvin"], 0, 0)
        exposure["wb_cyan"] = exposure.get("wb_cyan", 0) + cyan
        exposure["wb_yellow"] = exposure.get("wb_yellow", 0) + yellow
    if exposure:
        out = np.ascontiguousarray(out, dtype=np.float32)
        config = replace(
            ExposureConfig(),
            auto_exposure=False,
            auto_normalize_contrast=False,
            cast_removal_strength=0.0,
            **exposure,
        )
        context.metrics["final_bounds"] = LogNegativeBounds(*transfer_bounds())
        before = _range_receipt(out)
        density = -np.log10(np.maximum(out, np.float32(1e-30))) / TRANSFER_DENSITY_RANGE
        out = TransferProcessor(config, positive_source=True).process(density, context)
        name = "NegPy.TransferProcessor(positive_source=True)"
        stages.append(name)
        bounded_stages.append(_bounded_stage_receipt(name, before, out))
    if hue := values.get(HUE_SPEC.key, 0):
        import cv2

        # An HSV rotation keeps the maximum channel and chroma magnitude,
        # including HDR headroom and neutral pixels in linear working RGB.
        rotated = np.empty(out.shape, dtype=np.float32)
        rows = max(1, 524288 // out.shape[1])
        for start in range(0, out.shape[0], rows):
            block = slice(start, start + rows)
            hsv = cv2.cvtColor(np.ascontiguousarray(out[block], dtype=np.float32), cv2.COLOR_RGB2HSV)
            hue_channel = hsv[..., 0]
            hue_channel += hue
            np.remainder(hue_channel, 360.0, out=hue_channel)
            cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB, dst=rotated[block])
        out = rotated
        stages.append("HSV hue rotation")
    lab = section("lab")
    if lab:
        config = replace(
            LabConfig(), sharpen=0.0, chroma_denoise=0.0, skin_protection=0.0
        )
        config = replace(config, **lab)
        # Ancillary sharpening settings do not enable sharpening. Calling the
        # original processor without an enabled effect would still run its
        # unconditional final clip01 and silently erase floating headroom.
        active = (
            config.clahe_strength > 0
            or config.chroma_denoise > 0
            or config.saturation != 1
            or config.skin_protection > 0
            or config.sharpen > 0
            or config.glow_amount > 0
            or config.halation_strength > 0
        )
        if active:
            out = np.ascontiguousarray(out, dtype=np.float32)
            # Sharpness radius is specified in original output pixels.
            config = replace(
                config, sharpen_radius=max(0.01, config.sharpen_radius / sample_step)
            )
            before = _range_receipt(out)
            out = apply_clahe(out, config.clahe_strength)
            out = PhotoLabProcessor(config).process(out, context)
            name = "NegPy.CLAHE+PhotoLabProcessor"
            stages.append(name)
            bounded_stages.append(_bounded_stage_receipt(name, before, out))
        else:
            skipped_groups.append("lab: all effect strengths neutral")
    toning = section("toning")
    if toning:
        out = np.ascontiguousarray(out, dtype=np.float32)
        config = ToningConfig(**toning)
        before = _range_receipt(out)
        out = ToningProcessor(config).process(out, context)
        name = "NegPy.ToningProcessor(split)"
        stages.append(name)
        if config.shadow_tint_strength > 0 or config.highlight_tint_strength > 0:
            bounded_stages.append(_bounded_stage_receipt(name, before, out))
    finish = section("finish")
    if finish:
        out = np.ascontiguousarray(out, dtype=np.float32)
        config = FinishConfig(**finish)
        before = _range_receipt(out)
        out = FinishProcessor(config).process(out, context)
        name = "NegPy.FinishProcessor"
        stages.append(name)
        if config.vignette_stops != 0:
            bounded_stages.append(_bounded_stage_receipt(name, before, out))
    if not np.isfinite(out).all():
        raise ValueError("公共后期出现非有限结果")
    if not stages:
        out = linear
    return out, {
        "identity": not stages,
        "version": 2,
        "stages": stages,
        "skipped_groups": skipped_groups,
        "domain": "positive linear working RGB; no negative/scanner reinterpretation",
        "sample_step": sample_step,
        "input_range": _range_receipt(linear),
        "bounded_stages": bounded_stages,
        "output_range": _range_receipt(out),
        "range_observations_are_hard_clip_counts": False,
        "source_linear_retained_by_caller": True,
        "existing_clipped_samples_recoverable": False,
    }


def post_codes(pixels, document=None, *, sample_step=1):
    if not document or not checked_controls(document, POST_SPECS)["values"]:
        return pixels, {"identity": True}
    linear = (pixels.astype(np.float32) / 65535) ** GAMMA
    out, metadata = post_linear(linear, document, sample_step=sample_step)
    from .editing_render import encode_positive_rows

    metadata["RGB16_encoding"] = {
        "input_was_already_quantized_RGB16": True,
        "linear_clamp_bounds": [0.0, 1.0],
        "lower_truncated_fraction_rgb": (out < 0).mean(axis=(0, 1)).tolist(),
        "upper_truncated_fraction_rgb": (out > 1).mean(axis=(0, 1)).tolist(),
        "scope": "Exact final encoder clamp only; earlier original-processor clipping remains unmeasured",
    }
    return encode_positive_rows(np.maximum(out, 0)), metadata
