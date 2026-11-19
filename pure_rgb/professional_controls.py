"""Versioned, sparse controls with explicit Native/positive-output domains.

No range recovery is performed here. Native overrides never enter the shared
density analysis or the neutral reference used to fit experimental renderers.
"""

from dataclasses import dataclass, replace
import math

import numpy as np

from .tone_study import GAMMA


@dataclass(frozen=True)
class Control:
    key: str
    title: str
    group: str
    default: float | bool | str
    low: float = 0.0
    high: float = 1.0
    step: float = 0.01
    choices: tuple = ()


def native_papers():
    from negpy.features.exposure.papers import profiles_for_mode
    from negpy.features.process.models import ProcessMode

    return tuple(
        (key, profile.label) for key, profile in profiles_for_mode(ProcessMode.C41)
    )


def native_specs():
    specs = []

    def add(field, title, group, default=0.0, low=-1.0, high=1.0, step=0.01):
        specs.append(
            Control("exposure." + field, title, group, default, low, high, step)
        )

    add("density", "印相密度（增加变暗）", "明暗与反差", 1.0, 0, 2)
    add("grade", "ISO-R 反差（低值更硬）", "明暗与反差", 115.0, 50, 180, 1)
    for field, title in (
        ("shadow_density", "阴影密度"),
        ("highlight_density", "高光密度"),
    ):
        add(field, title, "明暗与反差")
    for field, title in (("shadow_grade", "阴影反差"), ("highlight_grade", "高光反差")):
        add(field, title, "明暗与反差", 0, -50, 50, 1)
    for field, title, default in (
        ("auto_exposure", "自动印相密度", True),
        ("auto_normalize_contrast", "自动反差", True),
        ("hold_black", "提亮阴影时保持黑点", False),
    ):
        add(field, title, "明暗与反差", default)
    for field, title, default in (
        ("contrast_mask_strength", "反差遮罩", 0.0),
        ("preflash", "相纸预曝光", 0.0),
        ("diffusion", "扩散", 0.0),
        ("cast_removal_strength", "偏色去除", 0.5),
    ):
        add(field, title, "明暗与反差", default, 0, 1)
    for region, label in (("wb", "全局"), ("shadow", "阴影"), ("highlight", "高光")):
        for color, name in (("cyan", "青"), ("magenta", "品红"), ("yellow", "黄")):
            add(f"{region}_{color}", f"{label} · {name}", "色彩平衡")
    for field, title, default, low, high in (
        ("toe", "趾部强度", 0, -1, 1),
        ("toe_width", "趾部宽度", 2.5, 0.1, 5),
        ("shoulder", "肩部强度", 0, -1, 1),
        ("shoulder_width", "肩部宽度", 2.5, 0.1, 5),
        ("midtone_gamma", "中间调曲线", 0, -0.5, 0.5),
        ("dye_separation", "染料分离", 1, 0.5, 1.5),
        ("separation_damping", "分离阻尼", 0, 0, 1),
        ("paper_dmin", "显示相纸白底", False, 0, 1),
        ("paper_black", "显示相纸原生黑位", False, 0, 1),
    ):
        add(field, title, "相纸曲线", default, low, high)
    specs.append(
        Control(
            "exposure.paper_profile",
            "印相材料",
            "相纸曲线",
            "__reference__",
            choices=(("__reference__", "跟随基准／拟合参考"),) + native_papers(),
        )
    )
    for channel, name in (("red", "R"), ("green", "G"), ("blue", "B")):
        group = f"{name} 通道微调"
        for field, title, limit, step in (
            ("grade", "反差", 30, 1),
            ("shadow_grade", "阴影反差", 50, 1),
            ("highlight_grade", "高光反差", 50, 1),
            ("toe", "趾部", 1, 0.01),
            ("shoulder", "肩部", 1, 0.01),
            ("toe_width", "趾部宽度", 2, 0.01),
            ("shoulder_width", "肩部宽度", 2, 0.01),
            ("midtone_gamma", "中间调", 0.5, 0.01),
            ("dye_separation", "染料分离", 0.4, 0.01),
        ):
            add(f"{field}_trim_{channel}", title, group, 0, -limit, limit, step)
    for field, title, default, low, high in (
        ("saturation", "色度", 1, 0, 2),
        ("skin_protection", "肤色保护", 0.5, 0, 1),
        ("chroma_denoise", "色度降噪", 2.5, 0, 5),
        ("sharpen", "锐化", 0.25, 0, 1),
        ("sharpen_radius", "锐化半径", 1, 0.5, 3),
        ("sharpen_masking", "锐化蒙版", 0.1, 0, 1),
        ("sharpen_auto_masking", "自动颗粒蒙版", False, 0, 1),
        ("clahe_strength", "局部反差 CLAHE", 0, 0, 1),
        ("glow_amount", "辉光", 0, 0, 1),
        ("halation_strength", "光晕", 0, 0, 1),
    ):
        specs.append(Control("lab." + field, title, "细节与效果", default, low, high))
    specs.append(
        Control(
            "lab.sharpen_method",
            "锐化方式",
            "细节与效果",
            "usm",
            choices=(("usm", "反差锐化 USM"), ("rl", "反卷积 RL")),
        )
    )
    for field, title, default, low, high in (
        ("vignette_stops", "暗角（正值压暗）", 0, -2, 2),
        ("vignette_size", "暗角范围", 0.5, 0, 1),
        ("vignette_roundness", "暗角方圆", 0, 0, 1),
    ):
        specs.append(
            Control("finish." + field, title, "细节与效果", default, low, high)
        )
    return tuple(specs)


NATIVE_SPECS = native_specs()
OUTPUT_SPECS = (
    Control("exposure_ev", "该模式曝光", "正像调整", 0.0, -4, 4, 0.05),
    Control("red_ev", "R 通道平衡", "正像调整", 0.0, -2, 2, 0.01),
    Control("green_ev", "G 通道平衡", "正像调整", 0.0, -2, 2, 0.01),
    Control("blue_ev", "B 通道平衡", "正像调整", 0.0, -2, 2, 0.01),
    Control("contrast", "反差（线性 18% 锚点）", "正像调整", 1.0, 0.25, 2, 0.01),
    Control("saturation", "饱和度（工作 RGB）", "正像调整", 1.0, 0, 2, 0.01),
)
MODE_KEYS = ("native", "cool-shared-rgb", "flex-endpoint")
ENGINE_SPECS = {
    "cool-shared-rgb": (
        Control("density_gamma", "密度中间调", "引擎曲线", 1.0, 0.4, 3.0),
        Control("curve_contrast", "拟合曲线反差微调", "引擎曲线", 0.0, -1, 1),
        Control("curve_gain_ev", "拟合曲线增益", "引擎曲线", 0.0, -2, 2),
    ),
    "flex-endpoint": (
        Control(
            "curve_kernel",
            "Gradations 插值内核",
            "引擎曲线",
            "legacy-linear",
            choices=(
                ("legacy-linear", "现有线性近似（基准）"),
                ("point-14", "重建 CPointCurve · 14-bit"),
            ),
        ),
        Control(
            "sliders_kernel",
            "亮度／反差计算内核",
            "引擎曲线",
            "legacy-linear",
            choices=(
                ("legacy-linear", "现有线性近似（基准）"),
                ("flex-14", "重建 Flex Sliders · 14-bit"),
            ),
        ),
        Control("film_gamma", "负片 Gamma", "引擎曲线", 2.0, 0.5, 4),
        Control("brightness", "反演前亮度", "引擎曲线", 0.0, -64, 64, 1),
        Control("contrast", "反演前反差", "引擎曲线", 0.0, -80, 100, 1),
        Control("black_code", "输入下端点", "引擎曲线", 0.0, 0, 64, 1),
        Control("white_code", "输入上端点", "引擎曲线", 255.0, 192, 255, 1),
        Control("curve_contrast", "端点曲线反差微调", "引擎曲线", 0.0, -1, 1),
        Control("curve_offset", "端点曲线位置", "引擎曲线", 0.0, -2, 2),
    ),
}


def checked_engines(document):
    if not isinstance(document, dict) or set(document) - ENGINE_SPECS.keys():
        raise ValueError("未知实验引擎控件")
    return {
        mode: checked
        for mode, value in document.items()
        if (checked := checked_engine(mode, value))["values"]
    }


def checked_engine(mode, document):
    if not isinstance(document, dict):
        raise ValueError("无效引擎参数文档")
    if document.get("version") == 2 and mode == "flex-endpoint":
        if type(document["version"]) is not int:
            raise ValueError("无效引擎参数版本")
        checked_controls({**document, "version": 1}, ENGINE_SPECS[mode])
        _check_slider_fields(document["values"])
        return {"version": 2, "values": dict(document["values"])}
    result = checked_controls(document, ENGINE_SPECS[mode])
    if mode == "flex-endpoint":
        _check_slider_fields(result["values"])
    return result


def _check_slider_fields(values):
    if values.get("sliders_kernel") == "flex-14":
        for field in ("contrast", "brightness"):
            value = values.get(field, 0)
            if value != int(value):
                raise ValueError("重建 Flex 亮度／反差内核只接受整数参数")


def checked_controls(document, specs):
    if (
        not isinstance(document, dict)
        or set(document) != {"version", "values"}
        or type(document["version"]) is not int
        or document["version"] != 1
    ):
        raise ValueError("不支持的专业控件配方版本")
    values = document["values"]
    allowed = {s.key: s for s in specs}
    if not isinstance(values, dict) or set(values) - allowed.keys():
        raise ValueError("专业控件包含未支持字段")
    result = {}
    for key, value in values.items():
        spec = allowed[key]
        if isinstance(spec.default, bool):
            valid = type(value) is bool
        elif spec.choices:
            valid = value in tuple(c[0] for c in spec.choices)
        else:
            valid = (
                type(value) in (int, float)
                and math.isfinite(value)
                and spec.low <= value <= spec.high
            )
        if not valid:
            raise ValueError(f"专业控件参数越界或类型错误：{key}")
        if value != spec.default:
            result[key] = value
    return {"version": 1, "values": result}


def checked_outputs(document):
    if not isinstance(document, dict) or set(document) - set(MODE_KEYS):
        raise ValueError("未知呈现模式控件")
    result = {}
    for mode, value in document.items():
        checked = checked_controls(value, OUTPUT_SPECS)
        if checked["values"]:
            result[mode] = checked
    return result


def native_config(config, document):
    values = checked_controls(document, NATIVE_SPECS)["values"]
    for section in ("exposure", "lab", "finish"):
        changes = {
            k.split(".")[1]: v for k, v in values.items() if k.startswith(section + ".")
        }
        if changes:
            config = replace(
                config, **{section: replace(getattr(config, section), **changes)}
            )
    return config


def output_codes(pixels, document, common_ev=0.0):
    """Positive linear working RGB; one final clip/quantization, never an ICC edit."""
    overrides = checked_controls(document, OUTPUT_SPECS)["values"]
    if not overrides and common_ev == 0:
        return pixels, {"identity": True}
    if pixels.dtype == np.uint16 and overrides.get("saturation", 1) == 1:
        from .positive_lut import exact_cpu_lut_codes

        return exact_cpu_lut_codes(pixels, document, common_ev)
    rows = max(1, 262144 // pixels.shape[1])
    if pixels.shape[0] > rows:
        codes = np.empty_like(pixels)
        below = above = 0
        for start in range(0, pixels.shape[0], rows):
            block, diagnostics = output_codes(
                pixels[start : start + rows], document, common_ev
            )
            codes[start : start + rows] = block
            below += round(diagnostics["below_zero_fraction"] * block.size)
            above += round(diagnostics["above_one_fraction"] * block.size)
        return codes, {
            **diagnostics,
            "below_zero_fraction": below / pixels.size,
            "above_one_fraction": above / pixels.size,
        }
    values = {s.key: overrides.get(s.key, s.default) for s in OUTPUT_SPECS}
    linear = (pixels.astype(np.float64) / 65535) ** GAMMA
    linear *= 2 ** np.array(
        [
            common_ev + values["exposure_ev"] + values[f"{c}_ev"]
            for c in ("red", "green", "blue")
        ]
    )
    if values["contrast"] != 1:
        linear = 0.18 * (linear / 0.18) ** values["contrast"]
    if values["saturation"] != 1:
        # Adobe RGB working-space luminance, not display-sRGB coefficients.
        luma = (linear @ np.array([0.297344975, 0.627363566, 0.075291459]))[..., None]
        linear = luma + values["saturation"] * (linear - luma)
    diagnostics = {
        "identity": False,
        "domain": "Positive linear Adobe RGB after inversion; no density recovery",
        "below_zero_fraction": float(np.mean(linear < 0)),
        "above_one_fraction": float(np.mean(linear > 1)),
        "common_exposure_ev": common_ev,
    }
    codes = np.rint(np.clip(linear, 0, 1) ** (1 / GAMMA) * 65535).astype(np.uint16)
    return codes, diagnostics
