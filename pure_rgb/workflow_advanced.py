"""Stage-scoped advanced controls for the existing, validated recipe contracts.

This is a catalog and an atomic recipe adapter, not another rendering path.
Group updates replace only the fields owned by that group.  Other engines,
source selections and resource descriptors remain in the caller's recipe.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
import plistlib
from pathlib import Path

from .professional_controls import Control, NATIVE_SPECS, OUTPUT_SPECS, PRINT_EXPOSURE_EV, checked_controls
from .native_headroom import HEADROOM_CONTROL, HEADROOM_STRENGTH
from .positive_float import FLOAT_SPECS
from .positive_post import POST_SPECS
from .native_spatial_v30 import LAB_FIELDS
from .workflow_ui_model import RESULT_MODES
from .workspace import checked_recipe
from .native_metering_controls import RANGE_FIELDS, native_to_slider, slider_to_native


@dataclass(frozen=True)
class AdvancedGroup:
    key: str
    title: str
    note: str
    specs: tuple[Control, ...]


def _group(key, title, note, specs):
    return AdvancedGroup(key, title, note, tuple(specs))


def _spec(key, title, default, low=0, high=1, step=0.01):
    return Control(key, title, "高级参数", default, low, high, step)


def _density_specs(recipe):
    result = [
        _spec("zones_rows", "分区行数", 8, 1, 256, 1),
        _spec("zones_columns", "分区列数", 12, 1, 256, 1),
        _spec("blocks_rows", "每区子块行数", 2, 1, 256, 1),
        _spec("blocks_columns", "每区子块列数", 2, 1, 256, 1),
        _spec("padding", "矩形排除区扩边（源像素）", 0, 0, 4096, 1),
        _spec("min_area_fraction", "最小有效面积比例", 0.25, 0.01, 1),
    ]
    if recipe["native_spatial_controls"].get("automatic_avoidance"):
        result.append(_spec("max_fraction", "异常候选面积上限", 0.12, 0.01, 0.2))
    return tuple(result)


def _native_detail_specs(recipe, bw):
    from negpy.features.lab.models import LabConfig

    spatial = bool(recipe.get("native_spatial_controls"))
    legacy_bw_meter = bool(recipe.get("bw_metering")) and not spatial
    baseline = LabConfig()
    result = []
    for spec in NATIVE_SPECS:
        if not spec.key.startswith("lab."):
            continue
        field = spec.key[4:]
        if bw and field in ("saturation", "skin_protection", "chroma_denoise"):
            continue
        if legacy_bw_meter and field in LAB_FIELDS:
            continue
        if spatial and field == "sharpen_auto_masking":
            continue
        if spatial and field in LAB_FIELDS:
            default = getattr(baseline, field)
            if not bw and field in {
                "sharpen",
                "chroma_denoise",
                "clahe_strength",
                "glow_amount",
                "halation_strength",
            }:
                default = 0.0
            spec = replace(spec, default=str(default) if spec.choices else default)
        result.append(spec)
    return tuple(result)


def _flex_defaults(recipe):
    """Reset to this profile, including the verified default camera adapter."""
    from .workflow_ui_model import flex_controls

    document = recipe["flex_research_controls"]
    fields = plistlib.loads(Path(document["profile"]["path"]).read_bytes())[
        "ImageSetting"
    ]["ImageCorrection"]
    # The V35 adapter's Saturation=7 is intentional; a group reset must not
    # silently restore the original profile's different Saturation=15.
    try:
        baseline = flex_controls()
    except (OSError, ValueError, KeyError):
        baseline = None
    if baseline and baseline["profile"] == document["profile"]:
        fields.update(baseline["curve_overrides"])
        fields.update(baseline["color_overrides"])
    return fields


def _flex_specs(recipe):
    baseline = _flex_defaults(recipe)
    curve = [
        _spec("Gamma", "原曲线 Gamma", baseline["Gamma"], 0.8, 8, 0.05),
        _spec("Brightness", "原亮度（整数码值）", baseline["Brightness"], -128, 127, 1),
        _spec("Contrast", "原反差（整数码值）", baseline["Contrast"], -128, 127, 1),
    ]
    for field, label in (
        ("ApplyCurves", "启用原 Gradations 曲线"),
        ("ApplySliders", "启用原亮度／反差"),
        ("ApplyHistogram", "启用原直方图曲线"),
        ("EnhancedShadow", "原增强阴影分支"),
    ):
        curve.append(_spec(field, label, baseline[field]))
    color = (
        _spec("ApplyCC", "启用原六色色彩校正", baseline["ApplyCC"]),
        _spec(
            "Saturation", "原饱和度（整数码值）", baseline["Saturation"], -100, 100, 1
        ),
    )
    return tuple(curve), color


# Input hypotheses shared by both Nikon research kernels, so one dropdown stays
# valid regardless of the selected chain mode.
_NIKON_FILM_HYPOTHESES = (
    ("unbound-scanner-final-rgb16", "未绑定扫描器最终 RGB16"),
    ("camera-transmission-as-scanner-rgb16", "相机透射按扫描器码处理（未标定）"),
    ("synthetic-rgb16-probe", "合成 RGB16 探针"),
)


def _flex_film_specs(recipe):
    doc = (recipe.get("experimental_film_controls") or {}).get("flex") or {}
    return (
        _spec("gamma_neg", "启用 GammaNeg 负片反演曲线", bool(doc.get("gamma_neg", False))),
        _spec("neg_red", "R 通道 Negative 曲线", bool(doc.get("channel_negative", [False, False, False])[0])),
        _spec("neg_green", "G 通道 Negative 曲线", bool(doc.get("channel_negative", [False, False, False])[1])),
        _spec("neg_blue", "B 通道 Negative 曲线", bool(doc.get("channel_negative", [False, False, False])[2])),
        _spec("enhanced_shadow", "GammaNeg 增强阴影", bool(doc.get("enhanced_shadow", False))),
    )


def _nikon_film_specs(recipe):
    doc = (recipe.get("experimental_film_controls") or {}).get("nikon") or {}
    return (
        Control(
            "mode", "实验链模式", "实验胶片",
            doc.get("mode", "cms-only"),
            choices=(
                ("cms-only", "仅 CMS（生产同构）"),
                ("parameterized", "参数化 analyzer/LUT 反演（未标定）"),
            ),
        ),
        _spec(
            "fixed_strength", "fixed 色调曲线强度",
            float(doc.get("fixed_strength", 0.0)), 0.0, 1.0, 0.05,
        ),
        Control(
            "input_hypothesis", "未绑定输入假设", "实验胶片",
            doc.get("input_hypothesis", "synthetic-rgb16-probe"),
            choices=_NIKON_FILM_HYPOTHESES,
        ),
    )


def advanced_groups(recipe, engine, bw, stage):
    """Return applicable controls for one stage and the currently selected route."""
    if engine not in RESULT_MODES or (bw and engine != "native"):
        raise ValueError("高级参数与当前输入／引擎不匹配")
    if stage not in (2, 3, 4):
        return ()
    groups = []
    native = engine == "native"
    spatial = bool(recipe.get("native_spatial_controls"))
    if stage == 2:
        if native and not bw and recipe.get("measurement_method") == "negpy":
            groups.append(_group(
                "native_metering", "NegPy 原生测量参数",
                "与 NegPy 相同的白点、黑点及范围测量参数。截断使用 NegPy 的 −100 至 100 标尺；负值更温和，复位恢复原生默认值。",
                tuple(replace(s, default=0., low=-100., high=100., step=1.)
                      if s.key in RANGE_FIELDS else s
                      for s in NATIVE_SPECS if s.group == "原生密度测量"),
            ))
        if native and spatial:
            groups.append(
                _group(
                    "density",
                    "分区测量高级参数",
                    "改变后需应用测量。分区与子块总数最多65536；面积上限只控制候选／避让范围。",
                    _density_specs(recipe),
                )
            )
        return tuple(groups)
    if stage == 3:
        if native:
            blocked = {
                "exposure.grade", "exposure.paper_profile", PRINT_EXPOSURE_EV,
                HEADROOM_CONTROL, HEADROOM_STRENGTH,
            }
            if spatial or recipe.get("bw_metering"):
                blocked |= {"exposure.contrast_mask_strength", "exposure.diffusion"}
            print_specs = [
                s
                for s in NATIVE_SPECS
                if s.group == "明暗与反差" and s.key not in blocked
            ]
            if bw:
                print_specs = [
                    s for s in print_specs if s.key != "exposure.cast_removal_strength"
                ]
            note = "印相参数。" + (
                "当前分区使用冻结分析；修改本组后需保存新版本重新分析。"
                if spatial
                else "自动印相密度／反差开启时，这里的数值是计算基准；本次实际印相密度与 ISO-R 反差显示在更新后的画面左上角。"
            )
            groups.append(_group("native_print", "原生印相", note, print_specs))
            curve = [
                s
                for s in NATIVE_SPECS
                if s.group == "相纸曲线" and s.key != "exposure.paper_profile"
            ]
            if bw:
                curve = [
                    s
                    for s in curve
                    if s.key
                    not in {"exposure.dye_separation", "exposure.separation_damping"}
                ]
            groups.append(_group("native_curve", "相纸肩趾与中间调", note, curve))
            if not bw:
                groups.append(
                    _group(
                        "native_cmy",
                        "原生 CMY 平衡",
                        note,
                        (s for s in NATIVE_SPECS if s.group == "色彩平衡"),
                    )
                )
                for channel, title in (("red", "R"), ("green", "G"), ("blue", "B")):
                    groups.append(
                        _group(
                            "native_" + channel,
                            title + " 色层曲线",
                            note,
                            (s for s in NATIVE_SPECS if s.group == title + " 通道微调"),
                        )
                    )
            if not bw or spatial:
                groups.append(
                    _group(
                        "native_float",
                        "独立正像曲线",
                        "原生输出后的浮点正像调整；不重新估计密度，也不恢复已丢失的信息。",
                        (
                            s
                            for s in FLOAT_SPECS
                            if s.key in {"positive_gamma", "shadow_ev", "highlight_ev"}
                        ),
                    )
                )
        if not bw:
            groups.append(
                _group(
                    "output_" + engine,
                    "当前引擎正像调整",
                    "仅作用于当前引擎的正像；额外曝光叠加在上方公共曝光上。",
                    (s for s in OUTPUT_SPECS if s.key in {"exposure_ev", "saturation"}),
                )
            )
        groups.append(
            _group(
                "common_positive",
                "跨引擎公共正光",
                "对所有引擎生效的亮度曲线调整（阴影提亮、高光压暗、中间调 S 曲线）；关闭时与引擎原始输出逐字节一致。",
                (
                    _spec("highlight", "高光压暗", 0.0, 0.0, 100.0, 1.0),
                    _spec("shadow", "阴影提亮", 0.0, 0.0, 100.0, 1.0),
                    _spec("tone", "中间调曲线", 0.0, -100.0, 100.0, 1.0),
                ),
            )
        )
        if native and (bw or recipe.get("measurement_method") != "negpy") and recipe.get("highlight_controls"):
            from .highlight_reference_guarded import DEFAULTS

            specs = [
                _spec(
                    "range_start",
                    "起效亮度（线性）",
                    DEFAULTS["range_start"],
                    0.001,
                    0.99,
                    0.001,
                ),
                _spec(
                    "budget_reference",
                    "曲率参考亮度（线性）",
                    DEFAULTS["budget_reference"],
                    0.002,
                    1,
                    0.001,
                ),
                _spec(
                    "max_burn_ev", "压暗上限（EV）", DEFAULTS["max_burn_ev"], 0, 8, 0.1
                ),
                _spec("min_log_slope", "最小亮度斜率", DEFAULTS["min_log_slope"], 0, 1),
                _spec(
                    "protection_strength",
                    "人物／中间调保护强度",
                    DEFAULTS["protection_strength"],
                    0,
                    1,
                ),
            ]
            if recipe["highlight_controls"].get("protection"):
                specs.append(
                    _spec("feather_pixels", "保护边缘羽化（源像素）", 64, 0, 512, 1)
                )
            groups.append(
                _group(
                    "highlight",
                    "高光找回高级参数",
                    "起效亮度须小于曲率参考亮度。压暗上限不是实际恢复的底片档数；最小斜率为1或上限为0会关闭此功能。",
                    specs,
                )
            )
        if engine == "flex" and recipe.get("flex_research_controls"):
            curve, color = _flex_specs(recipe)
            groups.append(
                _group(
                    "flex_curve",
                    "Flex 原曲线参数",
                    "使用已验证的原曲线计算；亮度／反差是整数码值，不是曝光EV或百分比。",
                    curve,
                )
            )
            groups.append(
                _group(
                    "flex_color",
                    "Flex 原六色色彩",
                    "原六色模型位于曲线后、ICC前；饱和度0不等于所有配置的中性值。",
                    color,
                )
            )
            groups.append(
                _group(
                    "experimental_flex",
                    "Flex 实验胶片曲线（未标定）",
                    "显式未标定实验：重新打开生产旁路的 GammaNeg／通道 Negative。不代表原厂完整预设或扫描器标定；全部关闭时与生产输出一致。",
                    _flex_film_specs(recipe),
                )
            )
        if engine == "nikon" and recipe.get("nikon_research_controls"):
            groups.append(
                _group(
                    "experimental_nikon",
                    "Nikon 实验胶片链（未标定）",
                    "参数化 analyzer/LUT 反演尚未标定，可能增加处理时间并出现负像。切换为「仅 CMS（生产同构）」可使用标准路径。",
                    _nikon_film_specs(recipe),
                )
            )
        return tuple(groups)
    if native:
        detail = _native_detail_specs(recipe, bw)
        if detail:
            groups.append(
                _group(
                    "native_detail",
                    "原生细节与效果",
                    "使用100%预览判断颗粒与锐化。"
                    + (
                        "空间细节复用冻结密度分析；色彩相关参数仍需保存新分析。"
                        if spatial
                        else ""
                    ),
                    detail,
                )
            )
        groups.append(
            _group(
                "native_finish",
                "原生暗角",
                "原生画面收尾；启用分区时修改后需保存新版本。",
                (s for s in NATIVE_SPECS if s.key.startswith("finish.")),
            )
        )
    if not bw:
        post_groups = [
            ("geometry", "后期几何", "后期 · 几何"),
            ("toning", "分离色调", "后期 · 分离色调"),
        ]
        if not native:
            post_groups = [
                ("tone", "正像肩趾曲线", "后期 · 色调与肩趾"),
                ("balance", "正像 CMY 与色温", "后期 · 白平衡"),
                ("rgb", "正像 RGB 曲线", "后期 · RGB 曲线"),
                ("detail", "正像细节与效果", "后期 · 色度与细节"),
                ("vignette", "正像暗角", "后期 · 暗角"),
            ] + post_groups
        for key, title, category in post_groups:
            groups.append(
                _group(
                    "post_" + key + "_" + engine,
                    title,
                    "当前引擎独立的有界正像后期。几何变换不改变源坐标密度选区；细节请用100%预览。",
                    (s for s in POST_SPECS if s.group == category),
                )
            )
    return tuple(groups)


def _all_groups(recipe, engine, bw):
    return {
        group.key: group
        for stage in (2, 3, 4)
        for group in advanced_groups(recipe, engine, bw, stage)
    }


def _group_context(recipe, key):
    bw = recipe.get("source_mode") == "bw-single-v1"
    engine = (
        "nikon"
        if recipe.get("nikon_research_controls")
        else "flex"
        if recipe.get("flex_research_controls")
        else "native"
    )
    if key.startswith("output_") or key.startswith("post_"):
        engine = key.rsplit("_", 1)[1]
    return engine, bw


def advanced_values(recipe, group_key):
    """Read sparse widget values; absent values use the current catalog defaults."""
    engine, bw = _group_context(recipe, group_key)
    group = _all_groups(recipe, engine, bw).get(group_key)
    if group is None:
        raise ValueError("此高级参数组不适用于当前配方")
    if group_key == "density":
        spatial = recipe["native_spatial_controls"]
        meter = spatial["metering"]
        values = {
            "zones_rows": meter["zones"][0],
            "zones_columns": meter["zones"][1],
            "blocks_rows": meter["blocks_per_zone"][0],
            "blocks_columns": meter["blocks_per_zone"][1],
            "padding": meter["padding"],
            "min_area_fraction": meter["min_area_fraction"],
            "max_fraction": spatial.get("automatic_avoidance", {}).get(
                "max_fraction", 0.12
            ),
        }
    elif group_key == "highlight":
        high = recipe["highlight_controls"]
        values = {
            **high["controls"],
            "feather_pixels": (high.get("protection") or {}).get("feather_pixels", 64),
        }
    elif group_key.startswith("flex_"):
        doc = recipe["flex_research_controls"]
        values = {
            **_flex_defaults(recipe),
            **doc["curve_overrides"],
            **doc["color_overrides"],
        }
    elif group_key == "native_float":
        values = recipe.get("native_float_controls", {}).get("values", {})
    elif group_key == "common_positive":
        doc = recipe.get("common_positive_controls")
        values = (
            {
                "highlight": doc.get("highlight", 0.0),
                "shadow": doc.get("shadow", 0.0),
                "tone": doc.get("tone", 0.0),
            }
            if doc
            else {}
        )
    elif group_key == "experimental_flex":
        doc = (recipe.get("experimental_film_controls") or {}).get("flex") or {}
        channels = doc.get("channel_negative", [False, False, False])
        values = {
            "gamma_neg": bool(doc.get("gamma_neg", False)),
            "neg_red": bool(channels[0]),
            "neg_green": bool(channels[1]),
            "neg_blue": bool(channels[2]),
            "enhanced_shadow": bool(doc.get("enhanced_shadow", False)),
        }
    elif group_key == "experimental_nikon":
        doc = (recipe.get("experimental_film_controls") or {}).get("nikon") or {}
        values = {
            "mode": doc.get("mode", "cms-only"),
            "fixed_strength": float(doc.get("fixed_strength", 0.0)),
            "input_hypothesis": doc.get(
                "input_hypothesis", "synthetic-rgb16-probe"
            ),
        }
    elif group_key.startswith("output_"):
        values = (
            recipe.get("output_controls", {})
            .get(RESULT_MODES[engine], {})
            .get("values", {})
        )
    elif group_key.startswith("post_"):
        values = (
            recipe.get("post_controls", {})
            .get(RESULT_MODES[engine], {})
            .get("values", {})
        )
    else:
        values = dict(recipe.get("native_controls", {}).get("values", {}))
        if group_key == "native_detail" and recipe.get("native_spatial_controls"):
            values.update(
                {
                    "lab." + k: v
                    for k, v in recipe["native_spatial_controls"]["lab"].items()
                }
            )
    if group_key == "native_metering":
        values = {key: native_to_slider(key, value) for key, value in values.items()}
    return {
        s.key: deepcopy(values[s.key])
        for s in group.specs
        if s.key in values and values[s.key] != s.default
    }


def _replace_values(document, specs, values):
    result = deepcopy(document or {"version": 1, "values": {}})
    for spec in specs:
        result["values"].pop(spec.key, None)
    result["values"].update(values)
    return result


def _integer(value):
    if type(value) not in (int, float) or value != int(value):
        raise ValueError("此高级参数必须使用整数")
    return int(value)


def apply_advanced(recipe, engine, bw, changes):
    """Atomically merge applicable group edits, then run the complete validator."""
    if type(changes) is not dict:
        raise ValueError("高级参数需要分组文档")
    result = deepcopy(recipe)
    groups = _all_groups(result, engine, bw)
    if set(changes) - groups.keys():
        raise ValueError("包含当前引擎不支持的高级参数组")
    for key, submitted in changes.items():
        group = groups[key]
        values = checked_controls({"version": 1, "values": submitted}, group.specs)[
            "values"
        ]
        if key == "density":
            effective = {s.key: values.get(s.key, s.default) for s in group.specs}
            spatial = result["native_spatial_controls"]
            meter = spatial["metering"]
            meter.update(
                zones=[
                    _integer(effective["zones_rows"]),
                    _integer(effective["zones_columns"]),
                ],
                blocks_per_zone=[
                    _integer(effective["blocks_rows"]),
                    _integer(effective["blocks_columns"]),
                ],
                padding=_integer(effective["padding"]),
                min_area_fraction=effective["min_area_fraction"],
            )
            if "max_fraction" in effective:
                spatial["automatic_avoidance"]["max_fraction"] = effective[
                    "max_fraction"
                ]
            if result.get("bw_metering"):
                result["bw_metering"] = deepcopy(meter)
        elif key == "highlight":
            high = result["highlight_controls"]
            for spec in group.specs:
                value = values.get(spec.key, spec.default)
                if spec.key == "feather_pixels":
                    high["protection"]["feather_pixels"] = value
                else:
                    high["controls"][spec.key] = value
        elif key.startswith("flex_"):
            target = result["flex_research_controls"][
                "curve_overrides" if key == "flex_curve" else "color_overrides"
            ]
            for spec in group.specs:
                value = values.get(spec.key, spec.default)
                target[spec.key] = (
                    _integer(value)
                    if spec.key in {"Brightness", "Contrast", "Saturation"}
                    else value
                )
        elif key == "native_float":
            result["native_float_controls"] = _replace_values(
                result.get("native_float_controls"), group.specs, values
            )
        elif key == "common_positive":
            from .workflow_ui_model import common_positive_controls

            result["common_positive_controls"] = common_positive_controls(
                highlight=values.get("highlight", 0.0),
                shadow=values.get("shadow", 0.0),
                tone=values.get("tone", 0.0),
            )
        elif key in ("experimental_flex", "experimental_nikon"):
            from .workflow_ui_model import experimental_film_controls

            existing = result.get("experimental_film_controls") or {}
            flex_doc = existing.get("flex")
            nikon_doc = existing.get("nikon")
            if key == "experimental_flex":
                flex_doc = {
                    "gamma_neg": values.get("gamma_neg", False),
                    "channel_negative": [
                        values.get("neg_red", False),
                        values.get("neg_green", False),
                        values.get("neg_blue", False),
                    ],
                    "enhanced_shadow": values.get("enhanced_shadow", False),
                }
            else:
                nikon_mode = values.get("mode", "cms-only")
                nikon_doc = {
                    "mode": nikon_mode,
                    "fixed_strength": values.get("fixed_strength", 0.0),
                }
                if nikon_mode == "parameterized":
                    # The panel dropdown only offers parameterized-kernel
                    # hypotheses; cms-only uses its own validated default.
                    nikon_doc["input_hypothesis"] = values.get(
                        "input_hypothesis", "synthetic-rgb16-probe"
                    )
            experimental = experimental_film_controls(
                flex=flex_doc, nikon=nikon_doc
            )
            if experimental is None:
                result.pop("experimental_film_controls", None)
            else:
                result["experimental_film_controls"] = experimental
        elif key.startswith("output_") or key.startswith("post_"):
            field = "output_controls" if key.startswith("output_") else "post_controls"
            mode = RESULT_MODES[engine]
            target = result.setdefault(field, {})
            target[mode] = _replace_values(target.get(mode), group.specs, values)
        elif key == "native_detail" and result.get("native_spatial_controls"):
            spatial_fields = [s for s in group.specs if s.key[4:] in LAB_FIELDS]
            native_fields = [s for s in group.specs if s.key[4:] not in LAB_FIELDS]
            lab = result["native_spatial_controls"]["lab"]
            native = result.setdefault("native_controls", {"version": 1, "values": {}})
            for spec in spatial_fields:
                lab.pop(spec.key[4:], None)
                native["values"].pop(spec.key, None)
                if spec.key in values:
                    lab[spec.key[4:]] = values[spec.key]
            result["native_controls"] = _replace_values(
                native,
                native_fields,
                {
                    k: v
                    for k, v in values.items()
                    if k in {s.key for s in native_fields}
                },
            )
        else:
            if key == "native_metering":
                values = {field: slider_to_native(field, value) for field, value in values.items()}
            result["native_controls"] = _replace_values(
                result.get("native_controls"), group.specs, values
            )
    return checked_recipe(result)
