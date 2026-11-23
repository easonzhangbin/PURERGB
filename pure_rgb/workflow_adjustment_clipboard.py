"""Copy portable edits between photo drafts, with a selective paste dialog."""

from copy import deepcopy
import math
from pathlib import Path

from PyQt6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from .native_spatial_v30 import LAB_FIELDS
from .positive_post import POST_SPECS
from .professional_controls import NATIVE_SPECS
from .workflow_attention import attach_attention
from .workflow_glass import prepare_glass
from .workflow_ui_model import RESULT_MODES

GROUPS = {
    "exposure": ("曝光与反差", True),
    "balance": ("色温与色彩平衡", True),
    "tone": ("色调曲线与分离色调", True),
    "details": ("色度、锐化、降噪与暗角", True),
    "paper": ("印相相纸与胶片配置", True),
    "engine": ("当前引擎专用调整", True),
    "highlight": ("高光找回参数（保留目标选区）", False),
    "orientation": ("输出旋转与翻转", False),
    "crop": ("输出裁切（按画幅比例）", False),
    "geometry": ("拉直、透视与后期几何", False),
}


def copy_adjustments(project, photo):
    engine = project.color_engine(photo)
    return dict(
        photo_id=photo["id"], name=Path(photo["base"]["raw"][0]).stem,
        source_mode=photo["source_mode"], engine=engine,
        shape=deepcopy(photo.get("source_shape")), recipe=project.recipe(photo, engine),
        suspended_highlight_controls=deepcopy(photo.get("suspended_highlight_controls")),
    )


def available_groups(source, photo, engine):
    same_mode = source["source_mode"] == photo["source_mode"]
    same_native = same_mode and source["engine"] == engine == "native"
    rgb = same_mode and photo["source_mode"] == "rgb"
    result = {"exposure", "orientation", "geometry"}
    if rgb:
        result.update(("balance", "tone", "details"))
    elif same_native:
        result.update(("tone", "details"))
    if same_native:
        result.add("paper")
        target = photo["drafts"][engine]
        origin_high = (source["recipe"].get("highlight_controls")
                       or source.get("suspended_highlight_controls"))
        if (not origin_high or target.get("highlight_controls")
                or photo.get("suspended_highlight_controls")
                or target.get("roi") or photo.get("analysis_roi")):
            result.add("highlight")
    if same_mode and source["engine"] == engine:
        origin = source["recipe"]
        target = photo["drafts"][engine]
        research = f"{engine}_research_controls"
        if engine == "native":
            if rgb or target.get("native_spatial_controls"):
                result.add("engine")
        elif bool(origin.get(research)) == bool(target.get(research)):
            if engine != "nikon" or bool(origin.get("cool_headroom_controls")) == bool(target.get("cool_headroom_controls")):
                result.add("engine")
    if not source["recipe"].get("output_roi") or (source["shape"] and photo.get("source_shape")):
        result.add("crop")
    return result


def _fields(target, source, keys):
    for key in keys:
        if key in source:
            target[key] = deepcopy(source[key])
        else:
            target.pop(key, None)


def _values(recipe, field, mode=None, *, create=False):
    document = recipe.setdefault(field, {}) if create else recipe.get(field, {})
    if mode is not None:
        document = document.setdefault(mode, {}) if create else document.get(mode, {})
    if create:
        document.setdefault("version", 1)
        return document.setdefault("values", {})
    return document.get("values", {})


def _transfer_values(target, source, field, keys, source_mode=None, target_mode=None):
    if keys:
        _fields(
            _values(target, field, target_mode, create=True),
            _values(source, field, source_mode),
            keys,
        )


def merged_adjustments(source, photo, engine, target, groups):
    if not groups or not groups <= available_groups(source, photo, engine):
        raise ValueError("所选调整不适用于目标照片")
    result = deepcopy(target)
    recipe = source["recipe"]
    native = source["source_mode"] == photo["source_mode"] and source["engine"] == engine == "native"
    rgb = source["source_mode"] == photo["source_mode"] == "rgb"
    source_mode, target_mode = RESULT_MODES[source["engine"]], RESULT_MODES[engine]
    output_keys, post_categories, native_categories = set(), set(), set()
    if "exposure" in groups:
        _fields(result, recipe, ("exposure_ev",))
        if rgb:
            _fields(result, recipe, ("common_positive_controls",))
            output_keys.update(("exposure_ev", "contrast"))
        if native:
            native_categories.add("明暗与反差")
    if "balance" in groups:
        output_keys.update(("red_ev", "green_ev", "blue_ev"))
        post_categories.add("后期 · 白平衡")
        if native:
            native_categories.add("色彩平衡")
    if "tone" in groups:
        if rgb:
            post_categories.update(("后期 · 色调与肩趾", "后期 · RGB 曲线", "后期 · 分离色调"))
        if native:
            native_categories.update(("相纸曲线", "R 通道微调", "G 通道微调", "B 通道微调"))
    if "details" in groups:
        if rgb:
            output_keys.add("saturation")
            post_categories.update(("后期 · 色度与细节", "后期 · 暗角"))
        if native:
            native_categories.add("细节与效果")
    if "paper" in groups:
        _fields(result, recipe, ("paper", "native_film") if rgb else ("paper",))
    if "orientation" in groups:
        _fields(result, recipe, ("output_orientation",))
    if "geometry" in groups:
        post_categories.add("后期 · 几何")
    if "engine" in groups:
        _copy_engine_adjustments(result, recipe, engine, target_mode)
    if "highlight" in groups:
        _copy_highlight_adjustments(result, source, photo)
    if "crop" in groups:
        _fields(result, recipe, ("crop",))
        result.pop("output_roi", None)
        if (bounds := recipe.get("output_roi")) is not None:
            sh, sw = source["shape"][:2]
            th, tw = photo["source_shape"][:2]
            x0, y0, x1, y1 = bounds
            result["output_roi"] = [
                min(tw - 1, math.floor(x0 * tw / sw)),
                min(th - 1, math.floor(y0 * th / sh)),
                min(tw, math.ceil(x1 * tw / sw)),
                min(th, math.ceil(y1 * th / sh)),
            ]

    _transfer_values(result, recipe, "output_controls", output_keys, source_mode, target_mode)
    _transfer_values(result, recipe, "post_controls",
                     [spec.key for spec in POST_SPECS if spec.group in post_categories],
                     source_mode, target_mode)
    native_keys = [spec.key for spec in NATIVE_SPECS
                   if spec.group in native_categories and spec.key != "exposure.paper_profile"]
    if target.get("native_spatial_controls") or target.get("bw_metering"):
        blocked = {"exposure.contrast_mask_strength", "exposure.diffusion", "lab.sharpen_auto_masking"}
        native_keys = [key for key in native_keys if key not in blocked]
    if "paper" in groups:
        native_keys.append("exposure.paper_profile")
    _transfer_values(result, recipe, "native_controls", native_keys)
    if native and "details" in groups:
        source_spatial = recipe.get("native_spatial_controls")
        target_spatial = result.get("native_spatial_controls")
        if source_spatial or target_spatial:
            source_values = _values(recipe, "native_controls")
            source_lab = (source_spatial["lab"] if source_spatial else
                          {key: source_values["lab." + key] for key in LAB_FIELDS
                           if "lab." + key in source_values})
            target_values = _values(result, "native_controls", create=True)
            if target_spatial:
                _fields(target_spatial["lab"], source_lab, LAB_FIELDS)
                _fields(target_values, {}, ("lab." + key for key in LAB_FIELDS))
            else:
                _fields(target_values, {"lab." + key: value for key, value in source_lab.items()},
                        ("lab." + key for key in LAB_FIELDS))
    return result


def _copy_highlight_adjustments(target, source, photo):
    """Copy controls while retaining the target photo's source-coordinate shapes."""
    source_high = source["recipe"].get("highlight_controls")
    source_settings = source_high or source.get("suspended_highlight_controls")
    target_high = deepcopy(target.get("highlight_controls")
                           or photo.get("suspended_highlight_controls"))
    if target_high is None and source_settings is not None:
        roi = target.get("roi") or photo.get("analysis_roi")
        if roi is None:
            raise ValueError("目标照片尚无可用的高光参考区域")
        target_high = dict(version=1, reference_roi=deepcopy(roi), controls={})
        photo["highlight_reference_origin"] = "automatic"
        photo.pop("highlight_reference_drawn_roi", None)
        photo.pop("highlight_reference_aligned", None)
    if target_high is not None and source_settings is not None:
        target_high["controls"] = deepcopy(source_settings["controls"])
    if source_high is not None:
        target["highlight_controls"] = target_high
        photo.pop("suspended_highlight_controls", None)
    else:
        target.pop("highlight_controls", None)
        if target_high is not None:
            photo["suspended_highlight_controls"] = target_high


def _copy_engine_adjustments(target, source, engine, mode):
    if engine == "native":
        _fields(target, source, ("native_float_controls",))
        return
    if engine == "nikon" and target.get("nikon_research_controls"):
        target["nikon_research_controls"]["fixed_strength"] = source["nikon_research_controls"].get(
            "fixed_strength", 0
        )
    elif engine == "flex" and target.get("flex_research_controls"):
        origin = source["flex_research_controls"]
        for field, keys in (
            ("curve_overrides", ("Gamma", "Brightness", "Contrast", "ApplyCurves",
                                 "ApplySliders", "ApplyHistogram", "EnhancedShadow")),
            ("color_overrides", ("Saturation", "ApplyCC")),
        ):
            _fields(target["flex_research_controls"][field], origin.get(field, {}), keys)
    else:
        _fields(target.setdefault("engine_controls", {}), source.get("engine_controls", {}), (mode,))


class PasteAdjustmentsDialog(QDialog):
    def __init__(self, parent, source, target, others):
        super().__init__(parent)
        prepare_glass(self)
        self.setWindowTitle("粘贴调整")
        self.setMinimumWidth(370)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        source_label = QLabel(f"从 {source['name']} 复制到 {target[1]}")
        source_label.setWordWrap(True)
        layout.addWidget(source_label)
        self.batch = QCheckBox(f"同时应用到另外 {len(others)} 张已选照片")
        layout.addWidget(self.batch)
        self.batch.setVisible(bool(others))

        self.options = {}
        for key, (title, default) in GROUPS.items():
            control = QCheckBox(title)
            control.setChecked(default)
            layout.addWidget(control)
            self.options[key] = control
        note = QLabel("仅替换所选调整。跨引擎只应用兼容参数。密度选区、高光参考区、作用区、保护区、平场和除尘笔画按各照片保留。")
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.apply_button.setText("粘贴调整")
        self.apply_button.setObjectName("primary")
        attach_attention(self.apply_button)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._target, self._others = target, others
        self.batch.toggled.connect(self._sync)
        for control in self.options.values():
            control.toggled.connect(self._sync)
        self._sync()

    def targets(self):
        return [self._target[0], *(entry[0] for entry in self._others if self.batch.isChecked())]

    def groups(self):
        return {key for key, control in self.options.items() if control.isEnabled() and control.isChecked()}

    def _sync(self, *_):
        enabled = self._target[2].intersection(
            *(entry[2] for entry in self._others if self.batch.isChecked())
        )
        for key, control in self.options.items():
            control.setEnabled(key in enabled)
        self.apply_button.setEnabled(bool(self.groups()))
