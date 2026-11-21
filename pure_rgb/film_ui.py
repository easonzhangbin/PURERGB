"""Read-only NegPy film catalog and recipe-local density calibration controls."""

from copy import deepcopy

from PyQt6.QtCore import pyqtSignal
from .workflow_glass import GlassMessageBox  # noqa: E402
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
)

from .film_profiles import checked_film, film_catalog, generic_film, read_film
from .workflow_disclosure import DisclosureButton
from .workflow_combo import WorkflowComboBox
from .motion_sliders import ScrubberControl


class NativeFilmSelector(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.loading = True
        self.profiles, errors = film_catalog()
        self._matrix_reference_profiles = deepcopy(self.profiles)
        self._matrix_default_context = None
        self._matrix_default_key = None
        self._matrix_defaults = None
        self._matrix_default_cache = {}
        self.document = None
        column = QVBoxLayout(self)
        column.setContentsMargins(2, 2, 2, 2)
        column.addWidget(QLabel("C-41 胶片配置 · 负片密度域"))
        self.choice = WorkflowComboBox()
        self.choice.setAccessibleName("C-41 胶片配置")
        self.choice.setMinimumContentsLength(16)
        self.choice.setSizeAdjustPolicy(
            WorkflowComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        for profile in self.profiles:
            self.choice.addItem(self.title(profile))
        column.addWidget(self.choice)
        row = QHBoxLayout()
        choose = QPushButton("导入 TOML…")
        choose.clicked.connect(self.choose)
        reset = QPushButton("旁路／复位")
        reset.clicked.connect(self.reset)
        row.addWidget(choose)
        row.addWidget(reset)
        column.addLayout(row)
        self.strength = ScrubberControl("串扰校准强度", 0, 1, 0.01, 0)
        self.strength.label.setToolTip("串扰校准强度；0 为旁路")
        self.strength.slider.setToolTip("0 为旁路；更改强度后重新测量校准密度")
        column.addWidget(self.strength)
        self.details = DisclosureButton("3×3 密度矩阵")
        self.details.setAccessibleName("展开／收起 3×3 密度矩阵")
        column.addWidget(self.details)
        self.matrix_body = QWidget()
        cells = QVBoxLayout(self.matrix_body)
        cells.setContentsMargins(0, 0, 0, 0)
        cells.setSpacing(10)
        self.matrix = []
        for row, channel in enumerate("RGB"):
            group = QWidget()
            group_column = QVBoxLayout(group)
            group_column.setContentsMargins(0, 0, 0, 0)
            group_column.setSpacing(6)
            title = QLabel(f"{channel} 密度通道")
            title.setObjectName("muted")
            group_column.addWidget(title)
            for column_index, source_channel in enumerate("RGB"):
                cell = ScrubberControl(
                    f"{channel} ← {source_channel}", -8, 8, 0.000001, 0
                )
                cell.setAccessibleName(
                    f"胶片密度矩阵 {row + 1} 行 {column_index + 1} 列"
                )
                cell.valueChanged.connect(self.matrix_changed)
                group_column.addWidget(cell)
                self.matrix.append(cell)
            cells.addWidget(group)
        column.addWidget(self.matrix_body)
        self.matrix_body.hide()
        self.details.toggled.connect(self.matrix_body.setVisible)
        self.status = QLabel()
        self.status.setWordWrap(True)
        column.addWidget(self.status)
        if errors:
            skipped = QLabel(f"{len(errors)} 份配置不适用于当前 C-41 路线（悬停查看）")
            skipped.setWordWrap(True)
            skipped.setToolTip("\n".join(e["path"] + ": " + e["error"] for e in errors))
            column.addWidget(skipped)
        self.choice.currentIndexChanged.connect(self.select)
        self.strength.valueChanged.connect(self.strength_changed)
        self.set_document(None)

    @staticmethod
    def title(profile):
        types = {
            "measured": "实测",
            "tuned": "调校",
            "specsheet-based": "数据表近似",
            "built-in": "内置",
        }
        return (
            profile["name"]
            + " · "
            + types.get(profile["type"], profile["type"] or "自定义")
        )

    def set_document(self, document, *, default_context=None, reset_defaults=False):
        self.loading = True
        try:
            self.document = deepcopy(checked_film(document)) if document else None
            value = self.document or generic_film()
            self._set_matrix_defaults(value, default_context, reset_defaults)
            index = next(
                (
                    i
                    for i, p in enumerate(self.profiles)
                    if p == {**value, "strength": 0.0}
                ),
                None,
            )
            if index is None:
                self.profiles.append({**value, "strength": 0.0})
                self.choice.addItem(self.title(value))
                index = len(self.profiles) - 1
            self.choice.setCurrentIndex(index)
            self.strength.setValue(value["strength"])
            for cell, number, default in zip(self.matrix, value["matrix"], self._matrix_defaults):
                cell.setDefaultValue(default, description="所选配置初始值")
                cell.setValue(number)
            origin = value["origin"]
            self.status.setText(
                "负片密度串扰校准；作用后重测密度。仅用于 negpy · 原生。\n"
                + (
                    "当前旁路。"
                    if value["strength"] == 0
                    else "已启用；配置和矩阵随照片保存。"
                )
            )
            self.status.setToolTip(
                origin["path"] if origin else "NegPy 内置矩阵；未修改原配置文件"
            )
        finally:
            self.loading = False

    def _set_matrix_defaults(self, value, context, explicit):
        """Keep the chosen profile's coefficients across live recipe rebases."""
        name = value["name"]
        base_name = name.removesuffix(" · 自定义") if value["type"] == "tuned" else name
        key = (base_name, value["process"])
        if (explicit or self._matrix_defaults is None or key != self._matrix_default_key
                or context != self._matrix_default_context):
            reference = next((profile for profile in self._matrix_reference_profiles
                              if value.get("origin") and profile.get("origin") == value["origin"]), None)
            if reference is None:
                reference = next((profile for profile in self._matrix_reference_profiles
                                  if profile["name"] in (name, base_name)), None)
            cache_key = (context, key)
            if explicit:
                self._matrix_defaults = list(value["matrix"])
            elif cache_key in self._matrix_default_cache:
                self._matrix_defaults = list(self._matrix_default_cache[cache_key])
            else:
                self._matrix_defaults = list(value["matrix"] if reference is None else reference["matrix"])
            self._matrix_default_cache[cache_key] = list(self._matrix_defaults)
        self._matrix_default_key, self._matrix_default_context = key, context

    def select(self, index):
        if self.loading:
            return
        value = deepcopy(self.profiles[index])
        value["strength"] = 1.0
        self.set_document(value, default_context=self._matrix_default_context, reset_defaults=True)
        self.changed.emit()

    def strength_changed(self):
        if self.loading:
            return
        value = deepcopy(self.document or self.profiles[self.choice.currentIndex()])
        value["strength"] = self.strength.value()
        self.document = checked_film(value)
        self.status.setText(
            "当前旁路。"
            if value["strength"] == 0
            else "已启用负片密度校准；仅作用于 negpy · 原生。"
        )
        self.changed.emit()

    def matrix_changed(self):
        if self.loading:
            return
        value = deepcopy(self.document or self.profiles[self.choice.currentIndex()])
        value.update(
            name=value["name"] + " · 自定义"
            if value["type"] != "tuned"
            else value["name"],
            type="tuned",
            matrix=[cell.value() for cell in self.matrix],
            origin=None,
        )
        try:
            self.document = checked_film(value)
        except ValueError as error:
            self.status.setText(str(error) + "；尚未应用，请修正矩阵。")
            return
        self.status.setText(
            "自定义密度矩阵；仅保存在照片配方中，不写入原 NegPy 配置目录。"
        )
        self.changed.emit()

    def choose(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入 NegPy 胶片密度矩阵", "", "NegPy 胶片配置 (*.toml)"
        )
        if path:
            try:
                value = read_film(path)
                value["strength"] = 1.0
                self._matrix_reference_profiles.insert(0, deepcopy(value))
                self.set_document(value, default_context=self._matrix_default_context, reset_defaults=True)
                self.changed.emit()
            except Exception as error:
                GlassMessageBox.warning(self, "胶片校准未载入", str(error))

    def reset(self):
        self.set_document(None, default_context=self._matrix_default_context, reset_defaults=True)
        self.changed.emit()

    def value(self):
        return deepcopy(self.document)
