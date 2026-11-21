"""Professional recipe editor; every exposed control has a render mapping."""

import math

from PyQt6.QtCore import QSignalBlocker, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .professional_controls import (
    MODE_KEYS,
    NATIVE_SPECS,
    OUTPUT_SPECS,
    checked_controls,
    checked_outputs,
    ENGINE_SPECS,
    checked_engines,
)
from .workflow_combo import WorkflowComboBox as QComboBox
from .motion_sliders import ScrubberSlider
from .preview_data import MODES


class NumericControl(QWidget):
    changed = pyqtSignal()

    def __init__(self, spec, parent=None):
        super().__init__(parent)
        self.spec = spec
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.slider = ScrubberSlider(
            self, show_labels=False, readout=lambda: f"{spec.low + self.slider.value() * spec.step:g}"
        )
        self.slider.setAccessibleName(spec.title)
        self.slider.setRange(0, round((spec.high - spec.low) / spec.step))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(spec.low, spec.high)
        self.spin.setDecimals(max(0, -math.floor(math.log10(spec.step))))
        self.spin.setSingleStep(spec.step)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(90)
        self.spin.setAccessibleName(spec.title + " 数值")
        reset = QPushButton("↺")
        reset.setFixedWidth(34)
        reset.setToolTip(f"复位至 {spec.default:g}")
        reset.clicked.connect(lambda: self.set_value(spec.default))
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)
        row.addWidget(reset)
        self.slider.valueChanged.connect(self.slide_changed)
        self.spin.valueChanged.connect(self.spin_changed)
        self.set_value(spec.default)

    def slide_changed(self, value):
        self.spin.setValue(round(self.spec.low + value * self.spec.step, 6))

    def spin_changed(self, value):
        with QSignalBlocker(self.slider):
            self.slider.setValue(round((value - self.spec.low) / self.spec.step))
        self.changed.emit()

    def set_value(self, value):
        with QSignalBlocker(self.spin):
            self.spin.setValue(value)
        self.spin_changed(self.spin.value())

    def value(self):
        return self.spin.value()


class ControlPage(QScrollArea):
    changed = pyqtSignal()

    def __init__(self, specs, note, parent=None):
        super().__init__(parent)
        self.specs = specs
        self.controls = {}
        self.setWidgetResizable(True)
        content = QWidget()
        self.setWidget(content)
        column = QVBoxLayout(content)
        hint = QLabel(note)
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        column.addWidget(hint)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for spec in specs:
            if type(spec.default) is bool:
                widget = QCheckBox()
                widget.toggled.connect(self.changed)
            elif spec.choices:
                widget = QComboBox()
                for value, title in spec.choices:
                    widget.addItem(title, value)
                widget.currentIndexChanged.connect(self.changed)
            else:
                widget = NumericControl(spec)
                widget.changed.connect(self.changed)
            widget.setAccessibleName(spec.title)
            widget.setObjectName(spec.key)
            self.controls[spec.key] = widget
            form.addRow(spec.title, widget)
        column.addLayout(form)
        reset = QPushButton("复位本页")
        reset.clicked.connect(lambda: self.set_values({}))
        column.addWidget(reset, alignment=Qt.AlignmentFlag.AlignRight)
        column.addStretch()
        self.set_values({})

    def set_values(self, values):
        for spec in self.specs:
            widget = self.controls[spec.key]
            value = values.get(spec.key, spec.default)
            if type(spec.default) is bool:
                widget.setChecked(value)
            elif spec.choices:
                widget.setCurrentIndex(widget.findData(value))
            else:
                widget.set_value(value)

    def values(self):
        values = {}
        for spec in self.specs:
            widget = self.controls[spec.key]
            value = (
                widget.isChecked()
                if type(spec.default) is bool
                else widget.currentData()
                if spec.choices
                else widget.value()
            )
            if value != spec.default:
                values[spec.key] = value
        return values


class ProfessionalEditor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.native_pages, self.output_pages, self.engine_pages = [], {}, {}
        column = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setObjectName("section")
        column.addWidget(self.summary)
        self.modes = QTabWidget()
        column.addWidget(self.modes, 1)
        native = QTabWidget()
        self.modes.addTab(native, MODES["native"])
        self.output_pages["native"] = self.output_page()
        native.addTab(self.output_pages["native"], "正像")
        notes = {
            "明暗与反差": "Native 印相参数；增加密度使正像变暗，较低 ISO-R 反差更硬。",
            "色彩平衡": "Native 全局／阴影／高光 CMY；不改变其他两路的拟合参考。",
            "相纸曲线": "Native 相纸肩部、趾部及染料呈现；这些参数不代表密度范围找回。",
            "细节与效果": "Native 原生细节处理。请用 100% 预览判断锐化和颗粒。",
        }
        channels = QTabWidget()
        for group, title in (
            ("明暗与反差", "印相"),
            ("色彩平衡", "CMY"),
            ("相纸曲线", "曲线"),
            ("R 通道微调", "R"),
            ("G 通道微调", "G"),
            ("B 通道微调", "B"),
            ("细节与效果", "细节"),
        ):
            page = ControlPage(
                tuple(s for s in NATIVE_SPECS if s.group == group),
                notes.get(group, "Native 相纸色层微调；叠加在全局参数上。"),
            )
            page.changed.connect(self.refresh_summary)
            self.native_pages.append(page)
            if group in ("R 通道微调", "G 通道微调", "B 通道微调"):
                if channels.count() == 0:
                    native.addTab(channels, "通道")
                channels.addTab(page, title)
            else:
                native.addTab(page, title)
        for mode, title in (
            ("cool-shared-rgb", MODES["cool-shared-rgb"]),
            ("flex-endpoint", MODES["flex-endpoint"]),
        ):
            tools = QTabWidget()
            self.output_pages[mode] = self.output_page()
            tools.addTab(self.output_pages[mode], "正像")
            self.engine_pages[mode] = ControlPage(
                ENGINE_SPECS[mode], "已接入的实验引擎参数；非原厂完整 CMS。"
            )
            self.engine_pages[mode].changed.connect(self.refresh_summary)
            tools.addTab(self.engine_pages[mode], "引擎曲线")
            self.modes.addTab(tools, title)
        self.refresh_summary()

    def output_page(self):
        page = ControlPage(
            OUTPUT_SPECS,
            "仅作用于该模式反演后的正像；叠加三路公共曝光。通道平衡、反差和饱和度可能影响亮暗部裁切，不能恢复已丢失的层次。",
        )
        page.changed.connect(self.refresh_summary)
        return page

    def documents(self):
        native = {}
        for page in self.native_pages:
            native.update(page.values())
        output = {
            mode: {"version": 1, "values": page.values()}
            for mode, page in self.output_pages.items()
        }
        result = {}
        if native:
            result["native_controls"] = checked_controls(
                {"version": 1, "values": native}, NATIVE_SPECS
            )
        output = checked_outputs(output)
        if output:
            result["output_controls"] = output
        engines = checked_engines(
            {
                mode: {"version": 1, "values": page.values()}
                for mode, page in self.engine_pages.items()
            }
        )
        if engines:
            result["engine_controls"] = engines
        if getattr(self, "post_documents", None):
            result["post_controls"] = self.post_documents
        if getattr(self, "native_film", None):
            result["native_film"] = self.native_film
        if getattr(self, "flex_profile", None):
            result["flex_profile"] = self.flex_profile
            # This editor is retained for input/preset compatibility. Preserve explicit
            # v2 defaults even while the professional tab is hidden in the workbench.
            if getattr(self, "flex_explicit", None):
                result.setdefault("engine_controls", {})["flex-endpoint"] = (
                    self.flex_explicit
                )
        return result

    def set_documents(self, recipe):
        from .positive_post import checked_post

        self.post_documents = checked_post(recipe.get("post_controls", {}))
        self.native_film = recipe.get("native_film")
        self.flex_profile = recipe.get("flex_profile")
        self.flex_explicit = recipe.get("engine_controls", {}).get("flex-endpoint")
        native = checked_controls(
            recipe.get("native_controls", {"version": 1, "values": {}}), NATIVE_SPECS
        )
        output = checked_outputs(recipe.get("output_controls", {}))
        engines = checked_engines(recipe.get("engine_controls", {}))
        for page in self.native_pages:
            page.set_values(native["values"])
        for mode in MODE_KEYS:
            self.output_pages[mode].set_values(output.get(mode, {}).get("values", {}))
        for mode, page in self.engine_pages.items():
            page.set_values(engines.get(mode, {}).get("values", {}))
        self.refresh_summary()

    def refresh_summary(self):
        count = (
            sum(len(p.values()) for p in self.native_pages)
            + sum(len(p.values()) for p in self.output_pages.values())
            + sum(len(p.values()) for p in self.engine_pages.values())
        )
        self.summary.setText(
            f"{count} 项自定义调整 · 加入队列后从 RAW 生成新版本，当前照片保留。"
        )
