"""Explicit RGB-role and processing recipe review before jobs enter a queue."""

from pathlib import Path
import json
import uuid

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .controls_ui import ProfessionalEditor
from .range_probe import PRIMARY
from .workspace import (
    checked_recipe,
    write_json,
    checked_native_float_controls,
    checked_highlight_controls,
    checked_cool_headroom_controls,
    checked_flex_research_controls,
    checked_nikon_research_controls,
)
from .flatfield import FILE_FILTER, RAW_EXTENSIONS
from .native import sha256
from .preview_data import ROOT
from .tasks import Tasks
from .workflow_combo import WorkflowComboBox as QComboBox
from .workflow_glass import GlassMessageBox
from .professional_controls import native_papers


class RecipeDialog(QDialog):
    def __init__(self, groups, parent=None, initial=None, geometry=None):
        super().__init__(parent)
        from .workflow_glass import prepare_glass

        prepare_glass(self)
        self.setWindowTitle("PURE RGB · 输入与专业调色")
        self.resize(980, 820)
        self.groups = groups
        self._native_float_recipe = {}
        # Structural input metadata follows an edited recipe; no new UI option.
        self._raw_import_policy = {
            key: initial[key]
            for key in ("raw_import", "registration_check")
            if initial and key in initial
        }
        outer = QVBoxLayout(self)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        general = QWidget()
        self.tabs.addTab(general, "输入与输出")
        layout = QVBoxLayout(general)
        title = QLabel(f"核对 {len(groups)} 组通道顺序；原始 RAW 保持只读。")
        layout.addWidget(title)
        self.table = QTableWidget(len(groups), 4)
        self.table.setMinimumHeight(120)
        self.table.setHorizontalHeaderLabels(
            ["组", "R 红光曝光", "G 绿光曝光", "B 蓝光曝光"]
        )
        self.roles = []
        for row, paths in enumerate(groups):
            self.table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            selectors = []
            for col, selected in enumerate(paths):
                box = QComboBox()
                for p in paths:
                    box.addItem(Path(p).name, p)
                box.setCurrentIndex(paths.index(selected))
                self.table.setCellWidget(row, col + 1, box)
                selectors.append(box)
            self.roles.append(selectors)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 45)
        for col in (1, 2, 3):
            self.table.setColumnWidth(col, 230)
        layout.addWidget(self.table, 1)
        form = QFormLayout()
        self.auto = QCheckBox("自动识别画面内部；失败时明确报错，不退回全图")
        is_primary = all(Path(p).resolve().parent == PRIMARY for g in groups for p in g)
        self.auto.setChecked(not is_primary)
        self.roi = QLineEdit("1000 350 5100 3650" if is_primary else "")
        self.roi.setPlaceholderText("x0 y0 x1 y1（合成并完成方向处理后的像素坐标）")
        self.roi.setEnabled(not self.auto.isChecked())
        self.auto.toggled.connect(lambda checked: self.roi.setEnabled(not checked))
        form.addRow("密度取样", self.auto)
        form.addRow("手工取样区域", self.roi)
        if geometry:
            self.geometry_info = geometry
            choose_roi = QPushButton("在合成预览中框选区域…")
            choose_roi.clicked.connect(self.choose_roi)
            form.addRow(choose_roi)
        self.density = QComboBox()
        self.density.addItem("NegPy 密度窗口", "negpy")
        self.density.addItem("画面百分位（实验对照）", "percentile")
        form.addRow("密度", self.density)
        self.paper = QComboBox()
        for key, title in native_papers():
            self.paper.addItem(title, key)
        self.paper.setToolTip(
            "基准 Native 印相及 Nikon CMS／FlexColor 亮度拟合参考；Native 控件仅改变 Native。"
        )
        form.addRow("基准／拟合参考相纸", self.paper)
        self.cool_tone = QComboBox()
        self.cool_tone.addItem("Soft shoulder · 柔和肩部提亮（默认）", "soft-shoulder")
        self.cool_tone.addItem("原保守版本（历史兼容）", "conservative")
        form.addRow("Nikon CMS 呈现", self.cool_tone)
        self.ev = QDoubleSpinBox()
        self.ev.setRange(-4, 4)
        self.ev.setSingleStep(0.1)
        self.ev.setSuffix(" EV")
        form.addRow("三路输出曝光", self.ev)
        self.crop = QCheckBox("使用下方独立裁切区域（不跟随密度取样框）")
        self.crop.setChecked(False)
        form.addRow("输出画幅", self.crop)
        self.output_roi = QLineEdit()
        self.output_roi.setPlaceholderText(
            "独立输出裁切 x0 y0 x1 y1；不勾选则保留全画幅"
        )
        self.output_roi.setEnabled(False)
        self.crop.toggled.connect(self.output_roi.setEnabled)
        form.addRow("输出裁切区域", self.output_roi)
        self.flat = QLineEdit()
        self.flat.setPlaceholderText("选择单张平场 RAW，或已有增益 NPZ")
        self.flat.setToolTip(
            "同一机位、镜头/光圈、对焦状态下，移开底片拍摄均匀白光；三通道需有足够信号。"
        )
        row = QHBoxLayout()
        row.addWidget(self.flat)
        self.choose_flat_button = QPushButton("选择…")
        self.choose_flat_button.clicked.connect(self.choose_flat)
        row.addWidget(self.choose_flat_button)
        clear_flat = QPushButton("清除")
        clear_flat.clicked.connect(self.clear_flat)
        row.addWidget(clear_flat)
        form.addRow("平场配置", row)
        self.flat_status = QLabel("RAW 会自动生成可复用增益；不修改原文件。")
        self.flat_status.setWordWrap(True)
        form.addRow("", self.flat_status)
        self.flat_tasks = Tasks(ROOT / "outputs/flatfield-imports/requests", self)
        self.flat_tasks.done.connect(self.flat_imported)
        self.flat_tasks.failed.connect(self.flat_failed)
        layout.addLayout(form)
        note = QLabel(
            "平场不代表色彩标定；曝光补偿不等于高光找回。\n专业调色按模式独立保存，应用后从 RAW 重新生成；高光／阴影范围找回仍待独立验证。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.professional = ProfessionalEditor()
        self.tabs.addTab(self.professional, "专业调色")
        layout = outer
        preset_row = QHBoxLayout()
        save = QPushButton("保存处理预设…")
        load = QPushButton("读取处理预设…")
        save.clicked.connect(self.save_preset)
        load.clicked.connect(self.load_preset)
        preset_row.addWidget(save)
        preset_row.addWidget(load)
        layout.addLayout(preset_row)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons = buttons
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("加入处理队列")
        from .workflow_attention import attach_attention

        attach_attention(buttons.button(QDialogButtonBox.StandardButton.Ok))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if initial:
            self.set_parameters(initial)
            self.tabs.setCurrentIndex(1)

    def set_parameters(self, initial):
        point = (
            checked_native_float_controls(initial["native_float_controls"])
            if "native_float_controls" in initial
            else None
        )
        highlight = (
            checked_highlight_controls(initial["highlight_controls"])
            if "highlight_controls" in initial
            else None
        )
        headroom = (
            checked_cool_headroom_controls(initial["cool_headroom_controls"])
            if "cool_headroom_controls" in initial
            else None
        )
        flex_research = (
            checked_flex_research_controls(initial["flex_research_controls"])
            if "flex_research_controls" in initial
            else None
        )
        nikon_research = (
            checked_nikon_research_controls(initial["nikon_research_controls"])
            if "nikon_research_controls" in initial
            else None
        )
        self.professional.set_documents(initial)
        self.cool_tone.setCurrentIndex(
            self.cool_tone.findData(initial.get("cool_tone", "conservative"))
        )
        self.auto.setChecked(initial["roi"] is None)
        self.roi.setText(" ".join(map(str, initial["roi"] or [])))
        self.density.setCurrentIndex(self.density.findData(initial["density"]))
        self.paper.setCurrentIndex(self.paper.findData(initial["paper"]))
        self.ev.setValue(initial["exposure_ev"])
        self.crop.setChecked(initial["crop"])
        self.output_roi.setText(
            " ".join(map(str, initial.get("output_roi", initial.get("roi")) or []))
        )
        self.flat.setText(initial["flatfield"] or "")
        self._native_float_recipe = {"native_float_controls": point} if point else {}
        if highlight:
            self._native_float_recipe["highlight_controls"] = highlight
        if headroom:
            self._native_float_recipe["cool_headroom_controls"] = headroom
        if flex_research:
            self._native_float_recipe["flex_research_controls"] = flex_research
        if nikon_research:
            self._native_float_recipe["nikon_research_controls"] = nikon_research

    def preset(self):
        return {
            "schema": "pure-rgb-processing-preset",
            "version": 1,
            "parameters": {k: v for k, v in self.recipes()[0].items() if k != "raw"},
        }

    def save_preset(self):
        try:
            preset = self.preset()
            path, _ = QFileDialog.getSaveFileName(
                self, "保存处理预设", "PURE-RGB-preset.json", "JSON (*.json)"
            )
            if path:
                write_json(path, preset)
        except Exception as error:
            GlassMessageBox.warning(self, "预设未保存", str(error))

    def load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "读取处理预设", "", "JSON (*.json)")
        if path:
            try:
                preset = json.loads(Path(path).read_text())
                if (
                    preset.get("schema") != "pure-rgb-processing-preset"
                    or preset.get("version") != 1
                ):
                    raise ValueError("不支持的预设版本")
                parameters = checked_recipe(
                    {
                        **preset["parameters"],
                        "raw": [c.currentData() for c in self.roles[0]],
                    }
                )
                self.set_parameters(parameters)
            except Exception as error:
                GlassMessageBox.warning(self, "预设未载入", str(error))

    def choose_flat(self):
        if self.flat_tasks.busy:
            self.flat_tasks.cancel()
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择单张平场 RAW 或已有增益 NPZ", "", FILE_FILTER
        )
        if path:
            self.import_flat(path)

    def import_flat(self, path):
        path = Path(path).resolve(strict=True)
        if path.suffix.lower() == ".npz":
            self.flat.setText(str(path))
            self.flat_status.setText("已选择增益配置；处理时核验来源与数据。")
            return
        if path.suffix.lower() not in RAW_EXTENSIONS:
            self.flat_status.setText("请选择 RAW 或增益 NPZ。")
            return
        root = getattr(
            getattr(self.parent(), "workspace", None),
            "root",
            ROOT / "outputs/flatfield-imports",
        )
        directory = root / "calibrations" / uuid.uuid4().hex
        self.flat_status.setText("正在解码 RAW 并生成平场增益…")
        self.choose_flat_button.setText("取消导入")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.flat_tasks.start(
            {
                "action": "flatfield_import",
                "calibration": {"path": str(path), "sha256": sha256(path)},
                "directory": str(directory),
                "targets": [p for group in self.groups for p in group],
            }
        )

    def flat_imported(self, request, result):
        report = result["calibration"]
        self.flat.setText(report["gain_path"])
        q = report["provenance"]["quality"]
        residual = max(-q["after"]["low"], q["after"]["high"])
        warnings = report["compatibility"]["warnings"]
        self.flat_status.setText(
            f"已导入 {Path(report['path']).name} · 校正后明暗偏差约 {residual:.1%}"
            + ("\n" + "\n".join(warnings) if warnings else "")
        )
        self.choose_flat_button.setText("选择…")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)

    def flat_failed(self, request, message, cancelled):
        self.flat_status.setText(
            "平场导入已取消；原配置保留。" if cancelled else "平场导入失败：" + message
        )
        self.choose_flat_button.setText("选择…")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)

    def clear_flat(self):
        if self.flat_tasks.busy:
            self.flat_tasks.cancel()
        self.flat.clear()
        self.flat_status.setText("未使用平场校正。")

    def done(self, result):
        if result == QDialog.DialogCode.Accepted and self.flat_tasks.busy:
            return
        self.flat_tasks.shutdown()
        super().done(result)

    def choose_roi(self):
        from .geometry_ui import GeometryDialog

        dialog = GeometryDialog(self.geometry_info, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selection:
            self.auto.setChecked(False)
            self.roi.setText(" ".join(map(str, dialog.selection)))

    def recipes(self):
        roi = (
            None
            if self.auto.isChecked()
            else [int(v) for v in self.roi.text().replace(",", " ").split()]
        )
        return [
            checked_recipe(
                {
                    "raw": [c.currentData() for c in selectors],
                    **self._raw_import_policy,
                    **self._native_float_recipe,
                    "roi": roi,
                    "density": self.density.currentData(),
                    "paper": self.paper.currentData(),
                    "exposure_ev": self.ev.value(),
                    "crop": self.crop.isChecked(),
                    "output_roi": [
                        int(v) for v in self.output_roi.text().replace(",", " ").split()
                    ]
                    if self.crop.isChecked()
                    else None,
                    "flatfield": self.flat.text().strip() or None,
                    "cool_tone": self.cool_tone.currentData(),
                    **self.professional.documents(),
                }
            )
            for selectors in self.roles
        ]

    def validate(self):
        if self.flat_tasks.busy:
            return
        try:
            self.recipes()
        except Exception as error:
            GlassMessageBox.warning(self, "配方需修正", str(error))
            return
        self.accept()
