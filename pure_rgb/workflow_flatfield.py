"""Optional source flat-field calibration in the staged desktop workflow."""

from copy import deepcopy
from pathlib import Path
import uuid

from .workflow_disclosure import DisclosureButton
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .flatfield import FILE_FILTER, RAW_EXTENSIONS
from .native import sha256
from .workflow_combo import WorkflowComboBox as QComboBox
from .motion_sliders import ScrubberControl


def flatfield_support(photo):
    if not photo:
        return False, "先导入照片，再选择匹配的平场参考。"
    if photo["source_mode"] == "bw":
        return False, "黑白单张尚未接通此平场链；当前不启用。"
    if photo["base"].get("raw_import") == "standard-cfa-dng-v1":
        return False, "CFA DNG 尚未接通此平场链；RGB RW2 可用。"
    if photo["base"].get("raw_import") == "standard-mono-v1":
        return True, "单色 RGB 可使用带完整传感器画幅记录的平场增益 NPZ。"
    return True, "使用同机位、镜头、光圈与对焦状态的均匀白光参考；先移开底片。"


class FlatfieldUI:
    def build_flatfield_tools(self, layout):
        self.flatfield_pending = {}
        self.flatfield_messages = {}
        self.flatfield_toggle = DisclosureButton("平场矫正（可选）")
        self.flatfield_toggle.setAccessibleName("平场矫正")
        layout.addWidget(self.flatfield_toggle)
        self.flatfield_tools = QWidget()
        column = QVBoxLayout(self.flatfield_tools)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)
        self.flatfield_source = QLabel("未启用平场")
        self.flatfield_source.setWordWrap(True)
        column.addWidget(self.flatfield_source)
        self.flatfield_choose = QPushButton("选择平场 RAW / 增益 NPZ…")
        self.flatfield_choose.clicked.connect(self.choose_flatfield)
        column.addWidget(self.flatfield_choose)
        self.flatfield_library_choice = QComboBox()
        self.flatfield_library_choice.setAccessibleName("已保存平场")
        self.flatfield_library_choice.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.flatfield_library_choice.setMinimumContentsLength(10)
        column.addWidget(self.flatfield_library_choice)
        library_row = QHBoxLayout()
        self.flatfield_use = QPushButton("从库中选用")
        self.flatfield_use.clicked.connect(self.use_library_flatfield)
        library_row.addWidget(self.flatfield_use)
        self.flatfield_remove = QPushButton("从库中移除")
        self.flatfield_remove.clicked.connect(self.remove_library_flatfield)
        library_row.addWidget(self.flatfield_remove)
        column.addLayout(library_row)
        self.flatfield_strength = ScrubberControl("平场强度（%）", 0, 200, 1, 100)
        self.flatfield_strength.label.setToolTip(
            "0 为旁路，100 为 NegPy 完整校正，200 为两倍曝光校正量"
        )
        self.flatfield_strength.slider.setToolTip(
            "0 为旁路，100 为 NegPy 完整校正，200 为两倍曝光校正量。更改后点击应用。"
        )
        self.flatfield_strength.valueChanged.connect(self.flatfield_strength_changed)
        column.addWidget(self.flatfield_strength)
        self._flatfield_strength_syncing = False
        self._flatfield_strength_photo = None
        self._flatfield_strength_value = 1.0
        self.flatfield_clear = QPushButton("移除平场")
        self.flatfield_clear.clicked.connect(self.clear_flatfield)
        column.addWidget(self.flatfield_clear)
        self.flatfield_apply = QPushButton("应用平场并重新处理")
        self.flatfield_apply.clicked.connect(self.apply_flatfield)
        column.addWidget(self.flatfield_apply)
        self.flatfield_status = QLabel()
        self.flatfield_status.setWordWrap(True)
        self.flatfield_status.setObjectName("muted")
        column.addWidget(self.flatfield_status)
        self.flatfield_note = QLabel()
        self.flatfield_note.setWordWrap(True)
        self.flatfield_note.setObjectName("muted")
        column.addWidget(self.flatfield_note)
        scope = QLabel("沿用现有合成后增益校正，三种色彩引擎共用。此入口为单张白光平场，不是逐 LED／暗场校准。")
        scope.setWordWrap(True)
        scope.setObjectName("muted")
        column.addWidget(scope)
        layout.addWidget(self.flatfield_tools)
        self.flatfield_tools.hide()
        self.flatfield_toggle.toggled.connect(self.flatfield_tools.setVisible)

    def open_flatfield_tools(self):
        if self.tasks.busy:
            return
        self.set_stage(0)
        self.flatfield_toggle.setChecked(True)
        self.pages.widget(0).ensureWidgetVisible(self.flatfield_tools)

    def sync_flatfield_controls(self):
        if not hasattr(self, "flatfield_tools"):
            return
        photo = self.current_photo()
        supported, note = flatfield_support(photo)
        self.flatfield_note.setText(note)
        pending = self.flatfield_pending.get(photo["id"]) if photo else None
        recipe = self.project.recipe() if photo else {}
        current = recipe.get("flatfield") if photo else None
        current_strength = float(recipe.get("flatfield_strength", 1.0)) if current else 1.0
        selected = pending.get("path") if pending is not None else current
        library = getattr(self.project, "data", {}).get("flatfield_library") or []
        self.flatfield_library_choice.clear()
        for entry in library:
            make, model = entry.get("camera") or [None, None]
            label = " · ".join(
                str(part) for part in (entry.get("name"), make, model) if part
            )
            self.flatfield_library_choice.addItem(label, entry)
        has_library = bool(library)
        self.flatfield_library_choice.setEnabled(
            bool(has_library and supported and not self.tasks.busy)
        )
        self.flatfield_use.setEnabled(
            bool(has_library and supported and not self.tasks.busy)
        )
        self.flatfield_remove.setEnabled(bool(has_library and not self.tasks.busy))
        if (
            photo
            and (
                self._flatfield_strength_photo != photo["id"]
                or self._flatfield_strength_value != current_strength
            )
        ) or (photo is None and self._flatfield_strength_value != 1.0):
            self._flatfield_strength_photo = photo["id"] if photo else None
            self._flatfield_strength_value = current_strength
            self._flatfield_strength_syncing = True
            try:
                self.flatfield_strength.setValue(round(current_strength * 100))
            finally:
                self._flatfield_strength_syncing = False
        slider_strength = self.flatfield_strength.value() / 100.0
        strength_pending = bool(selected and slider_strength != current_strength)
        self.flatfield_strength.setEnabled(bool(supported and selected and not self.tasks.busy))
        self.flatfield_choose.setEnabled(supported and not self.tasks.busy)
        self.flatfield_clear.setEnabled(bool(supported and not self.tasks.busy and
                                            (current or pending and pending.get("path"))))
        self.flatfield_apply.setEnabled(bool(supported and not self.tasks.busy and
                                             (pending is not None or strength_pending)))
        if pending is not None:
            path = pending.get("path")
            self.flatfield_source.setText("待应用：" + (Path(path).name if path else "移除当前平场"))
            self.flatfield_source.setToolTip(path or "")
        else:
            self.flatfield_source.setText(
                f"当前平场：{Path(current).name} · {current_strength:.0%}" if current
                else "未启用平场。请选择或从库中选用参考。"
            )
            self.flatfield_source.setToolTip(current or "")
        self.flatfield_apply.setText(
            ("移除平场并重新处理" if pending is not None and not pending.get("path")
             else "应用平场并重新处理") if photo and photo["initial_job"] else "用于首次导入"
        )
        self.flatfield_status.setText(self.flatfield_messages.get(photo["id"], "") if photo else "")

    def choose_flatfield(self):
        if self.tasks.busy or not flatfield_support(self.current_photo())[0]:
            return
        mono = self.current_photo()["base"].get("raw_import") == "standard-mono-v1"
        title = "选择平场增益 NPZ" if mono else "选择单张白光平场 RAW 或增益 NPZ"
        file_filter = "增益 NPZ (*.npz *.NPZ)" if mono else FILE_FILTER
        path, _ = QFileDialog.getOpenFileName(self, title, "", file_filter)
        if path:
            self.import_flatfield(path)

    def import_flatfield(self, path):
        photo = self.current_photo()
        supported, note = flatfield_support(photo)
        if not supported or self.tasks.busy:
            self.error(note if not supported else "请先完成当前处理。")
            return
        if not self.flush_advanced():
            self.status_message("请先修正左侧参数组合。")
            return
        try:
            path = Path(path).resolve(strict=True)
            if photo["base"].get("raw_import") == "standard-mono-v1" and path.suffix.lower() != ".npz":
                raise ValueError("单色 RGB 需要带传感器画幅记录的增益 NPZ。")
            if path.suffix.lower() not in (*RAW_EXTENSIONS, ".npz"):
                raise ValueError("请选择平场 RAW 或增益 NPZ。")
            request = dict(action="flatfield_import", photo=photo["id"],
                calibration=dict(path=str(path), sha256=sha256(path)), freeze_gain=True,
                directory=str(self.project.workspace.root / "calibrations" / uuid.uuid4().hex),
                targets=deepcopy(photo["base"]["raw"]))
            self.stop_preview()
            self.flatfield_messages[photo["id"]] = "正在读取参考并检查平场增益，原配置保留。"
            self.tasks.start(request)
            self.sync_controls()
        except Exception as error:
            self.error(error)

    def flatfield_imported(self, request, result):
        identifier = request["photo"]
        report = result["calibration"]
        photo = self.project.photo(identifier)
        capture = (report.get("provenance") or {}).get("capture", {})
        if photo["base"].get("raw_import") in ("standard-crop-v1", "standard-mono-v1") and not capture.get("sensor_shape"):
            raise ValueError("增益配置缺少完整传感器尺寸，不能对标准裁切源定位。请选择平场 RAW 重新生成。")
        self.flatfield_pending[identifier] = dict(path=report["gain_path"], report=deepcopy(report))
        self.project.register_flatfield(report)
        warnings = report.get("compatibility", {}).get("warnings", [])
        self.flatfield_messages[identifier] = "参考已准备，应用后重新处理完整源。" + ("\n" + "\n".join(warnings) if warnings else "")
        self.status_message("平场参考已准备；点击应用后生效。")

    def clear_flatfield(self):
        photo = self.current_photo()
        if not photo or self.tasks.busy:
            return
        self.flatfield_pending[photo["id"]] = dict(path=None)
        self.flatfield_messages[photo["id"]] = "待移除平场；应用后重新处理，原版本保留。"
        self.sync_controls()

    def use_library_flatfield(self):
        photo = self.current_photo()
        if not photo or self.tasks.busy or not flatfield_support(photo)[0]:
            return
        entry = self.flatfield_library_choice.currentData()
        if not entry or not entry.get("gain_path"):
            return
        if photo["base"].get("raw_import") == "standard-mono-v1":
            self.import_flatfield(str(entry["gain_path"]))
            return
        self.flatfield_pending[photo["id"]] = dict(path=str(entry["gain_path"]))
        self.flatfield_messages[photo["id"]] = "已从库中选用平场；点击应用后重新处理完整源。"
        self.sync_controls()

    def remove_library_flatfield(self):
        if self.tasks.busy:
            return
        entry = self.flatfield_library_choice.currentData()
        if not entry:
            return
        if self.project.remove_flatfield_entry(entry.get("id")):
            self.status_message("已从平场库移除该项；正在使用中的平场文件不受影响。")
            self.sync_controls()

    def flatfield_strength_changed(self, value):
        if self._flatfield_strength_syncing:
            return
        photo = self.current_photo()
        if photo:
            self.flatfield_messages[photo["id"]] = "平场强度待应用；点击应用后重新处理。"
        self.sync_controls()

    def apply_flatfield(self):
        photo = self.current_photo()
        if not photo or self.tasks.busy:
            return
        if not self.flush_advanced():
            self.status_message("请先修正左侧参数组合。")
            return
        pending = self.flatfield_pending.get(photo["id"])
        strength = self.flatfield_strength.value() / 100.0
        recipe = self.project.recipe()
        current = recipe.get("flatfield")
        current_strength = float(recipe.get("flatfield_strength", 1.0)) if current else 1.0
        if pending is None and (not current or strength == current_strength):
            return
        try:
            if pending is not None:
                path = pending["path"]
                self.project.set_flatfield(path, strength=strength if path else 1.0)
                self.flatfield_pending.pop(photo["id"], None)
            else:
                self.project.set_flatfield_strength(strength)
            self.flatfield_messages[photo["id"]] = "已更新平场配置；源处理期间原成片保留。"
            self.crop_locator_cache = None
            self.snapshot_store.clear()
            self.sync_controls()
            self.refresh_photos()
            if photo["initial_job"]:
                self.save_version()
            else:
                self.status_message("平场已设置，核对并导入时生效。")
        except Exception as error:
            self.error(error)

    def flatfield_failed(self, request, message, cancelled):
        self.flatfield_messages[request["photo"]] = (
            "平场导入已取消；原配置保留。" if cancelled else "平场导入失败：" + str(message)
        )

    def flatfield_rendered(self, request, manifest):
        binding = self.project.data["jobs"][request["id"]]
        identifier = binding["photo"]
        if manifest.get("calibration"):
            calibration = manifest["calibration"]
            warnings = calibration.get("compatibility", {}).get("warnings", [])
            strength = calibration.get("strength", 1.0)
            self.flatfield_messages[identifier] = (
                ("当前完整源按 0% 旁路平场。" if strength == 0 else
                 f"当前完整源已应用 {strength:.0%} 平场。测量和各色彩引擎共用此来源。")
                + ("\n" + "\n".join(warnings) if warnings else "")
            )
        elif identifier in self.flatfield_messages:
            self.flatfield_messages[identifier] = "完整源已恢复为未应用平场的版本。"
