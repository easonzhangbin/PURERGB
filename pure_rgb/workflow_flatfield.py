"""Optional source flat-field calibration in the staged desktop workflow."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
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
from .raw_import import file_identity
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
    def restore_flatfield_defaults(self):
        """Read the previous import choice before a new photo is created."""
        path = getattr(self, "flatfield_defaults_path", None)
        if path is None:
            return
        setting = self.project.flatfield_default()
        if setting is None:
            try:
                setting = json.loads(path.read_text())
            except (OSError, ValueError):
                setting = None
        if setting is None:
            try:
                preferences = json.loads((path.parent / "preferences.json").read_text())
                for name in ("workspace", "last_task_workspace"):
                    prior = Path(preferences.get(name, "")) / "ui-workflow.json"
                    if not prior.is_file() or prior.resolve() == self.project.path.resolve():
                        continue
                    data = json.loads(prior.read_text())
                    setting = data.get("flatfield_default")
                    if setting is None:
                        for photo in reversed(data.get("photos", [])):
                            recipe = photo.get("drafts", {}).get(photo.get("engine"), photo.get("base", {}))
                            gain = photo.get("flatfield_input", recipe.get("flatfield"))
                            if gain:
                                entry = next((e for e in data.get("flatfield_library", [])
                                              if e.get("gain_path") == gain), {})
                                setting = dict(path=gain, enabled=True, strength=recipe.get("flatfield_strength", 1.0),
                                               name=entry.get("name") or Path(gain).name)
                                break
                    if setting is not None:
                        break
            except (OSError, ValueError, TypeError):
                setting = None
        if not isinstance(setting, dict):
            return
        gain = setting.get("path")
        strength = setting.get("strength", 1.0)
        if not isinstance(gain, (str, type(None))) or not isinstance(strength, (int, float)) or not 0 <= strength <= 2:
            return
        self.project.data.setdefault("flatfield_default", deepcopy(setting))
        if gain and not Path(gain).is_file():
            return
        self.remember_flatfield_default(setting)

    def remember_flatfield_default(self, setting):
        """Keep a frozen gain available after switching or removing a project."""
        path = getattr(self, "flatfield_defaults_path", None)
        if path is None:
            return
        from .workspace import write_json

        saved = deepcopy(setting)
        gain = saved.get("path")
        if gain and saved.get("enabled"):
            source = Path(gain)
            digest = sha256(source)
            directory = path.parent / "flatfield-presets"
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / (digest + ".npz")
            if source.absolute() != target.absolute() and (not target.is_file() or sha256(target) != digest):
                temporary = directory / (uuid.uuid4().hex + ".npz")
                try:
                    shutil.copyfile(source, temporary)
                    if sha256(temporary) != digest:
                        raise ValueError("平场增益复制期间发生变化")
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            saved["path"] = str(target)
        write_json(path, saved)

    def build_flatfield_tools(self, layout):
        self.restore_flatfield_defaults()
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
        self.flatfield_apply = QPushButton("应用到整卷并重新处理")
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
        scope = QLabel("应用后整卷兼容照片及后续导入组共用此平场；更换机位、镜头或光圈后请重新核对参考。")
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
        setting = self.project.flatfield_default()
        current = recipe.get("flatfield") if photo else None
        current_strength = float(recipe.get("flatfield_strength", 1.0)) if current else 1.0
        selected = pending.get("path") if pending is not None else current
        library = getattr(self.project, "data", {}).get("flatfield_library") or []
        entries = []
        for entry in library:
            make, model = entry.get("camera") or [None, None]
            label = " · ".join(
                str(part) for part in (entry.get("name"), make, model) if part
            )
            entries.append((label, entry))
        current_entry = self.flatfield_library_choice.currentData()
        self.flatfield_library_choice.syncItems(entries or [("暂无平场配置", None)])
        if current_entry is not None:
            index = self.flatfield_library_choice.findData(current_entry)
            self.flatfield_library_choice.setCurrentIndex(max(0, index))
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
                                            (current or pending and pending.get("path")
                                             or setting and setting.get("enabled"))))
        self.flatfield_apply.setEnabled(bool(supported and not self.tasks.busy and
                                             (pending is not None or strength_pending)))
        if pending is not None:
            path = pending.get("path")
            self.flatfield_source.setText("待应用：" + (Path(path).name if path else "移除当前平场"))
            self.flatfield_source.setToolTip(path or "")
        else:
            self.flatfield_source.setText(
                f"当前平场：{self.project.flatfield_name(current)} · {current_strength:.0%}" if current
                else "未启用平场。请选择或从库中选用参考。"
            )
            self.flatfield_source.setToolTip(current or "")
            if photo is None and setting and setting.get("enabled"):
                name = setting.get("name") or Path(setting["path"]).name
                self.flatfield_source.setText(f"后续导入平场：{name} · {setting.get('strength', 1.0):.0%}")
                self.flatfield_source.setToolTip(setting["path"])
        self.flatfield_apply.setText(
            ("移除整卷平场并重新处理" if pending is not None and not pending.get("path")
             else "应用到整卷并重新处理") if photo and photo["initial_job"] else "用于整卷及后续导入"
        )
        status = self.flatfield_messages.get(photo["id"], "") if photo else ""
        if photo and photo.get("flatfield_inheritance_warning"):
            status += "\n" + photo["flatfield_inheritance_warning"]
        elif setting:
            if setting.get("enabled") and not Path(setting.get("path", "")).is_file():
                status += "\n默认平场文件已不存在，请重新选择参考。"
            elif photo and (current != setting.get("path") or current_strength != setting.get("strength", 1.0)):
                status += "\n此照片的平场与卷默认不同；后续新组仍继承卷默认。"
            elif not status:
                status = "整卷及后续新组沿用此平场。" if setting.get("enabled") else "整卷平场已关闭，后续新组沿用关闭状态。"
        self.flatfield_status.setText(status.strip())

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
                calibration=dict(path=str(path), file_identity=file_identity(path)), freeze_gain=True,
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
        self.sync_flatfield_controls()

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
                self.project.apply_flatfield_default(path, strength=strength if path else 1.0)
                self.flatfield_pending.pop(photo["id"], None)
            else:
                self.project.apply_flatfield_default(current, strength=strength)
            self.remember_flatfield_default(self.project.flatfield_default())
            self.flatfield_messages[photo["id"]] = "整卷兼容照片及后续新组已更新平场；需要时按新来源重新处理。"
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
