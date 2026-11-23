"""PURE RGB workflow desktop: import, frozen recipes, queue, review and export."""

import argparse
import json
import os
from pathlib import Path
import sys
import uuid
import time
from copy import deepcopy

from PyQt6.QtCore import (
    QLockFile,
    QSize,
    Qt,
    QTimer,
    QUrl,
    QThreadPool,
    QRectF,
    QPointF,
)
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .workflow_progress import WorkflowProgressBar
from .workflow_glass import GlassMessageBox as QMessageBox
from .preview_data import ROOT, MODES, load_samples
from .workflow_combo import WorkflowComboBox as QComboBox
from .preview_ui import (
    PreviewWindow,
    STYLE,
    Work,
    button,
    label,
    prepare_plugins,
    display_image,
    startup_splash,
)
from .editing_ui import EditingSidebar
from .live_preview import LivePreview
from .live_worker import EDITING_DOCUMENT_KEYS
from .native import sha256
from .diagnostics import setup, event, ApplicationMemoryMonitor
from .recipe_dialog import RecipeDialog
from .repair_bridge import DEFAULT_BACKEND
from .repair_ui import RepairDialog
from .tasks import Tasks
from .workspace import Workspace, write_json, checked_recipe, output_rectangle

PREFERENCES = ROOT / "outputs/workbench-state/preferences.json"


class Workbench(PreviewWindow):
    def __init__(self, directory=None, srgb=False):
        self.log_path = setup("workbench")
        self.memory_monitor = ApplicationMemoryMonitor()
        self.settings = (
            json.loads(PREFERENCES.read_text()) if PREFERENCES.exists() else {}
        )
        self.workspace = None
        self.workspace_lock = None
        self.queue_active, self.closing, self.loading_demo = False, False, False
        self.import_pending, self.import_groups, self.import_errors = [], [], []
        self.drafts, self.edit_key, self.edit_token = {}, None, 0
        self.pending_saves = {}
        self.edit_started_at = None
        self.preview_full_resolution, self.pending_actual = False, False
        self.geometry_panel = None
        self.geometry_cache, self.region_pending = {}, None
        super().__init__(srgb=srgb, samples=[])
        self.setWindowTitle("PURE RGB · 工作台")
        self.caption.setText("RAW → 三路反演 → 除尘审核 → RGB16")
        self.toolbar_row.insertWidget(3, button("导入 RAW…", self.import_files))
        self.toolbar_row.insertWidget(4, button("工作区…", self.workspace_menu))
        self.toolbar_row.insertWidget(5, button("设置", self.settings_dialog))
        self.toolbar_row.insertWidget(
            6, button("密度取样", lambda: self.open_region("analysis"))
        )
        self.toolbar_row.insertWidget(
            7, button("裁切", lambda: self.open_region("crop"))
        )
        self.toolbar_row.insertWidget(
            8, button("导出 TIFF…", self.export_current, True)
        )
        self.caption.hide()
        self.collection_note.setText("勾选用于批量导出\n缩略图：negpy 原生基准")
        self.photos_layout.insertWidget(
            1, button("全选／全不选", self.toggle_export_checks)
        )
        # Make the existing control panel scroll rather than discarding controls.
        side = self.sidebar_layout.parentWidget()
        side.setMaximumWidth(340)
        host = QWidget()
        host.setLayout(self.sidebar_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(host)
        outside = QVBoxLayout(side)
        outside.setContentsMargins(0, 0, 0, 0)
        self.engine_selector = QComboBox()
        self.engine_selector.setAccessibleName("色彩引擎")
        for key, title in MODES.items():
            self.engine_selector.addItem(title, key)
        self.engine_selector.currentIndexChanged.connect(
            lambda: self.choose_mode(self.engine_selector.currentData())
        )
        outside.addWidget(self.engine_selector)
        self.edit_tabs = QTabWidget()
        self.editor = EditingSidebar()
        self.editor.changed.connect(self.edit_changed)
        self.editor.save.clicked.connect(self.save_edit)
        self.edit_tabs.addTab(self.editor, "调色")
        self.edit_tabs.addTab(scroll, "文件／流程")
        outside.addWidget(self.edit_tabs, 1)
        presets = QHBoxLayout()
        presets.addWidget(button("保存预设", self.save_edit_preset))
        presets.addWidget(button("读取预设", self.load_edit_preset))
        outside.addLayout(presets)
        self.sidebar_layout.insertWidget(0, button("输入与区域设置…", self.edit_recipe))
        self.repair_button = button("PURE Neg 除尘与审核…", self.start_repair, True)
        self.sidebar_layout.addWidget(self.repair_button)
        self.sidebar_layout.addWidget(button("恢复除尘审核…", self.resume_review))
        self.sidebar_layout.addWidget(button("载入原 rgb2 示例", self.load_demo))
        self.sidebar_layout.addWidget(button("原 NegPy 高级界面…", self.open_native))
        self.queue_list = QListWidget()
        self.queue_list.setMinimumHeight(120)
        self.queue_list.setMaximumHeight(230)
        self.queue_list.itemDoubleClicked.connect(self.job_details)
        self.photos_layout.addWidget(label("后台处理队列", "section"))
        self.photos_layout.addWidget(self.queue_list)
        self.progress = WorkflowProgressBar()
        self.progress.setRange(0, 1000)
        self.photos_layout.addWidget(self.progress)
        self.task_status = label("无活动任务", "muted", True)
        self.photos_layout.addWidget(self.task_status)
        actions = QHBoxLayout()
        actions.addWidget(button("开始", self.start_queue))
        self.cancel_button = button("取消", self.cancel_task)
        actions.addWidget(self.cancel_button)
        self.photos_layout.addLayout(actions)
        self.photos_layout.addWidget(button("重试选中任务", self.retry_job))
        self.photos_layout.addWidget(
            button("导出勾选项（当前模式）…", self.batch_export)
        )
        self.tasks = Tasks(ROOT / "outputs/workbench-state/requests", self)
        self.tasks.done.connect(self.task_done)
        self.tasks.failed.connect(self.task_failed)
        self.tasks.progress.connect(self.task_progress)
        self.export_tasks = Tasks(
            ROOT / "outputs/workbench-state/export-requests", self
        )
        self.export_tasks.done.connect(self.exports_finished)
        self.export_tasks.failed.connect(self.exports_failed)
        self.export_tasks.progress.connect(self.task_progress)
        self.region_tasks = Tasks(
            ROOT / "outputs/workbench-state/geometry-requests", self
        )
        self.region_tasks.done.connect(self.region_prepared)
        self.region_tasks.failed.connect(
            lambda request, message, cancelled: self.status.setText(
                "定位图准备已取消。" if cancelled else "定位图准备失败：" + message
            )
        )
        self.region_tasks.progress.connect(
            lambda event: self.status.setText(event.get("message", "准备定位图…"))
        )
        self.live = LivePreview(self)
        self.live.ready.connect(self.preview_ready)
        self.live.failed.connect(self.preview_error)
        self.live.progress.connect(lambda text: self.editor.note.setText(text))
        self.canvas.full_resolution_requested.connect(self.full_edit_preview)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(350)
        self.preview_timer.timeout.connect(self.start_edit_preview)
        self.live_pool = QThreadPool(self)
        self.live_pool.setMaxThreadCount(1)
        self.main_splitter.setSizes([320, 980, 200])
        target = directory or self.settings.get("workspace")
        if target and (Path(target) / "workspace.json").exists():
            self.switch_workspace(target)
        else:
            self.switch_workspace(
                directory or ROOT / "outputs/projects" / uuid.uuid4().hex[:12],
                create=True,
            )
        QTimer.singleShot(0, self.startup_status)

    def startup_status(self):
        self.status.setText("工作区已就绪：导入 RAW 或载入原 rgb2 示例。原始文件只读。")
        print("PURE RGB workbench ready: " + str(self.workspace.root), flush=True)

    def switch_workspace(self, path, create=False):
        if self.tasks.busy or self.exporting:
            raise ValueError("请先等待或取消当前任务")
        path = Path(path).resolve()
        if self.workspace and path == self.workspace.root:
            return
        self.persist_drafts()
        if create:
            workspace = Workspace(path, create=True)
        lock = QLockFile(str(path / ".ui.lock"))
        if not lock.tryLock(0):
            raise ValueError("工作区已在另一个窗口打开")
        try:
            if not create:
                workspace = Workspace(path)
            samples = []
            for job in workspace.data["jobs"]:
                if job["state"] == "complete":
                    samples.append(workspace.sample(job))
        except Exception:
            lock.unlock()
            raise
        if self.workspace_lock:
            self.workspace_lock.unlock()
        self.workspace, self.workspace_lock = workspace, lock
        draft_path = workspace.root / "editing-drafts.json"
        self.drafts = (
            json.loads(draft_path.read_text()).get("drafts", {})
            if draft_path.exists()
            else {}
        )
        self.edit_key = None
        self.queue_active = False
        self.settings["workspace"] = str(path)
        write_json(PREFERENCES, self.settings)
        self.samples = samples
        self.cache.clear()
        self.refresh_samples()
        self.refresh_queue()

    def refresh_samples(self, select=None):
        previous = {
            self.photo_list.item(i).data(
                Qt.ItemDataRole.UserRole
            ): self.photo_list.item(i).checkState()
            for i in range(self.photo_list.count())
        }
        self.token += 1
        self.loaded_key = None
        self.canvas.current = self.canvas.reference = None
        self.canvas.update()
        self.photo_list.blockSignals(True)
        self.photo_list.clear()
        for i, sample in enumerate(self.samples):
            item = QListWidgetItem(
                f"{i + 1:02d}  {Path(sample.inputs[0]['path']).stem}\nRGB · 已计算"
            )
            item.setIcon(self.thumbnail(sample))
            item.setData(Qt.ItemDataRole.UserRole, sample.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(previous.get(sample.id, Qt.CheckState.Checked))
            self.photo_list.addItem(item)
        self.photo_list.blockSignals(False)
        self.photo_heading.setText(f"照片列表   {len(self.samples)} 组")
        old = self.pages.widget(1)
        board = QScrollArea()
        board.setWidgetResizable(True)
        content = QWidget()
        from PyQt6.QtWidgets import QGridLayout

        grid = QGridLayout(content)
        for i, sample in enumerate(self.samples):
            b = button(
                f"{i + 1:02d} · {Path(sample.inputs[0]['path']).stem}",
                lambda checked, row=i: self.open_sample(row),
            )
            b.setIcon(self.thumbnail(sample))
            b.setIconSize(QSize(180, 140))
            b.setMinimumHeight(180)
            grid.addWidget(b, i // 3, i % 3)
        board.setWidget(content)
        was_board = self.pages.currentIndex() == 1
        self.pages.removeWidget(old)
        old.deleteLater()
        self.pages.addWidget(board)
        if was_board:
            self.pages.setCurrentIndex(1)
        if self.samples:
            row = next(
                (i for i, s in enumerate(self.samples) if s.id == select),
                min(self.index, len(self.samples) - 1),
            )
            self.photo_list.setCurrentRow(row)
        else:
            self.request()

    def refresh_queue(self):
        self.queue_list.clear()
        names = {
            "queued": "等待",
            "running": "处理中",
            "complete": "完成",
            "failed": "失败",
            "cancelled": "已取消",
            "interrupted": "已中断",
        }
        for job in self.workspace.data["jobs"]:
            item = QListWidgetItem(
                f"{names.get(job['state'], job['state'])} · {Path(job['recipe']['raw'][0]).stem}"
            )
            item.setData(Qt.ItemDataRole.UserRole, job["id"])
            item.setToolTip(job.get("error", job["id"]))
            self.queue_list.addItem(item)

    def workspace_menu(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("工作区")
        layout = QVBoxLayout(dialog)
        layout.addWidget(label(str(self.workspace.root), "muted", True))
        layout.addWidget(button("新建工作区…", lambda: self.new_workspace(dialog)))
        layout.addWidget(button("打开已有工作区…", lambda: self.open_workspace(dialog)))
        layout.addWidget(
            button(
                "在 Finder 中查看",
                lambda: QDesktopServices.openUrl(
                    QUrl.fromLocalFile(str(self.workspace.root))
                ),
            )
        )
        dialog.exec()

    def new_workspace(self, dialog):
        parent = QFileDialog.getExistingDirectory(self, "选择新工作区的父目录")
        if parent:
            try:
                self.switch_workspace(
                    Path(parent) / ("PURE-RGB-" + uuid.uuid4().hex[:8]), create=True
                )
                dialog.accept()
            except Exception as error:
                QMessageBox.warning(self, "工作区未切换", str(error))

    def open_workspace(self, dialog):
        path = QFileDialog.getExistingDirectory(self, "选择含 workspace.json 的目录")
        if path:
            try:
                self.switch_workspace(path)
                dialog.accept()
            except Exception as error:
                QMessageBox.warning(self, "工作区未打开", str(error))

    def import_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择三张或多组三色 RW2（按文件名每三张分组后核对）",
            str(Path.home() / "Pictures"),
            "Panasonic RAW (*.RW2 *.rw2)",
        )
        if not paths:
            return
        self.begin_import(paths)

    def begin_import(self, paths):
        if self.tasks.busy or len(paths) % 3:
            QMessageBox.warning(
                self, "无法导入", "请等待当前任务，并选择完整的三张或三的倍数张 RW2。"
            )
            return
        paths = sorted(paths)
        self.queue_active = False
        self.import_pending = [paths[i : i + 3] for i in range(0, len(paths), 3)]
        self.import_groups, self.import_errors = [], []
        self.probe_next()

    def probe_next(self):
        if self.import_pending:
            raw = self.import_pending.pop(0)
            self.tasks.start({"action": "probe", "raw": raw})
            self.task_status.setText("核对三色通道：" + Path(raw[0]).name)
            return
        if self.import_errors:
            QMessageBox.warning(
                self,
                "部分分组未通过",
                "\n".join(self.import_errors)
                + "\n失败分组未入队，可重新选择三张文件。",
            )
        if self.import_groups:
            dialog = RecipeDialog(self.import_groups, self)
            dialog.tabs.setTabVisible(1, False)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                try:
                    for recipe in dialog.recipes():
                        self.workspace.add(recipe)
                    self.refresh_queue()
                    self.start_queue()
                except Exception as error:
                    QMessageBox.warning(self, "入队未完成", str(error))
        self.import_groups = []

    def edit_recipe(self):
        if not self.samples or self.tasks.busy:
            self.status.setText("请先选择已完成照片，并等待当前任务。")
            return
        sample = self.samples[self.index]
        original = next(
            (j for j in self.workspace.data["jobs"] if j["id"] == sample.id), None
        )
        initial = (
            self.editing_recipe()
            if getattr(self, "edit_base", None)
            else original["recipe"]
            if original
            else None
        )
        if (
            initial
            and initial["crop"]
            and initial.get("roi") is None
            and "output_roi" not in initial
        ):
            initial = {**initial, "output_roi": list(sample.roi)}
        dialog = RecipeDialog(
            [[r["path"] for r in sample.inputs]],
            self,
            initial,
            geometry=json.loads((sample.directory / "geometry.json").read_text())
            if (sample.directory / "geometry.json").exists()
            else None,
        )
        dialog.tabs.setCurrentIndex(0)
        # Import/geometry options may use a dialog; photo adjustments live in the sidebar.
        dialog.tabs.setTabVisible(1, False)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self.workspace.add(dialog.recipes()[0])
                self.refresh_queue()
                self.start_queue()
            except Exception as error:
                QMessageBox.warning(self, "配方未入队", str(error))

    def start_queue(self):
        self.queue_active = True
        self.next_job()

    def next_job(self):
        if self.closing or self.tasks.busy or not self.queue_active:
            return
        job = next(
            (j for j in self.workspace.data["jobs"] if j["state"] == "queued"), None
        )
        if job is None:
            self.queue_active = False
            counts = {
                key: sum(j["state"] == key for j in self.workspace.data["jobs"])
                for key in ("complete", "failed", "cancelled", "interrupted")
            }
            self.task_status.setText(
                f"队列结束：完成 {counts['complete']}，失败 {counts['failed']}，取消／中断 {counts['cancelled'] + counts['interrupted']}。双击任务查看详情。"
            )
            return
        self.workspace.set_state(job["id"], "running")
        self.refresh_queue()
        self.tasks.start(
            {
                **job,
                "action": "render",
                "directory": str(self.workspace.root / "runs" / job["id"]),
            }
        )

    def task_progress(self, event):
        fraction = event.get("fraction")
        self.progress.setRange(0, 0 if fraction is None else 1000)
        if fraction is not None:
            self.progress.setValue(round(fraction * 1000))
        self.task_status.setText(event.get("message", "处理中…"))

    def task_done(self, request, result):
        if request["action"] == "probe":
            self.import_groups.append(result["raw"])
            self.probe_next()
        elif request["action"] == "render":
            try:
                self.workspace.set_state(request["id"], "complete")
                sample = self.workspace.sample(self.workspace.job(request["id"]))
                self.samples.append(sample)
                saved = self.pending_saves.pop(request["id"], None)
                if saved and self.drafts.get(saved[0]) == saved[1]:
                    self.drafts.pop(saved[0], None)
                    self.persist_drafts()
                self.refresh_samples(sample.id)
            except Exception as error:
                self.task_status.setText("结果校验失败：" + str(error))
                self.queue_active = False
            self.refresh_queue()
            QTimer.singleShot(0, self.next_job)
        elif request["action"] == "repair_prepare":
            path = Path(request["directory"]) / "receipt.json"
            self.workspace.data.setdefault("repairs", []).append(str(path))
            self.workspace.save()
            self.show_review(path)
        self.progress.setValue(1000)

    def task_failed(self, request, message, cancelled):
        self.task_status.setText(message)
        if request["action"] == "render":
            self.workspace.set_state(
                request["id"], "cancelled" if cancelled else "failed", error=message
            )
            self.refresh_queue()
            if not cancelled:
                QTimer.singleShot(0, self.next_job)
        elif request["action"] == "probe":
            if cancelled:
                self.import_pending, self.import_groups = [], []
            else:
                self.import_errors.append(Path(request["raw"][0]).name + ": " + message)
                self.probe_next()
        elif not self.closing and not getattr(self, "automation", False):
            QMessageBox.warning(self, "任务未完成", message)

    def cancel_task(self):
        if self.export_tasks.busy:
            self.export_tasks.cancel()
            return
        if self.region_tasks.busy:
            self.region_tasks.cancel()
            return
        self.queue_active = False
        self.tasks.cancel()

    def retry_job(self):
        item = self.queue_list.currentItem()
        if item:
            job = self.workspace.job(item.data(Qt.ItemDataRole.UserRole))
            try:
                geometry = self.workspace.root / "runs" / job["id"] / "geometry.json"
                dialog = RecipeDialog(
                    [job["recipe"]["raw"]],
                    self,
                    job["recipe"],
                    geometry=json.loads(geometry.read_text())
                    if geometry.exists()
                    else None,
                )
                dialog.tabs.setCurrentIndex(0)
                dialog.tabs.setTabVisible(1, False)
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                self.workspace.add(dialog.recipes()[0])
                self.refresh_queue()
                self.start_queue()
            except Exception as error:
                QMessageBox.warning(self, "不能重试", str(error))

    def job_details(self, item):
        job = self.workspace.job(item.data(Qt.ItemDataRole.UserRole))
        QMessageBox.information(
            self, "任务配方与状态", json.dumps(job, ensure_ascii=False, indent=2)
        )

    def load_demo(self):
        try:
            present = {s.id for s in self.samples}
            self.samples.extend(s for s in load_samples() if s.id not in present)
            self.refresh_samples("primary-07")
        except Exception as error:
            QMessageBox.warning(self, "示例不可用", str(error))

    def start_repair(self):
        if self.loaded_key is None or self.tasks.busy:
            self.status.setText("请先载入当前图像并等待当前处理任务。")
            return
        self.queue_active = False
        source = self.samples[self.index].record(self.mode, self.probe)
        self.tasks.start(
            {
                "action": "repair_prepare",
                "source": source,
                "revision": source["sha256"],
                "directory": str(self.workspace.root / "repairs" / uuid.uuid4().hex),
                "backend": self.settings.get("backend", str(DEFAULT_BACKEND)),
                "detect": self.settings.get("detect", "dust,scratches,fibers"),
                "polarity": self.settings.get("polarity", "bright"),
                "sensitivity": self.settings.get("sensitivity", 1),
            }
        )
        self.task_status.setText("PURE Neg 开始分析；完成后进入审核，不自动发布。")

    def show_review(self, path):
        dialog = RepairDialog(path, self.workspace.root / "requests", self)
        dialog.setWindowState(dialog.windowState() | Qt.WindowState.WindowMaximized)
        dialog.exec()
        receipt = json.loads(Path(path).read_text())
        self.task_status.setText(
            "已导出：" + receipt["preview"]["path"]
            if receipt["state"] == "confirmed"
            else "审核已保存，可从“恢复除尘审核”继续。"
        )

    def resume_review(self):
        if self.tasks.busy:
            return
        paths = self.workspace.data.get("repairs", [])
        if not paths:
            self.status.setText("本工作区还没有除尘审核记录。")
            return
        from PyQt6.QtWidgets import QInputDialog

        selected, ok = QInputDialog.getItem(
            self, "恢复除尘审核", "选择审核记录", paths, len(paths) - 1, False
        )
        if ok:
            receipt = json.loads(Path(selected).read_text())
            if receipt["state"] not in ("ready", "previewed", "confirmed"):
                QMessageBox.warning(
                    self,
                    "审核不可恢复",
                    "上次分析或确认被中断，请从冻结输入重新除尘；不会改写 PURE Neg 签名记录。",
                )
            else:
                self.show_review(selected)

    def toggle_export_checks(self):
        all_checked = all(
            self.photo_list.item(i).checkState() == Qt.CheckState.Checked
            for i in range(self.photo_list.count())
        )
        for i in range(self.photo_list.count()):
            self.photo_list.item(i).setCheckState(
                Qt.CheckState.Unchecked if all_checked else Qt.CheckState.Checked
            )

    def export_records(self):
        return [
            (sample.id, sample.record(self.mode, "baseline"))
            for i, sample in enumerate(self.samples)
            if self.photo_list.item(i).checkState() == Qt.CheckState.Checked
        ]

    def batch_export(self):
        if self.exporting or not self.samples:
            return
        selected = [
            s
            for i, s in enumerate(self.samples)
            if self.photo_list.item(i).checkState() == Qt.CheckState.Checked
        ]
        if not selected:
            self.status.setText("请先勾选需要导出的照片。")
            return
        directory = QFileDialog.getExistingDirectory(
            self, "选择批量导出目录", str(self.workspace.root)
        )
        if not directory:
            return
        self.begin_exports(
            [
                self.export_item(s, Path(directory) / f"{s.id}-{self.mode}.tiff")
                for s in selected
            ]
        )

    def export_item(self, sample, destination):
        item = {
            "id": sample.id,
            "mode": self.mode,
            "destination": str(Path(destination).resolve()),
        }
        manifest = json.loads((sample.directory / "manifest.json").read_text())
        recipe = (
            self.editing_recipe()
            if sample.id == self.edit_key and getattr(self, "edit_base", None)
            else self.drafts.get(sample.id)
        )
        if recipe and checked_recipe(recipe) != checked_recipe(manifest["recipe"]):
            item["render"] = {
                "recipe": deepcopy(recipe),
                "inputs": manifest["inputs"],
                "calibration": manifest.get("calibration"),
            }
        else:
            item["record"] = sample.record(
                self.mode,
                self.probe if sample.id == self.samples[self.index].id else "baseline",
            )
        return item

    def begin_exports(self, items):
        if self.exporting:
            return
        self.exporting = True
        self.progress.setValue(0)
        self.export_button.setEnabled(False)
        self.export_tasks.start(
            {
                "action": "export_batch",
                "items": items,
                "directory": str(self.workspace.root / "exports" / uuid.uuid4().hex),
            }
        )
        self.status.setText("正在按导出时的参数处理；未修改原照片，可继续调节。")

    def exports_finished(self, request, event):
        self.exporting = False
        self.progress.setValue(1000)
        self.export_button.setEnabled(bool(self.samples))
        report = event["report"]
        paths = [row["path"] for row in report["saved"]]
        self.status.setText(
            f"已导出 {len(paths)} 张 RGB16 TIFF；失败 {len(report['errors'])} 张。"
            + (" " + paths[0] if len(paths) == 1 else "")
        )
        self.task_status.setText(self.status.text())
        if report["errors"] and not getattr(self, "automation", False):
            QMessageBox.warning(
                self,
                "导出未全部完成",
                "\n".join(row["message"] for row in report["errors"]),
            )

    def exports_failed(self, request, message, cancelled):
        self.exporting = False
        self.export_button.setEnabled(bool(self.samples))
        self.status.setText(
            "导出已取消；已写出的文件保留。" if cancelled else "导出失败：" + message
        )
        self.task_status.setText(self.status.text())

    def batch_exported(self, token, result):
        self.exporting = False
        self.export_button.setEnabled(self.loaded_key is not None)
        self.status.setText(
            f"批量导出完成 {len(result['saved'])} 张；失败 {len(result['errors'])} 张。"
        )
        if result["errors"]:
            QMessageBox.warning(self, "部分文件未导出", "\n".join(result["errors"]))

    def settings_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("PURE RGB 设置")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        backend = QLineEdit(self.settings.get("backend", str(DEFAULT_BACKEND)))
        form.addRow("PURE Neg Resources", backend)
        polarity = QComboBox()
        for title, key in [("亮灰尘", "bright"), ("暗灰尘", "dark"), ("自动", "auto")]:
            polarity.addItem(title, key)
        polarity.setCurrentIndex(
            polarity.findData(self.settings.get("polarity", "bright"))
        )
        form.addRow("灰尘极性", polarity)
        choices = {}
        current = self.settings.get("detect", "dust,scratches,fibers").split(",")
        for key, title in [
            ("dust", "灰尘"),
            ("scratches", "划痕"),
            ("fibers", "毛发／纤维"),
            ("patches", "柔斑（额外研究）"),
        ]:
            choice = QCheckBox(title)
            choice.setChecked(key in current)
            choices[key] = choice
            form.addRow(choice)
        layout.addLayout(form)
        layout.addWidget(
            button(
                "打开调试日志目录",
                lambda: QDesktopServices.openUrl(
                    QUrl.fromLocalFile(str(ROOT / "outputs/logs"))
                ),
            )
        )
        layout.addWidget(
            label(
                "原始 RAW 和原应用设置保持独立；除尘始终先审核再发布。\n当前输出为保留 ICC 的 16-bit TIFF。",
                "muted",
                True,
            )
        )
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            detect = ",".join(k for k, v in choices.items() if v.isChecked())
            if not detect:
                QMessageBox.warning(self, "设置未保存", "至少选择一类检测项目。")
                return
            self.settings.update(
                backend=backend.text().strip(),
                detect=detect,
                polarity=polarity.currentData(),
            )
            write_json(PREFERENCES, self.settings)

    def open_native(self):
        import subprocess

        env = os.environ.copy()
        env["NEGPY_USER_DIR"] = str(self.workspace.root / "advanced-native-state")
        # Same original application, independent database; not a recipe round-trip.
        launcher = ROOT / "run_native_ui.py"
        subprocess.Popen(
            [sys.executable, str(launcher)], cwd=ROOT, env=env, start_new_session=True
        )
        self.status.setText(
            "已打开原 NegPy 高级界面；使用独立图库，配方暂不自动回写工作台。"
        )

    def loaded(self, token, result):
        super().loaded(token, result)
        if token == self.token and self.loaded_key:
            if (
                self.mode == "cool-shared-rgb"
                and self.samples[self.index]
                .record(self.mode, self.probe)
                .get("cool_tone")
                == "soft-shoulder"
            ):
                self.mode_note.setText(
                    "公开密度算法 + 共同 RGB 曲线\nSoft shoulder 柔和肩部提亮。\n非 Nikon 原厂精确复现。"
                )
            self.status.setText(
                self.status.text().replace(
                    "离线结果，未执行 RAW 重算",
                    "冻结渲染结果；编辑配方可创建新版本重算",
                )
            )
            if self.edit_dirty():
                self.export_button.setEnabled(not self.exporting)
                self.preview_timer.start()

    def request(self, reset=False):
        if hasattr(self, "live"):
            self.close_region()
            if not self.canvas.fit_mode:
                self.canvas.zoom /= self.canvas.sample_scale
            self.canvas.sample_scale = 1
            if reset:
                self.preview_full_resolution = False
            self.preview_timer.stop()
            self.edit_token += 1
            self.live.invalidate(self.edit_token)
            self.canvas.reference_name = MODES["native"]
            self.compare.setText("分割对比 " + MODES["native"])
            self.engine_selector.blockSignals(True)
            self.engine_selector.setCurrentIndex(
                self.engine_selector.findData(self.mode)
            )
            self.engine_selector.blockSignals(False)
            self.editor.set_mode(self.mode)
            if self.samples:
                sample = self.samples[self.index]
                key = sample.id
                manifest = json.loads((sample.directory / "manifest.json").read_text())
                if f"{self.probe}__{self.mode}" not in sample.results:
                    self.probe = "baseline"
                self.edit_base = manifest.get("recipe")
                editable = self.edit_base is not None and self.probe == "baseline"
                self.editor.setEnabled(editable)
                if editable and key != self.edit_key:
                    self.edit_key = key
                    self.editor.set_documents(self.drafts.get(key, self.edit_base))
                self.editor.note.setText(
                    "滑动即可预览；保存时生成完整分辨率新版本。"
                    if editable
                    else "研究范围示例为只读；请导入 RAW 调色。"
                )
                self.repair_button.setEnabled(not self.edit_dirty())
            else:
                self.editor.setEnabled(False)
        super().request(reset)
        if hasattr(self, "export_tasks"):
            self.export_button.setEnabled(bool(self.samples) and not self.exporting)

    def open_region(self, kind, geometry=None):
        if not self.samples or self.tasks.busy or not getattr(self, "edit_base", None):
            self.status.setText("请先选择已完成的照片，并等待当前处理结束。")
            return
        self.close_region()
        sample = self.samples[self.index]
        path = sample.directory / "geometry.json"
        geometry = geometry or self.geometry_cache.get(sample.id)
        if geometry is None and path.exists():
            geometry = json.loads(path.read_text())
        if geometry is None:
            if self.region_tasks.busy:
                self.status.setText("正在准备定位图，请稍候…")
                return
            manifest = sample.directory / "manifest.json"
            self.region_pending = {"id": sample.id, "kind": kind}
            self.region_tasks.start(
                {
                    "action": "geometry_prepare",
                    "manifest": str(manifest),
                    "manifest_sha256": sha256(manifest),
                    "directory": str(
                        self.workspace.root / "geometry-cache" / uuid.uuid4().hex
                    ),
                }
            )
            self.status.setText("首次从 RAW 准备完整定位图，当前照片保持可见…")
            return
        from .geometry_ui import GeometryPanel

        panel = GeometryPanel(geometry)
        self.geometry_panel = panel
        self.geometry_sample_id, self.geometry_kind = sample.id, kind
        self.canvas_stack.addWidget(panel)
        self.canvas_stack.setCurrentWidget(panel)
        hint = panel.layout().itemAt(0).widget()
        hint.setText(
            "密度取样：只改变密度／测光统计，不裁切照片。"
            if kind == "analysis"
            else "输出裁切：只改变输出画幅，不改变密度取样区域。"
        )
        panel.ok.setText(
            "应用密度取样并重算" if kind == "analysis" else "应用裁切并重算"
        )
        if kind == "crop":
            reset = button("取消裁切，恢复全画幅", self.reset_crop)
            panel.layout().insertWidget(2, reset)
        manifest = json.loads((sample.directory / "manifest.json").read_text())
        rectangle = (
            sample.roi
            if kind == "analysis"
            else output_rectangle(
                self.editing_recipe(), list(sample.roi), manifest["source_shape"]
            )
        )
        sx, sy = (
            panel.pixmap.width() / geometry["width"],
            panel.pixmap.height() / geometry["height"],
        )
        a, b = (
            QPointF(rectangle[0] * sx, rectangle[1] * sy),
            QPointF(rectangle[2] * sx, rectangle[3] * sy),
        )
        panel.changed(QRectF(a, b).toRect(), a, b)
        panel.selection = tuple(rectangle)
        panel.applied.connect(self.apply_region)
        panel.cancelled.connect(self.close_region)

    def close_region(self):
        self.region_pending = None
        if self.geometry_panel:
            panel, self.geometry_panel = self.geometry_panel, None
            self.canvas_stack.setCurrentWidget(self.canvas)
            self.canvas_stack.removeWidget(panel)
            panel.deleteLater()

    def region_prepared(self, request, event):
        pending = self.region_pending
        self.region_pending = None
        if (
            pending
            and self.samples
            and pending["id"] == self.samples[self.index].id
            and not self.closing
        ):
            self.geometry_cache[pending["id"]] = event["geometry"]
            self.open_region(pending["kind"], event["geometry"])

    def region_recipe(self, rectangle, kind):
        recipe = self.editing_recipe()
        sample = self.samples[self.index]
        manifest = json.loads((sample.directory / "manifest.json").read_text())
        # Freeze the existing output rectangle BEFORE changing the analysis ROI.
        output = output_rectangle(recipe, list(sample.roi), manifest["source_shape"])
        if kind == "analysis":
            recipe.update(
                roi=list(rectangle), output_roi=output if recipe["crop"] else None
            )
        else:
            recipe["output_roi"] = list(rectangle) if rectangle is not None else None
        return checked_recipe(recipe)

    def apply_region(self):
        if (
            not self.geometry_panel
            or not self.geometry_panel.selection
            or self.samples[self.index].id != self.geometry_sample_id
        ):
            return
        recipe = self.region_recipe(self.geometry_panel.selection, self.geometry_kind)
        self.commit_region(recipe)

    def reset_crop(self):
        self.commit_region(self.region_recipe(None, "crop"))

    def commit_region(self, recipe):
        if self.tasks.busy:
            self.status.setText("请等待当前任务结束后应用区域。")
            return
        self.workspace.add(recipe)
        self.close_region()
        self.refresh_queue()
        self.start_queue()
        self.status.setText("正在重算区域设置；旧照片和调色草稿保留。")

    def editing_recipe(self):
        recipe = deepcopy(self.edit_base)
        hidden = self.drafts.get(self.edit_key, self.edit_base)
        # These existing workflow scalars have no ProfessionalEditor widgets.
        # Replay the current photo's draft; sparse old drafts retain its base.
        for key in ("exposure_ev", "cool_tone"):
            if key in hidden:
                recipe[key] = deepcopy(hidden[key])
        for key in EDITING_DOCUMENT_KEYS:
            recipe.pop(key, None)
        # No float widgets exist yet. Preserve the current draft's hidden
        # point recipe, including its explicit absence after loading a preset.
        for key in (
            "native_float_controls",
            "highlight_controls",
            "cool_headroom_controls",
            "flex_research_controls",
            "nikon_research_controls",
        ):
            if key in hidden:
                recipe[key] = deepcopy(hidden[key])
        return checked_recipe({**recipe, **self.editor.documents()})

    def edit_dirty(self):
        return bool(
            getattr(self, "edit_base", None)
            and self.edit_key
            and self.editing_recipe() != checked_recipe(self.edit_base)
        )

    def persist_drafts(self):
        if self.workspace:
            write_json(
                self.workspace.root / "editing-drafts.json",
                {"version": 1, "drafts": self.drafts},
            )

    def edit_changed(self):
        if not getattr(self, "edit_base", None) or not self.samples:
            return
        self.edit_token += 1
        self.edit_started_at = time.monotonic()
        self.live.invalidate(self.edit_token)
        recipe = self.editing_recipe()
        event(
            "edit_recipe",
            sample=self.edit_key,
            mode=self.mode,
            token=self.edit_token,
            recipe=recipe,
        )
        if self.edit_dirty():
            self.drafts[self.edit_key] = recipe
        else:
            self.drafts.pop(self.edit_key, None)
        self.persist_drafts()
        self.export_button.setEnabled(not self.exporting)
        self.repair_button.setEnabled(False)
        self.editor.note.setText("正在更新画面…")
        self.preview_timer.start()

    def start_edit_preview(self):
        if not self.samples or not self.editor.isEnabled() or self.closing:
            return
        # A render can be requested again without changing a slider. It still
        # needs a fresh artifact identity so a reader never races an overwrite.
        self.edit_token += 1
        self.live.invalidate(self.edit_token)
        sample = self.samples[self.index]
        manifest = sample.directory / "manifest.json"
        recipe = self.editing_recipe()
        controls = {
            k: recipe[k]
            for k in (*EDITING_DOCUMENT_KEYS, "exposure_ev", "cool_tone")
            if k in recipe
            and (
                self.mode == "native"
                or k not in ("native_float_controls", "highlight_controls")
            )
            and (self.mode == "cool-shared-rgb" or k != "cool_headroom_controls")
            and (self.mode == "cool-shared-rgb" or k != "nikon_research_controls")
            and (self.mode == "flex-endpoint" or k != "flex_research_controls")
        }
        self.live.submit(
            {
                "token": self.edit_token,
                "manifest": str(manifest),
                "manifest_sha256": sha256(manifest),
                "mode": self.mode,
                "controls": controls,
                "full_resolution": self.preview_full_resolution,
            }
        )

    def preview_ready(self, event):
        token, icc = event["token"], self.display_icc

        def read():
            try:
                return (
                    display_image(event["records"][0], icc),
                    display_image(event["records"][1], icc),
                    event,
                )
            finally:
                self.live.discard(event)

        work = Work(token, read)
        work.signals.done.connect(self.edit_preview_loaded)
        work.signals.failed.connect(self.preview_error)
        self.live_pool.start(work)

    def edit_preview_loaded(self, token, result):
        if token != self.edit_token or self.closing:
            return
        current, reference, event = result
        if self.canvas.current and not self.canvas.fit_mode:
            self.canvas.zoom *= self.canvas.current.width() / current.width()
        self.canvas.sample_scale = event.get("sample_step", 1)
        self.canvas.reference_name = "调整前"
        self.canvas.set_images(current, reference, MODES[self.mode] + " · 调色预览")
        if self.pending_actual and self.canvas.sample_scale == 1:
            self.pending_actual = False
            self.canvas.actual()
        self.compare.setText("分割对比调整前")
        self.last_preview_latency = (
            time.monotonic() - self.edit_started_at
            if self.edit_started_at
            else event["seconds"]
        )
        from .diagnostics import event as log_event

        log_event(
            "preview_presented",
            token=token,
            mode=self.mode,
            sample=self.edit_key,
            interaction_seconds=self.last_preview_latency,
            worker_seconds=event["seconds"],
            sample_step=self.canvas.sample_scale,
        )
        self.editor.note.setText(
            f"预览已更新 · {self.last_preview_latency:.2f}s · 保存后生成全尺寸版本"
        )
        self.status.setText("调色预览 · 原始取样与拟合参考保持冻结 · 原照片保留")
        self.export_button.setEnabled(not self.exporting)
        self.repair_button.setEnabled(not self.edit_dirty())

    def full_edit_preview(self):
        self.preview_full_resolution, self.pending_actual = True, True
        self.edit_started_at = time.monotonic()
        self.editor.note.setText("正在准备全尺寸 100% 预览，保留当前画面…")
        self.start_edit_preview()

    def preview_error(self, token, message):
        if token == self.edit_token:
            event("preview_error", sample=self.edit_key, token=token, error=message)
            self.editor.note.setText("预览未完成：" + message)
            self.status.setText("保留上一画面；当前调节尚未预览成功。")

    def save_edit(self):
        if not self.edit_dirty() or self.tasks.busy:
            self.editor.note.setText("没有待保存调整，或完整处理仍在运行。")
            return
        try:
            recipe = self.editing_recipe()
            job = self.workspace.add(recipe)
            self.pending_saves[job["id"]] = (self.edit_key, recipe)
            self.refresh_queue()
            self.start_queue()
        except Exception as error:
            self.editor.note.setText(str(error))

    def save_edit_preset(self):
        if not getattr(self, "edit_base", None):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存调色预设", "PURE-RGB-preset.json", "JSON (*.json)"
        )
        if path:
            write_json(
                path,
                {
                    "schema": "pure-rgb-processing-preset",
                    "version": 1,
                    "parameters": {
                        k: v for k, v in self.editing_recipe().items() if k != "raw"
                    },
                },
            )

    def load_edit_preset(self):
        if not getattr(self, "edit_base", None):
            return
        path, _ = QFileDialog.getOpenFileName(self, "读取调色预设", "", "JSON (*.json)")
        if path:
            try:
                preset = json.loads(Path(path).read_text())
                if (
                    preset.get("schema") != "pure-rgb-processing-preset"
                    or type(preset.get("version")) is not int
                    or preset["version"] != 1
                ):
                    raise ValueError("不支持的预设")
                # Missing scalars in an old preset mean this photo's original
                # values, rather than checked_recipe's global defaults or a
                # previously loaded preset's draft values.
                base = checked_recipe(self.edit_base)
                checked = checked_recipe(
                    {
                        **{key: base[key] for key in ("exposure_ev", "cool_tone")},
                        **preset["parameters"],
                        "raw": base["raw"],
                    }
                )
                previous = self.editing_recipe()
                try:
                    self.editor.set_documents(checked)
                except Exception as error:
                    # Resource/widget setters may fail after changing earlier
                    # groups. Restore their old documents before committing a
                    # new draft, so a rejected preset cannot partly recolor it.
                    try:
                        self.editor.set_documents(previous)
                    except Exception as restore_error:
                        raise ValueError(
                            "预设未应用；恢复原调色参数失败：" + str(restore_error)
                        ) from error
                    raise
                self.drafts[self.edit_key] = checked
                self.edit_changed()
            except Exception as error:
                self.editor.note.setText(str(error))

    def export_current(self):
        if not self.samples or self.exporting:
            return
        sample = self.samples[self.index]
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出当前效果 · 16-bit TIFF（保留 ICC）",
            str(self.workspace.root / f"{sample.id}-{self.mode}.tiff"),
            "TIFF (*.tiff *.tif)",
        )
        if path:
            self.begin_exports([self.export_item(sample, path)])

    def closeEvent(self, event):
        self.closing, self.queue_active = True, False
        self.preview_timer.stop()
        self.live.shutdown()
        self.persist_drafts()
        self.live_pool.waitForDone(3000)
        self.tasks.shutdown()
        self.export_tasks.shutdown()
        self.region_tasks.shutdown()
        self.memory_monitor.stop()
        super().closeEvent(event)
        if self.workspace_lock:
            self.workspace_lock.unlock()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--srgb", action="store_true")
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--ready-report", type=Path)
    args = parser.parse_args()
    os.environ["NEGPY_USER_DIR"] = str(ROOT / "outputs/workbench-state/native")
    prepare_plugins()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("PURE RGB")
    app.setFont(QFont("PingFang SC", 12))
    app.setStyleSheet(STYLE)
    from .workflow_glass import install_glass_surfaces

    install_glass_surfaces(app)
    splash = startup_splash()
    try:
        window = Workbench(args.workspace, args.srgb)
    except Exception as error:
        if splash:
            splash.close()
        if args.smoke_output:
            raise
        QMessageBox.critical(None, "无法打开工作台", str(error))
        return 1
    window.showMaximized()
    if splash:
        splash.finish(window)
    if args.ready_report:
        started = time.monotonic()
        ready_timer = QTimer(window)
        ready_timer.setInterval(200)

        def record_ready():
            if window.loaded_key is None and time.monotonic() - started < 45:
                return
            ready_timer.stop()
            from collections import Counter
            from .positive_post import POST_SPECS
            from .professional_controls import native_papers

            ready = window.loaded_key is not None
            write_json(
                args.ready_report,
                {
                    "state": "complete" if ready else "failed",
                    "pid": os.getpid(),
                    "workspace": str(window.workspace.root),
                    "photo_count": len(window.samples),
                    "jobs": dict(
                        Counter(job["state"] for job in window.workspace.data["jobs"])
                    ),
                    "mode_labels": MODES,
                    "common_post_count_per_mode": len(POST_SPECS),
                    "native_paper_count": len(native_papers()),
                    "log_path": str(window.log_path),
                    "loaded_key": str(window.loaded_key),
                    "error": window.last_error,
                },
            )
            if ready:
                window.grab().save(str(args.ready_report.with_suffix(".png")))

        ready_timer.timeout.connect(record_ready)
        ready_timer.start()
    if args.smoke_output:
        window.automation = True
        from .workbench_smoke import exercise

        exercise(app, window, args.smoke_output)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
