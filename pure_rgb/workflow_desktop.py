"""Staged desktop test entry, connected to the existing isolated RAW workflow."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import uuid

from . import WORKSPACE_ROOT as ROOT
if __name__ == "__main__":
    os.environ.setdefault(
        "NEGPY_USER_DIR", str(ROOT / "outputs/workflow-ui-state/native")
    )
    from .workflow_application import prepare_application_identity

    prepare_application_identity()

from PyQt6.QtCore import QEvent, QLockFile, QPointF, QSize, Qt, QThreadPool, QTimer, pyqtSignal  # noqa: E402
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QKeySequence, QPainter, QPen  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListView,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .branding import application_icon  # noqa: E402
from .diagnostics import event as diagnostic_event, set_debug_logging, setup  # noqa: E402
from .live_preview import LivePreview  # noqa: E402
from .live_worker import EDITING_DOCUMENT_KEYS  # noqa: E402
from .native import sha256  # noqa: E402
from .positive_post import HUE_SPEC, TEMPERATURE_SPEC  # noqa: E402
from .preview_ui import Canvas, Work, display_image, prepare_plugins  # noqa: E402
from .tasks import Tasks  # noqa: E402
from .workflow_glass import (  # noqa: E402
    GlassMenu,
    GlassMessageBox,
    WORKSPACE_GAP, WORKSPACE_INSET, attach_glass, install_glass_surfaces, prepare_glass, stylesheet,
)
from .workflow_indicators import CheckIndicatorDelegate  # noqa: E402
from .workflow_progress import WorkflowProgressBar  # noqa: E402
from .workflow_selection import SelectionView  # noqa: E402
from .workflow_ui_model import (  # noqa: E402
    ENGINES,
    RESULT_MODES,
    STAGES,
    LIGHT_TABLE_STAGE,
    VISIBLE_STAGE_INDICES,
    VISIBLE_STAGE_LABELS,
    canonical_stage,
    stage_presentation,
    experimental_repair_block_reason,
    WorkflowProject,
    metering_document,
    nikon_controls,
    shape_bounds,
    spatial_document,
    fit_manual_highlight_reference,
)
from .workspace import source_recipe_identity, write_json  # noqa: E402
from .workflow_advanced import advanced_groups  # noqa: E402
from .workflow_advanced_ui import AdvancedPanel  # noqa: E402
from .workflow_disclosure import DisclosureButton  # noqa: E402
from .workflow_scroll import WorkflowScrollArea  # noqa: E402
from .motion_sliders import ScrubberControl  # noqa: E402
from .workflow_branding import create_splash, header_logo  # noqa: E402
from .workflow_materials import attach_materials  # noqa: E402
from .workflow_canvas import RoundedPreviewStack  # noqa: E402
from .workflow_attention import AttentionButton  # noqa: E402
from .workflow_history_ui import EditHistoryUI, PreviewToolbar  # noqa: E402
from .workflow_flatfield import FlatfieldUI  # noqa: E402
from .workflow_combo import WorkflowComboBox as QComboBox  # noqa: E402
from .motion_menus import SpringComboBox  # noqa: E402
from .workflow_display import DisplayTracker  # noqa: E402
from .workflow_performance import (  # noqa: E402
    load_performance_settings, save_performance_settings, resolve_performance,
    process_rss_bytes, should_release_preview,
)


def text(value, muted=False):
    widget = QLabel(value)
    widget.setWordWrap(True)
    if muted:
        widget.setObjectName("muted")
    return widget


def button(value, action, primary=False):
    widget = QPushButton(value)
    widget.clicked.connect(action)
    if primary:
        widget.setObjectName("primary")
    return widget


def saved_display_source_step(record):
    """Exact fit grid for a full saved image; absent shapes keep old behavior."""
    shape = record.get("shape")
    if (isinstance(shape, (list, tuple)) and len(shape) == 3
            and all(type(value) is int and value > 0 for value in shape)
            and shape[2] == 3):
        return max(1, (max(shape[:2]) + 2047) // 2048)
    return None


class StraightenLineOverlay(QWidget):
    """画线拉直的直线浮层：真正的线段，而非 macOS 会画成矩形的选框。

    对鼠标事件透明；拖动由画布的 eventFilter 接管并驱动 set_line。
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.start = QPointF()
        self.end = QPointF()
        self._visible_line = False
        self.setGeometry(parent.rect())

    def set_line(self, start, end):
        self.start, self.end = start, end
        self._visible_line = True
        self.update()

    def clear_line(self):
        self._visible_line = False
        self.update()

    def paintEvent(self, event):
        if not self._visible_line:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#F0A055"), 2)
        painter.setPen(pen)
        painter.drawLine(self.start, self.end)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#F2EDE4"))
        for point in (self.start, self.end):
            painter.drawEllipse(point, 3.5, 3.5)


class WorkflowWindow(FlatfieldUI, EditHistoryUI, QMainWindow):
    job_saved = pyqtSignal(str)
    image_presented = pyqtSignal()

    def event(self, event):
        from .workflow_materials import material_frame_event

        return material_frame_event(self, event, super().event)

    def __init__(self, directory, *, srgb=False):
        super().__init__()
        # A Qt event filter can outlive Python attributes briefly during GC.
        # Enable it only after build(), and disable before detaching filters.
        self._event_filters_active = False
        setup("workflow-ui")
        self.project = WorkflowProject(
            directory, create=not (Path(directory) / "workspace.json").exists()
        )
        self.lock = QLockFile(str(self.project.workspace.root / ".ui.lock"))
        if not self.lock.tryLock(0):
            raise ValueError("此项目已在另一个窗口打开")
        self.closing = self.loading = False
        self.last_error = None
        self.token = 0
        self.display_serial = 0
        self.display_revision = 0
        self.base_display_pending = None
        self.shown_context = None
        self.preview_contexts = {}
        self.preview_reusable = False
        self.detail_contexts = {}
        self.pending_probe = []
        self.geometry_role = None
        self.adjustment_clipboard = None
        self.meter_grid_hidden = False
        self.current_record = None
        self.pending_actual = False
        self.display_icc = None
        self.force_srgb = srgb
        self.display_target = None
        self.display_description = "sRGB 测试显示" if srgb else "正在识别当前屏幕…"
        self.proof_space = "off"
        self.display_preferences = ROOT / "outputs/workflow-ui-state/display.json"
        if self.display_preferences.exists():
            proof = json.loads(self.display_preferences.read_text()).get("proof", "off")
            if proof in {"off", "srgb", "display-p3", "adobe-rgb"}:
                self.proof_space = proof
        self.performance_path = ROOT / "outputs/workflow-ui-state/performance.json"
        self.performance = load_performance_settings(self.performance_path)
        set_debug_logging(self.performance["debug_logging"])
        self._logged_preview_wait = None
        self.session_events = []
        self.settings_open = False
        self.last_activity = time.monotonic()
        from .workflow_preview_activity import PreviewActivity

        self.preview_activity = PreviewActivity(self)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.tasks = Tasks(self.project.workspace.root / "requests", self)
        self.tasks.performance = self.performance
        self.tasks.started.connect(self.task_started)
        self.tasks.done.connect(self.task_done)
        self.tasks.failed.connect(self.task_failed)
        self.tasks.progress.connect(self.task_progress)
        self._make_live()
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(450)
        self.preview_timer.timeout.connect(self.request_preview)
        self.import_preview_timer = QTimer(self)
        self.import_preview_timer.setSingleShot(True)
        self.import_preview_timer.setInterval(70)
        self.import_preview_timer.timeout.connect(self.request_import_preview)
        self._measurement_hint_photo = None
        self._measurement_hint_text = ""
        self._measurement_hint_timer = QTimer(self)
        self._measurement_hint_timer.setSingleShot(True)
        self._measurement_hint_timer.timeout.connect(self.clear_measurement_hint)
        self.import_attention_photo = None
        self.import_lens_context = None
        self.setWindowTitle("PURE RGB")
        self.setWindowIcon(application_icon())
        self.resize(1400, 920)
        # 侧栏控件（滑杆读数 + 复位按钮）在窄于此宽度时会被截断。
        self.setMinimumSize(1000, 700)
        self.setMinimumSize(1000, 650)
        prepare_glass(self)
        self.build()
        self.preview_activity.changed.connect(self.preview_activity_changed)
        self.setup_background_previews()
        from .workflow_preview_snapshot import PreviewSnapshotStore

        self.snapshot_store = PreviewSnapshotStore(self.live.directory / "display-snapshots")
        self.glass_report = dict(mode="pending", attached=False)
        self._apply_glass_palette()
        QApplication.styleHints().colorSchemeChanged.connect(self._apply_glass_palette)
        if not self.force_srgb:
            self.display_tracker = DisplayTracker(self)
            self.display_tracker.changed.connect(self.display_target_changed)
            self.display_tracker.refresh()
        self.refresh_photos()
        self.set_stage(self.project.data.get("stage", 0))
        if self.project.data["current"]:
            self.select_photo(self.project.data["current"])
        QTimer.singleShot(0, self.install_glass)

    def _make_live(self):
        self.live = LivePreview(self)
        self.live.performance = self.performance
        self.live.prewarm_gate = getattr(self, "prewarm_gate", None)
        self.live.ready.connect(self.live_ready)
        self.live.failed.connect(self.preview_failed)
        self.live.progress.connect(self.preview_progress)
        self.live.progress_detail.connect(self.preview_activity.progress)

    def preview_context(self):
        photo = self.current_photo()
        if not photo:
            return None
        import hashlib

        recipe = self.project.recipe()
        repair = self.project.repair_context(recipe=recipe)
        return dict(
            photo=photo["id"], engine=photo["engine"],
            recipe_sha256=hashlib.sha256(json.dumps(recipe, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            recipe=deepcopy(recipe),
            repair_plan_sha256=repair["receipt_sha256"] if repair else None,
            generation=self.token,
        )

    def refresh_display_snapshot(self):
        lease = self.snapshot_store.acquire()
        if lease is not None:
            photo = self.current_photo()
            if not photo or lease.context.get("photo") != photo["id"]:
                lease.release()
                return False
            self.load_records(
                lease.current, lease.reference, sample_step=lease.sample_step,
                context=lease.context, display_only=True, lease=lease,
            )
            return True
        if self.current_record is not None and hasattr(self, "last_display_records"):
            current, reference, sample_step = self.last_display_records
            self.load_records(
                current, reference, sample_step=sample_step,
                context=self.shown_context, display_only=True,
            )
            return True
        return False

    def preview_progress(self, message):
        if not self.tasks.busy:
            self.status_message(message)
        if self.preview_overlay.isVisible():
            self.preview_overlay.setText(str(message))
            self.set_preview_updating(True)

    def preview_activity_changed(self, event):
        if self.closing:
            return
        active = event["active"]
        elapsed = event["elapsed_seconds"]
        message = event["message"]
        if active:
            message += f" · 已用 {elapsed:.1f} 秒"
            if event["since_progress_seconds"] >= 12:
                message += " · 等待本阶段完成"
                marker = (event["token"], event["stage"])
                if self._logged_preview_wait != marker:
                    self._logged_preview_wait = marker
                    details = dict(
                        token=event["token"], stage=event["stage"],
                        seconds=round(event["elapsed_seconds"], 2),
                        worker_pid=int(self.live.process.processId()) if self.live.process else None,
                        worker_active=bool(self.live.active), queued=bool(self.live.pending),
                        display_pending=bool(self.base_display_pending),
                    )
                    diagnostic_event("preview_waiting", **details)
                    self.session_events.append(dict(event="preview_waiting", **details))
            self.preview_overlay.setText(message)
            self.set_preview_updating(True)
        else:
            self.set_preview_updating(False)
            if event["state"] != "idle":
                message += f" · {elapsed:.1f} 秒"
        if not self.tasks.busy:
            self.progress.setRange(0, 0 if active and event["fraction"] is None else 1000)
            self.progress.setValue(round(1000 * (event["fraction"] or 0)))
            self.progress.setVisible(active)
            self.cancel_button.setEnabled(active)
            if active:
                self._failed_preview = False
            if event["state"] == "failed":
                self._failed_preview = True
            failed_action = getattr(self, "failed_request", {}).get("action")
            self.retry_button.setEnabled(not active and (
                getattr(self, "_failed_preview", False)
                or failed_action in ("render", "probe", "geometry_prepare", "flatfield_import")
            ))
            if message:
                self.status_message(message)

    def begin_preview_activity(self, token, message="正在准备预览…", *, stage="queued"):
        queued = bool(self.live.active and self.live.active.get("token") != token)
        self.preview_activity.start(token, "正在等待上一项处理让出资源…" if queued else message,
                                    stage=stage)

    def show_preview_notice(self, message=None):
        if not hasattr(self, "preview_notice"):
            return
        if message is None and self.current_photo():
            repair = self.project.repair_status()
            if repair["state"] == "incompatible":
                message = repair["message"]
        self.preview_notice.setText(message or "")
        self.preview_notice.setMaximumWidth(max(150, self.canvas_stack.width() - 32))
        self.preview_notice.adjustSize()
        self.preview_notice.move(
            max(8, (self.canvas_stack.width() - self.preview_notice.width()) // 2),
            max(8, self.canvas_stack.height() - self.preview_notice.height() - 12),
        )
        self.preview_notice.setVisible(bool(message))
        self.preview_notice.raise_()

    def mark_preview_pending(self, *, clear_detail=True):
        if clear_detail:
            self.canvas.clear_detail()
        photo = self.current_photo()
        if photo and self.canvas.current is not None:
            old = ENGINES.get((self.shown_context or {}).get("engine"), "上一版本")
            self.show_preview_notice(f"当前显示：{old} · 最新调整正在更新")

    def schedule_preview(self):
        """Make an edit visible automatically; saving is a separate action."""
        if (self.geometry_role == "crop" or
                (self.geometry_role in ("highlight", "highlight_effect", "highlight_protect")
                 and getattr(self, "highlight_full_source", False))):
            self.request_preview()
            return
        self.token += 1
        self.live.invalidate(self.token)
        self.set_preview_updating(False)
        self.mark_preview_pending()
        self.begin_preview_activity(self.token, "正在准备当前调整…", stage="debounce")
        # A warm session reuses its transient anchor; waiting the full cold
        # debounce only adds latency to every slider tick.
        self.preview_timer.setInterval(120 if self.preview_reusable else 450)
        self.preview_timer.start()

    def stop_preview(self):
        self.preview_timer.stop()
        if self.preview_activity.active:
            self.preview_activity.finish(self.preview_activity.token, state="cancelled", message="已停止预览更新")
        self.set_preview_updating(False)
        self.token += 1
        self.live.invalidate(self.token)
        self.live.shutdown()
        self.live.deleteLater()
        if not self.closing:
            self._make_live()

    def install_glass(self):
        self.glass_report = attach_glass(self)
        self._apply_glass_palette()
        if self.glass_report["attached"]:
            self.glass_report["controls"] = attach_materials(self)

    def _apply_glass_palette(self, *_):
        self.setStyleSheet(stylesheet(self.glass_report["attached"]))

    def display_target_changed(self, target):
        self.canvas.clear_detail()
        self.display_target = target
        self.display_icc = target.icc
        self.display_description = target.description
        caption = ("sRGB 回退" if target.assumed else "自动 ICC") + " · " + target.screen_name
        self.display_status.setText(
            self.display_status.fontMetrics().elidedText(caption, Qt.TextElideMode.ElideRight, 240)
        )
        self.display_status.setToolTip(
            f"屏幕：{target.monitor_description}\n绘图空间：{target.profile_name}"
            + ("\n" + target.reason if target.reason else "")
        )
        if hasattr(self, "settings_panel"):
            self.settings_panel.set_display_status(target.description)
        self.display_revision += 1
        self.refresh_display_snapshot()
        if hasattr(self, "step"):
            self.schedule_import_preview()
        if self.geometry_role == "crop":
            self.request_preview()

    def show_about(self):
        from .release import APP_VERSION
        from .workflow_support import DONORS

        GlassMessageBox.about(
            self, "关于 PURE RGB",
            f"PURE RGB {APP_VERSION}\n\n三色负片与黑白影像工作流。\n"
            "导入归组、画幅与测量、转换调色、修整审核、观片台与 RGB16／ICC 导出。\n\n"
            f"感谢以下捐助者\n{DONORS}",
        )

    def _settings(self):
        if not hasattr(self, "settings_panel"):
            from .workflow_settings_window import SettingsWindow

            self.settings_window = SettingsWindow(self.performance, self)
            self.settings_panel = self.settings_window.panel
            self.settings_window.visibility_changed.connect(
                lambda visible: setattr(self, "settings_open", visible)
            )
            self.settings_panel.settings_changed.connect(self.performance_changed)
            self.settings_panel.proof_changed.connect(self.proof_changed)
            self.settings_panel.diagnostics_requested.connect(self.save_diagnostics)
            self.settings_panel.clear_cache_requested.connect(self.clear_preview_cache)
            self.settings_panel.release_memory_requested.connect(self.release_preview_memory)
            self.settings_panel.set_display_status(self.display_description)
            self.settings_panel.set_proof(self.proof_space)
        return self.settings_window

    def open_settings(self):
        if not self.flush_advanced():
            return
        self._settings().present()
        self.refresh_runtime_status()

    def check_updates(self):
        settings = self._settings()
        settings.panel.tabs.setCurrentIndex(0)
        settings.present()
        settings.check_updates()

    def performance_changed(self, settings):
        self.performance = save_performance_settings(self.performance_path, settings)
        self.tasks.performance = self.performance
        self.live.performance = self.performance
        set_debug_logging(self.performance["debug_logging"])
        if hasattr(self, "idle_scheduler"):
            self.idle_scheduler.set_enabled(self.performance["idle_previews"])
        policy = resolve_performance(self.performance, include_backend=False)
        if hasattr(self, "display_cache"):
            self.display_cache.set_budget(min(512 * 1024**2, policy["preview_cache_budget_bytes"] // 4))
            self.display_cache.set_workers(policy["display_threads"])
        self.session_events.append(dict(event="settings_changed", status="complete", mode=self.performance["mode"]))
        was_updating = bool(self.live.active or self.live.pending or self.pool.activeThreadCount())
        self.stop_preview()
        if was_updating:
            self.preview_timer.start()
        self.refresh_runtime_status()
        self.status_message("性能设置已保存，将用于后续处理与预览。")

    def proof_changed(self, space):
        if space not in {"off", "srgb", "display-p3", "adobe-rgb"}:
            self.error("未知软打样色域")
            return
        self.proof_space = space
        self.canvas.clear_detail()
        write_json(self.display_preferences, dict(proof=space))
        self.display_revision += 1
        self.refresh_display_snapshot()
        if self.geometry_role == "crop":
            self.request_preview()

    def refresh_runtime_status(self):
        if not hasattr(self, "settings_panel"):
            return
        from .diagnostics import memory

        rss = memory().get("rss_bytes") or 0
        worker_rss = process_rss_bytes(int(self.live.process.processId())) if self.live.process else 0
        policy = resolve_performance(self.performance, include_backend=False)
        self.settings_panel.set_runtime_status(
            f"CPU · {policy['cpu_threads']} 线程\n"
            f"界面内存 {rss / 1024**2:.0f} MB · 预览进程 {(worker_rss or 0) / 1024**2:.0f} MB\n"
            f"预览缓存预算 {policy['preview_cache_budget_bytes'] / 1024**2:.0f} MB"
        )
        if hasattr(self, "display_cache"):
            stats = self.display_cache.stats()
            self.settings_panel.set_cache_status(
                f"显示缓存：{stats['entries']} 幅 · {stats['bytes'] / 1024**2:.0f} MB\n"
                "可将 7 天前的临时预览移到废纸篓。"
            )
        busy = bool(self.tasks.busy or self.live.active or self.live.pending or self.pool.activeThreadCount())
        self.settings_panel.clear_cache_button.setEnabled(not busy)
        self.settings_panel.release_memory_button.setEnabled(not busy)

    def release_preview_memory(self):
        if self.live.active or self.live.pending or self.tasks.busy or self.pool.activeThreadCount():
            self.status_message("处理完成后可释放预览内存。")
            return
        self.stop_preview()
        if hasattr(self, "display_cache"):
            self.display_cache.clear()
        self.refresh_runtime_status()
        self.status_message("已释放预览缓存；当前画面保留。")

    def clear_preview_cache(self):
        from .workflow_maintenance import cleanup_preview_cache

        if self.tasks.busy or self.live.active or self.live.pending:
            self.status_message("请等待当前处理完成后清理缓存。")
            return
        result = cleanup_preview_cache(ROOT, active_directories=[self.live.directory])
        self.session_events.append(dict(event="cache_cleanup", status=result["state"], cleared_count=result["cleared_count"], cleared_bytes=result["cleared_bytes"]))
        self.status_message(
            f"已将 {result['cleared_count']} 个旧预览文件移到废纸篓，"
            f"共 {result['cleared_bytes'] / 1024**2:.1f} MB。"
            if result["state"] != "unavailable"
            else "暂时无法访问废纸篓，缓存未清理。"
        )

    def save_diagnostics(self):
        from .workflow_maintenance import write_diagnostic_report

        path, _ = QFileDialog.getSaveFileName(
            self, "保存诊断报告", str(Path.home() / "Desktop/PURE-RGB-diagnostics.json"), "JSON (*.json)"
        )
        if path:
            write_diagnostic_report(path, self.performance, events=self.session_events[-100:])
            self.status_message("诊断报告已保存。")

    def setup_background_previews(self):
        from .workflow_display_cache import DisplayImageCache
        from .workflow_idle import IdlePreviewScheduler, PrewarmGate

        policy = resolve_performance(self.performance, include_backend=False)
        self.display_cache = DisplayImageCache(
            budget_bytes=min(512 * 1024**2, policy["preview_cache_budget_bytes"] // 4),
            workers=policy["display_threads"],
        )
        self.idle_pool = QThreadPool(self)
        self.idle_pool.setMaxThreadCount(1)
        self.idle_cancelled = set()
        self.idle_scheduler = IdlePreviewScheduler(
            self.prefetch_display, self.cancel_prefetch, parent=self
        )
        self.prewarm_gate = PrewarmGate(self.idle_scheduler.user_activity, self)
        self.tasks.prewarm_gate = self.prewarm_gate
        self.live.prewarm_gate = self.prewarm_gate
        self.idle_scheduler.set_enabled(self.performance["idle_previews"])
        self._event_filters_active = True
        QApplication.instance().installEventFilter(self)
        self.maintenance_timer = QTimer(self)
        self.maintenance_timer.setInterval(2000)
        self.maintenance_timer.timeout.connect(self.service_background)
        self.maintenance_timer.start()

    def eventFilter(self, watched, event):
        if not getattr(self, "_event_filters_active", False) or getattr(self, "closing", True):
            return False
        if (
            watched is self.canvas
            and self._straighten_panel is not None
            and self.straighten_draw_event(event)
        ):
            return True
        if hasattr(self, "_history_slider"):
            self.history_slider_event(watched, event)
        if (
            hasattr(self, "idle_scheduler")
            and isinstance(watched, QWidget)
            and watched.window() is self
            and (
                event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.KeyPress, QEvent.Type.Wheel)
                or (event.type() == QEvent.Type.Resize and watched is self)
                or (event.type() == QEvent.Type.MouseMove and event.buttons() != Qt.MouseButton.NoButton)
            )
        ):
            self.last_activity = time.monotonic()
            self.idle_scheduler.user_activity()
        return super().eventFilter(watched, event)

    def queue_idle_previews(self, saved_jobs=None):
        if not hasattr(self, "idle_scheduler"):
            return
        photos = self.project.photos
        current = self.project.data.get("current")
        current_index = next(
            (index for index, photo in enumerate(photos) if photo["id"] == current),
            None,
        )
        ordered = list(enumerate(photos))
        if current_index is not None:
            ordered.sort(key=lambda item: (
                abs(item[0] - current_index), item[0] < current_index
            ))
        queue = []
        for _, photo in ordered:
            if photo["id"] == current:
                continue
            job = (self.project.saved_job(photo) if saved_jobs is None
                   else saved_jobs.get(photo["id"]))
            if job:
                queue.append((job["id"], photo["engine"]))
        self.idle_scheduler.set_queue(queue)

    def cancel_prefetch(self, item, generation):
        self.idle_cancelled.add(generation)

    def prefetch_display(self, item, generation):
        if (
            self.closing or self.tasks.busy or self.live.active or self.live.pending
            or self.pool.activeThreadCount() or self.display_target is None
        ):
            self.idle_scheduler.user_activity()
            self.idle_cancelled.discard(generation)
            return
        target, proof = self.display_target, self.proof_space

        def read():
            if generation in self.idle_cancelled or self.closing:
                return None
            sample = self.project.workspace.sample(self.project.workspace.job(item[0]))
            for mode in dict.fromkeys((RESULT_MODES[item[1]], "native")):
                if generation in self.idle_cancelled or self.closing:
                    break
                if f"baseline__{mode}" not in sample.results:
                    continue
                record = sample.record(mode, "baseline")
                self.display_cache.get_image(
                    record, target, proof=proof,
                    source_step=saved_display_source_step(record) or 1,
                )
            if hasattr(self.project.workspace, "root"):
                from .decode_prewarm import prewarm

                manifest = self.project.workspace.root / "runs" / item[0] / "manifest.json"
                def cancelled():
                    return generation in self.idle_cancelled or self.closing
                prewarm(manifest, cancelled,
                        start_guard=lambda: self.prewarm_gate.enter(cancelled),
                        end_guard=self.prewarm_gate.leave)
            return None

        work = Work(generation, read)
        work.signals.done.connect(lambda token, value: self.prefetch_finished(token))
        work.signals.failed.connect(lambda token, message: self.prefetch_finished(token))
        self.idle_pool.start(work)

    def prefetch_finished(self, generation):
        self.idle_cancelled.discard(generation)
        self.idle_scheduler.complete(generation)

    def service_background(self):
        if self.closing:
            return
        if self.settings_open:
            self.refresh_runtime_status()
        busy = bool(
            self.tasks.busy or self.live.active or self.live.pending
            or self.pool.activeThreadCount() or self.idle_pool.activeThreadCount()
        )
        idle = time.monotonic() - self.last_activity
        if not busy and self.live.process and idle >= 8:
            rss = process_rss_bytes(int(self.live.process.processId()))
            if should_release_preview(
                self.performance, idle_seconds=idle, busy=False, rss_bytes=rss,
                retained_source_bytes=self.live.cache_report.get("retained_source_bytes", 0),
            ):
                self.stop_preview()

    def build(self):
        self.application_menu = self.menuBar().addMenu("PURE RGB")
        self.about_action = QAction("关于 PURE RGB", self)
        self.about_action.setMenuRole(QAction.MenuRole.AboutRole)
        self.about_action.triggered.connect(self.show_about)
        self.application_menu.addAction(self.about_action)
        for title, slot in (
            ("使用指南", lambda: self._settings().open_manual()),
            ("捐赠 PURE Studio 项目", lambda: self._settings().open_donation()),
            ("检查更新", self.check_updates),
        ):
            action = QAction(title, self)
            action.setMenuRole(QAction.MenuRole.ApplicationSpecificRole)
            action.triggered.connect(slot)
            self.application_menu.addAction(action)
        self.settings_action = QAction("设置…", self)
        self.settings_action.setShortcut(QKeySequence.StandardKey.Preferences)
        self.settings_action.setMenuRole(QAction.MenuRole.PreferencesRole)
        self.settings_action.triggered.connect(self.open_settings)
        self.application_menu.addAction(self.settings_action)
        self.application_menu.addSeparator()
        self.quit_action = QAction("退出 PURE RGB", self)
        self.quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.quit_action.setMenuRole(QAction.MenuRole.QuitRole)
        self.quit_action.triggered.connect(lambda: self.close())
        self.application_menu.addAction(self.quit_action)
        self.build_history_actions()
        from .workflow_application import ApplicationMenuTitles

        self.application_menu_titles = ApplicationMenuTitles(self)
        root = QWidget()
        root.setObjectName("workflowRoot")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(WORKSPACE_INSET, WORKSPACE_INSET, WORKSPACE_INSET, WORKSPACE_INSET)
        outer.setSpacing(WORKSPACE_GAP)
        header = QFrame()
        header.setObjectName("workflowHeader")
        header.setFixedHeight(64)
        head = QHBoxLayout(header)
        head.setContentsMargins(16, 8, 16, 8)
        head.setSpacing(16)
        self.header_logo = QLabel()
        self.header_logo.setPixmap(header_logo(32))
        self.header_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header_logo.setAccessibleName("PURE RGB Logo")
        self.header_logo.setFixedSize(160, 40)
        head.addWidget(self.header_logo, 0, Qt.AlignmentFlag.AlignVCenter)
        self.context = text("请选择输入方式并导入素材", True)
        self.context.setParent(header)
        self.context.hide()
        outer.addWidget(header)
        stage_bar = QFrame()
        stage_bar.setObjectName("workflowStages")
        stage_bar.setFixedHeight(38)
        stage_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        stages = QHBoxLayout(stage_bar)
        stages.setContentsMargins(3, 3, 3, 3)
        stages.setSpacing(2)
        self.stage_buttons = []
        self.stage_group = QButtonGroup(self)
        for index, name in enumerate(STAGES):
            number, visible_name = stage_presentation(index)
            # macOS accessibility can select a checkable button by changing
            # its checked state without emitting clicked(). Route all input
            # methods through the group's checked-state transition instead.
            item = QPushButton(f"{number}  {visible_name}")
            item.setCheckable(True)
            item.setAccessibleName(visible_name)
            self.stage_group.addButton(item, index)
            if index not in VISIBLE_STAGE_INDICES:
                item.setParent(stage_bar)
                item.hide()
            self.stage_buttons.append(item)
        for index in VISIBLE_STAGE_INDICES:
            stages.addWidget(self.stage_buttons[index], 1)
        self.stage_group.idToggled.connect(self.stage_toggled)
        from .workflow_stage_navigation import StageNavigation

        self.stage_navigation = StageNavigation(stage_bar, {
            index: self.stage_buttons[index] for index in VISIBLE_STAGE_INDICES})
        self.stage_highlight = self.stage_navigation.indicator
        head.addWidget(stage_bar, 1, Qt.AlignmentFlag.AlignVCenter)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(WORKSPACE_GAP)
        outer.addWidget(split, 1)
        tools = QFrame()
        tools.setObjectName("glassPanel")
        tools.setMinimumWidth(320)
        tools.setMaximumWidth(390)
        left = QVBoxLayout(tools)
        left.setContentsMargins(15, 14, 15, 14)
        left.setSpacing(9)
        self.stage_title = text("")
        self.stage_title.setObjectName("section")
        left.addWidget(self.stage_title)
        self.engine_box = QWidget()
        engine_layout = QVBoxLayout(self.engine_box)
        engine_layout.setContentsMargins(0, 4, 0, 8)
        engine_layout.addWidget(text("转换引擎"))
        self.engine_choice = SpringComboBox()
        self.engine_choice.setAccessibleName("转换引擎")
        for key, name in ENGINES.items():
            self.engine_choice.addItem(name, key)
        self.engine_choice.currentIndexChanged.connect(self.engine_changed)
        engine_layout.addWidget(self.engine_choice)
        left.addWidget(self.engine_box)
        self.pages = QStackedWidget()
        self.advanced_buttons = {}
        self.advanced_panels = {}
        self.advanced_contexts = {}
        self.advanced_applying = False
        left.addWidget(self.pages, 1)
        for index in range(len(STAGES)):
            host = QWidget()
            layout = QVBoxLayout(host)
            layout.setContentsMargins(0, 6, 0, 8)
            layout.setSpacing(12)
            getattr(self, f"build_stage_{index}")(layout)
            layout.addStretch()
            scroll = WorkflowScrollArea()
            scroll.setWidgetResizable(True)
            # Sidebar pages are single-column vertical layouts. A horizontal
            # bar only appears when a child overshoots the viewport; it steals
            # vertical space and clips the bottom controls, so forbid it and
            # let widget-resizable layout constrain widths instead.
            scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            scroll.setWidget(host)
            from .workflow_scroll_gutter_v45 import reserve_scrollbar_gutter

            reserve_scrollbar_gutter(scroll, 12)
            self.pages.addWidget(scroll)
        self.primary_stack = QStackedWidget()
        self.primary_stack.setFixedHeight(42)
        for action in (
            self.import_button,
            self.frame_next,
            self.meter_apply,
            self.save_button,
            self.retouch_save,
            self.export_button,
            self.light_table_current,
        ):
            self.primary_stack.addWidget(action)
        left.addWidget(self.primary_stack)
        split.addWidget(tools)
        center = QWidget()
        middle = QVBoxLayout(center)
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(WORKSPACE_GAP)
        preview_toolbar = PreviewToolbar()
        bar = preview_toolbar.edit_row
        self.canvas_title = text("尚未载入照片", True)
        self.canvas_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas_title.setWordWrap(False)
        self.canvas_title.setMinimumWidth(80)
        self.canvas_title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        bar.addWidget(self.canvas_title, 1)
        self.history_toolbar_buttons(bar)
        bar = preview_toolbar.view_row
        self.fit_button = button("适应", lambda: self.canvas.fit())
        self.fit_button.setAccessibleName("适应画面")
        bar.addWidget(self.fit_button)
        self.actual_button = button("100%", lambda: self.canvas.actual())
        self.actual_button.setAccessibleName("100% 原始像素")
        bar.addWidget(self.actual_button)
        self.loupe_button = QPushButton()
        self.loupe_button.setObjectName("photoLoupe")
        self.loupe_button.setAccessibleName("照片放大镜")
        self.loupe_button.setToolTip("放大镜 · 点击照片定位，拖动镜面移动，按住临时放大。蓝点调大小，绿点或滚轮调倍率。方向键移动，Esc 退出。")
        self.loupe_button.setCheckable(True)
        self.loupe_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.loupe_button.setIcon(QIcon(str(Path(__file__).resolve().parents[1] / "assets/ui/loupe.svg")))
        self.loupe_button.setIconSize(QSize(20, 20))
        self.loupe_button.setFixedSize(32, 30)
        bar.addWidget(self.loupe_button)
        self.compare = button("A/B", self.toggle_compare)
        self.compare.setCheckable(True)
        bar.addWidget(self.compare)
        self.orientation_buttons = {}
        icon_directory = Path(__file__).resolve().parents[1] / "assets/ui"
        for action, icon_name, title in (
            ("left", "rotate-left", "向左旋转 90°"),
            ("right", "rotate-right", "向右旋转 90°"),
            ("flip_horizontal", "flip-horizontal", "水平翻转"),
            ("flip_vertical", "flip-vertical", "垂直翻转"),
        ):
            control = button(
                "",
                lambda checked=False, action=action: self.change_output_orientation(action),
            )
            control.setFixedSize(32, 30)
            control.setProperty("orientationControl", True)
            control.setIcon(QIcon(str(icon_directory / (icon_name + ".svg"))))
            control.setIconSize(QSize(22, 22))
            control.setObjectName("orientation_" + action)
            control.setAccessibleName(title)
            control.setToolTip(title + "；仅改变照片输出方向")
            control.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.orientation_buttons[action] = control
            bar.addWidget(control)
        middle.addWidget(preview_toolbar)
        self.canvas_stack = RoundedPreviewStack()
        self.canvas = Canvas()
        self.canvas.empty_message = "选择 RAW 文件或素材文件夹开始"
        from .motion_lens import PhotoLens

        self.canvas.photo_lens = PhotoLens(self.canvas, self.loupe_button)
        self.canvas.set_detail_enabled(True)
        self.canvas.installEventFilter(self)
        self._straighten_panel = None
        self._straighten_line = None
        self._straighten_anchor = None
        self.canvas.detail_requested.connect(self.request_detail_preview)
        self.canvas.full_resolution_requested.connect(self.request_actual)
        self.canvas_stack.addWidget(self.canvas)
        self.selection = SelectionView()
        self.selection.stroke_changed.connect(self.sync_drawing_hint)
        self.selection.selected.connect(self.shape_selected)
        self.selection.crop_rotated.connect(self.crop_post_changed)
        self.selection.zone_selected.connect(self.open_meter_zone_menu)
        self.canvas_stack.addWidget(self.selection)
        self.canvas_stack.currentChanged.connect(lambda _: self.sync_loupe_button())
        self.preview_overlay = QLabel("正在更新预览…", self.canvas_stack)
        self.preview_overlay.setObjectName("previewUpdateHUD")
        self.preview_overlay.setProperty("nativeHUD", True)
        self.preview_overlay.setAccessibleName("预览更新状态")
        self.preview_overlay.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self.preview_overlay.setStyleSheet(
            "QLabel {background:rgba(30,29,28,230);color:#f2ede4;"
            "border:1px solid rgba(255,255,255,40);border-radius:10px;padding:8px 14px;}"
            "QLabel[nativeMaterial=\"true\"] {background:transparent;"
            "border-color:transparent;color:transparent;}"
        )
        self.preview_overlay.adjustSize()
        self.preview_overlay.hide()
        self.preview_notice = QLabel(self.canvas_stack)
        self.preview_notice.setObjectName("previewNoticeHUD")
        self.preview_notice.setProperty("nativeHUD", True)
        self.preview_notice.setWordWrap(True)
        self.preview_notice.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.preview_notice.setStyleSheet(
            "QLabel {background:rgba(30,29,28,230);color:#f2ede4;"
            "border:1px solid rgba(255,255,255,40);border-radius:10px;padding:7px 12px;}"
            "QLabel[nativeMaterial=\"true\"] {background:transparent;"
            "border-color:transparent;color:transparent;}"
        )
        self.preview_notice.hide()
        middle.addWidget(self.canvas_stack, 1)
        self.highlight_frame_toggle = button("显示完整源画面", self.toggle_highlight_frame)
        self.highlight_frame_toggle.hide()
        middle.addWidget(self.highlight_frame_toggle)
        self.region_done = AttentionButton("退出绘制")
        self.region_done.clicked.connect(self.leave_selection)
        self.region_done.hide()
        middle.addWidget(self.region_done)
        self.zone_note = text("点击网格选择分区", True)
        self.zone_note.hide()
        middle.addWidget(self.zone_note)
        self.zone_menu = GlassMenu(self.selection)
        self.zone_menu.setObjectName("meterZoneMenu")
        self.zone_menu.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.zone_exclude_action = self.zone_menu.addAction("排除该分区")
        self.zone_restore_action = self.zone_menu.addAction("恢复该分区")
        self.zone_exclude_action.setCheckable(True)
        self.zone_restore_action.setCheckable(True)
        self.zone_exclude_action.triggered.connect(lambda: self.change_meter_zone(exclude=True))
        self.zone_restore_action.triggered.connect(lambda: self.change_meter_zone(exclude=False))
        split.addWidget(center)
        photos = QFrame()
        photos.setObjectName("glassPanel")
        photos.setMinimumWidth(240)
        photos.setMaximumWidth(300)
        right = QVBoxLayout(photos)
        from .workflow_scopes import ScopePanel

        self.scope_panel = ScopePanel()
        right.addWidget(self.scope_panel)
        self.photo_heading = text("照片")
        right.addWidget(self.photo_heading)
        self.photo_list = QListWidget()
        self.photo_list.setObjectName("workflowPhotos")
        self.photo_list.setItemDelegate(CheckIndicatorDelegate(self.photo_list, "#ff8052", round_shape=True))
        self.photo_list.setIconSize(QSize(48, 36))
        self.photo_list.setViewMode(QListView.ViewMode.ListMode)
        self.photo_list.setFlow(QListView.Flow.TopToBottom)
        self.photo_list.setWrapping(False)
        self.photo_list.setMovement(QListView.Movement.Static)
        self.photo_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.photo_list.setWordWrap(True)
        self.photo_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.photo_list.currentItemChanged.connect(self.photo_selected)
        self.photo_list.itemChanged.connect(self.check_changed)
        self.photo_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.photo_list.customContextMenuRequested.connect(self.photo_list_menu)
        right.addWidget(self.photo_list, 1)
        self.batch_note = text("未勾选时处理当前照片", True)
        right.addWidget(self.batch_note)
        from .workflow_update_notice import UpdateNotice

        self.update_notice = UpdateNotice(photos)
        right.addWidget(self.update_notice)
        self.status = text("请选择素材开始导入。", True)
        self.status.setAccessibleName("任务状态")
        self.status.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.status.setMaximumHeight(3 * self.status.fontMetrics().lineSpacing())
        right.addWidget(self.status)
        self.progress = WorkflowProgressBar()
        self.progress.setAccessibleName("任务进度")
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(10)
        self.progress.hide()
        right.addWidget(self.progress)
        self.cancel_button = button("取消当前任务", self.cancel_task)
        right.addWidget(self.cancel_button)
        self.retry_button = button("重试上次失败任务", self.retry_last)
        right.addWidget(self.retry_button)
        self.display_status = text(self.display_description, True)
        self.display_status.setWordWrap(False)
        self.display_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        right.addWidget(self.display_status)
        split.addWidget(photos)
        # 强制限制两侧栏宽度：不可拖拽折叠，窄窗口时由画布吸收空间。
        split.setCollapsible(0, False)
        split.setCollapsible(1, False)
        split.setCollapsible(2, False)
        split.setStretchFactor(1, 1)
        split.setSizes([320, 840, 238])

    def build_stage_0(self, layout):
        layout.addWidget(text("输入方式"))
        self.input_choice = QComboBox()
        self.input_choice.setAccessibleName("新素材输入方式")
        self.input_choice.addItem("拜耳 RGB · 三张 RAW", "rgb")
        self.input_choice.addItem("单色 RGB · 三张 RAW", "mono_rgb")
        self.input_choice.addItem("黑白负片 · 单张 RAW", "bw")
        layout.addWidget(self.input_choice)
        layout.addWidget(button("选择 RAW 文件…", self.import_files))
        layout.addWidget(button("选择素材文件夹…", self.import_folder))
        layout.addWidget(
            text("输入方式用于新导入。单色 RGB 按文件名顺序分配通道，请在导入前核对。", True)
        )
        self.source_note = text("尚未导入源文件", True)
        layout.addWidget(self.source_note)
        self.roles = []
        self.role_host = QWidget()
        roles = QFormLayout(self.role_host)
        roles.setContentsMargins(0, 0, 0, 0)
        roles.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        roles.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        roles.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for role in "RGB":
            choice = QComboBox()
            choice.setAccessibleName(role + " 通道文件")
            choice.currentIndexChanged.connect(self.schedule_import_preview)
            roles.addRow(role, choice)
            self.roles.append(choice)
        layout.addWidget(self.role_host)
        layout.addWidget(text("机身与镜头"))
        self.equipment_note = text("选择 RAW 后读取器材与内嵌矫正参数。", True)
        layout.addWidget(self.equipment_note)
        self.lens_correction_button = button("RAW 内嵌镜头矫正", self.change_lens_correction)
        self.lens_correction_button.setCheckable(True)
        layout.addWidget(self.lens_correction_button)
        self.build_flatfield_tools(layout)
        self.import_button = AttentionButton("核对并导入")
        self.import_button.setObjectName("primary")
        self.import_button.clicked.connect(self.confirm_import)
        self.import_button.setToolTip("核对当前照片的源文件后导入，继续画幅与测量。")
        layout.addWidget(self.import_button)

    def build_stage_1(self, layout):
        # Keep the historical stack/primary-action slot for persisted IDs;
        # visible navigation maps this page to the combined stage below.
        layout.addWidget(text("画幅工具已合并到“画幅与测量”。", True))
        self.frame_next = button(
            "进入画幅与测量", lambda: self.navigate_stage(2), True
        )
        layout.addWidget(self.frame_next)

    def change_output_orientation(self, action):
        from .output_orientation import change_orientation

        photo = self.current_photo()
        if self.tasks.busy or not photo or not photo["initial_job"]:
            return
        self.end_history_gesture()
        if not self.flush_advanced():
            return
        if self.geometry_role != "crop":
            self.finish_selection()
        self.canvas.fit()
        self.preview_timer.stop()
        self.token += 1
        self.live.invalidate(self.token)
        self.set_preview_updating(False)
        labels = {"left": "向左旋转 90°", "right": "向右旋转 90°",
                  "flip_horizontal": "水平翻转", "flip_vertical": "垂直翻转"}
        self.project.set_recipe(change_orientation(self.project.recipe(), action),
                                label=labels[action])
        self.sync_controls()
        self.request_preview()

    def build_stage_2(self, layout):
        from .workflow_parameter_reset import add_parameter_reset

        self.meter_tools = QWidget()
        body = QVBoxLayout(self.meter_tools)
        body.setContentsMargins(0, 0, 0, 0)
        self.frame_toggle = DisclosureButton("画幅与输出裁切")
        self.frame_toggle.setAccessibleName("展开画幅与输出裁切")
        body.addWidget(self.frame_toggle)
        self.frame_tools = QWidget()
        frame = QVBoxLayout(self.frame_tools)
        frame.setContentsMargins(0, 0, 0, 0)
        frame.setSpacing(9)
        self.source_info = text("完成首次导入后，可查看完整源画幅与实际尺寸。", True)
        frame.addWidget(self.source_info)
        frame.addWidget(button("平场矫正…", self.open_flatfield_tools))
        frame.addWidget(text("旋转与翻转也可使用画面上方工具栏。", True))
        self.orientation_note = text("输出方向：原始方向", True)
        frame.addWidget(self.orientation_note)
        self.frame_inspect = button("查看完整源画幅", lambda: self.begin_selection("inspect"))
        frame.addWidget(self.frame_inspect)
        self.frame_crop = button("裁切、拉直与旋转…", self.open_crop_tools)
        frame.addWidget(self.frame_crop)
        frame.addWidget(text("输出裁切在各色彩引擎间共用。", True))
        body.addWidget(self.frame_tools)
        self.frame_tools.hide()
        self.frame_toggle.toggled.connect(self.frame_tools.setVisible)
        self.shape_choice = QComboBox()
        self.shape_choice.setAccessibleName("测量选区形状")
        for value, name in (
            ("rectangle", "矩形"),
            ("rounded_rectangle", "圆角矩形"),
            ("ellipse", "椭圆"),
            ("polygon", "手绘套索"),
        ):
            self.shape_choice.addItem(name, value)
        self.shape_choice.setCurrentIndex(0)
        self.shape_choice.currentIndexChanged.connect(self.selection_shape_changed)
        body.addWidget(self.shape_choice)
        body.addWidget(
            button("绘制密度测量区", lambda: self.begin_selection("include"))
        )
        body.addWidget(button("减去不良区域", lambda: self.begin_selection("exclude")))
        body.addWidget(
            button("保护正常区域（矩形）", lambda: self.begin_selection("protect"))
        )
        body.addWidget(button("清除排除与保护", self.clear_exclusions))
        self.confirm_candidates_button = AttentionButton("确认候选并加入排除")
        self.confirm_candidates_button.clicked.connect(self.confirm_candidates)
        body.addWidget(self.confirm_candidates_button)
        self.zones_note = text("分区密度 · 8 × 12；RGB 三通道共用支持区域。", True)
        body.addWidget(self.zones_note)
        grid_controls = QHBoxLayout()
        self.meter_grid_button = button("显示分区网格与数值", self.show_meter_grid)
        self.meter_grid_reset = button("复位默认", self.reset_meter_grid)
        self.meter_grid_reset.setToolTip("恢复 8 × 12 分区、每区 2 × 2 子块，保留测量范围与排除设置。")
        grid_controls.addWidget(self.meter_grid_button, 1)
        grid_controls.addWidget(self.meter_grid_reset)
        body.addLayout(grid_controls)
        self.avoidance = QComboBox()
        self.avoidance.setAccessibleName("异常处理方式")
        for value, name in (
            ("suggest", "只给建议 · 默认"),
            ("exclude", "自动避让 · 可选"),
            ("off", "关闭异常建议"),
        ):
            self.avoidance.addItem(name, value)
        self.avoidance.currentIndexChanged.connect(self.avoidance_changed)
        self.avoidance_reset = add_parameter_reset(body, self.avoidance, "suggest", "异常处理方式")
        body.addWidget(text("避让只改变测量支持，不修补图像，也不改变输出裁切。", True))
        layout.addWidget(self.meter_tools)
        self.meter_note = text("先核对导入，随后可绘制源坐标选区。", True)
        layout.addWidget(self.meter_note)
        self.advanced_button(layout, 2, "分区测量高级参数…")
        self.meter_apply = button("刷新测量预览", self.apply_metering, True)
        layout.addWidget(self.meter_apply)

    def spin(self, layout, name, low, high, step, value, callback, *, control_type=ScrubberControl):
        control = control_type(name, low, high, step, value)
        control.valueChanged.connect(callback)
        layout.addWidget(control)
        return control

    def build_stage_3(self, layout):
        from .film_ui import NativeFilmSelector
        from .motion_sliders import JellySliderControl
        from .workflow_parameter_reset import add_parameter_reset

        spec = TEMPERATURE_SPEC
        self.temperature = self.spin(
            layout, "色温（K）", spec.low, spec.high, spec.step, spec.default,
            lambda value: self.color_slider_changed(TEMPERATURE_SPEC, value),
        )
        self.temperature.setToolTip("NegPy 正像色温；5500 K 为中性")
        spec = HUE_SPEC
        self.hue = self.spin(
            layout, spec.title, spec.low, spec.high, spec.step, spec.default,
            lambda value: self.color_slider_changed(HUE_SPEC, value),
        )
        self.hue.setToolTip("旋转整张正像的色相；0° 为原色")
        self.native_film = NativeFilmSelector()
        self.native_film.setObjectName("nativeFilmSelector")
        self.native_film.changed.connect(self.profile_changed)
        layout.addWidget(self.native_film)
        self.film_note = text("", True)
        layout.addWidget(self.film_note)
        self.exposure = self.spin(
            layout, "曝光（EV）", -4, 4, 0.1, 0, self.tone_changed
        )
        self.contrast = ScrubberControl("反差", 50, 180, 1, 115)
        self.contrast_label = self.contrast.label
        self.contrast.valueChanged.connect(self.tone_changed)
        layout.addWidget(self.contrast)
        self.paper = QComboBox()
        self.paper.setAccessibleName("印相相纸")
        self.paper.currentIndexChanged.connect(self.tone_changed)
        self.paper_label = text("印相相纸")
        layout.addWidget(self.paper_label)
        self.paper_reset = add_parameter_reset(layout, self.paper, "neutral", "印相相纸")
        self.color_box = QWidget()
        color_outer = QVBoxLayout(self.color_box)
        color_outer.setContentsMargins(0, 0, 0, 0)
        self.balance_toggle = DisclosureButton("RGB 通道平衡")
        color_outer.addWidget(self.balance_toggle)
        self.balance_body = QWidget()
        colors = QVBoxLayout(self.balance_body)
        colors.setContentsMargins(0, 0, 0, 0)
        self.balance = {}
        for key, name in (
            ("red_ev", "R 通道平衡（EV）"),
            ("green_ev", "G 通道平衡（EV）"),
            ("blue_ev", "B 通道平衡（EV）"),
        ):
            self.balance[key] = self.spin(
                colors, name, -2, 2, 0.01, 0, self.tone_changed
            )
        color_outer.addWidget(self.balance_body)
        self.balance_body.hide()
        self.balance_toggle.toggled.connect(self.balance_body.setVisible)
        layout.addWidget(self.color_box)
        self.nikon_box = QWidget()
        nikon = QVBoxLayout(self.nikon_box)
        nikon.setContentsMargins(0, 0, 0, 0)
        nikon.addWidget(text("CMS 已启用，fixed 默认关闭。", True))
        self.fixed = QCheckBox("官方链路 fixed 曲线复现")
        self.fixed.toggled.connect(self.tone_changed)
        self.fixed_reset = add_parameter_reset(nikon, self.fixed, False, "fixed 曲线")
        self.fixed_strength = self.spin(
            nikon, "fixed 强度（%）", 0, 100, 5, 50, self.tone_changed
        )
        layout.addWidget(self.nikon_box)
        self.high_box = QWidget()
        high = QVBoxLayout(self.high_box)
        high.setContentsMargins(0, 0, 0, 0)
        self.highlight = QCheckBox("启用高光找回 · 独立功能")
        self.highlight.toggled.connect(self.tone_changed)
        self.highlight_reset = add_parameter_reset(high, self.highlight, False, "高光找回")
        self.highlight_strength = self.spin(
            high, "高光找回强度（%）", 0, 100, 5, 50, self.tone_changed, control_type=JellySliderControl
        )
        high.addWidget(
            button("绘制高光参考区", lambda: self.begin_selection("highlight"))
        )
        self.highlight_effect_draw = button(
            "添加高光作用区", lambda: self.begin_selection("highlight_effect")
        )
        high.addWidget(self.highlight_effect_draw)
        self.highlight_effect_clear = button("恢复高光全图作用", self.clear_highlight_effect)
        high.addWidget(self.highlight_effect_clear)
        self.highlight_effect_feather = self.spin(
            high, "作用区羽化半径（源像素）", 0, 512, 1, 64,
            self.highlight_effect_feather_changed,
        )
        self.highlight_effect_feather.setToolTip(
            "边缘会自动柔化；此值在自动柔化基础上向外增加羽化，单位为源像素。"
        )
        high.addWidget(
            button(
                "绘制人物／中间调保护",
                lambda: self.begin_selection("highlight_protect"),
            )
        )
        self.highlight_note = text(
            "默认参考当前测量区域，可另选局部；保护区由人工绘制。", True
        )
        high.addWidget(self.highlight_note)
        self.highlight_effect_note = text(
            "添加作用区后按画面亮度筛选高光；保护区内减弱高光找回。", True
        )
        high.addWidget(self.highlight_effect_note)
        layout.addWidget(self.high_box)
        self.engine_note = text("", True)
        layout.addWidget(self.engine_note)
        self.advanced_button(layout, 3, "印相、曲线与高光高级参数…")
        self.save_button = button("生成完整版本（可选）", self.save_version)
        layout.addWidget(self.save_button)

    def build_stage_4(self, layout):
        from .workflow_crop_ui import install_crop_panel

        self.repair_button = button("PURE Neg 除尘与审核…", self.start_repair, True)
        layout.addWidget(self.repair_button)
        layout.addWidget(button("恢复已有除尘审核", self.resume_repair))
        self.repair_status_note = text("", True)
        layout.addWidget(self.repair_status_note)
        layout.addWidget(
            text("调整参数自动保存；打开除尘时自动准备当前完整版本，确认后回到工作台。", True)
        )
        install_crop_panel(self, layout, 4)
        self.advanced_button(layout, 4, "细节、暗角与几何高级参数…")
        self.retouch_save = button("生成完整版本（可选）", self.save_version)
        layout.addWidget(self.retouch_save)

    def build_stage_5(self, layout):
        layout.addWidget(text("TIFF · 16-bit · 嵌入实际 ICC"))
        self.export_note = text("当前照片", True)
        layout.addWidget(self.export_note)
        self.export_directory = QLineEdit(
            str(self.project.workspace.root / "deliveries")
        )
        self.export_directory.setAccessibleName("导出目录")
        layout.addWidget(self.export_directory)
        layout.addWidget(button("选择导出目录…", self.choose_export_directory))
        self.export_button = button("导出当前／勾选照片", self.export_selected, True)
        layout.addWidget(self.export_button)
        self.export_old_highlight_button = button(
            "导出当前照片的已保存旧版", self.export_previous_highlight
        )
        self.export_old_highlight_button.setToolTip("导出旧算法保存的成片及其已确认除尘。")
        self.export_old_highlight_button.hide()
        layout.addWidget(self.export_old_highlight_button)
        layout.addWidget(text("参数自动保存；导出时自动生成当前配方的完整 TIFF，同名文件不会覆盖。", True))

    def build_stage_6(self, layout):
        from .workflow_light_table import ProjectLightTable

        self.light_table = ProjectLightTable(self)
        self.light_table_current = button("查看当前照片", self.light_table.open_current, True)
        self.light_table_roll = button("整卷观片", self.light_table.open_roll)
        self.light_table_files = button("打开图像文件…", self.light_table.open_files)
        layout.addWidget(self.light_table_roll)
        layout.addWidget(self.light_table_files)
        self.light_table_note = text("单张可用放大镜检查细节。整卷可挑选要导出的照片。", True)
        layout.addWidget(self.light_table_note)

    def current_photo(self):
        return self.project.photo() if self.project.data["current"] else None

    @property
    def advanced_panel(self):
        return self.advanced_panels.get(getattr(self, "step", 3))

    def advanced_button(self, layout, stage, title):
        panel = AdvancedPanel()
        panel.expand_all_button.setAccessibleName(title)
        self.advanced_buttons[stage] = panel.expand_all_button
        panel.recipe_changed.connect(
            lambda recipe, stage=stage: self.advanced_recipe_changed(recipe, stage)
        )
        self.advanced_panels[stage] = panel
        layout.addWidget(panel)

    def open_advanced(self, stage):
        photo = self.current_photo()
        if not photo or not photo["initial_job"] or self.tasks.busy:
            return
        if self.step != stage:
            self.navigate_stage(stage)
        if self.step != stage:
            return
        self.sync_controls()
        panel = self.advanced_panels[stage]
        panel.expand_all()
        self.pages.widget(stage).ensureWidgetVisible(panel)

    def flush_advanced(self):
        panel = self.advanced_panel
        return panel.apply_changes() if panel else True

    def close_advanced(self):
        if not self.flush_advanced():
            self.status_message("请先修正左侧参数组合。")
            return
        if self.advanced_panel:
            self.advanced_panel.collapse_all()

    def advanced_recipe_changed(self, recipe, stage):
        photo = self.current_photo()
        if not photo or self.advanced_contexts.get(stage) != (
            photo["id"],
            photo["engine"],
            self.step,
        ):
            return
        self.advanced_applying = True
        try:
            self.apply_advanced_recipe(recipe, stage)
        finally:
            self.advanced_applying = False

    def apply_advanced_recipe(self, recipe, stage):
        """Publish a validated sidebar edit, invalidating any old preview."""
        try:
            if recipe == self.project.recipe():
                return
            self.preview_timer.stop()
            self.token += 1
            self.live.invalidate(self.token)
            self.project.set_recipe(recipe)
            self.sync_controls()
            self.mark_preview_pending()
            self.request_preview()
        except Exception as error:
            self.error(error)
            self.sync_controls()

    def status_message(self, message):
        if not self.closing:
            message = str(message)
            self.status.setText(message)
            self.status.setToolTip(message)

    def set_preview_updating(self, updating):
        if not hasattr(self, "preview_overlay"):
            return
        self.preview_overlay.adjustSize()
        self.preview_overlay.move(
            max(0, (self.canvas_stack.width() - self.preview_overlay.width()) // 2), 12
        )
        self.preview_overlay.setVisible(updating)
        if updating:
            self.preview_overlay.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "preview_overlay") and self.preview_overlay.isVisible():
            self.set_preview_updating(True)
        if hasattr(self, "preview_notice") and self.preview_notice.isVisible():
            self.show_preview_notice(self.preview_notice.text())

    def error(self, message):
        self.last_error = str(message)
        self.status_message("未完成：" + str(message))

    def sync_drawing_hint(self):
        message = ""
        if self.geometry_role is not None and self._measurement_hint_timer.isActive():
            self._measurement_hint_timer.stop()
            self._measurement_hint_photo = None
            self._measurement_hint_text = ""
        if getattr(self, "step", None) == 2 and not self.selection.points:
            if self.geometry_role == "import_roi":
                message = "未能自动识别有效画面\n请拖出矩形密度检测区域。松开后自动测量，完成测量后才能进入下一步。"
            elif self.geometry_role == "include":
                message = "请绘制密度检测区域\n从画面内按住鼠标拖动，松开后开始测量。"
            elif (self.geometry_role is None and self.current_photo()
                  and self._measurement_hint_timer.isActive()
                  and self.current_photo()["id"] == self._measurement_hint_photo
                  and not self.tasks.busy):
                message = self._measurement_hint_text
        self.canvas_stack.set_drawing_hint(message)

    def clear_measurement_hint(self):
        self._measurement_hint_timer.stop()
        self._measurement_hint_photo = None
        self._measurement_hint_text = ""
        if not self.closing:
            self.sync_drawing_hint()

    def stage_toggled(self, index, checked):
        """Apply a selected stage once for mouse, keyboard and accessibility."""
        if not checked or not hasattr(self, "step"):
            return
        target = canonical_stage(index)
        if target == self.step:
            # The historical hidden stage 1 aliases the visible stage 2.
            if target != index:
                self.stage_buttons[target].setChecked(True)
            return
        self.navigate_stage(target)

    def navigate_stage(self, index):
        index = canonical_stage(index)
        photo = self.current_photo()
        if index == self.step and (self.geometry_role or self._straighten_panel is not None):
            return
        if index != LIGHT_TABLE_STAGE and index > 2 and photo and not photo["initial_job"] and not self.tasks.busy:
            # An undetected first frame must be explicitly located before
            # its density can be measured. Reopen the same source-bound
            # locator if the user exited it and then tries to continue.
            pending = getattr(self, "_import_roi_retry", None)
            locating = self.geometry_role == "import_roi"
            if not locating and pending:
                from .workflow_import_roi import offer_import_roi

                try:
                    locating = offer_import_roi(self, pending, "No reliable picture rectangle")
                except (OSError, TypeError, ValueError):
                    # The temporary locator may have been removed. A normal
                    # import retry can create a fresh, source-bound locator.
                    self.__dict__.pop("_import_roi_retry", None)
            if locating:
                self.stage_buttons[2].setChecked(True)
                self.region_done.remind()
                self.sync_drawing_hint()
                self.status_message("请在有效画面内拖出矩形密度检测区域，完成测量后才能进入下一步")
                return
        if stage_presentation(index)[0] > stage_presentation(self.step)[0] and self.remind_pending_drawing():
            self.stage_buttons[self.step].setChecked(True)
            return
        if index not in (0, LIGHT_TABLE_STAGE) and photo and not photo["initial_job"]:
            if self.tasks.busy:
                self.stage_buttons[self.step].setChecked(True)
                self.status_message("请等待当前任务完成。")
            else:
                self.set_stage(0)
                self.import_button.remind()
                self.schedule_import_preview()
                self.status_message("请先点击「核对并导入」，再进入下一步。")
            return
        self.set_stage(index)

    def remind_pending_drawing(self):
        if self.geometry_role not in (None, "inspect", "meter_grid"):
            self.region_done.remind()
            self.status_message("请先完成或退出当前绘制，再进入下一步。")
            return True
        if self._straighten_panel is not None:
            self._straighten_panel.draw_straighten.remind()
            self.status_message("请先画完参照线或退出画线拉直。")
            return True
        return False

    def set_stage(self, index):
        self.end_history_gesture()
        try:
            index = canonical_stage(index)
        except ValueError:
            return
        if index != 2 and self._measurement_hint_timer.isActive():
            self.clear_measurement_hint()
        if not self.flush_advanced():
            if hasattr(self, "step"):
                self.stage_buttons[self.step].setChecked(True)
            return
        for panel in self.advanced_panels.values():
            panel.cancel_pending()
        had_stage = hasattr(self, "step")
        route_changed = self.project.enter_stage(index)
        if route_changed:
            self.preview_timer.stop()
            self.token += 1
            self.live.invalidate(self.token)
            self.mark_preview_pending()
        from .workflow_materials import stable_material_layout

        with stable_material_layout(self):
            self.step = index
            self.primary_stack.show()
            self.pages.setCurrentIndex(index)
            self.primary_stack.setCurrentIndex(index)
            self.stage_buttons[index].setChecked(True)
            number, title = stage_presentation(index)
            self.stage_title.setText(f"{number:02d}  {title}")
            self.finish_selection()
            self.sync_controls()
            if route_changed:
                self.refresh_photos()
            self.stage_navigation.select(index, animate=had_stage)
        if route_changed and had_stage and self.current_photo()["initial_job"]:
            job = self.project.saved_job()
            if job and not self.project.dirty():
                sample = self.project.workspace.sample(job)
                self.load_records(*self.saved_images(sample))
            elif not self.tasks.busy:
                self.request_preview()
        if index == 0:
            self.schedule_import_preview()

    def refresh_metering_status(self):
        photo = self.current_photo()
        if not photo:
            return
        recipe = self.project.recipe()
        native = photo["engine"] == "native"
        spatial = recipe.get("native_spatial_controls", {})
        receipt = self.project.metering_receipt() if native else {}
        automatic = receipt.get("report", {})
        boxes = automatic.get("candidate_exclusions", [])
        matches = bool(receipt and recipe.get("roi") == receipt.get("roi")
                       and spatial.get("metering") == receipt.get("metering")
                       and spatial.get("automatic_avoidance") == receipt.get("automatic"))
        note = ("统一使用 NegPy 测量；选区、分区与避让结果供后续色彩引擎共用。"
                "调整后自动更新正像。")
        if automatic:
            summary = ("当前测量" if matches else "上次测量") + f"：{len(boxes)} 个候选片段"
            if automatic.get("application_action") == "suggest":
                summary += "，仅建议，未自动排除。"
            elif automatic.get("fallback"):
                summary += "，避让未应用，已保留原支持。"
            else:
                summary += f"，已自动排除 {len(automatic.get('automatic_exclusions', []))} 个。"
            if not matches:
                summary += " 当前设置等待更新。"
            note += "\n" + summary
        self.meter_note.setText(note)
        self.meter_note.setToolTip("源坐标测量区域：" + str(recipe.get("roi")))
        self.confirm_candidates_button.setEnabled(
            bool(photo["initial_job"]) and native and not self.tasks.busy
            and matches and 0 < len(boxes) <= 64
        )
        self.confirm_candidates_button.set_attention(self.confirm_candidates_button.isEnabled())

    def sync_controls(self):
        self.loading = True
        photo = self.current_photo()
        busy = self.tasks.busy
        try:
            self.light_table.sync_buttons()
            self.input_choice.setEnabled(not busy)
            self.pages.widget(3).setEnabled(
                bool(photo and photo["initial_job"]) and not busy
            )
            self.pages.widget(4).setEnabled(
                bool(photo and photo["initial_job"]) and not busy
            )
            self.engine_box.setVisible(
                bool(photo and photo["source_mode"] == "rgb" and self.step == 3)
            )
            self.engine_choice.setEnabled(not busy)
            self.cancel_button.setEnabled(
                (busy and self.tasks.request.get("action") != "repair_confirm")
                or (not busy and self.preview_activity.active)
            )
            failed = getattr(self, "failed_request", {})
            locating_import = self.geometry_role == "import_roi"
            self.retry_button.setEnabled(
                not busy and not locating_import and not self.preview_activity.active
                and (failed.get("action") in ("render", "probe", "geometry_prepare", "flatfield_import")
                     or getattr(self, "_failed_preview", False))
            )
            self.import_button.setEnabled(bool(photo) and not busy and not locating_import)
            ready = bool(photo and photo["initial_job"])
            grid_available = ready and not busy and photo["engine"] == "native"
            self.meter_grid_button.setEnabled(grid_available)
            self.meter_grid_reset.setEnabled(grid_available)
            attention_photo = (
                photo["id"] if photo and not ready and not busy and not locating_import
                and self.step == 0 else None
            )
            if attention_photo != self.import_attention_photo:
                self.import_button.set_attention(False)
                self.import_attention_photo = attention_photo
                self.import_button.set_attention(attention_photo is not None)
            self.sync_loupe_button()
            self.actual_button.setEnabled(ready)
            self.compare.setEnabled(ready)
            self.lens_correction_button.setEnabled(
                bool(photo and self.import_lens_context
                     and self.import_lens_context[0] == self.import_preview_context()
                     and self.import_lens_context[1] and not busy
                     and photo["base"].get("raw_import") != "standard-mono-v1")
            )
            self.native_film.setEnabled(ready and not busy)
            for control in self.orientation_buttons.values():
                control.setEnabled(ready and not busy)
            self.engine_choice.setEnabled(ready and not busy and self.step == 3
                                          and photo["source_mode"] == "rgb")
            for control in (
                self.meter_apply,
                self.save_button,
                self.retouch_save,
                self.frame_next,
                self.repair_button,
                self.export_button,
            ):
                control.setEnabled(ready and not busy)
            if not photo:
                self.equipment_note.setText("选择 RAW 后读取器材与内嵌矫正参数。")
                self.lens_correction_button.setChecked(False)
                self.meter_tools.setEnabled(False)
                for control in self.advanced_buttons.values():
                    control.setEnabled(False)
                return
            engine = photo["engine"]
            recipe = self.project.recipe()
            self.lens_correction_button.setChecked(bool(recipe.get("lens_correction")))
            orientation = recipe.get("output_orientation", {})
            direction = [f"顺时针 {orientation.get('quarter_turns', 0) * 90}°"]
            if orientation.get("flip_horizontal"):
                direction.append("水平翻转")
            if orientation.get("flip_vertical"):
                direction.append("垂直翻转")
            self.orientation_note.setText(
                "当前照片输出方向：" + " · ".join(direction)
                + "\n方向在各色彩引擎间保持一致；密度选区跟随显示方向。"
            )
            native = engine == "native"
            bw = photo["source_mode"] == "bw"
            self.native_film.setVisible(native and not bw)
            self.film_note.setVisible(native and not bw)
            self.native_film.set_document(recipe.get("native_film") if native and not bw else None,
                                          default_context=photo["id"])
            self.native_film.setEnabled(ready and not busy)
            self.film_note.setText(
                "保留当前选区与避让支持；先在胶片校正密度域重新测量，再完成原生印相。"
                if recipe.get("native_spatial_controls")
                else "胶片密度配置独立于印相相纸；数据表配置均标为近似，矩阵随照片配方保存。"
            )
            self.engine_choice.setCurrentIndex(self.engine_choice.findData(engine))
            for stage, panel in self.advanced_panels.items():
                context = (photo["id"], engine, stage)
                if context != self.advanced_contexts.get(stage):
                    self.advanced_contexts[stage] = context
                    panel.configure(recipe, engine, bw, stage)
                elif (
                    not self.advanced_applying
                    and not panel.timer.isActive()
                    and not panel.error.text()
                    and panel.original != recipe
                ):
                    panel.rebase(recipe)
                panel.setEnabled(ready and not busy)
                panel.setVisible(bool(panel.groups))
            input_mode = ("mono_rgb" if recipe.get("raw_import") == "standard-mono-v1"
                          else photo["source_mode"])
            self.input_choice.setCurrentIndex(self.input_choice.findData(input_mode))
            self.context.setText(
                ("黑白 · 单张 RAW" if bw else "单色 RGB · 三张 RAW"
                 if input_mode == "mono_rgb" else "拜耳 RGB · 三张 RAW")
                + "   |   "
                + ("NegPy · 黑白原生" if bw else "NegPy · 密度测量参考"
                   if self.step == 2 else ENGINES[engine])
            )
            raw = photo["base"]["raw"]
            self.source_note.setText("\n".join(Path(p).name for p in raw))
            self.role_host.setVisible(not bw)
            for index, choice in enumerate(self.roles):
                choice.clear()
                for path in raw:
                    choice.addItem(Path(path).name, path)
                choice.setCurrentIndex(index if not bw else 0)
                choice.setEnabled(not busy)
            shape = photo["source_shape"]
            self.source_info.setText(
                (
                    f"完整源：{shape[1]} × {shape[0]} 像素\n"
                    if shape
                    else "尚未生成完整源定位图\n"
                )
                + (
                    "黑白单张不需要 RGB 配准。"
                    if bw
                    else "已使用原导入／配准链；请查看边缘确认。"
                )
            )
            self.meter_tools.setVisible(native)
            self.meter_tools.setEnabled(ready and not busy)
            self.meter_apply.setEnabled(ready and native and not busy)
            spatial = recipe.get("native_spatial_controls", {})
            for stage, control in self.advanced_buttons.items():
                groups = advanced_groups(recipe, engine, bw, stage)
                control.setEnabled(ready and not busy and bool(groups))
                control.setToolTip(
                    f"{len(groups)} 组、{sum(len(group.specs) for group in groups)} 项参数"
                    if groups
                    else "先绘制密度选区并启用分区测量。"
                )
            zones = spatial.get("metering", {}).get("zones", [8, 12])
            self.zones_note.setText(
                f"分区密度 · {zones[0]} × {zones[1]}；"
                + ("黑白原生测量。" if bw else "RGB 三通道共用支持区域。")
            )
            action = (
                spatial.get("automatic_avoidance", {}).get("action", "suggest")
                if spatial
                else "suggest"
            )
            if spatial and "automatic_avoidance" not in spatial:
                action = "off"
            self.avoidance.setCurrentIndex(self.avoidance.findData(action))
            self.refresh_metering_status()
            color_values = recipe.get("post_controls", {}).get(RESULT_MODES[engine], {}).get("values", {})
            for control, spec in ((self.temperature, TEMPERATURE_SPEC), (self.hue, HUE_SPEC)):
                control.setVisible(not bw)
                control.setEnabled(ready and not busy)
                control.setValue(color_values.get(spec.key, spec.default))
            self.exposure.setValue(recipe["exposure_ev"])
            self.contrast_label.setText(
                "ISO-R 反差（低值更硬）" if native else "正像反差（18% 锚点）"
            )
            self.contrast.setRange(50 if native else 0.25, 180 if native else 2)
            self.contrast.setDefaultValue(115 if native else 1)
            self.contrast.setSingleStep(1 if native else 0.01)
            output = (
                recipe.get("output_controls", {})
                .get(RESULT_MODES[engine], {})
                .get("values", {})
            )
            self.contrast.setValue(
                recipe.get("native_controls", {})
                .get("values", {})
                .get("exposure.grade", 115)
                if native
                else output.get("contrast", 1)
            )
            self.paper.clear()
            if native:
                from .bw_recipe import bw_papers
                from .professional_controls import native_papers

                for key, name in bw_papers() if bw else native_papers():
                    self.paper.addItem(name, key)
                self.paper.setCurrentIndex(self.paper.findData(recipe["paper"]))
            self.paper.setVisible(native)
            self.paper_reset.setVisible(native)
            self.paper_label.setVisible(native)
            self.color_box.setVisible(not bw)
            for key, control in self.balance.items():
                control.setValue(output.get(key, 0))
            self.nikon_box.setVisible(engine == "nikon")
            fixed = recipe.get("nikon_research_controls", {}).get("fixed_strength", 0)
            self.fixed.setChecked(fixed > 0)
            self.fixed_strength.setValue((fixed or photo["fixed_last_strength"]) * 100)
            self.fixed_strength.setEnabled(fixed > 0)
            self.high_box.setVisible(native)
            high = recipe.get("highlight_controls")
            self.highlight.setChecked(bool(high))
            self.highlight_strength.setValue(
                high["controls"]["strength"] * 100 if high else 50
            )
            self.highlight_strength.setEnabled(bool(high))
            effect_area = (high or {}).get("effect_area")
            self.highlight_effect_feather.setValue(
                effect_area.get("feather_pixels", 64) if effect_area else 64
            )
            self.highlight_effect_feather.setEnabled(bool(effect_area) and ready and not busy)
            self.highlight_effect_clear.setEnabled(bool(effect_area) and ready and not busy)
            self.highlight_effect_draw.setEnabled(ready and not busy)
            self.highlight_effect_note.setText(
                f"已添加 {len(effect_area['include'])} 个作用区，区内按亮度筛选高光。橙色为作用区，青色为参考区，绿色为保护区。"
                if effect_area else
                "高光找回作用于全图。添加作用区后按亮度筛选高光；保护区内减弱找回。"
            )
            self.highlight_note.setText(
                ("正在使用手动高光参考。青色框为参考，绿色框为保护。"
                 if photo.get("highlight_reference_origin") == "manual" else
                 "正在使用自动高光参考。可在源画面内另选局部。")
                if high
                else "默认关闭；启用后建立独立参考，人物保护由人工绘制。"
            )
            if high and photo.get("highlight_reference_aligned"):
                self.highlight_note.setText(
                    self.highlight_note.text()
                    + "\n手工框已按原生分析网格收边。"
                )
            flex_note = (
                "本机资源配方 · 采用已核对的 FlexColor 研究配置。"
                if recipe.get("flex_research_controls")
                else "中性曲线 · 使用公开算法演算，胶片原曲线面板需要本机资源配方。"
            )
            self.engine_note.setText(flex_note if engine == "flex" else "")
            highlight_refresh = ready and self.project.highlight_reprocess_required(photo)
            self.canvas_title.setText(
                Path(raw[0]).stem
                + (" · 待核对导入" if not ready else
                   " · 高光算法已更新，需重新生成完整版本" if highlight_refresh else
                   " · 调整已自动保存" if self.project.dirty() else " · 完整版本已就绪")
            )
            self.canvas_title.setToolTip(self.canvas_title.text())
            self.repair_button.setEnabled(ready and not busy)
            repair_status = self.project.repair_status(photo)
            self.repair_status_note.setText(repair_status["message"])
            if repair_status["state"] == "incompatible":
                self.canvas_title.setText(self.canvas_title.text() + " · 除尘需重新审核")
                self.canvas_title.setToolTip(repair_status["message"])
            self.repair_button.setText(
                "保存并打开 PURE Neg…" if self.project.dirty() else "PURE Neg 除尘与审核…")
            self.repair_button.setToolTip(
                "高光算法已更新。生成新版时保留同来源、同画幅的已确认除尘选区；原版本与审核保留。"
                if highlight_refresh else
                "保存当前完整分辨率版本后自动打开除尘审核。" if self.project.dirty()
                else "使用当前保存版本进入 PURE Neg 检测、画笔修整及审核。")
            checked_photos = [self.project.photo(key) for key in self.project.data["checked"]]
            export_refresh = any(self.project.highlight_reprocess_required(item) for item in checked_photos) if checked_photos else highlight_refresh
            self.export_old_highlight_button.setVisible(bool(highlight_refresh))
            self.export_old_highlight_button.setEnabled(bool(highlight_refresh) and not busy)
            self.export_note.setText(
                "高光算法已更新，导出会生成新版；同来源、同画幅的已确认除尘选区继续应用。"
                if export_refresh else
                f"已勾选 {len(self.project.data['checked'])} 张；按各照片自动保存的当前参数导出"
                if self.project.data["checked"]
                else "按当前自动保存的参数导出；需要时自动生成完整成片"
            )
        finally:
            self.loading = False
            self.sync_history_actions()
            self.sync_crop_controls()
            self.sync_flatfield_controls()
            self.sync_drawing_hint()

    def refresh_photos(self):
        self.photo_list.blockSignals(True)
        self.photo_list.clear()
        current = self.project.data["current"]
        checked = set(self.project.data["checked"])
        saved_jobs = {}
        for photo in self.project.photos:
            job = self.project.saved_job(photo)
            if job:
                saved_jobs[photo["id"]] = job
            state = ("高光需重新生成" if job and self.project.highlight_reprocess_required(photo, saved=job)
                     else "完整版本" if job and not self.project.dirty(photo, saved=job)
                     else "调整已保存" if photo["initial_job"] else "待导入")
            item = QListWidgetItem(
                Path(photo["base"]["raw"][0]).stem
                + "\n"
                + ("黑白 · 单张" if photo["source_mode"] == "bw" else "R · G · B")
                + " · " + state
            )
            item.setData(Qt.ItemDataRole.UserRole, photo["id"])
            item.setSizeHint(QSize(180, 54))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if photo["id"] in checked
                else Qt.CheckState.Unchecked
            )
            thumbnail_job = job or (self.project.workspace.job(photo["initial_job"])
                                    if photo["initial_job"] else None)
            if thumbnail_job:
                path = (
                    self.project.workspace.root
                    / "runs"
                    / thumbnail_job["id"]
                    / ((RESULT_MODES[photo["engine"]] if job else "native") + "-preview.png")
                )
                if path.exists():
                    item.setIcon(QIcon(str(path)))
            self.photo_list.addItem(item)
            if photo["id"] == current:
                self.photo_list.setCurrentItem(item)
        self.photo_heading.setText(f"照片 · {len(self.project.photos)}")
        self.photo_list.blockSignals(False)
        self.queue_idle_previews(saved_jobs)

    def sync_loupe_button(self):
        photo = self.current_photo()
        available = bool(photo and photo["initial_job"] and self.canvas_stack.currentWidget() is self.canvas
                         and self._straighten_panel is None)
        self.loupe_button.setEnabled(available)
        if not available:
            self.loupe_button.setChecked(False)

    def begin_straighten_draw(self, panel):
        """进入画布画线拉直模式；激活时按钮保持高亮，再次点击退出。"""
        if self.tasks.busy:
            return
        if self._straighten_panel is not None:
            self._exit_straighten_draw()
            return
        if not self.current_photo() or self.canvas.current is None:
            self.status_message("请先完成一次预览，再使用画线拉直。")
            return
        if self.geometry_role is not None:
            self.finish_selection()
        self._straighten_panel = panel
        self.sync_loupe_button()
        self._straighten_anchor = None
        panel.draw_straighten.setChecked(True)
        panel.draw_straighten.setText("退出画线拉直")
        panel.draw_straighten.set_attention(True)
        self.canvas.setCursor(Qt.CursorShape.CrossCursor)
        self.status_message(
            "画线拉直已激活：在画面上沿需要拉平的线拖动，松开即填入拉直角度；"
            "再次点击“画线拉直”可退出。"
        )

    def _exit_straighten_draw(self, message="已退出画线拉直。"):
        panel, self._straighten_panel = self._straighten_panel, None
        self.sync_loupe_button()
        self._straighten_anchor = None
        if self._straighten_line is not None:
            self._straighten_line.clear_line()
            self._straighten_line.hide()
        if panel is not None:
            panel.draw_straighten.setChecked(False)
            panel.draw_straighten.setText(panel.straighten_draw_label)
            panel.draw_straighten.set_attention(False)
        self.canvas.unsetCursor()
        if message:
            self.status_message(message)

    def straighten_draw_event(self, event):
        kind = event.type()
        if kind == QEvent.Type.MouseButtonPress and (
            event.button() == Qt.MouseButton.LeftButton
        ):
            self._straighten_anchor = QPointF(event.position())
            if self._straighten_line is None:
                self._straighten_line = StraightenLineOverlay(self.canvas)
            self._straighten_line.setGeometry(self.canvas.rect())
            self._straighten_line.set_line(
                self._straighten_anchor, self._straighten_anchor
            )
            self._straighten_line.show()
            self._straighten_line.raise_()
            return True
        if kind == QEvent.Type.MouseMove and self._straighten_anchor is not None:
            self._straighten_line.set_line(
                self._straighten_anchor, QPointF(event.position())
            )
            return True
        if kind == QEvent.Type.MouseButtonRelease and (
            self._straighten_anchor is not None
        ):
            start, end = self._straighten_anchor, QPointF(event.position())
            panel = self._straighten_panel
            self._exit_straighten_draw(message=None)
            delta = end - start
            if delta.manhattanLength() < 12:
                self.status_message("拖动太短，已取消画线拉直。")
                return True
            # fine_rotation 正值为逆时针：水平线右端下沉的拖动（dy>0）
            # 需要逆时针转正，故直接取屏幕角的 atan2(dy, dx)。
            import math

            angle = math.degrees(math.atan2(delta.y(), delta.x()))
            if abs(angle) > 45:
                # 竖直参照线换算成让画面竖线立正所需的角度。
                angle -= math.copysign(90, angle)
            angle = round(min(45.0, max(-45.0, angle)) / 0.05) * 0.05
            panel.straighten.setValue(angle)
            self.status_message(f"已按拖动方向填入拉直 {angle:g}°；可继续用滑杆微调。")
            return True
        return False

    def photo_list_menu(self, point):
        item = self.photo_list.itemAt(point)
        if item is None or self.tasks.busy:
            return
        photo_id = item.data(Qt.ItemDataRole.UserRole)
        photo = next((p for p in self.project.photos if p["id"] == photo_id), None)
        if photo is None:
            return
        menu = GlassMenu(self.photo_list)
        copy = menu.addAction("复制调整")
        copy.setEnabled(bool(photo["initial_job"]))
        paste = menu.addAction("粘贴调整…")
        paste.setEnabled(bool(photo["initial_job"] and self.adjustment_clipboard
                              and self.adjustment_clipboard["photo_id"] != photo_id))
        reset = menu.addAction("还原所有设置…")
        reset.setEnabled(bool(photo["initial_job"]))
        menu.addSeparator()
        remove = menu.addAction("从列表移除（卸载）")
        try:
            action = menu.exec(self.photo_list.viewport().mapToGlobal(point))
        finally:
            menu.deleteLater()
        if action is copy:
            self.copy_photo_adjustments(photo_id)
        elif action is paste:
            self.paste_photo_adjustments(photo_id)
        elif action is reset:
            self.reset_photo_adjustments(photo_id)
        elif action is remove:
            self.remove_photo_from_list(photo_id)

    def copy_photo_adjustments(self, photo_id):
        from .workflow_adjustment_clipboard import copy_adjustments

        self.end_history_gesture()
        if not self.flush_advanced():
            return
        photo = self.project.photo(photo_id)
        if photo["initial_job"]:
            self.adjustment_clipboard = copy_adjustments(self.project, photo)
            self.status_message("已复制 " + self.adjustment_clipboard["name"] + " 的调整")

    def paste_photo_adjustments(self, photo_id):
        from .workflow_adjustment_clipboard import PasteAdjustmentsDialog, available_groups

        source = self.adjustment_clipboard
        if not source or source["photo_id"] == photo_id or self.tasks.busy:
            return
        self.end_history_gesture()
        if not self.flush_advanced():
            return
        target = self.project.photo(photo_id)
        if not target["initial_job"]:
            return

        def entry(photo):
            engine = self.project.color_engine(photo)
            return (photo["id"], Path(photo["base"]["raw"][0]).stem,
                    available_groups(source, photo, engine))

        checked = set(self.project.data.get("checked") or ())
        others = [entry(photo) for photo in self.project.photos
                  if photo["id"] in checked and photo["initial_job"]
                  and photo["id"] not in (photo_id, source["photo_id"])]
        dialog = PasteAdjustmentsDialog(self, source, entry(target), others)
        try:
            if dialog.exec() != dialog.DialogCode.Accepted:
                return
            groups = dialog.groups()
            updated = self.project.paste_adjustments(source, dialog.targets(), groups)
        except (OSError, ValueError) as error:
            self.error(error)
            return
        finally:
            dialog.deleteLater()

        if updated:
            self.refresh_photos()
            if self.project.data["current"] in updated:
                self.finish_selection()
                self.sync_controls()
                if groups & {"crop", "orientation", "geometry"}:
                    self.canvas.fit()
                self.schedule_preview()
        self.status_message(f"已粘贴调整到 {len(updated)} 张照片" if updated else "所选调整已一致")

    def reset_photo_adjustments(self, photo_id):
        if self.tasks.busy:
            return
        photo = self.project.photo(photo_id)
        if not photo["initial_job"]:
            return
        name = Path(photo["base"]["raw"][0]).stem
        if GlassMessageBox.question(
            self, "还原所有设置",
            f"将 {name} 还原到首次导入状态？这会还原各引擎的调整、裁切和测量选区，"
            "并从当前照片移除已应用的除尘。可撤回，已保存版本保留。",
            default=QMessageBox.StandardButton.No,
            button_labels={
                QMessageBox.StandardButton.Yes: "确认还原",
                QMessageBox.StandardButton.No: "取消",
            },
        ) != QMessageBox.StandardButton.Yes:
            return
        self.end_history_gesture()
        if not self.flush_advanced():
            return
        try:
            changed = self.project.reset_photo_adjustments(photo)
        except (OSError, ValueError) as error:
            self.error(error)
            return
        pending = self.flatfield_pending.pop(photo_id, None)
        self.flatfield_messages.pop(photo_id, None)
        if self.project.data["current"] == photo_id:
            self._flatfield_strength_photo = None
            self.sync_controls()
            if changed:
                self.finish_selection()
                self.crop_locator_cache = None
                self.snapshot_store.clear()
                self.canvas.fit()
                self.schedule_preview()
        if changed:
            self.refresh_photos()
        self.status_message(f"{name} 已还原到首次导入状态" if changed or pending
                            else f"{name} 已是首次导入状态")

    def remove_photo_from_list(self, photo_id, *, confirm=True):
        photo = next((p for p in self.project.photos if p["id"] == photo_id), None)
        if photo is None:
            return
        name = Path(photo["base"]["raw"][0]).stem
        if confirm and GlassMessageBox.question(
            self,
            "移除照片",
            f"从列表移除 {name}？已保存的版本仍在工作区，"
            "重新导入相同 RAW 可恢复其全部设置与版本。",
        ) != QMessageBox.StandardButton.Yes:
            return
        was_current = self.project.data["current"] == photo_id
        self.project.remove_photo(photo)
        self.refresh_photos()
        if self.project.data["current"]:
            # refresh_photos 在信号屏蔽区内重设当前项，需显式完成切换同步。
            if was_current:
                self.select_photo(self.project.data["current"])
            self.status_message(f"已从列表移除 {name}；重新导入相同 RAW 可恢复。")
        else:
            self.sync_controls()
            self.status_message("已移除全部照片；请导入新的 RAW 三色通道。")

    def photo_selected(self, item, previous=None):
        if item:
            self.select_photo(item.data(Qt.ItemDataRole.UserRole))

    def select_photo(self, identifier):
        self.end_history_gesture()
        if not self.flush_advanced():
            self.refresh_photos()
            return
        self.preview_timer.stop()
        self.import_preview_timer.stop()
        self.import_lens_context = None
        self.preview_reusable = False
        self.preview_timer.setInterval(450)
        self.preview_activity.reset()
        self.token += 1
        self.live.invalidate(self.token)
        self.snapshot_store.clear()
        self.shown_context = None
        self.__dict__.pop("last_display_records", None)
        self.show_preview_notice()
        self.project.data["current"] = identifier
        photo = self.project.photo()
        if not photo["initial_job"] and not self.tasks.busy and self.step not in (0, LIGHT_TABLE_STAGE):
            self.set_stage(0)
        self.project.enter_stage(self.step)
        self.finish_selection()
        self.current_record = None
        self.pending_actual = False
        self.canvas.current = self.canvas.reference = None
        self.canvas.empty_message = "选择 RAW 文件或素材文件夹开始"
        self.scope_panel.set_scope_data(None)
        self.canvas.clear_detail()
        self.canvas.set_output_size(None, None)
        self.canvas.update()
        self.sync_controls()
        if photo["initial_job"]:
            job = self.project.saved_job()
            if job:
                sample = self.project.workspace.sample(job)
                self.load_records(*self.saved_images(sample), reset=True)
            if self.project.dirty():
                self.preview_timer.start()
        self.schedule_import_preview()

    def import_preview_context(self):
        photo = self.current_photo()
        if self.closing or self.step != 0 or not photo or self.geometry_role:
            return None
        paths = (tuple(box.currentData() for box in self.roles)
                 if photo["source_mode"] == "rgb" else tuple(photo["base"]["raw"]))
        return self.token, self.display_revision, photo["id"], paths

    def schedule_import_preview(self):
        context = self.import_preview_context()
        if self.loading or context is None:
            return
        if not self.current_photo()["initial_job"]:
            self.canvas.empty_message = "正在生成导入预览…"
            self.canvas.update()
        if not self.import_lens_context or self.import_lens_context[0] != context:
            self.equipment_note.setText("正在核对机身与镜头参数…")
            self.lens_correction_button.setEnabled(False)
        self.import_preview_timer.start()

    def request_import_preview(self):
        context = self.import_preview_context()
        if context is None:
            return
        self._import_preview_request = getattr(self, "_import_preview_request", 0) + 1
        request_id = self._import_preview_request
        from .workflow_import_preview import import_preview
        from .raw_import import is_standard_import
        from .raw_lens_correction import camera_lens_info

        paths = context[-1]
        if not all(paths) or len(set(paths)) != len(paths):
            self.import_preview_failed(context, "请为 R/G/B 分别选择不同的 RAW。")
            return
        icc = self.display_icc
        needs_image = not self.current_photo()["initial_job"]
        policy = self.current_photo()["base"].get("raw_import")
        supported = is_standard_import(policy) and policy != "standard-mono-v1"

        def prepare():
            equipment = camera_lens_info(paths, supported)
            if policy == "standard-mono-v1":
                equipment = (equipment[0], False, "单色 RGB 暂不支持 RAW 内嵌镜头矫正。")
            try:
                image = import_preview(paths, icc) if needs_image else None
            except Exception:
                image = None
            return image, equipment

        work = Work(context, prepare)
        work.signals.done.connect(self.show_import_preview)
        work.signals.failed.connect(self.import_preview_failed)
        def start():
            if (request_id == self._import_preview_request
                    and context == self.import_preview_context()):
                self.pool.start(work)

        gate = getattr(self, "prewarm_gate", None)
        if gate is None or not gate.defer(start, lambda error: self.import_preview_failed(context, str(error))):
            start()

    def show_import_preview(self, context, result):
        if context != self.import_preview_context():
            return
        image, (description, available, detail) = result
        self.import_lens_context = context, available
        self.equipment_note.setText(description + "\n" + detail)
        self.lens_correction_button.setEnabled(available and not self.tasks.busy)
        if self.current_photo()["initial_job"]:
            return
        if image is None or image.isNull():
            self.import_preview_failed(context, "RAW 缩略图无法读取。")
            return
        self.canvas.sample_scale = 1
        self.canvas.split = False
        self.compare.setChecked(False)
        self.canvas.set_images(image, None, "导入预览 · 未校色", reset=True)

    def import_preview_failed(self, context, message):
        if context != self.import_preview_context():
            return
        if self.current_photo()["initial_job"]:
            self.equipment_note.setText("无法读取当前 RAW 的机身与镜头参数。")
            return
        self.canvas.current = self.canvas.reference = None
        self.canvas.empty_message = "无法显示导入预览，可继续核对 RAW"
        self.canvas.update()
        self.status_message(message)

    def change_lens_correction(self, enabled):
        photo = self.current_photo()
        if not photo or self.tasks.busy:
            return
        try:
            self.project.set_lens_correction(enabled)
            if photo["initial_job"]:
                self.save_version()
            else:
                self.status_message("镜头矫正设置已保存，核对并导入时应用。")
        except Exception as error:
            self.lens_correction_button.setChecked(bool(self.project.recipe().get("lens_correction")))
            self.error(error)

    def check_changed(self, item):
        key = item.data(Qt.ItemDataRole.UserRole)
        checked = set(self.project.data["checked"])
        if item.checkState() == Qt.CheckState.Checked:
            checked.add(key)
        else:
            checked.discard(key)
        self.project.data["checked"] = [
            p["id"] for p in self.project.photos if p["id"] in checked
        ]
        self.project.save()
        self.batch_note.setText(
            f"勾选 {len(checked)} 张；与当前查看分开"
            if checked
            else "未勾选时处理当前照片"
        )
        self.sync_controls()

    def input_extensions(self):
        if self.input_choice.currentData() != "mono_rgb":
            return {".rw2", ".dng"}
        from negpy.infrastructure.loaders.constants import (
            SUPPORTED_JPEG_EXTENSIONS, SUPPORTED_JXL_EXTENSIONS,
            SUPPORTED_RAW_EXTENSIONS, SUPPORTED_TIFF_EXTENSIONS,
        )

        return (SUPPORTED_RAW_EXTENSIONS - SUPPORTED_JPEG_EXTENSIONS
                - SUPPORTED_JXL_EXTENSIONS - SUPPORTED_TIFF_EXTENSIONS)

    def import_files(self):
        extensions = sorted(self.input_extensions())
        patterns = " ".join(f"*{suffix} *{suffix.upper()}" for suffix in extensions)
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择原始 RAW",
            str(Path.home() / "Pictures"),
            f"RAW ({patterns});;所有文件 (*)",
        )
        if paths:
            self.import_paths(paths, self.input_choice.currentData(), explicit_files=True)

    def import_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "选择素材文件夹", str(Path.home() / "Pictures")
        )
        if not path:
            return
        self.status_message("正在扫描素材文件夹…")
        QApplication.processEvents()
        extensions = self.input_extensions()
        paths = sorted(
            p
            for p in Path(path).rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )
        self.import_paths(paths, self.input_choice.currentData())

    def import_paths(self, paths, source_mode, *, explicit_files=False):
        if self.tasks.busy:
            self.error("请先完成或取消当前任务")
            return
        try:
            groups = {}
            ordered_paths = sorted(str(Path(p).resolve()) for p in paths)
            if source_mode == "mono_rgb" and explicit_files and len(ordered_paths) == 3:
                groups[None] = ordered_paths
            else:
                for path in ordered_paths:
                    groups.setdefault(
                        (str(Path(path).parent), Path(path).suffix.lower()), []
                    ).append(path)
            if not groups:
                raise ValueError("未找到支持的 RAW 文件")
            if source_mode == "bw":
                for values in groups.values():
                    for path in values:
                        self.project.add_photo([path], "bw")
                self.refresh_photos()
                self.set_stage(0)
                self.select_photo(self.project.data["current"])
                return
            pending = []
            skipped = {}
            for values in groups.values():
                usable = len(values) - len(values) % 3
                pending.extend(values[i : i + 3] for i in range(0, usable, 3))
                if usable != len(values):
                    directory = str(Path(values[0]).parent)
                    skipped[directory] = skipped.get(directory, 0) + len(values) - usable
            if skipped:
                summary = "、".join(
                    f"{Path(directory).name or directory} 剩 {count} 张"
                    for directory, count in skipped.items()
                )
                if pending:
                    GlassMessageBox.warning(
                        self,
                        "批量导入跳过未配组文件",
                        f"已跳过 {sum(skipped.values())} 张无法按三色配组的 RAW"
                        f"（{summary}），开始导入其余 {len(pending)} 组。",
                    )
                else:
                    GlassMessageBox.warning(
                        self,
                        "批量导入未找到三色组",
                        f"没有目录能凑出完整三色组：{summary}。"
                        "请让每个目录的 RAW 数量为 3 的倍数，或用「选择 RAW 文件」逐组选择。",
                    )
                    return
            if source_mode == "mono_rgb":
                for raw in pending:
                    self.project.add_photo(raw, source_mode)
                self.refresh_photos()
                self.set_stage(0)
                self.select_photo(self.project.data["current"])
                self.status_message("单色 RGB 已按文件名顺序分配 R/G/B，请核对后导入。")
            else:
                self.pending_probe = pending
                self.probe_next()
        except Exception as error:
            self.error(error)

    def probe_next(self):
        if self.pending_probe:
            raw = self.pending_probe.pop(0)
            policy = (
                "standard-cfa-dng-v1"
                if Path(raw[0]).suffix.lower() == ".dng"
                else "standard-crop-v1"
            )
            self.tasks.start(dict(action="probe", raw=raw, raw_import=policy))
            self.status_message("核对三色通道…")
        else:
            self.refresh_photos()
            self.set_stage(0)
            if self.project.data["current"]:
                self.select_photo(self.project.data["current"])
            self.status_message("通道核对完成；确认 R/G/B 后点击核对并导入。")
        self.sync_controls()

    def confirm_import(self):
        photo = self.current_photo()
        if not photo:
            return
        if photo["source_mode"] == "rgb":
            raw = [box.currentData() for box in self.roles]
            if len(set(raw)) != 3:
                self.error("通道角色重复，请保证 R/G/B 各一张")
                return
            if raw != photo["base"]["raw"]:
                recipe = self.project.recipe()
                flatfield = recipe.get("flatfield")
                source_mode = ("mono_rgb" if photo["base"].get("raw_import") == "standard-mono-v1"
                               else "rgb")
                photo = self.project.add_photo(raw, source_mode)
                if flatfield:
                    self.project.set_flatfield(flatfield, strength=recipe.get("flatfield_strength", 1.0))
                if recipe.get("lens_correction"):
                    self.project.set_lens_correction(True)
        if photo["initial_job"] and self.project.source_job(photo) is not None:
            self.set_stage(1)
            self.status_message("当前源文件与平场已准备完成；调整参数自动保存，可继续测量与调色。")
            return
        self.save_version(import_stage=self.step)

    def engine_changed(self):
        if self.loading or not self.current_photo():
            return
        if self.step != 3 or self.tasks.busy:
            self.sync_controls()
            return
        if self.remind_pending_drawing():
            self.sync_controls()
            return
        desired = self.engine_choice.currentData()
        self.end_history_gesture()
        if not self.flush_advanced():
            self.sync_controls()
            return
        try:
            if desired == "nikon":
                from .nikon_research_workflow import checked_nikon_resources

                checked_nikon_resources()
            self.preview_timer.stop()
            self.preview_reusable = False
            self.token += 1
            self.live.invalidate(self.token)
            self.project.select_engine(desired)
            self.mark_preview_pending()
            self.sync_controls()
            self.refresh_photos()
            if self.current_photo()["initial_job"]:
                job = self.project.saved_job()
                if job and not self.project.dirty():
                    sample = self.project.workspace.sample(job)
                    self.load_records(*self.saved_images(sample))
                else:
                    self.request_preview()
        except Exception as error:
            self.error(error)
            self.sync_controls()

    def profile_changed(self):
        if self.loading or not self.current_photo() or self.tasks.busy:
            return
        photo = self.current_photo()
        if photo["engine"] != "native" or photo["source_mode"] == "bw":
            self.sync_controls()
            return
        document = self.native_film.value()
        if not self.flush_advanced():
            self.sync_controls()
            return
        try:
            recipe = self.project.recipe()
            if document is None:
                recipe.pop("native_film", None)
            else:
                recipe["native_film"] = document
            self.project.set_recipe(recipe)
            self.sync_controls()
            self.schedule_preview()
            self.status_message("胶片密度配置已更新，正在重新测量并生成预览…")
        except Exception as error:
            self.error(error)
            self.sync_controls()

    def color_slider_changed(self, spec, value):
        photo = self.current_photo()
        if self.loading or self.tasks.busy or not photo or not photo["initial_job"] or photo["source_mode"] != "rgb":
            return
        if not self.flush_advanced():
            self.sync_controls()
            return
        recipe = self.project.recipe()
        mode = RESULT_MODES[photo["engine"]]
        current = recipe.setdefault("post_controls", {}).setdefault(mode, {"version": 1, "values": {}})
        if value == spec.default:
            current["values"].pop(spec.key, None)
        else:
            current["values"][spec.key] = value
        self.project.set_recipe(recipe, label="调整" + spec.title.split("（")[0])
        self.sync_controls()
        self.schedule_preview()

    def tone_changed(self, *args):
        if self.loading or not self.current_photo() or self.tasks.busy:
            return
        desired = dict(
            exposure=self.exposure.value(),
            contrast=self.contrast.value(),
            paper=self.paper.currentData(),
            highlight=self.highlight.isChecked(),
            highlight_strength=self.highlight_strength.value(),
            balance={key: widget.value() for key, widget in self.balance.items()},
            fixed=self.fixed.isChecked(),
            fixed_strength=self.fixed_strength.value(),
        )
        if not self.flush_advanced():
            self.sync_controls()
            self.status_message("请先修正左侧高级参数组合。")
            return
        photo = self.project.photo()
        highlight_keys = (
            "suspended_highlight_controls", "highlight_reference_origin",
            "highlight_reference_drawn_roi", "highlight_reference_aligned",
        )
        previous_highlight_state = {
            key: deepcopy(photo[key]) for key in highlight_keys if key in photo
        }
        committed = False
        try:
            recipe = self.project.recipe()
            engine, bw = photo["engine"], photo["source_mode"] == "bw"
            recipe["exposure_ev"] = desired["exposure"]
            if engine == "native":
                if desired["paper"]:
                    recipe["paper"] = desired["paper"]
                values = recipe.setdefault(
                    "native_controls", dict(version=1, values={})
                )["values"]
                if desired["contrast"] != 115 or "exposure.grade" in values:
                    values["exposure.grade"] = desired["contrast"]
                if desired["highlight"]:
                    roi = recipe.get("roi") or photo["analysis_roi"]
                    if roi is None:
                        raise ValueError("请先完成源准备，再启用高光找回")
                    high = recipe.get("highlight_controls")
                    if high is None:
                        high = photo.pop("suspended_highlight_controls", None)
                        if high is None:
                            high = dict(version=1, reference_roi=list(roi), controls={})
                            photo["highlight_reference_origin"] = "automatic"
                            photo.pop("highlight_reference_drawn_roi", None)
                            photo.pop("highlight_reference_aligned", None)
                        else:
                            high = deepcopy(high)
                            desired["highlight_strength"] = high["controls"]["strength"] * 100
                        recipe["highlight_controls"] = high
                    elif desired["highlight_strength"] == 0:
                        # A zero-strength recipe canonicalizes to disabled; keep
                        # the last active settings and source-coordinate masks.
                        photo["suspended_highlight_controls"] = deepcopy(high)
                    high["controls"]["strength"] = desired["highlight_strength"] / 100
                else:
                    high = recipe.pop("highlight_controls", None)
                    if high is not None:
                        photo["suspended_highlight_controls"] = deepcopy(high)
            if not bw:
                mode = RESULT_MODES[engine]
                values = deepcopy(
                    recipe.get("output_controls", {}).get(mode, {}).get("values", {})
                )
                values.update(desired["balance"])
                if engine != "native":
                    values["contrast"] = desired["contrast"]
                recipe.setdefault("output_controls", {})[mode] = dict(
                    version=1, values=values
                )
            if engine == "nikon":
                strength = desired["fixed_strength"] / 100
                photo["fixed_last_strength"] = strength
                recipe["nikon_research_controls"] = nikon_controls(
                    strength if desired["fixed"] else 0
                )
            self.token += 1
            self.live.invalidate(self.token)
            self.set_preview_updating(False)
            self.project.set_recipe(recipe)
            committed = True
            if engine == "native":
                self.project.prepare_highlight_reference()
            self.sync_controls()
            self.mark_preview_pending()
            self.preview_timer.start()
        except Exception as error:
            if not committed:
                for key in highlight_keys:
                    photo.pop(key, None)
                photo.update(previous_highlight_state)
            self.error(error)
            self.sync_controls()

    def highlight_effect_feather_changed(self, value):
        if self.loading or self.tasks.busy or not self.current_photo():
            return
        if not self.flush_advanced():
            self.sync_controls()
            return
        try:
            recipe = self.project.recipe()
            effect_area = recipe.get("highlight_controls", {}).get("effect_area")
            if effect_area is None:
                return
            effect_area["feather_pixels"] = value
            self.project.set_recipe(recipe, label="调整高光作用区羽化")
            self.sync_controls()
            self.schedule_preview()
        except Exception as error:
            self.error(error)
            self.sync_controls()

    def clear_highlight_effect(self):
        if self.loading or self.tasks.busy or not self.current_photo():
            return
        if not self.flush_advanced():
            self.sync_controls()
            return
        try:
            recipe = self.project.recipe()
            high = recipe.get("highlight_controls")
            if not high or not high.get("effect_area"):
                return
            high.pop("effect_area")
            self.project.set_recipe(recipe, label="恢复高光全图作用")
            self.sync_controls()
            self.refresh_selection_preview()
            self.schedule_preview()
            self.status_message("高光找回已恢复全图作用。")
        except Exception as error:
            self.error(error)
            self.sync_controls()

    def meter_recipe(self):
        recipe = self.project.recipe()
        photo = self.project.photo()
        if photo["engine"] != "native" or not photo["initial_job"]:
            raise ValueError("请在第3步完成 NegPy 密度测量")
        if recipe["roi"] is None:
            recipe["roi"] = list(photo["analysis_roi"])
        previous = recipe.get("native_spatial_controls")
        values = recipe.get("native_controls", {}).get("values", {})
        if any(
            values.get("exposure." + field, 0)
            for field in ("contrast_mask_strength", "diffusion")
        ):
            raise ValueError("启用分区前，请在高级印相参数中将反差遮罩与扩散复位。")
        if values.get("lab.sharpen_auto_masking", False):
            raise ValueError("启用分区前，请在高级细节参数中关闭自动颗粒蒙版。")
        meter = deepcopy(previous["metering"]) if previous else metering_document()
        recipe["native_spatial_controls"] = spatial_document(
            meter, self.avoidance.currentData(), photo["protected_rois"], previous
        )
        # Existing non-spatial detail edits must follow the spatial render route.
        from .native_spatial_v30 import LAB_FIELDS

        values = recipe.get("native_controls", {}).get("values", {})
        for key in list(values):
            if key.startswith("lab.") and key[4:] in LAB_FIELDS:
                recipe["native_spatial_controls"]["lab"][key[4:]] = values.pop(key)
        return recipe

    def avoidance_changed(self):
        if (
            self.loading
            or not self.current_photo()
            or not self.current_photo()["initial_job"]
        ):
            return
        desired_action = self.avoidance.currentData()
        if not self.flush_advanced():
            self.sync_controls()
            self.status_message("请先修正左侧高级参数组合。")
            return
        self.loading = True
        self.avoidance.setCurrentIndex(self.avoidance.findData(desired_action))
        self.loading = False
        try:
            self.project.set_recipe(self.meter_recipe())
            self.sync_controls()
            if (desired_action != "off" and not self.meter_grid_hidden
                    and self.geometry_role in (None, "inspect", "meter_grid")):
                self.geometry_role = "meter_grid"
            self.refresh_selection_preview()
            self.schedule_preview()
            self.status_message("异常策略已更新，正在准备测量预览…")
        except Exception as error:
            self.error(error)

    def apply_metering(self):
        if not self.flush_advanced():
            self.status_message("请先修正左侧高级参数组合。")
            return
        try:
            self.project.set_recipe(self.meter_recipe())
            self.sync_controls()
            if not self.meter_grid_hidden and self.geometry_role in (None, "inspect", "meter_grid"):
                self.begin_selection("meter_grid")
            else:
                self.refresh_selection_preview()
            self.request_preview()
        except Exception as error:
            self.error(error)

    def show_meter_grid(self):
        if self.geometry_role == "meter_grid":
            self.leave_selection()
            return
        self.meter_grid_hidden = False
        if not self.project.recipe().get("native_spatial_controls"):
            self.apply_metering()
            return
        self.begin_selection("meter_grid")

    def reset_meter_grid(self):
        if self.tasks.busy or not self.flush_advanced():
            return
        try:
            recipe = self.meter_recipe()
            meter = recipe["native_spatial_controls"]["metering"]
            defaults = metering_document()
            for key in ("zones", "blocks_per_zone"):
                meter[key] = defaults[key]
            self.selection.selected_zone = None
            self.project.set_recipe(recipe, label="复位分区网格")
            self.sync_controls()
            self.refresh_selection_preview()
            self.request_preview()
        except Exception as error:
            self.error(error)

    def meter_zone_state(self):
        from .metering_hover_v45 import grid_hover

        selected = self.selection.selected_zone
        grid = self.selection.hover_model.get("grid")
        if self.geometry_role != "meter_grid" or not selected or not grid:
            return None
        a, b, c, d = selected["bounds"]
        zone = grid_hover(grid, ((a + c) / 2, (b + d) / 2))
        if not zone or zone.get("bounds") != selected["bounds"]:
            return None
        recipe = self.project.recipe()
        spatial = recipe["native_spatial_controls"]
        meter = spatial["metering"]
        shape = dict(type="rectangle", bounds=zone["bounds"])
        manual = shape in meter["exclude"]
        receipt = self.project.metering_receipt()
        padding = meter["padding"]
        automatic = grid["fresh"] and any(
            a < right + padding and left - padding < c
            and b < bottom + padding and top - padding < d
            for left, top, right, bottom in receipt.get("report", {}).get("automatic_exclusions", [])
        )
        return zone, shape, manual, automatic

    def open_meter_zone_menu(self, _zone, position):
        self.select_meter_zone()
        self.zone_menu.popup(position)

    def select_meter_zone(self):
        state = self.meter_zone_state()
        available = bool(state) and not self.tasks.busy
        if not state:
            self.selection.selected_zone = None
            self.zone_note.setText("点击网格选择分区")
            self.zone_note.setToolTip("")
            self.zone_exclude_action.setEnabled(False)
            self.zone_restore_action.setEnabled(False)
            self.zone_exclude_action.setChecked(False)
            self.zone_restore_action.setChecked(False)
            return
        zone, _, manual, automatic = state
        status = "已排除" if manual else "部分或全部自动排除" if automatic else zone["status"]
        self.zone_note.setText(f"第 {zone['row']} 行 · 第 {zone['column']} 列 · {status}")
        self.zone_note.setToolTip("\n".join(zone["details"]))
        has_support = zone.get("supported_pixels", 1) > 0
        self.zone_exclude_action.setEnabled(available and not manual and has_support)
        self.zone_restore_action.setEnabled(available and (manual or automatic))
        self.zone_exclude_action.setChecked(manual or (automatic and not has_support))
        self.zone_restore_action.setChecked(has_support and not manual and not automatic)

    def change_meter_zone(self, *, exclude):
        self.zone_menu.close()
        if self.tasks.busy or not self.flush_advanced():
            return
        state = self.meter_zone_state()
        if not state:
            return
        zone, shape, manual, automatic = state
        recipe = self.project.recipe()
        spatial = recipe["native_spatial_controls"]
        meter = spatial["metering"]
        protections = deepcopy(self.project.photo()["protected_rois"])
        action = spatial.get("automatic_avoidance", {}).get("action", "off")
        if exclude and not manual:
            meter["exclude"].append(shape)
        elif not exclude and (manual or automatic):
            if manual:
                meter["exclude"].remove(shape)
            if action == "exclude" and zone["bounds"] not in protections:
                protections.append(zone["bounds"])
        else:
            return
        photo = self.project.photo()
        previous_protections = photo["protected_rois"]
        try:
            recipe["native_spatial_controls"] = spatial_document(meter, action, protections, spatial)
            photo["protected_rois"] = protections
            try:
                self.project.set_recipe(recipe, label="排除密度分区" if exclude else "恢复密度分区")
            except Exception:
                photo["protected_rois"] = previous_protections
                raise
            self.last_error = None
            self.refresh_selection_preview()
            self.request_preview()
        except Exception as error:
            self.error(error)

    def selection_shape_changed(self):
        if self.loading or not self.geometry_role:
            return
        if self.geometry_role in ("include", "exclude", "highlight_protect"):
            self.selection.shape_kind = self.shape_choice.currentData()
            self.selection.points = []
            if self.selection.temporary is not None:
                self.selection.scene().removeItem(self.selection.temporary)
                self.selection.temporary = None
            self.selection.temporary_shape = None
            self.refresh_selection_preview()
            self.schedule_preview()

    def selection_overlays(self):
        photo, recipe, role = self.project.photo(), self.project.recipe(), self.geometry_role
        overlays, grid = [], None
        if role in ("include", "exclude", "protect", "inspect", "meter_grid"):
            spatial = recipe.get("native_spatial_controls", {})
            meter = spatial.get("metering", {})
            for key, color in (("include", "#62ccdf"), ("exclude", "#ed9b80")):
                overlays.extend((shape, color) for shape in meter.get(key, []))
            if not meter.get("include"):
                overlays.append((dict(type="rectangle", bounds=recipe.get("roi") or photo["analysis_roi"]), "#62ccdf"))
            overlays.extend((dict(type="rectangle", bounds=roi), "#a3d7b9") for roi in photo["protected_rois"])
            receipt = self.project.metering_receipt()
            matches = bool(receipt and receipt.get("roi") == recipe.get("roi")
                           and receipt.get("metering") == spatial.get("metering")
                           and receipt.get("automatic") == spatial.get("automatic_avoidance"))
            automatic = receipt.get("report", {}) if matches else {}
            applied = automatic.get("automatic_exclusions", [])
            color = "#ed9b80" if applied else "#e7c06e"
            overlays.extend((dict(type="rectangle", bounds=roi), color) for roi in automatic.get("candidate_exclusions", []))
            if role != "inspect":
                grid = (recipe.get("roi") or photo["analysis_roi"], meter, applied)
            from .metering_hover_v45 import build_metering_hover

            shown = self.shown_context or {}
            self.selection.set_hover_model(build_metering_hover(
                recipe, receipt,
                shown.get("recipe") if shown.get("photo") == photo["id"] else None,
                recipe.get("roi") or photo["analysis_roi"], photo["protected_rois"],
                show_grid=role != "inspect", review_grid=role == "meter_grid",
            ))
            if role == "meter_grid":
                self.select_meter_zone()
        elif role == "crop" and recipe.get("output_roi"):
            overlays.append((dict(type="rectangle", bounds=recipe["output_roi"]), "#f0a055"))
        elif role in ("highlight", "highlight_effect", "highlight_protect"):
            high = recipe.get("highlight_controls", {})
            reference_roi = high.get("reference_roi")
            if high.get("automatic_reference"):
                reference_roi = self.project.metering_receipt().get("highlight_reference_roi")
            if reference_roi:
                overlays.append((dict(type="rectangle", bounds=reference_roi), "#62ccdf"))
            overlays.extend((shape, "#f0a055") for shape in (high.get("effect_area") or {}).get("include", []))
            overlays.extend((shape, "#a3d7b9") for shape in (high.get("protection") or {}).get("include", []))
        if role not in ("include", "exclude", "protect", "inspect", "meter_grid"):
            self.selection.set_hover_model(None)
        return overlays, grid

    def refresh_selection_preview(self):
        """Keep source-coordinate masks on the currently displayed positive frame."""
        photo, context = self.current_photo(), self.shown_context
        if self.geometry_role in (None, "import_roi") or not photo or not photo.get("source_shape"):
            return False
        if (self.geometry_role == "crop" or
                (self.geometry_role in ("highlight", "highlight_effect", "highlight_protect")
                 and getattr(self, "highlight_full_source", False))):
            # Keep the complete source locator while the rendered canvas is
            # cropped/straightened; otherwise the next preview would hide
            # everything outside the existing crop and prevent expanding it.
            from .workflow_crop_preview import refresh_crop_frame

            return refresh_crop_frame(self)
        if self.selection.crop_handles.bounds is not None:
            self.selection.crop_handles.configure(None, 0)
        if (self.geometry_role == "meter_grid"
                and self.selection.source_photo_id == photo["id"]
                and self.selection.geometry_info
                and not self.selection.geometry_info.get("preview_kind")):
            overlays, grid = self.selection_overlays()
            return self.selection.refresh_source_overlays(overlays, grid)
        image = self.canvas.current
        if image is None or image.isNull() or not context or context.get("photo") != photo["id"]:
            return False
        from .viewport_preview_v43 import output_size_for_recipe
        from .workspace import output_rectangle

        shown_recipe = context["recipe"]
        if output_size_for_recipe(shown_recipe, photo["source_shape"], photo["analysis_roi"], RESULT_MODES[context["engine"]]) is None:
            # General finishing geometry has no exact source inverse here.
            # Its existing full-source positioning view remains authoritative.
            return False
        shown_rectangle = output_rectangle(shown_recipe, photo["analysis_roi"], photo["source_shape"])
        if self.geometry_role == "meter_grid":
            a, b, c, d = self.project.recipe()["roi"] or photo["analysis_roi"]
            desired_rectangle = output_rectangle(
                self.project.recipe(), photo["analysis_roi"], photo["source_shape"]
            )
            if not all(x0 <= a < c <= x1 and y0 <= b < d <= y1
                       for x0, y0, x1, y1 in (shown_rectangle, desired_rectangle)):
                return False
        overlays, grid = self.selection_overlays()
        self.selection.set_preview(
            image, photo["source_shape"],
            output_roi=shown_rectangle,
            output_orientation=shown_recipe.get("output_orientation"),
            overlays=overlays, grid=grid, preserve_view=True,
        )
        self.selection.source_photo_id = photo["id"]
        self.selection.drawing_enabled = self.geometry_role not in ("inspect", "meter_grid")
        self.selection.shape_kind = (self.shape_choice.currentData()
                                     if self.geometry_role in ("include", "exclude", "highlight_protect") else "rectangle")
        if self.geometry_role == "meter_grid":
            self.canvas_stack.setCurrentWidget(self.selection)
            self.show_region_exit()
        return True

    def begin_selection(self, role):
        if not self.flush_advanced():
            self.status_message("请先修正左侧高级参数组合。")
            return
        try:
            photo = self.project.photo()
            if not photo["initial_job"] or self.tasks.busy:
                raise ValueError("请先完成源准备，并等待当前处理结束")
            if role != self.geometry_role:
                self.selection.cancel_drawing()
                if (self.geometry_role in ("highlight", "highlight_effect", "highlight_protect")
                        and getattr(self, "highlight_full_source", False)):
                    self.cancel_highlight_locator()
            if role == "crop":
                from .workflow_crop_preview import begin_crop_selection

                try:
                    begin_crop_selection(self)
                except Exception as error:
                    self.preview_failed(self.token, str(error))
                return
            if role in ("highlight", "highlight_effect", "highlight_protect"):
                self.highlight_full_source = False
            previous_role, self.geometry_role = self.geometry_role, role
            if self.refresh_selection_preview():
                self.canvas_stack.setCurrentWidget(self.selection)
                self.show_region_exit()
                self.status_message("格内为 RGB 平均光密度；悬停查看 RGB 分量、范围及有效样本。"
                                    if role == "meter_grid" else
                                    "框选需要找回高光的区域。橙色为作用区，青色为参考区，绿色为保护区。"
                                    if role == "highlight_effect" else
                                    "在画面上绘制高光参考。青色为参考区，绿色为保护区。"
                                    if role == "highlight" else
                                    "在当前正像上绘制；选区和异常策略自动更新预览，可连续绘制。")
                return
            if role in ("highlight", "highlight_effect", "highlight_protect"):
                self.highlight_full_source = True
                self.canvas_stack.setCurrentWidget(self.canvas)
                self.show_region_exit()
                from .workflow_crop_preview import request_crop_frame

                try:
                    request_crop_frame(self)
                except Exception as error:
                    self.preview_failed(self.token, str(error))
                return
            self.geometry_role = previous_role
            source_job = self.project.source_job()
            if source_job is None:
                raise ValueError("当前平场／源配置尚未重新处理完成。")
            path = (
                self.project.workspace.root
                / "runs"
                / source_job["id"]
                / "geometry.json"
            )
            geometry = photo.get("geometry") or (json.loads(path.read_text()) if path.exists() else {})
            if not geometry.get("positive_preview") or not Path(geometry["positive_preview"]).is_file():
                manifest = path.with_name("manifest.json")
                self.pending_geometry = dict(
                    photo=photo["id"], role=role, stage=self.step
                )
                if self.live.process is not None:
                    self.stop_preview()
                self.tasks.start(
                    dict(
                        action="geometry_prepare",
                        manifest=str(manifest),
                        manifest_sha256=sha256(manifest),
                        directory=str(
                            self.project.workspace.root
                            / "geometry-cache"
                            / uuid.uuid4().hex
                        ),
                    )
                )
                self.status_message("正在准备完整源画幅的正像定位图，当前画面保留…")
                self.sync_controls()
                return
            self.geometry_role = role
            overlays, grid = self.selection_overlays()
            self.selection.set_geometry(geometry, overlays, grid=grid, source_id=photo["id"], positive=True)
            self.selection.drawing_enabled = role not in ("inspect", "meter_grid")
            self.selection.shape_kind = (
                self.shape_choice.currentData()
                if role in ("include", "exclude", "highlight_protect")
                else "rectangle"
            )
            self.canvas_stack.setCurrentWidget(self.selection)
            self.show_region_exit()
            self.status_message(
                "完整源分区网格；点击选择分区，悬停查看 RGB 光密度。"
                if role == "meter_grid" else
                "在完整源画面框选需要找回高光的区域。橙色为作用区，青色为参考区，绿色为保护区。"
                if role == "highlight_effect" else
                "在完整源画面绘制高光参考。青色为参考区，绿色为保护区。"
                if role == "highlight" else
                "使用完整源画幅正像定位；选区仍自动生效。"
            )
        except Exception as error:
            self.error(error)

    def toggle_highlight_frame(self):
        if self.geometry_role not in ("highlight", "highlight_effect", "highlight_protect"):
            return
        self.selection.cancel_drawing()
        if not getattr(self, "highlight_full_source", False):
            self.highlight_full_source = True
            self.canvas_stack.setCurrentWidget(self.canvas)
            self.show_region_exit()
            from .workflow_crop_preview import request_crop_frame

            try:
                request_crop_frame(self)
            except Exception as error:
                self.preview_failed(self.token, str(error))
            return
        self.highlight_full_source = False
        if not self.refresh_selection_preview():
            self.highlight_full_source = True
            self.show_region_exit()
            self.status_message("当前画面的几何变换不能精确映射选区，请使用完整源画面。")
            return
        self.cancel_highlight_locator()
        self.canvas_stack.setCurrentWidget(self.selection)
        self.show_region_exit()
        shown, desired = self.shown_context or {}, self.preview_context() or {}
        if any(shown.get(key) != desired.get(key) for key in ("photo", "engine", "recipe_sha256", "repair_plan_sha256")):
            self.preview_timer.start()

    def cancel_highlight_locator(self):
        pending = getattr(self, "crop_locator_pending", None)
        self.crop_locator_pending = None
        if pending and pending.get("token") == self.token:
            self.token += 1
            self.live.invalidate(self.token)
            if self.preview_activity.active and self.preview_activity.token == pending["token"]:
                self.preview_activity.finish(pending["token"], message="已取消完整画面预览")
            self.set_preview_updating(False)

    def show_region_exit(self):
        self.sync_drawing_hint()
        self.region_done.setText({
            "crop": "完成裁切",
            "meter_grid": "关闭分区网格",
            "import_roi": "退出导入画面定位",
        }.get(self.geometry_role, "退出绘制"))
        self.region_done.show()
        highlight_role = self.geometry_role in ("highlight", "highlight_effect", "highlight_protect")
        self.highlight_frame_toggle.setVisible(highlight_role)
        self.highlight_frame_toggle.setText(
            "返回当前画面" if getattr(self, "highlight_full_source", False) else "显示完整源画面"
        )
        self.region_done.set_attention(True)
        self.zone_note.setVisible(self.geometry_role == "meter_grid")
        self.meter_grid_button.setText(
            "关闭分区网格与数值" if self.geometry_role == "meter_grid" else "显示分区网格与数值"
        )
        if self.geometry_role != "meter_grid":
            self.zone_menu.close()

    def leave_selection(self):
        was_crop = self.geometry_role == "crop"
        was_import = self.geometry_role == "import_roi"
        if self.geometry_role == "meter_grid":
            self.meter_grid_hidden = True
        self.finish_selection()
        if was_crop:
            self.preview_timer.stop()
            self.request_preview()
        elif was_import:
            self.set_stage(0)
            self.schedule_import_preview()

    def finish_selection(self):
        was_crop = self.geometry_role == "crop"
        was_full_highlight = (self.geometry_role in ("highlight", "highlight_effect", "highlight_protect")
                              and getattr(self, "highlight_full_source", False))
        if was_full_highlight:
            self.cancel_highlight_locator()
        self.highlight_full_source = False
        self.__dict__.pop("_import_roi_context", None)
        self.geometry_role = None
        if hasattr(self, "canvas_stack"):
            self.selection.cancel_drawing()
            self.selection.crop_handles.configure(None, 0)
            if self._straighten_panel is not None:
                self._exit_straighten_draw(message=None)
            self.canvas_stack.setCurrentWidget(self.canvas)
            self.sync_drawing_hint()
            self.region_done.set_attention(False)
            self.region_done.setText("退出绘制")
            self.meter_grid_button.setText("显示分区网格与数值")
            self.region_done.hide()
            self.highlight_frame_toggle.hide()
            self.zone_note.hide()
            self.zone_menu.close()
            self.selection.selected_zone = None
        if was_crop or was_full_highlight:
            self.crop_locator_pending = None
            photo = self.current_photo()
            shown, desired = self.shown_context or {}, self.preview_context() or {}
            if (not self.closing and photo and photo.get("initial_job")
                    and any(shown.get(key) != desired.get(key)
                            for key in ("photo", "engine", "recipe_sha256", "repair_plan_sha256"))):
                self.preview_timer.start()

    def shape_selected(self, shape):
        if self.tasks.busy or not self.geometry_role or self.geometry_role == "inspect":
            return
        if not self.flush_advanced():
            self.status_message("请先修正左侧高级参数组合。")
            return
        photo = None
        previous_highlight_state = None
        committed = False
        try:
            role = self.geometry_role
            bounds = shape_bounds(shape)
            if role == "import_roi":
                from .workflow_import_roi import accept_import_roi

                accept_import_roi(self, bounds)
                return
            photo = self.project.photo()
            if role in ("highlight", "highlight_effect", "highlight_protect"):
                previous_highlight_state = {
                    key: deepcopy(photo[key]) for key in (
                        "suspended_highlight_controls", "highlight_reference_origin",
                        "highlight_reference_drawn_roi", "highlight_reference_aligned",
                    ) if key in photo
                }
            if role in ("include", "exclude", "protect"):
                recipe = self.meter_recipe()
                meter = recipe["native_spatial_controls"]["metering"]
                if role == "include":
                    recipe["roi"] = bounds
                    meter["include"] = [shape]
                elif role == "exclude":
                    meter["exclude"].append(shape)
                else:
                    photo["protected_rois"].append(bounds)
                recipe["native_spatial_controls"] = spatial_document(
                    meter,
                    self.avoidance.currentData(),
                    photo["protected_rois"],
                    recipe["native_spatial_controls"],
                )
            else:
                recipe = self.project.recipe()
                if role == "crop":
                    recipe.update(crop=True, output_roi=bounds)
                else:
                    high = recipe.get("highlight_controls")
                    if high is None:
                        high = photo.pop("suspended_highlight_controls", None)
                        if high is None:
                            high = dict(
                                version=1,
                                reference_roi=recipe.get("roi") or photo["analysis_roi"],
                                controls=dict(strength=self.highlight_strength.value() / 100),
                            )
                            photo["highlight_reference_origin"] = "automatic"
                            photo.pop("highlight_reference_drawn_roi", None)
                            photo.pop("highlight_reference_aligned", None)
                        else:
                            high = deepcopy(high)
                        recipe["highlight_controls"] = high
                    if role == "highlight":
                        aligned = fit_manual_highlight_reference(
                            recipe, photo["source_shape"], bounds, self.project.metering_receipt()
                        )
                        high.pop("automatic_reference", None)
                        high["reference_roi"] = aligned
                        photo["highlight_reference_origin"] = "manual"
                        photo["highlight_reference_drawn_roi"] = bounds
                        photo["highlight_reference_aligned"] = aligned != bounds
                    elif role == "highlight_effect":
                        if high.get("effect_area") is None:
                            high["effect_area"] = dict(include=[], feather_pixels=64)
                        high["effect_area"]["include"].append(shape)
                    else:
                        if high.get("protection") is None:
                            high["protection"] = dict(include=[], feather_pixels=0)
                        high["protection"]["include"].append(shape)
            self.project.set_recipe(recipe, label={
                "crop": "调整裁切范围", "include": "调整密度选区", "exclude": "添加测量避让",
                "protect": "添加保护区", "highlight": "调整高光参考区",
                "highlight_effect": "添加高光作用区", "highlight_protect": "添加高光保护区",
            }[role])
            committed = True
            self.sync_controls()
            if (
                role == "highlight"
                and aligned != bounds
                and self.selection.temporary is not None
            ):
                self.selection.temporary.setPath(
                    self.selection.path(dict(type="rectangle", bounds=aligned))
                )
            # The committed source mask now owns this outline. A background
            # positive refresh should preserve only a stroke still being drawn.
            if self.selection.temporary is not None:
                self.selection.scene().removeItem(self.selection.temporary)
                self.selection.temporary = None
            self.selection.temporary_shape = None
            if role == "include":
                self.meter_grid_hidden = False
                self.finish_selection()
                self.begin_selection("meter_grid")
                self.schedule_preview()
                return
            self.refresh_selection_preview()
            self.schedule_preview()
            self.region_done.remind()
            self.status_message(
                "高光参考已按分析网格收边，正在更新预览。"
                if role == "highlight" and aligned != bounds else
                "选区已生效，正在更新正像；可继续绘制，无需确认。"
            )
        except Exception as error:
            if previous_highlight_state is not None and not committed:
                for key in (
                    "suspended_highlight_controls", "highlight_reference_origin",
                    "highlight_reference_drawn_roi", "highlight_reference_aligned",
                ):
                    photo.pop(key, None)
                photo.update(previous_highlight_state)
                self.selection.cancel_drawing()
            self.error(error)

    def clear_exclusions(self):
        if not self.flush_advanced():
            self.status_message("请先修正左侧高级参数组合。")
            return
        try:
            recipe = self.meter_recipe()
            meter = recipe["native_spatial_controls"]["metering"]
            meter.update(exclude=[], excluded_rois=[])
            self.project.photo()["protected_rois"] = []
            recipe["native_spatial_controls"] = spatial_document(
                meter,
                self.avoidance.currentData(),
                previous=recipe["native_spatial_controls"],
            )
            self.project.set_recipe(recipe)
            self.sync_controls()
            self.refresh_selection_preview()
            self.schedule_preview()
        except Exception as error:
            self.error(error)

    def confirm_candidates(self):
        if self.tasks.busy or not self.current_photo():
            return
        if not self.flush_advanced():
            self.status_message("请先修正左侧高级参数组合。")
            return
        try:
            self.project.confirm_candidates()
            self.sync_controls()
            self.refresh_selection_preview()
            self.schedule_preview()
            self.status_message("候选已加入手工排除草稿，正在更新预览…")
        except Exception as error:
            self.error(error)

    def sync_crop_controls(self):
        photo = self.current_photo()
        ready = bool(photo and photo["initial_job"] and not self.tasks.busy)
        for panel in getattr(self, "crop_panels", {}).values():
            panel.setEnabled(ready)
            if photo:
                panel.set_recipe(self.project.recipe(), photo["engine"])

    def apply_auto_crop(self, mode):
        from .workflow_auto_crop import apply_auto_crop

        apply_auto_crop(self, mode)

    def crop_post_changed(self, angle, crop_to_valid=None):
        if self.loading or self.tasks.busy or not self.current_photo():
            return
        if not self.flush_advanced():
            self.status_message("请先修正左侧高级参数组合。")
            return
        from .workflow_crop_ui import straighten_recipe

        try:
            current = self.project.recipe()
            engine = self.current_photo()["engine"]
            if crop_to_valid is None:
                values = current.get("post_controls", {}).get(RESULT_MODES[engine], {}).get("values", {})
                crop_to_valid = values.get("geometry.crop_to_valid", False)
            recipe = straighten_recipe(current, engine, angle, crop_to_valid)
            if recipe == current:
                return
            if self.geometry_role == "crop":
                self.project.set_recipe(recipe, label="拉直")
                self.sync_controls()
                self.refresh_selection_preview()
                return
            self.finish_selection()
            self.canvas.fit()
            self.preview_timer.stop()
            self.token += 1
            self.live.invalidate(self.token)
            self.project.set_recipe(recipe, label="拉直")
            self.sync_controls()
            self.mark_preview_pending()
            self.schedule_preview()
        except Exception as error:
            self.error(error)
            self.sync_crop_controls()
            self.refresh_selection_preview()

    def clear_crop(self):
        if not self.flush_advanced():
            self.status_message("请先修正左侧高级参数组合。")
            return
        if self.current_photo():
            recipe = self.project.recipe()
            recipe["crop"] = False
            recipe.pop("output_roi", None)
            self.project.set_recipe(recipe, label="取消输出裁切")
            self.sync_controls()
            if self.geometry_role == "crop":
                self.selection.points = []
                self.selection.temporary_shape = None
            self.refresh_selection_preview()
            self.schedule_preview()

    def save_version(self, *, import_stage=None):
        if self.tasks.busy or not self.current_photo():
            return
        if not self.flush_advanced():
            self.status_message("请先修正左侧参数组合，再保存版本。")
            return
        try:
            self.stop_preview()
            job = self.project.enqueue()
            self.project.workspace.set_state(job["id"], "running")
            request = dict(
                **job,
                action="render",
                only_mode=RESULT_MODES[self.project.photo()["engine"]],
                directory=str(self.project.workspace.root / "runs" / job["id"]),
                # Prior completed decode lets the renderer reuse the
                # decoded-RAW cache instead of re-decoding the triplet.
                decode_anchor=self.project.decode_anchor(recipe=job["recipe"]),
                repair_plan=self.project.repair_context(recipe=job["recipe"]),
            )
            if import_stage is not None:
                request["import_stage"] = import_stage
            self.tasks.start(request)
            self.last_error = None
            self.status_message(
                "正在核对并导入，完成后进入画幅与测量。" if import_stage is not None
                else "正在按当前配方生成完整分辨率版本；已有成片保留。"
            )
            self.sync_controls()
        except Exception as error:
            self.error(error)

    def task_started(self):
        self.progress.setRange(0, 0)
        self.progress.show()
        self.status_message("正在准备任务…")

    def task_progress(self, event):
        fraction = event.get("fraction")
        self.progress.setRange(0, 0 if fraction is None else 1000)
        if fraction is not None:
            self.progress.setValue(round(float(fraction) * 1000))
        self.status_message(event.get("message", "处理中…"))
        preparing = getattr(self, "repair_preparing", None)
        if preparing is not None:
            preparing.update_progress(event)

    def task_done(self, request, result):
        self.progress.hide()
        action = request["action"]
        try:
            if action == "probe":
                self.project.add_photo(result["raw"], "rgb")
                self.probe_next()
                return
            if action == "render":
                binding = self.project.data["jobs"][request["id"]]
                first_import = self.project.photo(binding["photo"])["initial_job"] is None
                sample, manifest = self.project.completed(request["id"])
                self.flatfield_rendered(request, manifest)
                if request.get("import_stage") is not None:
                    failed_id = getattr(self, "failed_request", {}).get("id")
                    failed_binding = self.project.data["jobs"].get(failed_id, {})
                    if failed_binding.get("photo") == binding["photo"]:
                        self.failed_request = {}
                if (
                    binding["photo"] == self.project.data["current"]
                    and binding["engine"] == self.project.photo()["engine"]
                ):
                    if (request.get("import_stage") == self.step
                            and source_recipe_identity(request["recipe"])
                            == source_recipe_identity(self.project.recipe())):
                        self.set_stage(2)
                    if binding["engine"] == self.project.photo()["engine"]:
                        self.load_records(*self.saved_images(sample))
                self.refresh_photos()
                self.status_message(
                    "完整分辨率版本已保存，RGB16／ICC 已通过现有输出核对。"
                )
                if (first_import and request.get("import_stage") is not None and self.step == 2
                        and binding["photo"] == self.project.data["current"]
                        and manifest.get("density_window")):
                    from .picture import AUTO_PICTURE_INSET

                    region = manifest.get("picture_region") or {}
                    inset = region.get("inset_per_side") if isinstance(region, dict) else None
                    automatic_80 = (request["recipe"].get("roi") is None
                                    and type(inset) in (int, float)
                                    and abs(inset - AUTO_PICTURE_INSET) < 1e-12)
                    prefix = ("已自动测量中心约 80% 画面。" if automatic_80
                              else "密度测量已完成。")
                    self._measurement_hint_photo = binding["photo"]
                    self._measurement_hint_text = prefix + "可直接进入转换调色，也可手动重选区域。"
                    self._measurement_hint_timer.start(5000)
                    self.sync_drawing_hint()
                self.job_saved.emit(request["id"])
            elif action == "export_batch":
                report = result["report"]
                self.status_message(
                    f"导出完成：{len(report['saved'])} 张；失败 {len(report['errors'])} 张。"
                )
                if report["errors"]:
                    self.last_error = str(report["errors"])
            elif action == "repair_prepare":
                receipt = Path(request["directory"]) / "receipt.json"
                self.project.workspace.data.setdefault("repairs", []).append(
                    str(receipt)
                )
                self.project.workspace.save()
                self._close_repair_preparing()
                from .repair_ui import RepairDialog

                dialog = RepairDialog(
                    receipt, self.project.workspace.root / "requests", self
                )
                dialog.finished.connect(lambda: self.repair_finished(receipt))
                dialog.showMaximized()
                self.repair_dialog = dialog
            elif action == "geometry_prepare":
                pending = getattr(self, "pending_geometry", None)
                self.pending_geometry = None
                if pending:
                    photo = self.project.photo(pending["photo"])
                    photo["geometry"] = result["geometry"]
                    self.project.save()
                    if (
                        self.project.data["current"] == photo["id"]
                        and self.step == pending["stage"]
                    ):
                        if pending.get("auto_crop"):
                            self.apply_auto_crop(pending["auto_crop"])
                        else:
                            self.begin_selection(pending["role"])
                            if pending["role"] == "meter_grid":
                                self.request_preview()
            elif action == "flatfield_import":
                self.flatfield_imported(request, result)
            pending_repair = getattr(self, "pending_repair_save", None)
            if action == "render" and pending_repair == request["id"]:
                self.pending_repair_save = None
                binding = self.project.data["jobs"][request["id"]]
                if (binding["photo"] == self.project.data["current"]
                        and binding["engine"] == self.project.photo()["engine"]):
                    QTimer.singleShot(0, self.start_repair)
        except Exception as error:
            self.error(error)
            # 任务完成回调里的异常曾只写状态行，用户几乎不可见；
            # 弹窗直陈原因。仅真实桌面会话弹模态，避免阻塞测试。
            from PyQt6.QtGui import QGuiApplication

            if QGuiApplication.platformName() != "offscreen":
                GlassMessageBox.warning(self, "任务未完成", str(error))
        self.sync_controls()

    def task_failed(self, request, message, cancelled):
        self.progress.hide()
        if getattr(self, "pending_repair_save", None) == request.get("id"):
            self.pending_repair_save = None
        preparing = getattr(self, "repair_preparing", None)
        if preparing is not None and not preparing.failed:
            preparing.show_failure(message, cancelled)
        if request["action"] == "flatfield_import":
            self.flatfield_failed(request, message, cancelled)
        if request["action"] == "render":
            self.project.workspace.set_state(
                request["id"], "cancelled" if cancelled else "failed", error=message
            )
        self.error("任务已取消，旧版本保留" if cancelled else message)
        self.failed_request = deepcopy(request)
        if request["action"] == "probe":
            self.pending_probe = []
        self.sync_controls()
        if request["action"] == "render" and not cancelled:
            from .workflow_import_roi import offer_import_roi

            try:
                offer_import_roi(self, request, message)
            except Exception as error:
                self.error(error)

    def cancel_task(self):
        self.pending_probe = []
        if self.tasks.busy:
            self.tasks.cancel()
        elif self.preview_activity.active:
            token = self.preview_activity.token
            self.preview_timer.stop()
            self.token += 1
            self.live.invalidate(self.token)
            self.preview_activity.finish(token, state="cancelled", message="已取消更新，保留当前画面")
            self.show_preview_notice("已取消更新，当前画面保留")

    def retry_last(self):
        if getattr(self, "_failed_preview", False) and not self.tasks.busy:
            self.request_preview()
            return
        request = getattr(self, "failed_request", None)
        if not request or self.tasks.busy or self.geometry_role == "import_roi":
            return
        if request["action"] == "render":
            binding = self.project.data["jobs"][request["id"]]
            if request.get("import_stage") is not None:
                self.set_stage(0)
                if self.step == 0:
                    self.select_photo(binding["photo"])
                    self.confirm_import()
                return
            self.set_stage(3)
            if self.step != 3:
                return
            self.select_photo(binding["photo"])
            if self.project.data["current"] != binding["photo"]:
                return
            self.project.select_engine(binding["engine"])
            self.sync_controls()
            self.save_version()
        elif request["action"] == "probe":
            self.pending_probe = [request["raw"]]
            self.probe_next()
        elif request["action"] == "geometry_prepare":
            retry = {
                **request,
                "directory": str(
                    self.project.workspace.root / "geometry-cache" / uuid.uuid4().hex
                ),
            }
            self.last_error = None
            self.tasks.start(retry)
            self.sync_controls()
        elif request["action"] == "flatfield_import":
            retry = {**request, "directory": str(self.project.workspace.root / "calibrations" / uuid.uuid4().hex)}
            self.last_error = None
            self.tasks.start(retry)
            self.sync_controls()

    def request_actual(self):
        self.pending_actual = True
        self.request_preview(full=True)

    def _detail_context_matches(self, context, token):
        if self.closing or token != self.token or context["display_revision"] != self.display_revision:
            return False
        current = self.preview_context()
        return bool(current and current["photo"] == context["photo"]
                    and current["engine"] == context["engine"]
                    and current["recipe_sha256"] == context["recipe_sha256"]
                    and current.get("repair_plan_sha256") == context.get("repair_plan_sha256"))

    def request_detail_preview(self, view):
        request_id = view["request_id"]
        current, shown = self.preview_context(), self.shown_context
        base_in_flight = any(
            request and not request.get("viewport") and request.get("token") == self.token
            for request in (self.live.active, self.live.pending)
        )
        display_pending = getattr(self, "base_display_pending", None)
        if (self.closing or self.tasks.busy or not current or not shown
                or base_in_flight
                or (display_pending is not None and display_pending[1] == self.token)
                or self.canvas_stack.currentWidget() is not self.canvas
                or any(shown.get(key) != current.get(key)
                       for key in ("photo", "engine", "recipe_sha256", "repair_plan_sha256"))):
            # A changed recipe needs its matching base frame first. set_images
            # schedules the viewport again while preserving the current view.
            self.canvas.detail_failed(request_id)
            return
        try:
            x0, y0, x1, y1 = view["roi"]
            viewport = dict(x=x0, y=y0, width=x1 - x0, height=y1 - y0,
                            output_size=list(view["source_size"]),
                            sample_step=view["sample_step"])
            recipe = self.project.recipe()
            manifest = self.project.preview_base()
            self.token += 1
            context = {**current, "request_id": request_id,
                       "display_revision": self.display_revision,
                       "center_only": view.get("center_only", False)}
            self.detail_contexts[self.token] = context
            self.detail_contexts = {key: value for key, value in self.detail_contexts.items()
                                    if key >= self.token - 16}
            self.live.submit(dict(
                token=self.token, manifest=str(manifest), manifest_sha256=sha256(manifest),
                mode=RESULT_MODES[current["engine"]], draft_recipe=recipe,
                controls={key: value for key, value in recipe.items()
                          if key in (*EDITING_DOCUMENT_KEYS, "exposure_ev", "cool_tone")},
                full_resolution=False, viewport=viewport, detail_id=request_id,
                repair_plan=self.project.repair_context(recipe=recipe),
            ))
            self.begin_preview_activity(self.token, "正在加载当前区域的清晰预览…", stage="detail")
            self.status_message("正在加载当前区域的清晰预览…")
        except Exception as error:
            self.canvas.detail_failed(request_id)
            if not self.preview_activity.active or self.preview_activity.token != self.token:
                self.preview_activity.start(self.token)
            self.preview_activity.finish(self.token, state="failed", message="高清预览失败：" + str(error))
            self.status_message("高清预览未更新：" + str(error))

    def _start_display_work(self, work, current, abandon, failed):
        """Wait for decode prewarm without losing ownership of display inputs."""
        pending = getattr(self, "_deferred_display_work", None)
        if pending is None:
            pending = self._deferred_display_work = {}
        marker = object()

        def cancel():
            if pending.pop(marker, None) is not None:
                try:
                    abandon()
                except OSError as error:
                    from .diagnostics import failure

                    failure("deferred_display_cleanup_failed", error)

        def start(check_current=True):
            if marker not in pending:
                return
            if check_current and (self.closing or not current()):
                cancel()
                return
            self.pool.start(work)
            pending.pop(marker, None)

        def on_error(error):
            cancel()
            failed(str(error))

        pending[marker] = cancel
        try:
            gate = getattr(self, "prewarm_gate", None)
            if gate is None or not gate.defer(start, on_error):
                start(check_current=False)
        except Exception:
            cancel()
            raise

    def load_detail_records(self, event):
        token = event["token"]
        context = self.detail_contexts.get(token)
        if (context is None or event.get("detail_id") != context["request_id"]
                or not self._detail_context_matches(context, token)):
            self.live.discard(event)
            return
        target, proof = self.display_target, self.proof_space
        if target is None:
            from .workflow_display import DisplayTarget, named_reference_profile, profile_description

            icc = self.display_icc or named_reference_profile("srgb")
            target = DisplayTarget(
                icc=icc, description="高清显示回退", profile_name=profile_description(icc),
                monitor_icc=None, monitor_description="未确认屏幕ICC", screen_name="显示回退",
                screen_id=None, source="explicit-display-fallback", assumed=True,
            )
        viewport = event["viewport"]

        def read():
            try:
                first, second = event["records"]
                image = self.display_cache.get_image(first, target, proof=proof)
                other = (image if first["sha256"] == second["sha256"] else
                         self.display_cache.get_image(second, target, proof=proof))
                return image, other
            finally:
                self.live.discard(event)

        def presented(result_token, images):
            if not self._detail_context_matches(context, result_token):
                return
            x, y = viewport["x"], viewport["y"]
            accepted = self.canvas.set_detail_images(
                images[0], images[1], [x, y, x + viewport["width"], y + viewport["height"]],
                viewport["sample_step"], context["request_id"],
            )
            if accepted:
                self.set_preview_updating(False)
                self.show_preview_notice()
                self.preview_activity.finish(result_token, message="高清预览已更新")
                self.status_message("中心区域已更新为原像素预览" if context["center_only"]
                                    else "当前可见区域的清晰预览已更新")

        work = Work(token, read)
        work.signals.done.connect(presented)
        def failed(result_token, message):
            if self._detail_context_matches(context, result_token):
                self.canvas.detail_failed(context["request_id"])
                self.preview_activity.finish(result_token, state="failed", message="高清预览失败：" + str(message))
                self.status_message("高清预览未更新：" + str(message))
        work.signals.failed.connect(failed)
        WorkflowWindow._start_display_work(
            self,
            work,
            lambda: self._detail_context_matches(context, token),
            lambda: self.live.discard(event),
            lambda message: failed(token, message),
        )

    def request_preview(self, full=False):
        if (self.geometry_role == "crop"
                or (self.geometry_role in ("highlight", "highlight_effect", "highlight_protect")
                    and getattr(self, "highlight_full_source", False))):
            from .workflow_crop_preview import request_crop_frame

            try:
                request_crop_frame(self)
            except Exception as error:
                self.preview_failed(self.token, str(error))
            return
        self.set_preview_updating(False)
        if not full:
            self.canvas.clear_detail()
        if (
            self.tasks.busy
            or not self.current_photo()
            or not self.current_photo()["initial_job"]
        ):
            return
        try:
            recipe = self.project.prepare_highlight_reference()
            manifest = self.project.preview_base()
            self.last_error = None
            self._failed_preview = False
            self.token += 1
            self.begin_preview_activity(self.token)
            controls = {
                key: value
                for key, value in recipe.items()
                if key in (*EDITING_DOCUMENT_KEYS, "exposure_ev", "cool_tone")
            }
            self.preview_overlay.setText("正在更新预览…")
            self.set_preview_updating(True)
            self.mark_preview_pending(clear_detail=not full)
            self.preview_contexts[self.token] = self.preview_context()
            self.preview_contexts = {key: value for key, value in self.preview_contexts.items() if key >= self.token - 16}
            self.live.submit(
                dict(
                    token=self.token,
                    manifest=str(manifest),
                    manifest_sha256=sha256(manifest),
                    mode=RESULT_MODES[self.project.photo()["engine"]],
                    controls=controls,
                    draft_recipe=recipe,
                    full_resolution=bool(full),
                    repair_plan=self.project.repair_context(recipe=recipe),
                )
            )
            self.status_message("正在更新真实处理预览…")
        except Exception as error:
            self.set_preview_updating(False)
            self.preview_activity.finish(self.token, state="failed", message="预览失败：" + str(error))
            self.error(error)

    def live_ready(self, event):
        if event.get("viewport") is not None:
            self.load_detail_records(event)
            return
        context = deepcopy(self.preview_contexts.get(event.get("token")))
        if context is not None and context.get("crop_locator"):
            from .workflow_crop_preview import receive_crop_frame

            receive_crop_frame(self, event, context)
            return
        if context is not None:
            context["output_size"] = event.get("output_size")
        seconds = event.get("seconds")
        self.preview_reusable = bool(event.get("preview_reusable")) and (
            isinstance(seconds, (int, float)) and seconds <= 1.5
        )
        if context is not None and event.get("reference_label"):
            context["reference_label"] = event["reference_label"]
        if context is not None and event.get("preview_quality"):
            context["preview_quality"] = deepcopy(event["preview_quality"])
        if context is not None and event.get("metering_receipt") is not None:
            try:
                self.project.remember_preview_metering(context["recipe"], event["metering_receipt"])
                self.refresh_metering_status()
            except Exception as error:
                self.live.discard(event)
                self.preview_failed(event["token"], str(error))
                return
        self.load_records(
            event["records"][0],
            event["records"][1],
            sample_step=event.get("sample_step", 1),
            temporary=event,
            context=context,
        )

    def saved_images(self, sample):
        source = sample.record(RESULT_MODES[self.project.photo()["engine"]], "baseline")
        current = (
            self.project.confirmed_record(source)
            if not self.project.dirty()
            else source
        )
        if current["sha256"] != source["sha256"]:
            reference = source
        elif "baseline__native" in sample.results:
            reference = sample.record("native", "baseline")
        else:
            reference = {**source, "reference_label": "已保存版本"}
        return current, reference

    def load_records(
        self, current, reference, reset=False, sample_step=1, temporary=None,
        context=None, display_only=False, lease=None,
    ):
        token = self.token
        self.display_serial += 1
        display_serial, display_revision = self.display_serial, self.display_revision
        self.base_display_pending = (display_serial, token)
        activity = getattr(self, "preview_activity", None)
        context = deepcopy(context or self.preview_context())
        output_size = (context or {}).get("output_size")
        if not temporary and not display_only and current.get("shape"):
            output_size = [current["shape"][1], current["shape"][0]]
            context["output_size"] = output_size
        if not temporary and not display_only and self.current_photo():
            initial = self.current_photo()["initial_job"]
            job = self.project.saved_job() or (self.project.workspace.job(initial) if initial else None)
            if job:
                import hashlib

                recipe = deepcopy(job["recipe"])
                if recipe.get("roi") is None:
                    recipe["roi"] = self.current_photo()["analysis_roi"]
                context["recipe"] = recipe
                context["recipe_sha256"] = hashlib.sha256(json.dumps(recipe, sort_keys=True, allow_nan=False).encode()).hexdigest()

        def context_key(value):
            return tuple((value or {}).get(key) for key in (
                "photo", "engine", "recipe_sha256", "repair_plan_sha256", "generation",
            ))

        desired_context = context_key(self.preview_context())
        matching_generation = bool(context and context_key(context) == desired_context
                                   and context.get("generation") == token)
        # A proof/ICC refresh may be reading an older shown frame while a new
        # recipe is still rendering. Its pixels remain useful, but its token
        # at submission time does not make it the new render's display stage.
        display_photo = self.current_photo()
        idle_display = bool(
            context and display_photo and context.get("photo") == display_photo["id"]
            and not (activity is not None and activity.active)
            and not (self.live.active or self.live.pending)
        )
        activity_claim = matching_generation or idle_display
        activity_started = None
        if activity is not None and activity_claim:
            if not activity.active or activity.token != token:
                activity.start(token, "正在更新显示色彩…", stage="display")
            activity_started = activity.started
            activity.progress(dict(token=token, stage="display", message="正在处理显示色彩…", fraction=0.97))

        def owns_activity():
            photo = self.current_photo()
            return bool(
                activity_claim and token == self.token and context and photo
                and context.get("photo") == photo["id"]
                and desired_context == context_key(self.preview_context())
                and (activity is None or (
                    activity.active and activity.token == token
                    and activity.started == activity_started
                ))
            )

        current, reference = deepcopy(current), deepcopy(reference)
        source_step = current.get("display_source_step", 1)
        if not display_only and (temporary or self.pending_actual):
            # Worker packets already carry their sampling scale; a requested
            # full result must also replace any prior fit-only record hints.
            source_step = 1
            current.pop("display_source_step", None)
            reference.pop("display_source_step", None)
        if (not temporary and not display_only and sample_step == 1
                and not self.pending_actual):
            fit_step = saved_display_source_step(current)
            if fit_step is not None:
                source_step = fit_step
                current["display_source_step"] = source_step
                reference["display_source_step"] = source_step
        effective_step = sample_step * source_step
        icc = self.display_icc
        target, proof = self.display_target, self.proof_space
        want_scopes = hasattr(self, "scope_panel")
        if target is None and (source_step > 1 or want_scopes):
            # Explicit sRGB test mode and initial ICC discovery must also avoid
            # full-frame float64 conversion of a 96MP saved image.
            from .workflow_display import DisplayTarget, named_reference_profile, profile_description

            fallback_icc = icc or named_reference_profile("srgb")
            target = DisplayTarget(
                icc=fallback_icc, description="显式显示回退",
                profile_name=profile_description(fallback_icc), monitor_icc=None,
                monitor_description="未确认屏幕ICC", screen_name="显示回退",
                screen_id=None, source="explicit-display-fallback", assumed=True,
            )

        def displayed(record, include_scopes=False):
            if target is not None:
                return self.display_cache.get_image(
                    record, target, proof=proof, sample_step=sample_step,
                    source_step=record.get("display_source_step", 1),
                    include_scopes=include_scopes,
                )
            return display_image(record, icc)

        def read():
            from .preview_requests_v44 import read_preview_metering

            prepared = lease
            try:
                metering = read_preview_metering(temporary) if temporary else None
                if prepared is None:
                    prepared = self.snapshot_store.prepare(
                        current, reference, context, sample_step,
                        own_files=bool(temporary),
                    )
                current_result = displayed(prepared.current, include_scopes=want_scopes)
                image, scopes = current_result if want_scopes else (current_result, None)
                other = (
                    image
                    if prepared.current["sha256"] == prepared.reference["sha256"]
                    else displayed(prepared.reference)
                )
                return image, other, prepared, scopes, metering
            except BaseException:
                if prepared is not None:
                    prepared.release()
                raise
            finally:
                if temporary:
                    self.live.discard(temporary)

        work = Work(token, read)

        def shown(result_token, images):
            if self.base_display_pending == (display_serial, result_token):
                self.base_display_pending = None
            prepared = images[2] if len(images) > 2 else None
            photo = self.current_photo()
            if (self.closing or not photo or context.get("photo") != photo["id"]
                    or (not display_only and result_token != self.token)):
                if prepared:
                    prepared.release()
                return
            if len(images) > 4 and images[4] is not None:
                try:
                    self.project.remember_preview_metering(context["recipe"], images[4])
                    self.refresh_metering_status()
                except (ValueError, OSError) as error:
                    if prepared:
                        prepared.release()
                    self.preview_failed(result_token, str(error))
                    return
            published = self.snapshot_store.publish(prepared) if prepared and not display_only else False
            if display_serial != self.display_serial or display_revision != self.display_revision:
                if prepared:
                    prepared.release()
                if published:
                    self.refresh_display_snapshot()
                return
            if self.canvas.current is not None and not self.canvas.fit_mode:
                self.canvas.zoom *= effective_step / self.canvas.sample_scale
            self.canvas.sample_scale = effective_step
            self.canvas.reference_name = (
                context.get("reference_label")
                or reference.get("reference_label")
                or "调整前／原生"
            )
            self.canvas.set_images(
                images[0],
                images[1],
                ENGINES[(context or {}).get("engine", self.project.photo()["engine"])]
                + (" · 交互预览" if context.get("preview_quality", {}).get("approximate") else "")
                + (" · 已确认除尘" if "outside_mask_changed_pixels" in current else ""),
                reset=reset,
            )
            self.canvas.set_output_size(*(output_size or (None, None)))
            if self.pending_actual and effective_step == 1:
                self.pending_actual = False
                self.canvas.actual()
            if not display_only:
                self.current_record = current if not temporary else None
                if not temporary:
                    self.last_display_records = (current, reference, sample_step)
            self.shown_context = context
            if want_scopes:
                self.scope_panel.set_scope_data(images[3] if len(images) > 3 else None)
                self.scope_panel.caption.setText("输出 RGB · 预览取样")
            if hasattr(self, "refresh_selection_preview"):
                self.refresh_selection_preview()
            if prepared:
                prepared.release()
            desired = self.preview_context()
            matches = bool(context and desired and context["photo"] == desired["photo"]
                           and context["engine"] == desired["engine"]
                           and context["recipe_sha256"] == desired["recipe_sha256"])
            if matches:
                self.show_preview_notice()
                quality = context.get("preview_quality", {})
                if activity is None or owns_activity():
                    self.status_message((quality.get("label") or "预览已更新")
                                        + " · 调整已自动保存，除尘或导出时生成完整成片。")
                if activity is not None and owns_activity():
                    activity.finish(result_token, message=quality.get("label") or "预览已更新")
            else:
                self.mark_preview_pending()
                if activity is not None and owns_activity():
                    activity.finish(result_token, message="显示色彩已更新")
            self.last_activity = time.monotonic()
            if (not (activity is not None and activity.active)
                    and (matches or not (self.live.active or self.live.pending))):
                self.set_preview_updating(False)
            self.image_presented.emit()
            self.queue_idle_previews()

        work.signals.done.connect(shown)
        def failed(result_token, message):
            if self.base_display_pending == (display_serial, result_token):
                self.base_display_pending = None
            if (display_serial == self.display_serial and display_revision == self.display_revision
                    and owns_activity()):
                self.base_display_failed(result_token, message)
        work.signals.failed.connect(failed)
        def abandon():
            if self.base_display_pending == (display_serial, token):
                self.base_display_pending = None
            try:
                if lease is not None:
                    lease.release()
            finally:
                if temporary:
                    self.live.discard(temporary)

        def current_display():
            photo = self.current_photo()
            return bool(photo and context and context.get("photo") == photo["id"]
                        and (not display_only or (
                            display_serial == self.display_serial
                            and display_revision == self.display_revision
                        ))
                        and (display_only or token == self.token))

        WorkflowWindow._start_display_work(
            self, work, current_display, abandon, lambda message: failed(token, message),
        )

    def base_display_failed(self, token, message):
        if token == self.token:
            pending = getattr(self, "crop_locator_pending", None)
            if pending and pending.get("token") == token:
                self.crop_locator_pending = None
            activity = getattr(self, "preview_activity", None)
            diagnostic_event("preview_failed", token=token, message=str(message))
            self.session_events.append(dict(event="preview_failed", token=token, status="failed"))
            if activity is not None:
                activity.finish(token, state="failed", message="预览失败：" + str(message))
            self._failed_preview = True
            self.set_preview_updating(False)
            self.show_preview_notice("预览未更新：" + str(message))
            self.error(message)
            if hasattr(self, "retry_button"):
                self.retry_button.setEnabled(True)

    def preview_failed(self, token, message):
        detail = self.detail_contexts.pop(token, None)
        if detail is not None:
            if self._detail_context_matches(detail, token):
                self.canvas.detail_failed(detail["request_id"])
                self.preview_activity.finish(token, state="failed", message="高清预览失败：" + str(message))
                self.status_message("高清预览未更新：" + str(message))
            return
        self.base_display_failed(token, message)

    def toggle_compare(self):
        self.canvas.split = self.compare.isChecked()
        self.canvas.update()

    def choose_export_directory(self):
        path = QFileDialog.getExistingDirectory(
            self, "选择导出目录", self.export_directory.text()
        )
        if path:
            self.export_directory.setText(path)

    def export_selected(self):
        if self.tasks.busy or not self.current_photo():
            return
        try:
            destination = Path(self.export_directory.text()).expanduser().resolve()
            request = self.project.export_request(destination)
            destination.mkdir(parents=True, exist_ok=True)
            self.stop_preview()
            self.tasks.start(request)
            self.sync_controls()
        except Exception as error:
            self.error(error)

    def export_previous_highlight(self):
        if self.tasks.busy or not self.current_photo():
            return
        try:
            destination = Path(self.export_directory.text()).expanduser().resolve()
            request = self.project.export_previous_highlight_request(destination)
            destination.mkdir(parents=True, exist_ok=True)
            self.stop_preview()
            self.tasks.start(request)
            self.sync_controls()
        except Exception as error:
            self.error(error)

    def _ensure_repair_preparing(self):
        """Show the preparing sheet immediately, reusing it across save+detect."""
        dialog = getattr(self, "repair_preparing", None)
        if dialog is None:
            from .repair_ui import RepairPreparingDialog

            dialog = RepairPreparingDialog(self)
            dialog.cancel_requested.connect(self._cancel_repair_prepare)
            dialog.finished.connect(self._repair_preparing_finished)
            self.repair_preparing = dialog
            dialog.show()
        return dialog

    def _close_repair_preparing(self):
        dialog = getattr(self, "repair_preparing", None)
        if dialog is not None:
            self.repair_preparing = None
            dialog.accept()
            dialog.deleteLater()

    def _repair_preparing_finished(self):
        # The failure sheet is dismissed by the user; the subprocess lifecycle
        # stays owned by Tasks regardless of the sheet's visibility.
        self.repair_preparing = None

    def _cancel_repair_prepare(self):
        if self.tasks.busy:
            self.tasks.cancel()

    def start_repair(self):
        if self.tasks.busy or not self.current_photo():
            return
        photo = self.project.photo()
        reason = experimental_repair_block_reason(self.project.recipe(photo), photo["engine"])
        if reason is not None:
            GlassMessageBox.warning(self, "暂不能进入 PURE Neg 除尘", reason)
            return
        preparing = self._ensure_repair_preparing()
        try:
            if self.project.dirty():
                preparing.update_progress(
                    {"message": "正在保存当前完整版本；完成后自动开始尘点检测…"}
                )
                self.save_version()
                if self.tasks.busy and self.tasks.request.get("action") == "render":
                    self.pending_repair_save = self.tasks.request["id"]
                    self.status_message("正在保存当前完整版本；完成后自动打开 PURE Neg 除尘审核。")
                else:
                    # save_version refused to enqueue a render (e.g. invalid
                    # advanced edits); do not leave an orphan preparing sheet.
                    self._close_repair_preparing()
                return
            job = self.project.saved_job()
            record = self.project.workspace.sample(job).record(
                RESULT_MODES[self.project.photo()["engine"]], "baseline"
            )
            from .repair_bridge import DEFAULT_BACKEND

            self.stop_preview()
            preparing.update_progress(
                {"message": "正在检测尘点、划痕与纤维…"}
            )
            self.tasks.start(
                dict(
                    action="repair_prepare",
                    source=record,
                    revision=record["sha256"],
                    backend=str(DEFAULT_BACKEND),
                    directory=str(
                        self.project.workspace.root / "repairs" / uuid.uuid4().hex
                    ),
                    polarity="bright",
                    detect="dust,scratches,fibers",
                )
            )
            self.sync_controls()
        except Exception as error:
            self._close_repair_preparing()
            self.error(error)

    def repair_finished(self, receipt_path):
        if not self.current_photo() or self.project.dirty():
            return
        try:
            receipt = json.loads(Path(receipt_path).read_text())
            job = self.project.saved_job()
            source = self.project.workspace.sample(job).record(
                RESULT_MODES[self.project.photo()["engine"]], "baseline"
            )
            if (
                receipt.get("source", {}).get("sha256") == source["sha256"]
                and receipt.get("state") == "confirmed"
            ):
                self._show_confirmed_repair(source)
        except Exception as error:
            self.error(error)

    def _show_confirmed_repair(self, source):
        """Invalidate every older frame, including the same recipe before review."""
        current = self.project.confirmed_record(source)
        self.stop_preview()
        self.detail_contexts.clear()
        self.preview_contexts.clear()
        self.canvas.clear_detail()
        self.crop_locator_cache = None
        self.display_cache.clear()
        self.snapshot_store.clear()
        self.load_records(current, source)
        self.sync_controls()
        self.status_message("已确认除尘；导出将使用此版本的已核验修复结果。")

    def resume_repair(self):
        if self.tasks.busy or not self.current_photo() or self.project.dirty():
            self.status_message("请等待当前任务完成；使用 PURE Neg 入口可自动准备当前调整对应的完整版本。")
            return
        photo = self.project.photo()
        reason = experimental_repair_block_reason(self.project.recipe(photo), photo["engine"])
        if reason is not None:
            GlassMessageBox.warning(self, "暂不能恢复 PURE Neg 除尘", reason)
            return
        try:
            job = self.project.saved_job()
            source = self.project.workspace.sample(job).record(
                RESULT_MODES[self.project.photo()["engine"]], "baseline"
            )
            confirmed = self.project.confirmed_record(source)
            if confirmed["sha256"] != source["sha256"]:
                self._show_confirmed_repair(source)
                return
            for path in reversed(self.project.repair_paths(photo)):
                receipt = json.loads(Path(path).read_text())
                if receipt.get("source", {}).get("sha256") != source["sha256"]:
                    continue
                if receipt.get("state") == "confirmed":
                    self.repair_finished(path)
                    return
                from .repair_ui import RepairDialog

                dialog = RepairDialog(
                    path, self.project.workspace.root / "requests", self
                )
                dialog.finished.connect(lambda: self.repair_finished(path))
                dialog.showMaximized()
                self.repair_dialog = dialog
                return
            self.status_message("当前保存版本没有可恢复的除尘审核。")
        except Exception as error:
            self.error(error)

    def closeEvent(self, event):
        if not self.closing and not self.flush_advanced():
            event.ignore()
            return
        self.application_menu_titles.dispose()
        from .workflow_shutdown import shutdown_window

        shutdown_window(self)
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--srgb", action="store_true")
    parser.add_argument("--ready-report", type=Path)
    parser.add_argument("--stage", type=int, choices=range(len(STAGES)))
    parser.add_argument("--settings", choices=("general", "display", "performance", "maintenance"))
    args = parser.parse_args()
    from .workflow_application import prepare_application_identity

    prepare_application_identity()
    prepare_plugins()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("PURE RGB")
    app.setApplicationDisplayName("PURE RGB")
    app.setFont(QFont("PingFang SC", 12))
    app.setWindowIcon(application_icon())
    # Frost every dialog, alert, menu and combo popup with native vibrancy.
    install_glass_surfaces(app)
    from .workflow_startup import choose_workspace

    choice = choose_workspace(
        ROOT, explicit_workspace=args.workspace, ready_report=args.ready_report
    )
    if choice.project is None:
        from .workflow_shutdown import dispose_application

        dispose_application(app, None)
        return 0
    splash = create_splash()
    splash.show()
    app.processEvents()
    preferences = ROOT / "outputs/workflow-ui-state/preferences.json"
    project = choice.project
    window = WorkflowWindow(project, srgb=args.srgb)
    if args.stage is not None:
        window.set_stage(args.stage)
    if choice.preferences is not None:
        write_json(preferences, choice.preferences)
    window.showMaximized()
    window.update_notice.schedule()
    if args.settings:
        window.open_settings()
        window.settings_panel.tabs.setCurrentIndex(
            ("general", "display", "performance", "maintenance").index(args.settings)
        )
    QTimer.singleShot(600, lambda: splash.finish(window))
    if args.ready_report:
        QTimer.singleShot(
            1000,
            lambda: write_json(
                args.ready_report,
                dict(
                    pid=os.getpid(),
                    workspace=str(project),
                    glass=window.glass_report,
                    stages=VISIBLE_STAGE_LABELS,
                    actual_desktop=True,
                    display=(
                        dict(description=window.display_description,
                             assumed=window.display_target.assumed,
                             screen_name=window.display_target.screen_name,
                             profile_name=window.display_target.profile_name)
                        if window.display_target else dict(description=window.display_description)
                    ),
                    performance=resolve_performance(window.performance),
                ),
            ),
        )
    from .workflow_shutdown import dispose_application, shutdown_window

    app.aboutToQuit.connect(lambda: shutdown_window(window))
    try:
        return app.exec()
    finally:
        dispose_application(app, window)


if __name__ == "__main__":
    raise SystemExit(main())
