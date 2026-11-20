"""Inline application settings; image previews remain in the main canvas."""

from copy import deepcopy

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPalette, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .release import APP_VERSION
from .workflow_combo import WorkflowComboBox
from .workflow_glass import system_dark
from .workflow_performance import (
    MIB,
    checked_performance_settings,
    hardware_info,
    resolve_performance,
    backend_status,
)
from .workflow_scroll import WorkflowScrollArea
from .workflow_slider import SliderControl


def _note(caption):
    label = QLabel(caption)
    label.setWordWrap(True)
    label.setObjectName("muted")
    return label


class SettingsTabBar(QTabBar):
    """Keep Qt's tab navigation and draw labels above the AppKit menu material."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("glassMenuStrip", True)
        self.setExpanding(True)
        self.setDrawBase(False)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        dark = system_dark()
        if not self.property("nativeMaterial"):
            edge = QColor(255, 255, 255, 48) if dark else QColor(32, 38, 48, 48)
            painter.setPen(QPen(edge))
            painter.setBrush(QColor("#353331" if dark else "#e5e6e9"))
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 12, 12)
        for index in range(self.count()):
            rect = QRectF(self.tabRect(index)).adjusted(3, 3, -3, -3)
            selected = index == self.currentIndex()
            if selected:
                painter.setBrush(QColor(255, 103, 66, 68) if dark else QColor(221, 66, 32, 48))
                painter.setPen(QPen(QColor("#ff956f")) if self.hasFocus() else Qt.PenStyle.NoPen)
                painter.drawRoundedRect(rect, 9, 9)
            color = self.palette().color(
                QPalette.ColorGroup.Active if self.isTabEnabled(index) else QPalette.ColorGroup.Disabled,
                QPalette.ColorRole.WindowText,
            )
            painter.setPen(QColor("#ffb49b" if dark else "#861c09") if selected else color)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.tabText(index))


class SettingsPanel(QWidget):
    """Emit validated settings; persistence and worker ownership stay with host."""

    settings_changed = pyqtSignal(dict)
    proof_changed = pyqtSignal(str)
    diagnostics_requested = pyqtSignal()
    clear_cache_requested = pyqtSignal()
    release_memory_requested = pyqtSignal()

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.settings = checked_performance_settings(settings)
        self.hardware = hardware_info()
        self.loading = True
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabBar(SettingsTabBar(self.tabs))
        outer.addWidget(self.tabs)
        self._general_page()
        self._display_page()
        self._performance_page()
        self._maintenance_page()
        self.set_settings(self.settings)

    def _page(self, title):
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 12, 0, 0)
        column.setSpacing(12)
        scroll = WorkflowScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, title)
        return column

    def _general_page(self):
        column = self._page("通用")
        self.idle_previews = QCheckBox("闲置时自动生成显示预览")
        self.idle_previews.toggled.connect(
            lambda value: self._change("idle_previews", bool(value))
        )
        column.addWidget(self.idle_previews)
        column.addWidget(
            _note("停止操作 4 秒后，在后台准备其他照片的显示预览。继续操作时优先处理当前照片。")
        )
        column.addWidget(QLabel(f"PURE RGB {APP_VERSION}"))
        release_actions = QHBoxLayout()
        self.manual_button = QPushButton("操作手册")
        self.donation_button = QPushButton("捐赠支持")
        self.donation_button.setToolTip("打开 StudioPURE 的支持说明")
        self.update_button = QPushButton("检查更新")
        for button in (self.manual_button, self.donation_button, self.update_button):
            release_actions.addWidget(button)
        column.addLayout(release_actions)
        self.release_status = _note("")
        self.release_status.setTextFormat(Qt.TextFormat.PlainText)
        self.release_status.hide()
        column.addWidget(self.release_status)
        self.releases_button = QPushButton("查看 GitHub 发布")
        self.releases_button.hide()
        column.addWidget(self.releases_button)
        column.addStretch()

    def _display_page(self):
        column = self._page("显示")
        column.addWidget(QLabel("屏幕色彩管理"))
        self.display_status = _note("自动跟随当前窗口所在屏幕的色彩配置。")
        column.addWidget(self.display_status)
        column.addWidget(QLabel("色域软打样"))
        self.proof_choice = WorkflowComboBox()
        self.proof_choice.setAccessibleName("色域软打样")
        for key, caption in (
            ("off", "关闭 · 保留原色域"),
            ("display-p3", "Display P3"),
            ("adobe-rgb", "Adobe RGB (1998)"),
            ("srgb", "sRGB"),
        ):
            self.proof_choice.addItem(caption, key)
        self.proof_choice.currentIndexChanged.connect(self._proof_selected)
        column.addWidget(self.proof_choice)
        column.addWidget(
            _note("预览所选色域的呈现效果。实际显示仍按当前屏幕适配，保存文件的色彩配置保持独立。")
        )
        column.addStretch()

    def _performance_page(self):
        column = self._page("性能")
        column.addWidget(QLabel("处理模式"))
        self.mode_choice = WorkflowComboBox()
        self.mode_choice.setAccessibleName("处理模式")
        self.mode_choice.addItem("均衡", "balanced")
        self.mode_choice.addItem("高性能", "high")
        self.mode_choice.currentIndexChanged.connect(
            lambda: self._change("mode", self.mode_choice.currentData())
        )
        column.addWidget(self.mode_choice)
        column.addWidget(_note("均衡模式兼顾响应和资源占用；高性能模式使用更多可用 CPU 线程并保留更大的预览缓存。"))
        column.addWidget(QLabel("图像处理设备"))
        self.backend_choice = WorkflowComboBox()
        self.backend_choice.setAccessibleName("图像处理设备")
        for key, caption in (("auto", "自动择优"), ("cpu", "CPU · 精确"), ("gpu", "GPU · 正像预览")):
            self.backend_choice.addItem(caption, key)
        status = backend_status(self.settings)
        self.backend_choice.model().item(2).setEnabled(status["gpu_available"])
        self.backend_choice.currentIndexChanged.connect(
            lambda: self._change("backend", self.backend_choice.currentData())
        )
        column.addWidget(self.backend_choice)
        column.addWidget(_note(
            ("GPU 用于适用的正像预览调整；分区测量、高光分析及保存继续使用精确处理。自动模式优先保持逐码一致；显式GPU的饱和度预览可能有微小舍入差异。"
             if status["gpu_available"] else "GPU 预览不可用，当前使用 CPU。")
        ))
        self.cpu_threads = SliderControl(
            "CPU 线程", 0, min(256, self.hardware["cpu_count"]), 1, 0
        )
        self.cpu_threads.setDefaultValue(0, description="自动线程数")
        self.cpu_threads.valueChanged.connect(
            lambda value: self._change("cpu_threads", int(value))
        )
        column.addWidget(self.cpu_threads)
        self.cache_mb = SliderControl("预览缓存（MiB）", 0, 65536, 256, 0)
        self.cache_mb.setDefaultValue(0, description="自动缓存预算")
        self.cache_mb.valueChanged.connect(
            lambda value: self._change("cache_mb", int(value))
        )
        column.addWidget(self.cache_mb)
        self.memory_saver = QCheckBox("节省预览内存")
        self.memory_saver.setToolTip("将预览缓存预算设为 256 MiB；关闭后恢复自动预算。")
        self.memory_saver.toggled.connect(self._memory_saver_changed)
        column.addWidget(self.memory_saver)
        self.policy_status = _note("")
        column.addWidget(self.policy_status)
        self.runtime_status = _note("")
        column.addWidget(self.runtime_status)
        column.addWidget(_note("设置应用于后续启动的任务和预览。缓存预算控制缓存保留量，处理大图所需的瞬时内存另计。"))
        column.addStretch()

    def _maintenance_page(self):
        column = self._page("维护")
        column.addWidget(QLabel("本地缓存"))
        self.cache_status = _note("可清理 7 天前的临时显示预览。")
        column.addWidget(self.cache_status)
        clear = QPushButton("清理旧预览缓存")
        clear.clicked.connect(lambda checked=False: self.clear_cache_requested.emit())
        self.clear_cache_button = clear
        column.addWidget(clear)
        release = QPushButton("释放预览内存")
        release.clicked.connect(lambda checked=False: self.release_memory_requested.emit())
        self.release_memory_button = release
        column.addWidget(release)
        column.addWidget(_note("只清理临时预览；原始素材、已保存版本和审核记录继续保留。"))
        column.addWidget(QLabel("诊断"))
        self.debug_logging = QCheckBox("记录详细调试日志")
        self.debug_logging.toggled.connect(
            lambda value: self._change("debug_logging", bool(value))
        )
        column.addWidget(self.debug_logging)
        column.addWidget(_note("详细日志保存在工作区的 outputs/logs，并自动轮转。"))
        report = QPushButton("保存诊断报告…")
        report.clicked.connect(lambda checked=False: self.diagnostics_requested.emit())
        self.diagnostics_button = report
        column.addWidget(report)
        column.addWidget(_note("诊断报告包含版本、设置和操作结果，不包含照片内容。"))
        column.addStretch()

    def _change(self, key, value):
        if self.loading:
            return
        candidate = checked_performance_settings({**self.settings, key: value})
        if candidate != self.settings:
            self.settings = candidate
            self._refresh_policy()
            self.settings_changed.emit(deepcopy(candidate))

    def _memory_saver_changed(self, checked):
        if self.loading:
            return
        self._change("cache_mb", 256 if checked else 0)
        self.set_settings(self.settings)

    def _proof_selected(self):
        if not self.loading:
            self.proof_changed.emit(self.proof_choice.currentData())

    def _refresh_policy(self):
        policy = resolve_performance(self.settings, hardware=self.hardware, include_backend=False)
        self.cpu_threads.readout.setText(
            "自动" if not self.settings["cpu_threads"] else str(self.settings["cpu_threads"])
        )
        self.cache_mb.readout.setText(
            "自动" if not self.settings["cache_mb"] else str(self.settings["cache_mb"])
        )
        previous = self.memory_saver.blockSignals(True)
        self.memory_saver.setChecked(self.settings["cache_mb"] == 256)
        self.memory_saver.blockSignals(previous)
        self.policy_status.setText(
            f"当前策略：{policy['cpu_threads']} 个 CPU 线程；"
            f"预览缓存 {policy['preview_cache_budget_bytes'] // MIB:,} MiB。"
        )

    def set_settings(self, document):
        """Synchronize without emitting edits or changing the caller's document."""
        self.loading = True
        try:
            self.settings = checked_performance_settings(document)
            self.mode_choice.setCurrentIndex(self.mode_choice.findData(self.settings["mode"]))
            self.backend_choice.setCurrentIndex(self.backend_choice.findData(self.settings["backend"]))
            self.cpu_threads.setValue(self.settings["cpu_threads"])
            self.cache_mb.setValue(self.settings["cache_mb"])
            self.idle_previews.setChecked(self.settings["idle_previews"])
            self.debug_logging.setChecked(self.settings["debug_logging"])
            self._refresh_policy()
        finally:
            self.loading = False

    def set_proof(self, key):
        index = self.proof_choice.findData(key)
        if index < 0:
            raise ValueError("未知软打样色域")
        previous = self.loading
        self.loading = True
        try:
            self.proof_choice.setCurrentIndex(index)
        finally:
            self.loading = previous

    def set_display_status(self, caption):
        self.display_status.setText(str(caption))

    def set_runtime_status(self, caption):
        self.runtime_status.setText(str(caption))

    def set_cache_status(self, caption):
        self.cache_status.setText(str(caption))
