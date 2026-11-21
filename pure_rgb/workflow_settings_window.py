"""Independent, reusable settings window with the workflow's native materials.

The existing SettingsPanel retains its policy/proof/maintenance signal API.
This shell owns only presentation and its Cocoa resources; the main workspace
continues to own persistence, workers, the selected stage and image state.
"""

from pathlib import Path

from PyQt6.QtCore import Qt, QThreadPool, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QImage, QTextDocument
from PyQt6.QtWidgets import (
    QApplication, QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout, QWidget,
)

from .branding import application_icon
from .release import (
    APP_VERSION, OPERATING_MANUAL_PATH, OPERATING_MANUAL_URL, RELEASES_URL,
    check_newer_release,
)
from .workflow_branding import header_logo
from .workflow_glass import (
    WORKSPACE_INSET, WORKSPACE_GAP, attach_glass, detach_glass, prepare_glass, stylesheet,
)
from .workflow_materials import attach_materials
from .workflow_settings_ui import SettingsPanel
from .workflow_support import donation_dialog


class SettingsWindow(QWidget):
    """A nonmodal owned window; closing hides it and keeps the current tab."""

    visibility_changed = pyqtSignal(bool)

    def event(self, event):
        from .workflow_materials import material_frame_event

        return material_frame_event(self, event, super().event)

    def __init__(self, settings=None, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setObjectName("workflowRoot")
        self.setWindowTitle("PURE RGB · 设置")
        self.setWindowIcon(application_icon())
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setMinimumSize(580, 500)
        self.resize(680, 680)
        self.closing = self.disposed = False
        self._surface_installed = False
        self._positioned = False
        self.glass_report = dict(mode="pending", attached=False)
        self._material_timer = QTimer(self)
        self._material_timer.setSingleShot(True)
        self._material_timer.timeout.connect(self._install_materials)
        self._update_pending = False
        self._manual_dialog = None
        self._donation_dialog = None
        prepare_glass(self)
        self._apply_style(False)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(WORKSPACE_INSET, WORKSPACE_INSET, WORKSPACE_INSET, WORKSPACE_INSET)
        outer.setSpacing(WORKSPACE_GAP)
        header = QFrame()
        header.setObjectName("workflowHeader")
        row = QHBoxLayout(header)
        row.setContentsMargins(18, 14, 18, 14)
        logo = QLabel()
        logo.setPixmap(header_logo(26))
        logo.setAccessibleName("PURE RGB")
        row.addWidget(logo)
        row.addStretch()
        title = QLabel("设置")
        title.setObjectName("settingsTitle")
        row.addWidget(title)
        outer.addWidget(header)

        body = QFrame()
        body.setObjectName("glassPanel")
        content = QVBoxLayout(body)
        content.setContentsMargins(18, 16, 18, 16)
        self.panel = SettingsPanel(settings, body)
        self.panel.manual_button.clicked.connect(self.open_manual)
        self.panel.donation_button.clicked.connect(self.open_donation)
        self.panel.update_button.clicked.connect(self.check_updates)
        self.panel.releases_button.clicked.connect(self.open_releases)
        content.addWidget(self.panel)
        outer.addWidget(body, 1)

        footer = QFrame()
        footer.setObjectName("workflowFooter")
        buttons = QHBoxLayout(footer)
        buttons.setContentsMargins(16, 10, 12, 10)
        note = QLabel("设置自动保存")
        note.setObjectName("muted")
        buttons.addWidget(note)
        buttons.addStretch()
        self.close_button = QPushButton("完成")
        from .workflow_attention import attach_attention

        attach_attention(self.close_button)
        self.close_button.setObjectName("primary")
        self.close_button.setMinimumWidth(100)
        self.close_button.clicked.connect(self.close)
        buttons.addWidget(self.close_button)
        outer.addWidget(footer)

        application = QApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self.dispose)
            application.styleHints().colorSchemeChanged.connect(self._system_appearance_changed)
        if parent is not None:
            parent.destroyed.connect(self.dispose)

    def _release_status(self, message, *, show_releases=False):
        self.panel.release_status.setText(message)
        self.panel.release_status.show()
        self.panel.releases_button.setVisible(show_releases)

    def open_manual(self):
        if self._manual_dialog is not None:
            self._manual_dialog.raise_()
            self._manual_dialog.activateWindow()
            return
        manual = Path(__file__).resolve().parents[1] / OPERATING_MANUAL_PATH
        try:
            text = manual.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            if not QDesktopServices.openUrl(QUrl(OPERATING_MANUAL_URL)):
                self.present()
                self._release_status("无法打开使用指南。")
            return
        dialog = QDialog(self if self.isVisible() else self.parentWidget())
        prepare_glass(dialog)
        dialog.setObjectName("workflowRoot")
        dialog.setStyleSheet(stylesheet(False))
        dialog.setWindowTitle("PURE RGB · 使用指南")
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.resize(760, 680)
        layout = QVBoxLayout(dialog)
        viewer = QTextBrowser()
        document = viewer.document()
        document.setBaseUrl(QUrl(OPERATING_MANUAL_URL))
        viewer.setOpenExternalLinks(True)
        viewer.setMarkdown(text)
        for path in (manual.parent / "images").glob("*.png"):
            picture = QImage(str(path))
            if picture.isNull():
                continue
            name = document.baseUrl().resolved(QUrl(f"images/{path.name}"))
            document.addResource(
                QTextDocument.ResourceType.ImageResource,
                name,
                picture.scaledToWidth(min(680, picture.width()), Qt.TransformationMode.SmoothTransformation),
            )
        document.markContentsDirty(0, document.characterCount())
        layout.addWidget(viewer)
        close = QPushButton("关闭")
        close.clicked.connect(dialog.close)
        layout.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)
        self._manual_dialog = dialog
        dialog.finished.connect(lambda _: setattr(self, "_manual_dialog", None))
        dialog.show()

    def open_donation(self):
        if self._donation_dialog is not None:
            self._donation_dialog.raise_()
            self._donation_dialog.activateWindow()
            return
        dialog = donation_dialog(self if self.isVisible() else self.parentWidget())
        self._donation_dialog = dialog
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.finished.connect(lambda _: setattr(self, "_donation_dialog", None))
        dialog.show()

    def open_releases(self):
        if not QDesktopServices.openUrl(QUrl(RELEASES_URL)):
            self._release_status("无法打开 GitHub 发布页。", show_releases=True)

    def check_updates(self):
        if self._update_pending or self.disposed:
            return
        from .preview_ui import Work

        self._update_pending = True
        self.panel.update_button.setEnabled(False)
        self._release_status("正在检查 GitHub 正式版本…")
        work = Work(None, check_newer_release)
        work.signals.done.connect(self._update_finished)
        work.signals.failed.connect(self._update_failed)
        QThreadPool.globalInstance().start(work)

    def _update_finished(self, _, latest):
        self._update_pending = False
        if not self.disposed:
            self.panel.update_button.setEnabled(True)
            if latest is None:
                self._release_status(f"已是最新正式版本 {APP_VERSION}。")
            else:
                self._release_status(f"发现新版本 {latest}。", show_releases=True)

    def _update_failed(self, _, message):
        self._update_pending = False
        if not self.disposed:
            self.panel.update_button.setEnabled(True)
            self._release_status(message, show_releases=True)

    def _apply_style(self, glass):
        self.setStyleSheet(stylesheet(glass) + "\nQLabel#settingsTitle { font-size:18px; font-weight:600; }")

    def _system_appearance_changed(self, *_):
        self._apply_style(self._surface_installed and self.glass_report["attached"])

    def _position_initially(self):
        if self._positioned:
            return
        self._positioned = True
        owner = self.parentWidget()
        screen = owner.screen() if owner is not None else self.screen()
        if screen is None:
            return
        available = screen.availableGeometry()
        center = owner.frameGeometry().center() if owner is not None else available.center()
        # Keep the initial independent window on its owner's screen. Later
        # presentations preserve the user's chosen window position and size.
        width, height = min(self.width(), available.width()), min(self.height(), available.height())
        self.resize(width, height)
        x = min(max(center.x() - width // 2, available.left()), available.right() - width + 1)
        y = min(max(center.y() - height // 2, available.top()), available.bottom() - height + 1)
        self.move(x, y)

    def present(self):
        """Show/focus this instance without entering a modal event loop."""
        if self.disposed:
            raise RuntimeError("设置窗口已释放")
        self._position_initially()
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        self.raise_()
        self.activateWindow()

    def _install_materials(self):
        if self.closing or self.disposed or not self.isVisible() or self._surface_installed:
            return
        self.glass_report = attach_glass(self, base="under", veil=None)
        self._apply_style(self.glass_report["attached"])
        if self.glass_report["attached"]:
            self.glass_report["controls"] = attach_materials(self)
        self._surface_installed = True

    def _release_native(self):
        self._material_timer.stop()
        if not self._surface_installed and not hasattr(self, "_workflow_material_binder"):
            return
        detach_glass(self)
        # attach_materials normally reuses a binder. A hidden settings window
        # disposes its native siblings, so reopening needs a fresh binder.
        binder = getattr(self, "_workflow_material_binder", None)
        if binder is not None:
            del self._workflow_material_binder
            binder.deleteLater()
        self._surface_installed = False

    def showEvent(self, event):
        super().showEvent(event)
        if not self.disposed:
            self._material_timer.start(0)
            self.visibility_changed.emit(True)

    def hideEvent(self, event):
        self._release_native()
        super().hideEvent(event)
        self.visibility_changed.emit(False)

    def closeEvent(self, event):
        self._release_native()
        super().closeEvent(event)

    def dispose(self, *args):
        """Detach Cocoa siblings before owner/Qt teardown; safe to call twice."""
        if self.disposed:
            return
        self.closing = self.disposed = True
        self._release_native()
        self.hide()
