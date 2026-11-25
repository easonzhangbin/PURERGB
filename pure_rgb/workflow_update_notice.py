"""Nonblocking, opt-in-to-download notification of official releases."""

import threading

from PyQt6.QtCore import QObject, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from . import release
from .workflow_notices import NoticeLabel


class _UpdateSignals(QObject):
    checked = pyqtSignal(object)


class UpdateNotice(QWidget):
    """Check in a daemon worker; never download or replace the application."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._disposed = False
        self._checking = False
        self._started = False
        self._release_url = release.RELEASES_URL
        self._signals = _UpdateSignals(self)
        self._signals.checked.connect(self._checked)
        self._timer = QTimer(self)
        self._timer.setInterval(24 * 60 * 60 * 1000)
        self._timer.timeout.connect(self.check)
        self._startup = QTimer(self)
        self._startup.setSingleShot(True)
        self._startup.timeout.connect(self.check)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.message = NoticeLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QHBoxLayout()
        self.open_button = QPushButton("查看更新")
        self.open_button.clicked.connect(self.open_release)
        buttons.addWidget(self.open_button)
        self.dismiss_button = QPushButton("稍后")
        self.dismiss_button.clicked.connect(self.hide)
        buttons.addWidget(self.dismiss_button)
        layout.addLayout(buttons)
        self.hide()

    def schedule(self):
        if not self._disposed and not self._started:
            self._started = True
            self._startup.start(2000)
            self._timer.start()

    def check(self):
        if self._disposed or self._checking:
            return
        self._checking = True
        signals = self._signals
        check = release.check_newer_release

        def run():
            try:
                latest = check()
            except Exception:
                latest = None
            try:
                signals.checked.emit(latest)
            except RuntimeError:
                # The window can be destroyed while the network request runs.
                pass

        threading.Thread(target=run, name="PURE-RGB-update-check", daemon=True).start()

    def _checked(self, latest):
        self._checking = False
        if not self._disposed and latest:
            self._release_url = release.RELEASES_URL
            self.open_button.setText("查看更新")
            self.message.setText(f"发现 PURE RGB 新版本 {latest}。")
            self.show()

    def show_installed_release(self):
        self._release_url = f"{release.RELEASES_URL}/tag/v{release.APP_VERSION}"
        self.message.setText(
            f"已更新到 PURE RGB {release.APP_VERSION}\n"
            "RGB 灯光配平可从「导入归组」或「采集工具」打开。\n"
            "新增 RAW 曝光检查、平场选用、灯具参考和中英文说明。"
        )
        self.open_button.setText("查看更新说明")
        self.show()

    def open_release(self):
        if not QDesktopServices.openUrl(QUrl(self._release_url)):
            self.message.setText("无法打开发布页，请在设置中重试。")

    def dispose(self):
        self._disposed = True
        self._startup.stop()
        self._timer.stop()
        self.hide()
