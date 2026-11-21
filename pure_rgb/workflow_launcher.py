"""Launch the current desktop and present startup failures to desktop users."""

import runpy
import sys


ALREADY_OPEN = "此项目已在另一个窗口打开"


def show_startup_failure(message, *, already_open=False):
    try:
        from PyQt6.QtWidgets import QApplication, QSplashScreen

        from .workflow_glass import GlassMessageBox

        app = QApplication.instance()
        if app is None:
            from .preview_ui import prepare_plugins

            prepare_plugins()
            app = QApplication(sys.argv[:1])
        app.setApplicationName("PURE RGB")
        # A lock failure happens after the desktop creates its startup screen.
        # Only this new process's splash is closed; the existing app stays open.
        for widget in app.topLevelWidgets():
            if isinstance(widget, QSplashScreen):
                widget.close()
        if already_open:
            GlassMessageBox.information(
                None, "PURE RGB · 项目已打开",
                "此项目已在另一个 PURE RGB 窗口打开。\n请返回已打开的窗口继续编辑。",
            )
        else:
            GlassMessageBox.critical(None, "无法启动 PURE RGB", message)
    except Exception:
        # Keep a useful message even if the local Qt runtime itself is missing.
        print("无法启动 PURE RGB：" + message, file=sys.stderr)


def main():
    try:
        from .workflow_application import prepare_application_identity

        prepare_application_identity()
        # Preserve the desktop's own argument parsing, last-workspace restore,
        # SDK environment setup and QApplication lifetime without a second UI.
        runpy.run_module("pure_rgb.workflow_desktop", run_name="__main__", alter_sys=True)
    except Exception as error:
        already_open = isinstance(error, ValueError) and str(error) == ALREADY_OPEN
        show_startup_failure(str(error), already_open=already_open)
        return 0 if already_open else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
