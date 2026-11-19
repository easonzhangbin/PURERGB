"""One isolated engine process at a time, with progress and process-group cancellation."""

import json
import os
from pathlib import Path
import signal
import sys
import uuid

from PyQt6.QtCore import QObject, QProcess, QTimer, pyqtSignal

from .preview_data import ROOT
from .workspace import write_json
from .diagnostics import event


class Tasks(QObject):
    started = pyqtSignal()
    progress = pyqtSignal(object)
    done = pyqtSignal(object, object)
    failed = pyqtSignal(object, str, bool)

    def __init__(self, directory, parent=None):
        super().__init__(parent)
        self.directory = Path(directory)
        self.process = None
        self.request = None
        self.performance = None

    @property
    def busy(self):
        return self.process is not None

    def start(self, request):
        if self.busy:
            raise ValueError("已有处理任务，请等待或先取消")
        self.request = request
        self.buffer, self.errors, self.result, self.cancelled = b"", "", None, False
        path = self.directory / (uuid.uuid4().hex + ".json")
        write_json(path, request)
        event(
            "task_start",
            action=request.get("action"),
            request_path=str(path),
            job=request.get("id"),
        )
        process = QProcess(self)
        if self.performance is not None:
            from .workflow_performance import qprocess_environment

            process.setProcessEnvironment(qprocess_environment(self.performance))
        self.process = process
        process.setWorkingDirectory(str(ROOT))
        process.readyReadStandardOutput.connect(self.read)
        process.readyReadStandardError.connect(self.read_error)
        process.finished.connect(self.finished)
        process.errorOccurred.connect(self.process_error)
        self.started.emit()
        process.start(sys.executable, ["-m", "pure_rgb.workflow_worker", str(path)])

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart and self.process:
            process = self.process
            request, self.process = self.request, None
            process.deleteLater()
            self.failed.emit(request, "无法启动处理进程", False)

    def read(self):
        if not self.process:
            return
        self.buffer += bytes(self.process.readAllStandardOutput())
        if len(self.buffer) > 8_000_000:
            self.errors = "处理进程返回数据过大"
            self.cancel()
            return
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            self.consume(line)

    def consume(self, line):
        if not line.strip():
            return
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            self.errors = (self.errors + line.decode("utf-8", errors="replace"))[
                -16000:
            ]
            return
        if not isinstance(event, dict):
            return
        if event.get("type") == "result":
            self.result = event
        elif event.get("type") == "progress":
            fraction = event.get("fraction")
            if type(fraction) not in (int, float) or not 0 <= fraction <= 1:
                event["fraction"] = None
            if not isinstance(event.get("message"), str):
                event["message"] = "处理中…"
            self.progress.emit(event)
        elif event.get("type") == "error":
            self.errors = str(event.get("message") or "处理失败")

    def read_error(self):
        if self.process:
            chunk = bytes(self.process.readAllStandardError()).decode(
                "utf-8", errors="replace"
            )
            event("worker_stderr", action=self.request.get("action"), text=chunk)
            self.errors = (self.errors + chunk)[-16000:]

    def finished(self, code, status):
        if not self.process:
            return
        event(
            "task_finished",
            action=self.request.get("action"),
            code=code,
            exit_status=str(status),
            cancelled=self.cancelled,
        )
        self.read()
        self.read_error()
        if self.buffer.strip():
            self.consume(self.buffer)
        process, self.process = self.process, None
        process.deleteLater()
        if (
            not self.cancelled
            and code == 0
            and status == QProcess.ExitStatus.NormalExit
            and self.result
        ):
            self.done.emit(self.request, self.result)
        else:
            self.failed.emit(
                self.request,
                "任务已取消"
                if self.cancelled
                else self.errors or f"进程未完成（{code}）",
                self.cancelled,
            )

    def cancel(self):
        if not self.process or self.request.get("action") == "repair_confirm":
            return
        self.cancelled = True
        event("task_cancel_requested", action=self.request.get("action"))
        pid = int(self.process.processId())
        if not pid:
            self.process.kill()
            return
        try:
            if os.name == "posix" and os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGTERM)
                QTimer.singleShot(3000, lambda: self.kill_group(pid))
            else:
                self.process.terminate()
        except ProcessLookupError:
            pass

    @staticmethod
    def kill_group(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def shutdown(self):
        if self.process:
            self.cancel()
            if not self.process.waitForFinished(3500):
                self.process.kill()
                self.process.waitForFinished(1000)
