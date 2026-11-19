"""A single persistent preview process, newest snapshot wins, no queue revisions."""

import json
import os
from pathlib import Path
import signal
import shutil
import sys
import uuid

from PyQt6.QtCore import QObject, QProcess, QTimer, Qt, pyqtSignal

from .preview_data import ROOT
from .diagnostics import event as log_event
from .preview_requests_v44 import preview_metering_path


MAX_PREVIEW_LINE_BYTES = 1_000_000


class LivePreview(QObject):
    ready = pyqtSignal(object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(str)
    progress_detail = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.directory = ROOT / "outputs/live-preview" / uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        self.process, self.pending, self.active = None, None, None
        self.latest, self.stopping = 0, False
        self.errors, self.buffer = "", b""
        self.cache_report = {}
        self.cached_source_context = None
        self.performance = None
        self.restarting = False
        self.superseded_token = None
        self.supersession_timer = QTimer(self)
        self.supersession_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.supersession_timer.setSingleShot(True)
        self.supersession_timer.timeout.connect(self._abandon_stale_worker)

    def invalidate(self, token):
        self.latest, self.pending = token, None
        LivePreview._notify_superseded(self)

    def _notify_superseded(self):
        if (self.active and self.process is not None
                and self.active["token"] != self.latest
                and self.process.state() == QProcess.ProcessState.Running):
            self.process.write((json.dumps(dict(action="supersede", token=self.latest)) + "\n").encode())
            timer = getattr(self, "supersession_timer", None)
            if timer is not None and not timer.isActive() and not self.restarting:
                self.superseded_token = self.active["token"]
                # Cooperative row/phase checkpoints normally retain the RAW
                # cache. A blocked SDK stage must never hold newer edits for
                # minutes. Initial RAW preparation gets its own longer grace.
                warm = (self.cache_report.get("retained_source_bytes", 0) > 0
                        and self.cached_source_context == self._source_context(self.active))
                timer.start(2500 if warm else 20000)

    @staticmethod
    def _source_context(request):
        recipe = request.get("draft_recipe")
        if isinstance(recipe, dict):
            from .workspace import source_recipe_identity

            return json.dumps(source_recipe_identity(recipe), sort_keys=True, separators=(",", ":"))
        return (request.get("manifest"), request.get("manifest_sha256"))

    def _abandon_stale_worker(self):
        if (self.stopping or self.restarting or not self.active
                or self.active["token"] != self.superseded_token
                or self.active["token"] == self.latest or self.process is None):
            return
        self.restarting = True
        log_event("preview_supersession_restart", token=self.active["token"], latest=self.latest)
        message = "已停止过期处理，正在准备最新调整…"
        self.progress.emit(message)
        self.progress_detail.emit(dict(token=self.latest, stage="restart", message=message))
        process = self.process
        pid = int(process.processId())
        try:
            if pid > 0 and os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            process.kill()

    def submit(self, request):
        log_event(
            "preview_submit",
            token=request["token"],
            mode=request["mode"],
            replaces_pending=self.pending is not None,
            active_token=self.active.get("token") if self.active else None,
        )
        self.latest = request["token"]
        self.pending = {**request, "directory": str(self.directory), "metering_transport": "sidecar"}
        self._notify_superseded()
        self.pump()

    def pump(self):
        if self.active or not self.pending or self.stopping or self.restarting:
            return
        if self.process is None:
            self.buffer, self.errors = b"", ""
            self.cache_report = {}
            self.cached_source_context = None
            self.process = QProcess(self)
            if self.performance is not None:
                from .workflow_performance import qprocess_environment

                self.process.setProcessEnvironment(
                    qprocess_environment(self.performance)
                )
            self.process.setWorkingDirectory(str(ROOT))
            self.process.started.connect(self.pump)
            self.process.readyReadStandardOutput.connect(self.read)
            self.process.readyReadStandardError.connect(self.read_error)
            self.process.finished.connect(self.finished)
            self.process.errorOccurred.connect(self.process_error)
            self.process.start(sys.executable, ["-m", "pure_rgb.live_worker"])
            return
        if self.process.state() != QProcess.ProcessState.Running:
            return
        self.active, self.pending = self.pending, None
        log_event(
            "preview_dispatch",
            token=self.active["token"],
            mode=self.active["mode"],
            worker_pid=int(self.process.processId()),
        )
        self.process.write((json.dumps(self.active) + "\n").encode())

    def read_error(self):
        if self.process:
            chunk = bytes(self.process.readAllStandardError()).decode(errors="replace")
            log_event("preview_stderr", token=self.latest, text=chunk)
            self.errors = (self.errors + chunk)[-8000:]

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart and self.process is not None:
            process = self.process
            self.process, self.active, self.pending = None, None, None
            process.deleteLater()
            self.failed.emit(self.latest, "无法启动画面预览")

    def read(self):
        if self.process is None:
            return
        lines = (self.buffer + bytes(self.process.readAllStandardOutput())).split(b"\n")
        self.buffer = lines.pop()
        for line in lines:
            if len(line) > MAX_PREVIEW_LINE_BYTES:
                self._reject_oversized_response()
                return
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "progress":
                if self.active and self.active["token"] == self.latest:
                    if event.get("token", self.latest) != self.latest:
                        continue
                    message = str(event.get("message") or "正在更新预览…")
                    self.progress.emit(message)
                    self.progress_detail.emit({**event, "token": self.latest, "message": message})
            elif event.get("type") in ("result", "error"):
                if self.restarting:
                    self.discard(event)
                    continue
                cache = event.get("preview_cache")
                if (type(cache) is dict and all(type(cache.get(key)) is int and cache[key] >= 0
                                               for key in ("retained_source_bytes", "derived_bytes"))):
                    self.cache_report = cache
                    self.cached_source_context = (
                        self._source_context(self.active)
                        if self.active and cache["retained_source_bytes"] else None
                    )
                self.active = None
                self.supersession_timer.stop()
                self.superseded_token = None
                if event.get("token") == self.latest:
                    if event["type"] == "result":
                        self.ready.emit(event)
                    else:
                        self.failed.emit(self.latest, str(event.get("message") or "预览处理失败"))
                else:
                    log_event(
                        "preview_discard_stale",
                        token=event.get("token"),
                        latest=self.latest,
                    )
                    self.discard(event)
                self.pump()
        if len(self.buffer) > MAX_PREVIEW_LINE_BYTES:
            self._reject_oversized_response()

    def _reject_oversized_response(self):
        if self.restarting or self.stopping or self.process is None:
            return
        failed_current = not self.pending
        self.buffer = b""
        self.active = None
        self.restarting = True
        log_event("preview_response_rejected", token=self.latest, reason="line-too-large")
        self.process.kill()
        if failed_current:
            self.failed.emit(self.latest, "预览响应超过限制，可重试。")

    @staticmethod
    def discard(event):
        records = event.get("records", [])
        if records and event.get("metering_sidecar") is not None:
            preview_metering_path(records[0]).unlink(missing_ok=True)
        for record in records:
            Path(record["path"]).unlink(missing_ok=True)

    def finished(self, code, status):
        stale_pending = bool(self.active and self.pending
                             and self.active["token"] != self.latest)
        restart = (self.restarting or stale_pending) and not self.stopping
        pending = self.pending if restart else None
        if self.active and not self.stopping and not restart:
            self.failed.emit(self.latest, "预览进程结束：" + self.errors[-600:])
        process = self.process
        self.process, self.active, self.pending = None, None, None
        self.restarting = False
        self.supersession_timer.stop()
        self.superseded_token = None
        if process:
            process.deleteLater()
        self._cleanup_draft_analysis()
        if restart:
            self.pending = pending
            self.pump()

    def _cleanup_draft_analysis(self):
        """Only private full-grid anchors, after this worker has stopped."""
        directory = self.directory
        expected_parent = (ROOT / "outputs/live-preview").resolve()
        if (
            directory.is_symlink()
            or directory.parent.resolve() != expected_parent
            or len(directory.name) != 32
            or any(value not in "0123456789abcdef" for value in directory.name)
        ):
            return
        anchors = directory / "draft-analysis"
        if anchors.is_symlink() or not anchors.is_dir():
            return
        try:
            shutil.rmtree(anchors)
        except OSError as error:
            log_event("draft_anchor_cleanup_failed", message=str(error))

    def shutdown(self):
        log_event("preview_shutdown", token=self.latest)
        self.stopping, self.pending = True, None
        self.supersession_timer.stop()
        process = self.process
        if process and process.state() != QProcess.ProcessState.NotRunning:
            pid = int(process.processId())
            try:
                if os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
            if not process.waitForFinished(1000):
                process.kill()
                process.waitForFinished(1000)

        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            self._cleanup_draft_analysis()
