"""A single persistent preview process, newest snapshot wins, no queue revisions."""

import json
import os
from pathlib import Path
import signal
import shutil
import sys
import time
import uuid

from PyQt6.QtCore import QObject, QProcess, QTimer, Qt, pyqtSignal

from .preview_data import ROOT
from .diagnostics import event as log_event
from .preview_fit_transport import FIT_TRANSPORT, discard_fit_reference_anchor, discard_undelivered_fit
from .preview_requests_v44 import preview_metering_path


MAX_PREVIEW_LINE_BYTES = 1_000_000
_DRAG_PIPE_POLL_MS = 6
_DRAG_PIPE_POLL_WINDOW_SECONDS = 0.28


class LivePreview(QObject):
    ready = pyqtSignal(object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(str)
    progress_detail = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.directory = ROOT / "outputs/live-preview" / uuid.uuid4().hex
        self.directory.mkdir(parents=True, mode=0o700)
        self.fit_transport = False
        self.process, self.pending, self.active = None, None, None
        self._forwarded = None
        self.latest, self.stopping = 0, False
        self.errors, self.buffer = "", b""
        self.cache_report = {}
        self.cached_source_context = None
        self.performance = None
        self.prewarm_gate = None
        self._gate_waiting = False
        self.restarting = False
        self.superseded_token = None
        self._after_drag_packet = False
        self._drag_pipe_deadline = 0.0
        self.drag_pipe_timer = QTimer(self)
        self.drag_pipe_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.drag_pipe_timer.setInterval(_DRAG_PIPE_POLL_MS)
        self.drag_pipe_timer.timeout.connect(self._check_drag_pipe)
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
            active, process = self.active, self.process
            packet = dict(action="supersede", token=self.latest)
            pending = self.pending
            if (pending and pending.get("release_priority") is True
                    and "interactive_edge" in active and "interactive_edge" not in pending
                    and self._forwarded is None and not self.restarting and not self.stopping
                    and pending["token"] > active["token"] and pending["mode"] == active["mode"]
                    and pending.get("manifest") == active.get("manifest")
                    and pending.get("manifest_sha256") == active.get("manifest_sha256")
                    and self._source_context(pending) == self._source_context(active)):
                # The worker's reader sees the newer token before its current
                # render finishes. Track its normal ordered acknowledgement.
                self._forwarded = packet = pending
                self._after_drag_packet = False
                log_event("preview_forward_release", token=pending["token"],
                          active_token=active["token"])
            timer = getattr(self, "supersession_timer", None)
            if timer is not None and not timer.isActive() and not self.restarting:
                self.superseded_token = self.active["token"]
                # Cooperative row/phase checkpoints normally retain the RAW
                # cache. A blocked SDK stage must never hold newer edits for
                # minutes. Initial RAW preparation gets its own longer grace.
                warm = (self.cache_report.get("retained_source_bytes", 0) > 0
                        and self.cached_source_context == self._source_context(self.active))
                timer.start(2500 if warm else 20000)
            process.write((json.dumps(packet) + "\n").encode())
            if sys.platform == "darwin" and "interactive_edge" in active and process.bytesToWrite():
                process.waitForBytesWritten(0)

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
        if self.fit_transport:
            self.pending["preview_transport"] = FIT_TRANSPORT
        self._notify_superseded()
        self.pump()

    def pump(self, gate_checked=False):
        if self.active or not self.pending or self.stopping or self.restarting:
            return
        if self._gate_waiting:
            return
        if self.prewarm_gate is not None and not gate_checked:
            def resume():
                self._gate_waiting = False
                self.pump(gate_checked=True)

            def resume_failed(error):
                self._gate_waiting = False
                self.pending = None
                self.active = None
                if self.process is not None:
                    if self.process.state() == QProcess.ProcessState.NotRunning:
                        self.process.deleteLater()
                        self.process = None
                    else:
                        self.process.kill()
                if not self.stopping:
                    self.failed.emit(self.latest, "无法启动画面预览：" + str(error))

            if self.prewarm_gate.defer(resume, resume_failed):
                self._gate_waiting = True
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
        during_drag = "interactive_edge" in self.active
        fast_pipe = sys.platform == "darwin" and (during_drag or self._after_drag_packet)
        self._after_drag_packet = during_drag
        log_event(
            "preview_dispatch",
            token=self.active["token"],
            mode=self.active["mode"],
            worker_pid=int(self.process.processId()),
        )
        process = self.process
        process.write((json.dumps(self.active) + "\n").encode())
        if fast_pipe and self.process is process and self.active:
            # Flush a writable pipe immediately instead of depending on the
            # socket-notifier turn. Keep checking only while a drag frame or
            # its release frame is in flight, with a short bounded deadline.
            self._drag_pipe_deadline = time.monotonic() + _DRAG_PIPE_POLL_WINDOW_SECONDS
            self.drag_pipe_timer.start()
            if process.bytesToWrite():
                process.waitForBytesWritten(0)

    def _check_drag_pipe(self):
        process = self.process
        if (process is None or not self.active or self.stopping or self.restarting
                or process.state() != QProcess.ProcessState.Running
                or time.monotonic() >= self._drag_pipe_deadline):
            self.drag_pipe_timer.stop()
            return
        if not process.bytesAvailable():
            process.waitForReadyRead(0)
        # waitForReadyRead can deliver the normal signal synchronously. Only
        # drain anything the notifier has not already consumed.
        if self.process is process and process.bytesAvailable():
            self.read()

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
                if self._forwarded is not None and event.get("token") != self.active["token"]:
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
                self.active, self._forwarded = self._forwarded, None
                if self.active is not None and self.active is self.pending:
                    self.pending = None
                self.drag_pipe_timer.stop()
                self.supersession_timer.stop()
                self.superseded_token = None
                if self.active is not None:
                    if sys.platform == "darwin":
                        self._drag_pipe_deadline = time.monotonic() + _DRAG_PIPE_POLL_WINDOW_SECONDS
                        self.drag_pipe_timer.start()
                    self._notify_superseded()
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
        self.drag_pipe_timer.stop()
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
        if event.get("crop_source") is not None:
            records = [*records, event["crop_source"]]
        for path in {record["path"] for record in records}:
            Path(path).unlink(missing_ok=True)

    def finished(self, code, status):
        stale_pending = bool(self.active and self.pending
                             and self.active["token"] != self.latest)
        restart = (self.restarting or stale_pending) and not self.stopping
        pending = self.pending if restart else None
        if self.active and not self.stopping and not restart:
            self.failed.emit(self.latest, "预览进程结束：" + self.errors[-600:])
        for request in (self.active, self._forwarded):
            if request and request.get("preview_transport") == FIT_TRANSPORT:
                try:
                    discard_undelivered_fit(self.directory, request["token"])
                except OSError as error:
                    log_event("preview_fit_cleanup_failed", message=str(error))
        process = self.process
        self.process, self.active, self.pending = None, None, None
        self._forwarded = None
        self.restarting = False
        self._after_drag_packet = False
        self.drag_pipe_timer.stop()
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
        try:
            discard_fit_reference_anchor(directory)
        except OSError as error:
            log_event("preview_fit_cleanup_failed", message=str(error))
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
        self.drag_pipe_timer.stop()
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
