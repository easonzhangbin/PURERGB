"""Visible, generation-bound progress for queued rendering and display work."""

import math
import time

from PyQt6.QtCore import QObject, QTimer, pyqtSignal


class PreviewActivity(QObject):
    changed = pyqtSignal(object)

    def __init__(self, parent=None, *, clock=time.monotonic):
        super().__init__(parent)
        self.clock = clock
        self.token = None
        self.state = "idle"
        self.stage = ""
        self.message = ""
        self.fraction = None
        self.started = self.updated = self.clock()
        self.finished = None
        self.quiet_until = self.started
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.publish)

    @property
    def active(self):
        return self.state == "running"

    def start(self, token, message="正在准备预览…", *, stage="queued", fraction=None, quiet_for=0):
        self.token, self.state = token, "running"
        self.stage, self.message, self.fraction = stage, str(message), fraction
        self.started = self.updated = self.clock()
        self.quiet_until = self.started + quiet_for
        self.finished = None
        self.timer.start()
        self.publish()

    def progress(self, event):
        if not self.active or event.get("token") != self.token:
            return False
        self.message = str(event.get("message") or self.message)
        self.stage = str(event.get("stage") or self.stage)
        value = event.get("fraction")
        if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1:
            self.fraction = max(self.fraction or 0, float(value))
        self.updated = self.clock()
        self.publish()
        return True

    def finish(self, token, *, state="complete", message="预览已更新"):
        if token != self.token or not self.active:
            return False
        if state not in {"complete", "failed", "cancelled"}:
            raise ValueError("未知预览状态")
        self.state, self.message, self.finished = state, str(message), self.clock()
        self.fraction = 1.0 if state == "complete" else self.fraction
        self.timer.stop()
        self.publish()
        return True

    def reset(self):
        self.timer.stop()
        self.state, self.token, self.fraction = "idle", None, None
        self.message, self.stage, self.finished = "", "", self.clock()
        self.publish()

    def publish(self):
        now = self.finished if self.finished is not None else self.clock()
        if self.active and now < self.quiet_until:
            return
        self.changed.emit(dict(
            token=self.token, state=self.state, active=self.active,
            stage=self.stage, message=self.message, fraction=self.fraction,
            elapsed_seconds=max(0.0, now - self.started),
            since_progress_seconds=max(0.0, now - self.updated),
        ))
