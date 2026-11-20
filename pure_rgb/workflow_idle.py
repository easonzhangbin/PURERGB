"""One cancellable, generation-checked display prefetch after user inactivity."""

from collections import deque
from threading import Lock

from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal


class PrewarmGate(QObject):
    """Let foreground workers start after an active decode prewarm exits."""

    _released = pyqtSignal()

    def __init__(self, cancel, parent=None):
        super().__init__(parent)
        self._lock = Lock()
        self._cancel = cancel
        self._active = False
        self._resuming = False
        self._waiting = []
        self._released.connect(self._resume, Qt.ConnectionType.QueuedConnection)

    def enter(self, cancelled):
        """Register a prewarm atomically with the foreground cancellation."""
        with self._lock:
            if self._active or self._waiting or self._resuming or cancelled():
                return False
            self._active = True
            return True

    def leave(self):
        with self._lock:
            self._active = False
        self._released.emit()

    def defer(self, callback, on_error=None):
        """Cancel idle work and queue the foreground callback if necessary."""
        with self._lock:
            self._cancel()
            if not self._active:
                return False
            self._waiting.append((callback, on_error))
            return True

    def _resume(self):
        with self._lock:
            if self._active:
                return
            waiting, self._waiting = self._waiting, []
            self._resuming = True
        try:
            for callback, on_error in waiting:
                try:
                    callback()
                except Exception as error:
                    try:
                        if on_error is not None:
                            on_error(error)
                        else:
                            from .diagnostics import failure

                            failure("prewarm_foreground_resume_failed", error)
                    except Exception as handler_error:
                        from .diagnostics import failure

                        failure("prewarm_foreground_error_handler_failed", handler_error)
        finally:
            with self._lock:
                self._resuming = False


class IdlePreviewScheduler(QObject):
    """The owner performs IO; this object never edits photo selection or recipes."""

    started = pyqtSignal(object, int)
    cancelled = pyqtSignal(object, int)
    finished = pyqtSignal(object, int)
    failed = pyqtSignal(str)

    def __init__(self, callback, cancel_callback=None, idle_ms=4000, parent=None):
        super().__init__(parent)
        if type(idle_ms) is not int or idle_ms < 0:
            raise ValueError("闲置等待时间无效")
        self.callback, self.cancel_callback = callback, cancel_callback
        self.queue = deque()
        self.active = None
        self.generation = 0
        self.enabled = True
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(idle_ms)
        self.timer.timeout.connect(self._dispatch)

    @property
    def busy(self):
        return self.active is not None

    def _arm(self):
        if self.enabled and self.queue and not self.busy:
            self.timer.start()

    def _cancel(self, *, requeue):
        self.timer.stop()
        self.generation += 1
        previous, self.active = self.active, None
        if previous is None:
            return
        old_generation, item = previous
        if requeue:
            self.queue.appendleft(item)
        if self.cancel_callback is not None:
            try:
                self.cancel_callback(item, old_generation)
            except Exception as error:
                self.failed.emit(str(error))
        self.cancelled.emit(item, old_generation)

    def set_queue(self, items):
        self._cancel(requeue=False)
        self.queue = deque(items)
        self._arm()

    def set_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("闲置预览开关必须为布尔值")
        self.enabled = enabled
        if not enabled:
            self._cancel(requeue=True)
        else:
            self._arm()

    def user_activity(self):
        self._cancel(requeue=True)
        self._arm()

    def _dispatch(self):
        if not self.enabled or self.busy or not self.queue:
            return
        self.generation += 1
        item = self.queue.popleft()
        generation = self.generation
        self.active = (generation, item)
        self.started.emit(item, generation)
        try:
            self.callback(item, generation)
        except Exception as error:
            self.failed.emit(str(error))
            self.complete(generation)

    def complete(self, generation):
        if self.active is None or self.active[0] != generation:
            return False
        _, item = self.active
        self.active = None
        self.finished.emit(item, generation)
        # Stay idle between jobs so foreground interaction can always preempt.
        self._arm()
        return True

    def stop(self):
        self.enabled = False
        self._cancel(requeue=False)
        self.queue.clear()
