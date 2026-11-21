"""Interruptible scalar springs for the workflow's Qt controls."""

from time import monotonic

from PyQt6.QtCore import QObject, QTimer, Qt

from .workflow_glass import system_reduce_motion


class Spring(QObject):
    """Integrate the DesignBookmark oscillator in four substeps per frame."""

    def __init__(self, owner, initial, update, *, stiffness, damping, mass=1.0):
        super().__init__(owner)
        self.value = self.target = float(initial)
        self.velocity = 0.0
        self.mass = mass
        self.tune(stiffness, damping)
        self._update = update
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._advance)

    def tune(self, stiffness, damping):
        self.stiffness, self.damping = stiffness, damping

    def stop(self):
        self._timer.stop()
        self.target = self.value
        self.velocity = 0.0

    def jump(self, value):
        self.stop()
        self.value = self.target = float(value)
        self._update(self.value)

    def to(self, target):
        self.target = float(target)
        if self._timer.isActive() or (self.target == self.value and not self.velocity):
            return
        if system_reduce_motion():
            self.jump(target)
            return
        self._last = monotonic()
        self._timer.start()

    def _advance(self):
        now = monotonic()
        dt = min(now - self._last, 1 / 30) / 4
        self._last = now
        for _ in range(4):
            force = self.stiffness * (self.target - self.value) - self.damping * self.velocity
            self.velocity += force * dt / self.mass
            self.value += self.velocity * dt
        if abs(self.target - self.value) < .002 and abs(self.velocity) < .02:
            self.jump(self.target)
        else:
            self._update(self.value)
