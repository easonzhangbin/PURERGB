"""A device-independent, antialiased progress track with Qt accessibility."""

from PyQt6.QtCore import QElapsedTimer, QRectF, QSize, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPainterPath
from PyQt6.QtWidgets import QProgressBar, QStyleOptionProgressBar

from .workflow_glass import system_dark, system_reduce_motion


class WorkflowProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._clock = QElapsedTimer()
        self._animation = QTimer(self)
        self._animation.setInterval(33)
        self._animation.timeout.connect(self.update)

    def sizeHint(self):
        return QSize(120, 20 if self.isTextVisible() else 10)

    def minimumSizeHint(self):
        return QSize(36, self.sizeHint().height())

    def hideEvent(self, event):
        self._animation.stop()
        self._clock.invalidate()
        super().hideEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        dark = system_dark()
        accent = QColor("#ff794d" if dark else "#b83517")
        height = min(10, self.height()) - 1.0
        track = QRectF(0.5, (self.height() - height) / 2, self.width() - 1.0, height)
        radius = height / 2
        path = QPainterPath()
        path.addRoundedRect(track, radius, radius)
        painter.fillPath(path, QColor(255, 255, 255, 52) if dark else QColor(32, 38, 48, 54))
        inner = track.adjusted(0.8, 0.8, -0.8, -0.8)
        inner_path = QPainterPath()
        inner_path.addRoundedRect(inner, inner.height() / 2, inner.height() / 2)
        painter.fillPath(inner_path, QColor(255, 255, 255, 16) if dark else QColor(32, 38, 48, 12))

        busy = self.minimum() == self.maximum() == 0
        if busy:
            if not self._clock.isValid():
                self._clock.start()
                if not system_reduce_motion():
                    self._animation.start()
            width = inner.width() * 0.26
            phase = (self._clock.elapsed() % 1400) / 1400 if self._animation.isActive() else 0.5
            offset = (inner.width() + width) * phase - width
        else:
            self._animation.stop()
            self._clock.invalidate()
            span = max(1, self.maximum() - self.minimum())
            width = inner.width() * max(0, self.value() - self.minimum()) / span
            offset = 0.0
        option = QStyleOptionProgressBar()
        self.initStyleOption(option)
        if option.invertedAppearance != (self.layoutDirection() == Qt.LayoutDirection.RightToLeft):
            offset = inner.width() - offset - width
        if width > 0:
            fill = QRectF(inner.x() + offset, inner.y(), width, inner.height())
            painter.setClipPath(inner_path)
            painter.setBrush(accent)
            cap = min(width, fill.height()) / 2
            painter.drawRoundedRect(fill, cap, cap)
            painter.setClipping(False)
        if self.isTextVisible() and not busy:
            painter.setPen(QColor("#ffffff" if dark else "#17191e"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())
