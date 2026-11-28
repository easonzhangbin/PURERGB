"""A short travelling accent for the workbench header.

The visual reference is ismnoiet/BorderBeam. This implementation uses Qt's
rounded path and paints a finite animation on a mouse-transparent child.
"""

from PyQt6.QtCore import QElapsedTimer, QEvent, QRectF, QTimer, Qt
from PyQt6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from .workflow_glass import obsidian_enabled, observe_appearance, system_reduce_motion


class WorkflowBorderBeam(QWidget):
    _DURATION_MS = 5200
    _FADE_IN_MS = 550
    _FADE_OUT_MS = 1250
    _COLORS = (
        (107, 127, 219, 0), (108, 133, 228, 7), (112, 138, 237, 19),
        (132, 139, 238, 36), (158, 140, 231, 56), (188, 142, 219, 82),
        (217, 146, 201, 111), (243, 154, 176, 137), (250, 168, 150, 156),
        (255, 186, 151, 160), (255, 196, 161, 133), (255, 203, 168, 83),
        (255, 203, 168, 37), (255, 203, 168, 9), (255, 203, 168, 0),
    )

    def __init__(self, header):
        super().__init__(header)
        self.setObjectName("workflowBorderBeam")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._path = QPainterPath()
        self._length = 0.0
        self._phase = 0.0
        self._opacity = 0.0
        self._elapsed = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self._advance)
        self._stroke_layers = [
            (width, [QColor(r, g, b, round(alpha * strength)) for r, g, b, alpha in self._COLORS])
            for width, strength in ((10.0, 0.05), (7.0, 0.08), (4.0, 0.16), (1.25, 1.0))
        ]
        header.installEventFilter(self)
        self.setGeometry(header.rect())
        self.hide()

    def set_visible_for_theme(self, enabled):
        self.setVisible(enabled)
        if enabled:
            self.raise_()

    def pulse(self):
        if (self.isVisible() and not self.window().isMinimized()
                and not self._timer.isActive() and not system_reduce_motion()):
            self._elapsed.start()
            self._timer.start()

    def _rest(self):
        self._timer.stop()
        self._phase = 0.0
        if self._opacity:
            self._opacity = 0.0
            self.update()

    def _advance(self):
        elapsed = self._elapsed.elapsed()
        if elapsed >= self._DURATION_MS or self.window().isMinimized() or system_reduce_motion():
            self._rest()
            return
        self._phase = elapsed / self._DURATION_MS
        fade = min(1.0, elapsed / self._FADE_IN_MS, (self._DURATION_MS - elapsed) / self._FADE_OUT_MS)
        self._opacity = fade * fade * (3.0 - 2.0 * fade)
        self.update()

    def eventFilter(self, watched, event):
        if watched is self.parentWidget() and event.type() == QEvent.Type.Resize:
            self.setGeometry(watched.rect())
        return super().eventFilter(watched, event)

    def showEvent(self, event):
        super().showEvent(event)
        self.pulse()

    def hideEvent(self, event):
        self._rest()
        super().hideEvent(event)

    def resizeEvent(self, event):
        inset = QRectF(self.rect()).adjusted(2.25, 2.25, -2.25, -2.25)
        self._path = QPainterPath()
        self._path.addRoundedRect(inset, 10, 10)
        self._path.setCachingEnabled(True)
        self._length = self._path.length()
        super().resizeEvent(event)

    def paintEvent(self, event):
        if not self._length or not self._opacity:
            return
        path = self._path
        span = min(0.32, 300 / self._length)
        count = len(self._COLORS) - 1
        step = span / count
        start = self._phase - span
        segments = [path.trimmed(index * step, (index + 1) * step, start) for index in range(count)]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(self._opacity)
        for width, colors in self._stroke_layers:
            for index, segment in enumerate(segments):
                gradient = QLinearGradient(segment.pointAtPercent(0), segment.pointAtPercent(1))
                gradient.setColorAt(0, colors[index])
                gradient.setColorAt(1, colors[index + 1])
                pen = QPen(QBrush(gradient), width)
                pen.setCapStyle(Qt.PenCapStyle.FlatCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.drawPath(segment)


def attach_theme_beam(header):
    beam = WorkflowBorderBeam(header)

    def update():
        beam.set_visible_for_theme(obsidian_enabled())

    observe_appearance(beam, update)
    update()
    return beam
