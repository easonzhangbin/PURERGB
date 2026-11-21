"""A short, local pulse for a button that completes a required step."""

from PyQt6.QtCore import QEvent, QObject, QRectF, Qt, QVariantAnimation
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QPushButton

from .workflow_glass import system_reduce_motion


ATTENTION_TEXT_INSET = 8
ATTENTION_TEXT_FLAGS = int(Qt.AlignmentFlag.AlignCenter) | int(Qt.TextFlag.TextHideMnemonic)


def attention_font(button):
    font = button.font()
    font.setBold(True)
    return font


class _Attention(QObject):
    def __init__(self, button, needed):
        super().__init__(button)
        self.button = button
        self.needs_attention = needed
        self.pulse = QVariantAnimation(self)
        self.pulse.setDuration(1150)
        self.pulse.setLoopCount(3)
        self.pulse.setStartValue(0.15)
        self.pulse.setKeyValueAt(0.5, 1.0)
        self.pulse.setEndValue(0.15)
        self.pulse.valueChanged.connect(button.update)
        self.pulse.finished.connect(button.update)

    def set_attention(self, needed):
        if self.needs_attention == needed:
            return
        self.needs_attention = needed
        if needed:
            self.remind()
        else:
            self.pulse.stop()
            self.button.update()

    def remind(self):
        self.pulse.stop()
        if self.needs_attention and self.button.isEnabled() and not system_reduce_motion():
            self.pulse.start()
        self.button.update()

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind in (QEvent.Type.Show, QEvent.Type.EnabledChange):
            self.remind()
        elif kind == QEvent.Type.Hide:
            self.pulse.stop()
        elif kind == QEvent.Type.Paint and self.needs_attention and watched.isEnabled():
            self.paint()
            return True
        return False

    def paint(self):
        button = self.button
        level = self.pulse.currentValue() if self.pulse.state() != self.pulse.State.Stopped else 0.68
        if button.isDown():
            level = 0.12
        painter = QPainter(button)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(button.rect()).adjusted(1, 1, -1, -1)
        gradient = QLinearGradient(bounds.topLeft(), bounds.bottomRight())
        gradient.setColorAt(0, QColor(round(182 + 52 * level), round(34 + 20 * level), 22))
        gradient.setColorAt(1, QColor(round(168 + 54 * level), round(23 + 18 * level), 16))
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor(255, 164, 129, round(90 + 130 * level)), 1))
        radius = bounds.height() / 2 if button.property("nativeMaterial") else min(14.0, bounds.height() / 2)
        painter.drawRoundedRect(bounds, radius, radius)
        if button.hasFocus():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#fff0de"), 1, Qt.PenStyle.DotLine))
            painter.drawRoundedRect(bounds.adjusted(3, 3, -3, -3), radius - 3, radius - 3)
        painter.setFont(attention_font(button))
        painter.setPen(QColor("#ffffff"))
        text = bounds.adjusted(ATTENTION_TEXT_INSET, 0, -ATTENTION_TEXT_INSET, 0)
        painter.drawText(text, ATTENTION_TEXT_FLAGS, button.text())


def attach_attention(button, *, needed=True):
    """Highlight an existing Qt button without replacing its standard role."""
    attention = _Attention(button, needed)
    button._attention = attention
    button.installEventFilter(attention)
    return attention


class AttentionButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self._attention = attach_attention(self, needed=False)
        self.pulse = self._attention.pulse

    @property
    def needs_attention(self):
        return self._attention.needs_attention

    def set_attention(self, needed):
        self._attention.set_attention(needed)

    def remind(self):
        self._attention.remind()
