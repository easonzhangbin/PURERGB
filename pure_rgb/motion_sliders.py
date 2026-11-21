"""Native Qt adaptations of Jelly Slider and SmoothUI Scrubber.

The QSlider owns the parameter. Springs own only presentation coordinates, so
settling, hovering and elastic overdrag never generate extra recipe edits.
"""

import math

from PyQt6.QtCore import QEvent, QEasingCurve, QPointF, QRect, QRectF, Qt, QVariantAnimation
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QFontMetrics, QPainter, QPainterPath, QPen, QPalette
from PyQt6.QtWidgets import QAbstractSlider, QSlider

from .motion_spring import Spring
from .workflow_slider import SliderControl


def _clamp(value, low=0.0, high=1.0):
    return min(high, max(low, value))


def _alpha(color, opacity):
    result = QColor(color)
    result.setAlphaF(_clamp(opacity))
    return result


def scrubber_spring(duration=0.25, bounce=0.1):
    """Resolve SmoothUI's time/bounce spring with Motion's decay envelope.

    https://github.com/motiondivision/motion/blob/main/packages/motion-dom/src/animation/generators/spring.ts
    For this underdamped case the envelope root has a direct logarithmic form.
    """
    ratio = 1 - bounce
    frequency = math.log(ratio / (0.001 * math.sqrt(1 - ratio * ratio))) / (ratio * duration)
    return frequency * frequency, 2 * ratio * frequency


class _PaintedSlider(QSlider):
    """Keep Qt keyboard, range, focus and accessibility behavior intact."""

    def __init__(self, control=None):
        super().__init__(Qt.Orientation.Horizontal, control)
        self.control = control
        self.setProperty("motionPaintedSlider", True)
        self.dragging = self.hovering = self.keyboard_focus = False
        self._springs, self._goals = {}, {}
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        if control is not None:
            self.setAccessibleName(control.accessibleName())
        self.setMinimumWidth(50)

    def _spring(self, name, initial, stiffness, damping, update=None):
        def changed(value):
            if update is not None:
                update(value)
            self.update()
        self._goals[name] = initial
        self._springs[name] = Spring(self, initial, changed, stiffness=stiffness, damping=damping)

    def _to(self, name, value, *, immediate=False):
        self._goals[name] = value
        spring = self._springs[name]
        (spring.jump if immediate else spring.to)(value)

    def _ratio(self):
        return (self.value() - self.minimum()) / max(1, self.maximum() - self.minimum())

    def _span(self):
        return max(1.0, self.width() - 24.0)

    def _pointer_ratio(self, point):
        return (point.x() - 12.0) / self._span()

    def _logical_pointer(self, point):
        ratio = _clamp(self._pointer_ratio(point))
        step = max(1, self.singleStep())
        steps = round(ratio * (self.maximum() - self.minimum()) / step)
        self.setValue(self.maximum() if ratio == 1 else self.minimum() + steps * step)

    def _foreground(self):
        color = self.palette().color(QPalette.ColorRole.WindowText)
        return color if self.isEnabled() else _alpha(color, 0.35)

    def sync_visual(self, *, immediate=False):
        self.setAccessibleDescription(f"{self.accessibleName()} 当前值 {self.value_text()}")

    def value_text(self):
        return self.control.readout.text()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self.isEnabled():
            super().mousePressEvent(event)
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.keyboard_focus = False
        self.dragging = True
        self.setSliderDown(True)
        self._drag_point = event.position()
        self._logical_pointer(event.position())
        self._pointer_visual(event.position())
        self._interaction()
        event.accept()

    def mouseMoveEvent(self, event):
        if self.dragging:
            self._drag_point = event.position()
            self._logical_pointer(event.position())
            self._pointer_visual(event.position())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.dragging and event.button() == Qt.MouseButton.LeftButton:
            # An edit or recipe sync may update this value during the gesture.
            # Only apply a release coordinate if the pointer actually moved.
            if event.position() != self._drag_point:
                self._logical_pointer(event.position())
            self.dragging = False
            self.setSliderDown(False)
            self.sync_visual()
            self._interaction()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _pointer_visual(self, point):
        self.sync_visual(immediate=True)

    def enterEvent(self, event):
        self.hovering = True
        self._interaction()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hovering = False
        self._interaction()
        super().leaveEvent(event)

    def focusInEvent(self, event):
        self.keyboard_focus = event.reason() != Qt.FocusReason.MouseFocusReason
        self._interaction()
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        self.keyboard_focus = False
        self._interaction()
        super().focusOutEvent(event)

    def keyPressEvent(self, event):
        self.keyboard_focus = True
        super().keyPressEvent(event)
        self._interaction()

    def hideEvent(self, event):
        self.dragging = self.hovering = self.keyboard_focus = False
        self.setSliderDown(False)
        self._interaction()
        for name, spring in self._springs.items():
            spring.jump(self._goals[name])
        super().hideEvent(event)

    def changeEvent(self, event):
        if event.type() == QEvent.Type.EnabledChange and not self.isEnabled() and hasattr(self, "_springs"):
            self.dragging = self.hovering = self.keyboard_focus = False
            self.setSliderDown(False)
            self._interaction()
            if hasattr(self, "_position_motion"):
                self._position_motion.stop()
            for name, spring in self._springs.items():
                spring.jump(self._goals[name])
        super().changeEvent(event)

    def _interaction(self):
        self.update()


class JellySlider(_PaintedSlider):
    """Elastic thumb, trailing value drop, liquid neck and aligned scale."""

    def __init__(self, control):
        super().__init__(control)
        self.setFixedHeight(100)
        self._position, self._velocity = 0.0, 0.0
        self._spring("position", 0.0, 900, 50, self._move_position)
        self._spring("drop_x", 0.0, 260, 18)
        self._spring("lift", 0.0, 420, 22)
        self._spring("scale", 1.0, 520, 16)

    def _move_position(self, value):
        self._position = value
        self._velocity = self._springs["position"].velocity * self._span()
        if "drop_x" in self._springs:
            self._to("drop_x", value)

    def _position_to(self, name, target, *, immediate=False):
        stiffness, damping = {"follow": (900, 50), "release": (300, 14), "glide": (420, 34)}[name]
        self._springs["position"].tune(stiffness, damping)
        self._to("position", target, immediate=immediate)

    def sync_visual(self, *, immediate=False):
        super().sync_visual(immediate=immediate)
        if not self.dragging:
            self._position_to("release" if self.isSliderDown() else "glide", self._ratio(), immediate=immediate)
            if immediate:
                self._to("drop_x", self._ratio(), immediate=True)
                self._velocity = 0.0

    def mouseReleaseEvent(self, event):
        was_dragging = self.dragging
        super().mouseReleaseEvent(event)
        if was_dragging:
            self._position_to("release", self._ratio())

    def _pointer_visual(self, point):
        raw = self._pointer_ratio(point)
        low, high = 0.0, 1.0
        distance = 18.0 / self._span()
        if raw < low:
            raw = low - (1 - 1 / ((low - raw) * 0.55 / distance + 1)) * distance
        elif raw > high:
            raw = high + (1 - 1 / ((raw - high) * 0.55 / distance + 1)) * distance
        self._position_to("follow", raw)

    def _interaction(self):
        if not self._springs:
            return
        self._to("lift", float(self.dragging or self.keyboard_focus))
        self._to("scale", 1.16 if self.dragging else 1.08 if self.hovering else 1.0)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        foreground = self._foreground()
        x, cy, span = 12 + self._position * self._span(), 56.0, self._span()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_alpha(foreground, 0.14))
        painter.drawRoundedRect(QRectF(0, cy - 3, self.width(), 6), 3, 3)
        painter.setBrush(foreground)
        fill = _clamp(x, 0, self.width()) * _clamp((x - 12) / 12)
        painter.drawRoundedRect(QRectF(0, cy - 3, fill, 6), 3, 3)
        lift = _clamp(self._springs["lift"].value, -0.15, 1.2)
        scale = self._springs["scale"].value
        drop_x = 12 + self._springs["drop_x"].value * span
        drop_y, radius = cy - 38 * lift, 18 * (0.5 + 0.5 * lift)
        if lift > 0.02:
            neck = QPainterPath(QPointF(x - 8 * scale, cy))
            neck.cubicTo(x - 11, cy - 16, drop_x - radius * 0.8, drop_y + radius * 0.8, drop_x - radius * 0.6, drop_y)
            neck.lineTo(drop_x + radius * 0.6, drop_y)
            neck.cubicTo(drop_x + radius * 0.8, drop_y + radius * 0.8, x + 11, cy - 16, x + 8 * scale, cy)
            neck.closeSubpath()
            painter.drawPath(neck)
        painter.drawEllipse(QPointF(drop_x, drop_y), radius, radius)
        stretch = 1 + min(abs(self._velocity) / 2400, 0.4)
        painter.drawEllipse(QPointF(x, cy), 14 * stretch * scale, 14 * scale / stretch)
        if lift > 0.4:
            ink = QColor("#161719") if foreground.lightnessF() > 0.5 else QColor("#ffffff")
            painter.setPen(_alpha(ink, (lift - 0.4) / 0.45))
            font = painter.font()
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(QRectF(drop_x - 20, drop_y - 17, 40, 34), Qt.AlignmentFlag.AlignCenter,
                             self.control.readout.text().removesuffix(".0"))
        font = painter.font()
        font.setBold(False)
        font.setPointSize(8)
        painter.setFont(font)
        ratio = self._ratio()
        for index in range(21):
            at, major = 12 + span * index / 20, index % 5 == 0
            painter.setPen(QPen(_alpha(foreground, 0.85 if index / 20 <= ratio and ratio > 0 else 0.3), 1.2))
            painter.drawLine(QPointF(at, 77), QPointF(at, 85 if major else 81))
            if major:
                label = self.control.minimum() + index / 20 * (self.control.maximum() - self.control.minimum())
                painter.drawText(QRectF(at - 16, 85, 32, 15), Qt.AlignmentFlag.AlignCenter, f"{label:g}")


class ScrubberSlider(_PaintedSlider):
    """SmoothUI's labelled 36px track and gently expanding capsule thumb."""

    def __init__(self, parent=None, *, caption=None, readout=None, show_labels=True):
        super().__init__(parent)
        self._caption = caption or self.accessibleName
        self._readout = readout or (lambda: str(self.value()))
        self._show_labels = show_labels
        self._display_text = None
        self.setFixedHeight(36)
        self._spring("active", 0.0, *scrubber_spring())
        self._position = 0.0
        self._position_motion = QVariantAnimation(self)
        self._position_motion.setDuration(150)
        curve = QEasingCurve(QEasingCurve.Type.BezierSpline)
        curve.addCubicBezierSegment(QPointF(0.23, 1), QPointF(0.32, 1), QPointF(1, 1))
        self._position_motion.setEasingCurve(curve)
        self._position_motion.valueChanged.connect(self._paint_position)

    def _span(self):
        return max(1.0, self.width())

    def _pointer_ratio(self, point):
        return point.x() / self._span()

    def value_text(self):
        return self._readout()

    def sliderChange(self, change):
        super().sliderChange(change)
        if (hasattr(self, "_position_motion") and change in
                (QAbstractSlider.SliderChange.SliderValueChange, QAbstractSlider.SliderChange.SliderRangeChange)):
            self.sync_visual()

    def sync_visual(self, *, immediate=False):
        from .workflow_glass import system_reduce_motion

        text = (self._caption(), self.value_text())
        if self._display_text != text:
            self._display_text = text
            self.setAccessibleName(text[0])
            self.setAccessibleDescription(f"{text[0]} 当前值 {text[1]}")
            self.update()
        target = self._ratio()
        animation = self._position_motion
        if self._position == target and animation.state() == animation.State.Stopped:
            return
        if immediate or self.dragging or not self.isVisible() or system_reduce_motion():
            animation.stop()
            self._paint_position(target)
        elif animation.state() == animation.State.Running and animation.endValue() == target:
            return
        else:
            animation.stop()
            animation.setStartValue(self._position)
            animation.setEndValue(target)
            animation.start()

    def _paint_position(self, position):
        self._position = float(position)
        self.update()

    def _interaction(self):
        if self._springs:
            target = float(self.dragging or self.hovering or self.keyboard_focus)
            if target != self._goals["active"]:
                # Motion's duration-defined springs ignore inherited velocity.
                self._springs["active"].velocity = 0.0
            self._to("active", target)

    def hideEvent(self, event):
        self._position_motion.stop()
        self._position = self._ratio()
        super().hideEvent(event)

    def _caption_metrics(self, width):
        font = self.font()
        font.setPointSize(10)
        font.setWeight(QFont.Weight.Medium)
        number_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        number_font.setPointSize(10)
        value_width = QFontMetrics(number_font).horizontalAdvance(self.value_text()) + 2
        if isinstance(self.control, SliderControl):
            # Reserve the entire parameter range so a longer number cannot rewrap
            # the label and move the track in the middle of a drag.
            value_width = max(value_width, self.control.readout.minimumWidth())
        label_width = max(1, width - value_width - 30)
        return font, number_font, label_width, value_width

    def track_height(self, width):
        if not self._show_labels:
            return 36
        font, _, label_width, _ = self._caption_metrics(width)
        bounds = QFontMetrics(font).boundingRect(QRect(0, 0, label_width, 1_000_000),
                                                int(Qt.TextFlag.TextWordWrap), self._caption())
        return max(36, bounds.height() + 6)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        foreground = self._foreground()
        bounds = QRectF(0, 0, self.width(), self.height())
        clip = QPainterPath()
        clip.addRoundedRect(bounds, 8, 8)
        painter.setClipPath(clip)
        painter.fillRect(bounds, _alpha(foreground, 0.07))
        x = _clamp(self._position) * self.width()
        cy = self.height() / 2
        painter.fillRect(QRectF(0, 0, x, self.height()), _alpha(foreground, 0.14))
        painter.setPen(QPen(_alpha(foreground, 0.25), 1))
        for index in range(1, 10):
            at = self.width() * index / 10
            painter.drawLine(QPointF(at, cy - 3), QPointF(at, cy + 3))
        active = _clamp(self._springs["active"].value)
        size = 0.7 + 0.3 * active
        thumb_x = _clamp(x - 5, 2, self.width() - 2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_alpha(foreground, 0.15 + 0.65 * active))
        painter.drawRoundedRect(QRectF(thumb_x - 2 * size, cy - 11 * size, 4 * size, 22 * size), 2, 2)
        if not self._show_labels:
            return
        painter.setPen(foreground)
        font, number_font, label_width, value_width = self._caption_metrics(self.width())
        painter.setFont(font)
        painter.drawText(QRectF(12, 0, label_width, self.height()),
                         Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap, self._caption())
        painter.setFont(number_font)
        painter.drawText(QRectF(self.width() - 10 - value_width, 0, value_width, self.height()),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, self.value_text())


class _MotionControl(SliderControl):
    slider_type = JellySlider

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setProperty("motionComponent", self.slider_type.__name__)

    def _create_slider(self):
        return self.slider_type(self)

    def _sync(self):
        super()._sync()
        self.slider.sync_visual(immediate=not self.slider.isVisible())


class JellySliderControl(_MotionControl):
    pass


class ScrubberControl(_MotionControl):
    slider_type = ScrubberSlider

    def _create_slider(self):
        return ScrubberSlider(self, caption=self.label.text, readout=lambda: self.readout.text())

    def _layout_controls(self):
        self.label.hide()
        self.readout.hide()
        self._header.setSpacing(8)
        self._header.addWidget(self.slider, 1)
        self._header.addWidget(self.reset_button)
        self._column.addLayout(self._header)

    def heightForWidth(self, width):
        margins = self._column.contentsMargins()
        track_width = max(50, width - margins.left() - margins.right() - 8 - self.reset_button.width())
        return margins.top() + self.slider.track_height(track_width) + margins.bottom()

    def _update_text_height(self):
        if not hasattr(self, "reset_button"):
            return
        height = self.heightForWidth(self.width())
        margins = self._column.contentsMargins()
        track_height = height - margins.top() - margins.bottom()
        if self.slider.height() != track_height:
            self.slider.setFixedHeight(track_height)
        if self.minimumHeight() != height:
            self.setMinimumHeight(height)
            self.updateGeometry()

