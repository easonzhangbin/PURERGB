"""Spring dropdown with native Qt input and DesignBookmark motion tokens.

Reference: https://designbookmark.com/motion-src/spring-dropdown/
Animation timers settle; no frame loop runs while the control is idle.
"""

import math

from PyQt6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen, QRegion
from PyQt6.QtWidgets import QPushButton, QWidget

from .motion_spring import Spring
from .workflow_combo import WorkflowComboBox
from .workflow_glass import sync_popup_materials, system_reduce_motion


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _timer(owner, milliseconds, callback):
    timer = QTimer(owner)
    timer.setSingleShot(True)
    timer.timeout.connect(callback)
    timer.start(0 if system_reduce_motion() else milliseconds)
    return timer


def _screen_position(widget, point, size):
    screen = widget.screen()
    if screen is None:
        return point
    available = screen.availableGeometry()
    return QPoint(int(_clamp(point.x(), available.left(), max(available.left(), available.right() - size.width() + 1))),
                  int(_clamp(point.y(), available.top(), max(available.top(), available.bottom() - size.height() + 1))))


class _SpringRow(QPushButton):
    def __init__(self, popup, index):
        super().__init__(popup.combo.itemText(index), popup)
        self.popup, self.index = popup, index
        self.setAccessibleName(self.text())
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet("QPushButton {background:transparent;border:0;padding:0;} QPushButton:hover,QPushButton:focus,QPushButton:pressed {background:transparent;border:0;}")
        self.entered = Spring(self, 0, lambda value: self.update(), stiffness=300, damping=24)
        self.checked = Spring(self, float(index == popup.combo.currentIndex()), lambda value: self.update(), stiffness=260, damping=26)
        self.clicked.connect(lambda: popup.choose(index))
        self.setMouseTracking(True)

    def enterEvent(self, event):
        self.popup.hover(self.index)
        super().enterEvent(event)

    def focusInEvent(self, event):
        self.popup.hover(self.index)
        super().focusInEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Tab, Qt.Key.Key_Backtab,
                           Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space, Qt.Key.Key_Escape):
            self.popup.keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(_clamp(self.entered.value) * _clamp(self.popup.opacity.value))
        painter.translate(0, (self.entered.value - 1) * 14)
        painter.setPen(self.palette().color(QPalette.ColorRole.ButtonText))
        painter.setFont(self.font())
        painter.drawText(self.rect().adjusted(12, 0, -34, 0), Qt.AlignmentFlag.AlignVCenter,
                         self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, self.width() - 48))
        amount = _clamp(self.checked.value)
        if amount:
            points = [QPointF(self.width() - 24, 22), QPointF(self.width() - 20, 26), QPointF(self.width() - 12, 17)]
            path = QPainterPath(points[0])
            distance = amount * 22
            for first, second in zip(points, points[1:]):
                length = math.hypot(second.x() - first.x(), second.y() - first.y())
                if distance >= length:
                    path.lineTo(second)
                    distance -= length
                else:
                    path.lineTo(first + (second - first) * (distance / length))
                    break
            painter.setPen(QPen(painter.pen().color(), 2.2, Qt.PenStyle.SolidLine,
                                Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.drawPath(path)


class _SpringPopup(QWidget):
    ROW, PAD, GAP = 44, 6, 8

    def __init__(self, combo):
        super().__init__(combo, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint | Qt.WindowType.NoDropShadowWindowHint)
        self.combo, self.expanded, self.hovered = combo, False, -1
        self.setObjectName("springDropdownPopup")
        self.setProperty("glassPopup", True)
        self.setStyleSheet("QWidget#springDropdownPopup {background:transparent;border:0;}")
        self.setAccessibleName(combo.accessibleName() + "选项")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.rows, self.delays = [], []
        self.height_motion = Spring(self, 0, self._frame, stiffness=300, damping=23)
        self.lift = Spring(self, 0, self._frame, stiffness=260, damping=24)
        self.opacity = Spring(self, 0, self._frame, stiffness=400, damping=34)
        self.chevron = Spring(self, 0, self._frame, stiffness=300, damping=18)
        self.highlight_top = Spring(self, 6, self._frame, stiffness=620, damping=36)
        self.highlight_bottom = Spring(self, 50, self._frame, stiffness=620, damping=36)
        self.highlight_opacity = Spring(self, 0, self._frame, stiffness=400, damping=32)
        self.trigger_scale = Spring(self, 1, self._frame, stiffness=500, damping=18)
        self.base_y = 0

    def open(self):
        for row in self.rows:
            row.deleteLater()
        self.rows = [_SpringRow(self, index) for index in range(self.combo.count())]
        self.full_height = self.PAD * 2 + self.ROW * len(self.rows)
        self.setFixedSize(max(232, self.combo.width()) + 16, self.combo.height() + self.full_height + 50)
        anchor = self.combo.mapToGlobal(QPoint(0, 0))
        preferred = anchor - QPoint(8, int((self.GAP + self.full_height) / 2) + 12)
        self.move(_screen_position(self.combo, preferred, self.size()))
        self.base_y = anchor.y() - self.y()
        self.expanded, self.hovered = True, -1
        self.height_motion.jump(0)
        self.lift.jump(0)
        self.opacity.jump(0)
        self.chevron.jump(0)
        self.trigger_scale.jump(.965)
        self.show()
        self.setFocus(Qt.FocusReason.PopupFocusReason)
        self.height_motion.to(self.full_height)
        self.lift.to(-(self.GAP + self.full_height) / 2)
        self.opacity.to(1)
        self.chevron.to(1)
        self.trigger_scale.to(1)
        for index, row in enumerate(self.rows):
            self.delays.append(_timer(self, 50 + index * 45, lambda row=row: row.entered.to(1 if self.expanded else 0)))
        self.hover(max(0, self.combo.currentIndex()), snap=True)
        self._frame()

    def hover(self, index, snap=False):
        if not 0 <= index < len(self.rows) or index == self.hovered:
            return
        top = self.PAD + index * self.ROW
        if snap or self.hovered < 0:
            self.highlight_top.jump(top)
            self.highlight_bottom.jump(top + self.ROW)
        else:
            down = index > self.hovered
            lead = self.highlight_bottom if down else self.highlight_top
            trail = self.highlight_top if down else self.highlight_bottom
            lead.tune(620, 36)
            trail.tune(200, 20)
            self.highlight_top.to(top)
            self.highlight_bottom.to(top + self.ROW)
        self.hovered = index
        self.highlight_opacity.to(1)

    def choose(self, index):
        if not self.expanded or not self.combo.isEnabled():
            return
        item = self.combo.model().index(index, self.combo.modelColumn(), self.combo.rootModelIndex())
        if not self.combo.model().flags(item) & Qt.ItemFlag.ItemIsEnabled:
            return
        self.combo.setCurrentIndex(index)
        self.combo.activated.emit(index)
        for row in self.rows:
            row.checked.to(float(row.index == index))
        self.trigger_scale.jump(.955)
        self.trigger_scale.to(1)
        self.delays.append(_timer(self, 240, self.close_animated))

    def close_animated(self):
        if not self.isVisible():
            return
        self.expanded = False
        for delay in self.delays:
            delay.stop()
            delay.deleteLater()
        self.delays.clear()
        self.height_motion.to(0)
        self.opacity.to(0)
        self.lift.to(0)
        self.chevron.to(0)
        self.highlight_opacity.to(0)
        for index, row in enumerate(self.rows):
            row.setEnabled(False)
            self.delays.append(_timer(self, (len(self.rows) - 1 - index) * 20, lambda row=row: row.entered.to(0)))
        if system_reduce_motion():
            self.hide()

    def _frame(self, value=None):
        if not self.isVisible() or not hasattr(self, "full_height"):
            return
        menu_y = self.base_y + self.lift.value + self.combo.height() + self.GAP
        height = max(0, self.height_motion.value)
        for index, row in enumerate(self.rows):
            top = int(menu_y + self.PAD + index * self.ROW)
            row.setGeometry(14, top, self.width() - 28, self.ROW)
            clip = int(_clamp(menu_y + height - top, 0, self.ROW))
            row.setMask(QRegion(0, 0, row.width(), clip))
            row.setVisible(clip > 0)
            row.update()
        self.update()
        sync_popup_materials(self)
        if not self.expanded and height < .8 and self.opacity.value < .02:
            self.hide()

    def glass_regions(self):
        trigger = QRectF(8, self.base_y + self.lift.value, self.width() - 16, self.combo.height())
        menu = QRectF(8, trigger.bottom() + self.GAP, trigger.width(), max(0, self.height_motion.value))
        scale = self.trigger_scale.value
        dx, dy = (1 - scale) * trigger.width() / 2, (1 - scale) * trigger.height() / 2
        return ((trigger.adjusted(dx, dy, -dx, -dy), 16, 1.0), (menu, 18, _clamp(self.opacity.value)))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        trigger = QRectF(8, self.base_y + self.lift.value, self.width() - 16, self.combo.height())
        painter.save()
        painter.translate(trigger.center())
        painter.scale(self.trigger_scale.value, self.trigger_scale.value)
        painter.translate(-trigger.center())
        painter.setPen(Qt.PenStyle.NoPen)
        light_text = self.combo.palette().buttonText().color().lightnessF() > 0.5
        alpha = 56 if self.property("nativeMaterial") else 245
        painter.setBrush(QColor(64, 61, 59, alpha) if light_text else QColor(255, 255, 255, alpha))
        painter.drawRoundedRect(trigger, 16, 16)
        self.combo._paint_caption(painter, trigger, self.chevron.value)
        painter.restore()
        menu = QRectF(8, trigger.bottom() + self.GAP, self.width() - 16, max(0, self.height_motion.value))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setOpacity(_clamp(self.opacity.value))
        painter.setBrush(QColor(49, 47, 46, alpha) if light_text else QColor(247, 248, 250, alpha))
        painter.drawRoundedRect(menu, 18, 18)
        painter.setClipRect(menu)
        top = _clamp(min(self.highlight_top.value, self.highlight_bottom.value - 16), self.PAD, self.full_height - self.PAD - 16)
        bottom = _clamp(self.highlight_bottom.value, top + 16, self.full_height - self.PAD)
        painter.setOpacity(_clamp(self.opacity.value) * _clamp(self.highlight_opacity.value))
        painter.setBrush(QColor(255, 255, 255, 27) if light_text else QColor(32, 38, 48, 24))
        painter.drawRoundedRect(QRectF(14, menu.top() + top, self.width() - 28, bottom - top), 12, 12)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
            direction = -1 if key in (Qt.Key.Key_Up, Qt.Key.Key_Backtab) else 1
            self.hover((self.hovered + direction) % len(self.rows))
            self.rows[self.hovered].setFocus()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.choose(self.hovered)
        elif key == Qt.Key.Key_Escape:
            self.close_animated()
        else:
            return super().keyPressEvent(event)
        event.accept()

    def mousePressEvent(self, event):
        trigger = QRectF(8, self.base_y + self.lift.value, self.width() - 16, self.combo.height())
        if trigger.contains(event.position()):
            self.close_animated()
        super().mousePressEvent(event)

    def hideEvent(self, event):
        for spring in self.findChildren(Spring):
            spring.stop()
        for timer in self.delays:
            timer.stop()
        self.expanded = False
        self.combo.update()
        super().hideEvent(event)


class SpringComboBox(WorkflowComboBox):
    """QComboBox API/accessibility with the original spring picker choreography."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._motion_popup = None
        self._previous_caption = self._last_caption = ""
        self._caption_motion = Spring(self, 1, self._caption_frame, stiffness=340, damping=26)
        self.currentIndexChanged.connect(self._selected)

    def _caption_frame(self, value):
        self.update()
        if self._motion_popup is not None and self._motion_popup.isVisible():
            self._motion_popup.update()

    def _selected(self, index):
        self._previous_caption, self._last_caption = self._last_caption, self.currentText()
        if self._motion_popup is not None and self._motion_popup.isVisible():
            self._caption_motion.jump(0)
            self._caption_motion.to(1)
            self._motion_popup.update()
        else:
            self._caption_motion.jump(1)
        self.update()

    def showPopup(self):
        if self.isEditable() or not self.isEnabled() or not self.count():
            return super().showPopup() if self.isEditable() else None
        if self._motion_popup is None:
            self._motion_popup = _SpringPopup(self)
        if not self._motion_popup.isVisible():
            self._motion_popup.open()
            self.update()

    def hidePopup(self):
        if self._motion_popup is not None:
            self._motion_popup.close_animated()
        super().hidePopup()

    def _paint_caption(self, painter, rectangle, chevron=0):
        painter.save()
        painter.setFont(self.font())
        painter.setPen(self.palette().color(QPalette.ColorRole.ButtonText))
        text = rectangle.adjusted(14, 0, -34, 0)
        amount = self._caption_motion.value
        painter.setClipRect(text)
        painter.setOpacity(_clamp(1 - amount))
        painter.drawText(text.translated(0, -amount * 20), Qt.AlignmentFlag.AlignVCenter,
                         self.fontMetrics().elidedText(self._previous_caption, Qt.TextElideMode.ElideRight, int(text.width())))
        painter.setOpacity(_clamp(amount))
        painter.drawText(text.translated(0, (1 - amount) * 20), Qt.AlignmentFlag.AlignVCenter,
                         self.fontMetrics().elidedText(self.currentText(), Qt.TextElideMode.ElideRight, int(text.width())))
        painter.restore()
        painter.save()
        center = QPointF(rectangle.right() - 20, rectangle.center().y())
        painter.translate(center)
        painter.rotate(180 * chevron)
        path = QPainterPath(QPointF(-4.5, -2))
        path.lineTo(0, 2.5)
        path.lineTo(4.5, -2)
        painter.setPen(QPen(self.palette().color(QPalette.ColorRole.ButtonText), 1.5, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(path)
        painter.restore()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter,
                           Qt.Key.Key_Down, Qt.Key.Key_Up):
            if self._motion_popup is not None and self._motion_popup.isVisible():
                self._motion_popup.keyPressEvent(event)
            else:
                self.showPopup()
            event.accept()
        elif event.key() == Qt.Key.Key_Escape and self._motion_popup is not None:
            self.hidePopup()
        else:
            super().keyPressEvent(event)

    def changeEvent(self, event):
        if event.type() == QEvent.Type.EnabledChange and not self.isEnabled() and self._motion_popup is not None:
            self._motion_popup.hide()
        super().changeEvent(event)
