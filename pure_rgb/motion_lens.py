"""Draggable glass loupe over the color-managed photo canvas."""

from math import exp, sqrt

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen, QRadialGradient

from .motion_spring import Spring


class PhotoLens(QObject):
    # FilmLoupe's blue lower-right and green upper-right handles in Qt coordinates.
    handles = {"size": QPointF(1, 1) / sqrt(2), "zoom": QPointF(1, -1) / sqrt(2)}

    def __init__(self, canvas, button):
        super().__init__(canvas)
        self.canvas = canvas
        self.button = button
        self.diameter = 288.
        self.magnification = 3.
        self.active = False
        self.drag_offset = None
        self.adjustment = None
        self.painted = QRectF()
        # Position in normalized photo coordinates keeps the same detail under
        # the glass when the preview changes resolution or the canvas resizes.
        self.follow_x = Spring(self, .5, self._frame, stiffness=180, damping=17)
        self.follow_y = Spring(self, .5, self._frame, stiffness=180, damping=17)
        self.zoom = Spring(self, self.magnification, self._frame, stiffness=260, damping=26)
        canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        canvas.installEventFilter(self)
        button.toggled.connect(self._toggle)

    @property
    def radius(self):
        return min(self.diameter, max(1, min(self.canvas.width(), self.canvas.height()) - 12)) / 2

    def _toggle(self, checked):
        if checked:
            visible = self._photo_rect().intersected(QRectF(self.canvas.rect()))
            self.canvas.setFocus(Qt.FocusReason.OtherFocusReason)
            if not visible.isEmpty():
                self.follow(visible.center())
        else:
            self.dismiss()

    def _photo_rect(self):
        canvas = self.canvas
        if not canvas.current:
            return QRectF()
        return QRectF(canvas.origin, canvas.current.size().toSizeF() * canvas.zoom)

    def _focus(self, target=False):
        photo = self._photo_rect()
        x = self.follow_x.target if target else self.follow_x.value
        y = self.follow_y.target if target else self.follow_y.value
        return QPointF(photo.left() + photo.width() * x, photo.top() + photo.height() * y)

    def _center(self):
        # Move the glass inward at the canvas edges, but still sample the
        # requested point, including the very first and last source pixels.
        focus = self._focus()
        inset = self.radius + 2.
        return QPointF(min(max(inset, focus.x()), self.canvas.width() - inset),
                       min(max(inset, focus.y()), self.canvas.height() - inset))

    def _frame(self, _=None):
        center = self._center()
        radius = self.radius + 2
        next_rect = QRectF(center.x() - radius, center.y() - radius, 2 * radius, 2 * radius)
        self.canvas.update(self.painted.united(next_rect).toAlignedRect())
        self.painted = next_rect

    def follow(self, point, *, animate=False):
        photo = self._photo_rect()
        visible = photo.intersected(QRectF(self.canvas.rect()))
        if visible.isEmpty():
            return
        x = (min(max(point.x(), visible.left()), visible.right()) - photo.left()) / photo.width()
        y = (min(max(point.y(), visible.top()), visible.bottom()) - photo.top()) / photo.height()
        if self.active and animate:
            self.follow_x.to(x)
            self.follow_y.to(y)
        else:
            self.follow_x.jump(x)
            self.follow_y.jump(y)
        if not self.active:
            self.active = True
            self.zoom.jump(self.magnification)
        self.canvas._schedule_detail(180)

    def dismiss(self):
        self.drag_offset = self.adjustment = None
        if not self.active:
            return
        self.active = False
        for spring in (self.follow_x, self.follow_y, self.zoom):
            spring.stop()
        self.canvas.update(self.painted.toAlignedRect())
        self.painted = QRectF()
        if not self.canvas.fit_mode and self.canvas.isVisible():
            self.canvas._schedule_detail()

    def sample_view(self):
        """Region in canvas coordinates needed once the lens comes to rest."""
        radius = self.radius / self.zoom.target
        focus = self._focus(target=True)
        return QRectF(focus.x() - radius, focus.y() - radius, 2 * radius, 2 * radius)

    def _handle_center(self, name):
        return self._center() + self.handles[name] * (self.radius - 7.)

    def _adjust(self, kind, value):
        if kind == "size":
            maximum = max(120, min(480, self.canvas.width() - 12, self.canvas.height() - 12))
            self.diameter = min(maximum, max(120., value))
            self._frame()
            self.canvas._schedule_detail(180)
        else:
            self.magnification = min(8., max(1.5, value))
            self._set_zoom_target()

    def _set_zoom_target(self):
        pressed = self.drag_offset is not None
        self.zoom.to(min(8., self.magnification * (2.4 / 1.7 if pressed else 1.)))
        self.canvas._schedule_detail(180)

    def _press(self, point):
        if self.active:
            for name in self.handles:
                if (point - self._handle_center(name)).manhattanLength() <= 14:
                    value = self.radius * 2 if name == "size" else self.magnification
                    self.adjustment = name, point, value
                    return True
        distance = point - self._center()
        if self.active and QPointF.dotProduct(distance, distance) <= self.radius ** 2:
            self.drag_offset = self._focus() - point
        elif self._photo_rect().contains(point):
            self.follow(point)
            self.drag_offset = QPointF()
        else:
            return False
        self._set_zoom_target()
        return True

    def _move(self, point):
        if self.adjustment:
            name, start, value = self.adjustment
            delta = QPointF.dotProduct(point - start, self.handles[name])
            self._adjust(name, value + 2 * delta if name == "size" else value * exp(delta / 90.))
        elif self.drag_offset is not None:
            self.follow(point + self.drag_offset)
        else:
            return False
        return True

    def _key(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.button.setChecked(False)
            self.button.setFocus(Qt.FocusReason.ShortcutFocusReason)
            return True
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal, Qt.Key.Key_Minus):
            increase = key != Qt.Key.Key_Minus
            if event.modifiers() & Qt.KeyboardModifier.AltModifier:
                self._adjust("size", self.radius * 2 + (24 if increase else -24))
            else:
                self._adjust("zoom", self.magnification * (1.2 if increase else 1 / 1.2))
            return True
        offsets = {Qt.Key.Key_Left: (-8, 0), Qt.Key.Key_Right: (8, 0),
                   Qt.Key.Key_Up: (0, -8), Qt.Key.Key_Down: (0, 8)}
        if key in offsets:
            self.follow(self._focus(target=True) + QPointF(*offsets[key]), animate=True)
            return True
        return False

    def eventFilter(self, watched, event):
        if sip.isdeleted(self.button) or not self.button.isChecked():
            return False
        kind = event.type()
        if kind == QEvent.Type.Hide:
            self.button.setChecked(False)
        elif kind == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            return self._press(event.position())
        elif kind == QEvent.Type.MouseMove:
            return self._move(event.position())
        elif kind == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            if self.drag_offset is not None or self.adjustment:
                self._move(event.position())
                self.drag_offset = self.adjustment = None
                self._set_zoom_target()
                return True
        elif kind == QEvent.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton:
            self.button.setChecked(False)
            return True
        elif kind == QEvent.Type.KeyPress:
            return self._key(event)
        elif kind in (QEvent.Type.Wheel, QEvent.Type.NativeGesture) and self.active:
            distance = event.position() - self._center()
            if QPointF.dotProduct(distance, distance) <= self.radius ** 2:
                if kind == QEvent.Type.Wheel:
                    delta = event.pixelDelta().y() * .012 or event.angleDelta().y() / 120 * .16
                elif event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                    delta = event.value()
                else:
                    return False
                if delta:
                    self._adjust("zoom", self.magnification * exp(max(-2., min(2., delta))))
                    return True
        return False

    def paint(self, painter):
        if not self.active or not self.canvas.current:
            return
        center, focus, radius = self._center(), self._focus(), self.radius
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for inner, bend in ((0, 1.0), (0.86, 1.06)):
            path = QPainterPath()
            path.addEllipse(center, radius, radius)
            if inner:
                path.addEllipse(center, radius * inner, radius * inner)
            painter.save()
            painter.setClipPath(path, Qt.ClipOperation.IntersectClip)
            painter.fillRect(path.boundingRect(), QColor("#141517"))
            painter.translate(center)
            painter.scale(self.zoom.value * bend, self.zoom.value * bend)
            painter.translate(-focus)
            self.canvas.paint_photos(painter)
            painter.restore()
        rim = QRadialGradient(center, radius)
        rim.setColorAt(0.86, QColor(255, 255, 255, 0))
        rim.setColorAt(0.96, QColor(16, 19, 22, 42))
        rim.setColorAt(1, QColor(255, 255, 255, 100))
        painter.setPen(QPen(QColor(220, 225, 230, 170), 1))
        painter.setBrush(rim)
        painter.drawEllipse(center, radius, radius)
        for name, color in (("size", "#0a84ff"), ("zoom", "#30d158")):
            painter.setBrush(QColor(color))
            painter.drawEllipse(self._handle_center(name), 4.5, 4.5)
        pos = self._handle_center("zoom")
        font = painter.font()
        font.setPointSizeF(10.)
        font.setBold(True)
        painter.setFont(font)
        label = f"{self.zoom.target:.1f}".removesuffix(".0") + "×"
        width = painter.fontMetrics().horizontalAdvance(label) + 10
        badge = QRectF(pos.x() - width - 10, pos.y() - 9, width, 18)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 184))
        painter.drawRoundedRect(badge, 9, 9)
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, label)
        painter.restore()
