"""Viewport-sized handles for a source crop and its fine output rotation."""

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen


class CropHandles:
    def __init__(self, view):
        self.view = view
        self.bounds = None
        self.angle = 0.0
        self.drag = None

    def configure(self, bounds, angle):
        bounds = tuple(bounds) if bounds is not None else None
        if self.drag is not None and bounds is not None:
            action, previous = self.drag[:2]
            if action == "rotate":
                current = bounds == self.bounds and angle == self.drag[3]
            else:
                current = bounds == previous and angle == self.angle
            if current:
                return
        self.bounds = bounds
        self.angle = angle
        self.cancel()

    def cancel(self):
        self.drag = None
        self.view.viewport().unsetCursor()
        self.view.viewport().update()

    def _scene_rect(self, bounds):
        return self.view.path({"type": "rectangle", "bounds": bounds}).boundingRect()

    def _positions(self):
        frame = self.view.viewportTransform().mapRect(self._scene_rect(self.bounds))
        low, high = frame.topLeft(), frame.bottomRight()
        corners = (low, frame.topRight(), frame.bottomLeft(), high)
        y = low.y() - 25 if low.y() >= 40 else min(low.y() + 21, high.y() - 10)
        if frame.height() < 40:
            y = low.y() - 25 if low.y() >= 40 else high.y() + 21
        turn = QPointF(frame.center().x(), min(self.view.viewport().height() - 14, max(14, y)))
        return frame, corners, turn

    def _hit(self, position):
        _, corners, turn = self._positions()
        if (turn - position).manhattanLength() <= 19:
            return "rotate"
        distance, index = min(((corner - position).manhattanLength(), index)
                              for index, corner in enumerate(corners))
        if distance <= 15:
            return index
        return None

    def _model_sign(self):
        direction = self.view.mapping.output_orientation or {}
        return -1 if bool(direction.get("flip_horizontal")) != bool(direction.get("flip_vertical")) else 1

    @staticmethod
    def _polar(point, center):
        return math.degrees(math.atan2(point.y() - center.y(), point.x() - center.x()))

    def press(self, event):
        if self.bounds is None:
            return False
        hit = self._hit(event.position())
        if hit is None:
            return False
        frame, corners, _ = self._positions()
        if hit == "rotate":
            center = frame.center()
            self.drag = (hit, center, self._polar(event.position(), center), self.angle)
        else:
            # Determine which source corner the screen handle represents after
            # any quarter-turn or reflection.
            inverse, _ = self.view.viewportTransform().inverted()
            corner = inverse.map(corners[hit])
            size = self.view.pixmap.width(), self.view.pixmap.height()
            source = self.view.mapping.preview_to_source((corner.x(), corner.y()), size)
            x0, y0, x1, y1 = self.bounds
            sides = (0 if abs(source[0] - x0) < abs(source[0] - x1) else 2,
                     1 if abs(source[1] - y0) < abs(source[1] - y1) else 3)
            self.drag = (hit, self.bounds, self.view.source_point(event, clamp=False), sides)
        return True

    def _update_drag(self, event):
        if self.drag[0] == "rotate":
            _, center, start, initial = self.drag
            if (event.position() - center).manhattanLength() >= 6:
                delta = (self._polar(event.position(), center) - start + 180) % 360 - 180
                self.angle = round(max(-45, min(45, initial - delta * self._model_sign())) * 20) / 20
        else:
            _, original, start, sides = self.drag
            point = self.view.source_point(event, clamp=False)
            height, width = self.view.mapping.source_shape[:2]
            bounds = list(original)
            for axis, (side, high) in enumerate(zip(sides, (width, height))):
                candidate = round(original[side] + point[axis] - start[axis])
                opposite = original[(side + 2) % 4]
                minimum = min(2, abs(original[side] - opposite))
                bounds[side] = (max(0, min(opposite - minimum, candidate)) if side < 2
                                else min(high, max(opposite + minimum, candidate)))
            self.bounds = tuple(bounds)
        self.view.viewport().update()

    def move(self, event):
        if self.drag is not None:
            self._update_drag(event)
            return True
        if self.bounds is not None and event.buttons() == Qt.MouseButton.NoButton:
            hit = self._hit(event.position())
            if hit == "rotate":
                self.view.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            elif hit is not None:
                self.view.viewport().setCursor(Qt.CursorShape.SizeFDiagCursor if hit in (0, 3)
                                               else Qt.CursorShape.SizeBDiagCursor)
            else:
                self.view.viewport().unsetCursor()
        return False

    def release(self, event):
        if self.drag is None:
            return False
        self._update_drag(event)
        action, original = self.drag[:2]
        start_angle = self.drag[3] if action == "rotate" else None
        self.cancel()
        if action == "rotate":
            if self.angle != start_angle:
                self.view.crop_rotated.emit(self.angle)
        elif self.bounds != original:
            self.view.selected.emit({"type": "rectangle", "bounds": list(self.bounds)})
        return True

    def paint(self, painter):
        if self.bounds is None or self.view.points:
            return
        accent = QColor("#ffb478")
        scene_frame = self._scene_rect(self.bounds)
        if self.angle:
            painter.save()
            painter.setClipRect(scene_frame)
            painter.fillRect(scene_frame, QColor("#17181a"))
            center = scene_frame.center()
            painter.translate(center)
            painter.rotate(-self.angle * self._model_sign())
            painter.translate(-center)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawPixmap(scene_frame, self.view.pixmap, scene_frame)
            painter.restore()

        painter.save()
        painter.resetTransform()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        frame, corners, turn = self._positions()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(accent, 1.4))
        painter.drawRect(frame)
        painter.drawLine(QPointF(turn.x(), frame.top()), turn)
        painter.setBrush(QColor("#242220"))
        for corner in corners:
            painter.drawRoundedRect(QRectF(corner.x() - 4.5, corner.y() - 4.5, 9, 9), 2.5, 2.5)
        knob = QRectF(turn.x() - 11, turn.y() - 11, 22, 22)
        painter.drawEllipse(knob)
        painter.drawText(knob, Qt.AlignmentFlag.AlignCenter, "↻")
        if self.angle:
            label = QRectF(turn.x() + 16, turn.y() - 10, 52, 20)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(label, 5, 5)
            painter.setPen(accent)
            painter.drawText(label, Qt.AlignmentFlag.AlignCenter, f"{self.angle:g}°")
        painter.restore()
