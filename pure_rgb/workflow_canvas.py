"""An antialiased preview well, independent of image and selection coordinates."""

from PyQt6.QtCore import QEvent, QPoint, QPointF, QSize, QTimer
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication, QFrame, QStackedWidget, QWidget

from .preview_alpha_clip_v45 import (
    PreviewAlphaEffect, rounded_preview_alpha, rounded_preview_path,
)

__all__ = ["RoundedPreviewStack", "rounded_preview_alpha", "rounded_preview_path"]


class RoundedPreviewStack(QStackedWidget):
    """Clip composed pages with real alpha while retaining all logical geometry."""

    def __init__(self, parent=None, radius=12.0):
        super().__init__(parent)
        self._corner_radius = float(radius)
        rounded_preview_path(QSize(1, 1), self._corner_radius)
        self._alpha_effect = PreviewAlphaEffect(self)
        self._watch_timer = QTimer(self)
        self._watch_timer.setSingleShot(True)
        self._watch_timer.timeout.connect(self._watch_children)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setContentsMargins(0, 0, 0, 0)
        self.setObjectName("previewWell")
        self.setGraphicsEffect(self._alpha_effect)

    def cornerRadius(self):
        return self._corner_radius

    def setCornerRadius(self, radius):
        value = float(radius)
        rounded_preview_path(QSize(1, 1), value)
        if value != self._corner_radius:
            self._corner_radius = value
            self._update_clip()

    def containsPreviewPoint(self, point):
        """Logical hit geometry; no binary QWidget mask participates in paint."""
        x, y = point.x(), point.y()
        width, height = self.width(), self.height()
        if not (0 <= x < width and 0 <= y < height):
            return False
        radius = min(self._corner_radius, width / 2.0, height / 2.0)
        dx = max(radius - x, x - (width - radius), 0.0)
        dy = max(radius - y, y - (height - radius), 0.0)
        return dx * dx + dy * dy <= radius * radius

    def clipReport(self):
        return dict(backend="qt-alpha-effect", radius=self._corner_radius,
                    device_ratio=self.devicePixelRatioF(), binary_widget_mask=False,
                    alpha_bits=8, full_source_image_reprocessed=False)

    def _update_clip(self):
        self._alpha_effect.update()

    def _watch_children(self):
        for widget in (self, *self.findChildren(QWidget)):
            if widget.window() is self.window():
                widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.ChildAdded:
            self._watch_timer.start(0)
        if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick,
                            QEvent.Type.Wheel, QEvent.Type.ContextMenu):
            point = (event.position() if hasattr(event, "position") else QPointF(event.pos()))
            point += QPointF(watched.mapTo(self, QPoint()))
            if not self.containsPreviewPoint(point):
                parent = self.parentWidget()
                if parent is not None and isinstance(event, QMouseEvent):
                    local = point + QPointF(self.mapTo(parent, QPoint()))
                    forwarded = QMouseEvent(event.type(), local, event.globalPosition(), event.button(),
                                            event.buttons(), event.modifiers(), event.pointingDevice())
                    QApplication.sendEvent(parent, forwarded)
                # The forwarded parent press is the sole recipient. Leaving
                # this event ignored makes QApplication propagate it again.
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_corner_radius"):
            self._update_clip()

    def showEvent(self, event):
        super().showEvent(event)
        self._watch_children()
        self._update_clip()

    def event(self, event):
        result = super().event(event)
        if hasattr(self, "_corner_radius"):
            if event.type() == QEvent.Type.DevicePixelRatioChange:
                self._update_clip()
            elif event.type() == QEvent.Type.ChildAdded and hasattr(self, "_watch_timer"):
                self._watch_timer.start(0)
        return result
