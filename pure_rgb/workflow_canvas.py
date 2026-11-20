"""An antialiased preview well, independent of image and selection coordinates."""

from PyQt6.QtCore import QEvent, QPoint, QPointF, QSize, QTimer, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication, QFrame, QLabel, QStackedWidget, QWidget

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
        self.drawing_hint = QLabel(self)
        self.drawing_hint.setObjectName("drawingHint")
        self.drawing_hint.setAccessibleName("画布操作提示")
        self.drawing_hint.setTextFormat(Qt.TextFormat.PlainText)
        self.drawing_hint.setWordWrap(True)
        self.drawing_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drawing_hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.drawing_hint.setStyleSheet(
            "QLabel {background:rgba(27,26,25,225);color:#f2ede4;"
            "border:1px solid rgba(255,255,255,70);border-radius:10px;"
            "padding:10px 16px;font-size:14px;}"
        )
        self.drawing_hint.hide()
        self.currentChanged.connect(self._position_drawing_hint)

    def set_drawing_hint(self, text):
        """Show a click-through, centred instruction, or hide it when empty."""
        message = "" if text is None else str(text).strip()
        self.drawing_hint.setText(message)
        if message:
            self.drawing_hint.show()
            self._position_drawing_hint()
        else:
            self.drawing_hint.hide()

    def _position_drawing_hint(self, *_):
        self.drawing_hint.setMaximumWidth(max(100, self.width() - 48))
        self.drawing_hint.adjustSize()
        self.drawing_hint.move(
            max(0, (self.width() - self.drawing_hint.width()) // 2),
            max(0, (self.height() - self.drawing_hint.height()) // 2),
        )
        if not self.drawing_hint.isHidden():
            self.drawing_hint.raise_()

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
        if hasattr(self, "drawing_hint"):
            self._position_drawing_hint()

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
