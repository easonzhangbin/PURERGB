"""Visual metering/crop selection in oriented, merged source coordinates."""

import math

from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QColor, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QVBoxLayout,
    QWidget,
)


def source_rectangle(rect, preview_size, source_size):
    pw, ph = preview_size
    width, height = source_size
    x0, y0, x1, y1 = rect
    if pw <= 0 or ph <= 0 or width <= 0 or height <= 0:
        raise ValueError("无效预览尺寸")
    result = (
        max(0, math.floor(min(x0, x1) / pw * width)),
        max(0, math.floor(min(y0, y1) / ph * height)),
        min(width, math.ceil(max(x0, x1) / pw * width)),
        min(height, math.ceil(max(y0, y1) / ph * height)),
    )
    if result[2] - result[0] < 16 or result[3] - result[1] < 16:
        raise ValueError("请框选至少 16×16 像素的区域")
    return result


class RoiView(QGraphicsView):
    selected = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.drag_anchor = None
        self.setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_anchor = self.mapToScene(event.position().toPoint())
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_anchor is not None:
            self.selected.emit(
                QRectF(
                    self.drag_anchor, self.mapToScene(event.position().toPoint())
                ).normalized()
            )
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drag_anchor is not None and event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(
                QRectF(
                    self.drag_anchor, self.mapToScene(event.position().toPoint())
                ).normalized()
            )
            self.drag_anchor = None
        else:
            super().mouseReleaseEvent(event)


class GeometryPanel(QWidget):
    applied = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(self, geometry, parent=None):
        super().__init__(parent)
        self.geometry_info = geometry
        self.selection = None
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "框选密度／测光取样区域，避开片边和空光路；此操作不改变独立输出裁切。灰度预览仅供定位。"
            )
        )
        self.view = RoiView()
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.pixmap = QPixmap(geometry["preview"])
        if self.pixmap.isNull():
            raise ValueError("合成预览不存在，请重新处理生成定位预览")
        self.scene.addPixmap(self.pixmap)
        self.scene.setSceneRect(QRectF(self.pixmap.rect()))
        self.view.selected.connect(
            lambda rect: self.changed(rect.toRect(), rect.topLeft(), rect.bottomRight())
        )
        self.outline = self.scene.addRect(QRectF(), QPen(QColor("#F0A055"), 2))
        self.outline.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.outline.setZValue(1)
        layout.addWidget(self.view, 1)
        self.coordinate = QLabel("尚未选择区域")
        layout.addWidget(self.coordinate)
        controls = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.ok = controls.button(QDialogButtonBox.StandardButton.Ok)
        self.ok.setText("使用所选区域")
        from .workflow_attention import attach_attention

        attach_attention(self.ok)
        self.ok.setEnabled(False)
        controls.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        controls.accepted.connect(self.applied)
        controls.rejected.connect(self.cancelled)
        layout.addWidget(controls)
        QTimer.singleShot(
            0,
            lambda: self.view.fitInView(
                self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
            ),
        )

    def changed(self, rect, start, end):
        if rect.isEmpty():
            return
        selected = QRectF(start, end).normalized().intersected(self.scene.sceneRect())
        self.outline.setRect(selected)
        try:
            self.selection = source_rectangle(
                (selected.left(), selected.top(), selected.right(), selected.bottom()),
                (self.pixmap.width(), self.pixmap.height()),
                (self.geometry_info["width"], self.geometry_info["height"]),
            )
            self.coordinate.setText(
                "原始像素区域：" + "  ".join(map(str, self.selection))
            )
            self.ok.setEnabled(True)
        except ValueError as error:
            self.selection = None
            self.coordinate.setText(str(error))
            self.ok.setEnabled(False)


class GeometryDialog(QDialog):
    """Compatibility wrapper for initial-import geometry, not the main edit entry."""

    def __init__(self, geometry, parent=None):
        super().__init__(parent)
        from .workflow_glass import prepare_glass

        prepare_glass(self)
        self.setWindowTitle("框选密度取样区域 · 不改变输出裁切")
        self.resize(1000, 720)
        self.panel = GeometryPanel(geometry, self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.panel)
        self.panel.applied.connect(self.accept)
        self.panel.cancelled.connect(self.reject)
        for name in (
            "view",
            "scene",
            "pixmap",
            "outline",
            "coordinate",
            "ok",
            "geometry_info",
            "changed",
        ):
            setattr(self, name, getattr(self.panel, name))

    @property
    def selection(self):
        return self.panel.selection
