"""Antialiased preview clipping without changing the displayed photo pixels."""

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath
from PyQt6.QtWidgets import QGraphicsEffect


def rounded_preview_path(size, radius=12.0):
    radius = float(radius)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("Preview corner radius must be finite and nonnegative")
    path = QPainterPath()
    width, height = size.width(), size.height()
    if width > 0 and height > 0:
        radius = min(radius, width / 2.0, height / 2.0)
        path.addRoundedRect(QRectF(0, 0, width, height), radius, radius)
    return path


def rounded_preview_alpha(size, radius=12.0, device_ratio=1.0, *, offset=QPointF()):
    """Return an 8-bit coverage mask at the actual backing-store density."""
    path = rounded_preview_path(size, radius)
    ratio = float(device_ratio)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Preview device ratio must be finite and positive")
    if size.isEmpty():
        return QImage()
    image = QImage(math.ceil(size.width() * ratio), math.ceil(size.height() * ratio),
                   QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(ratio)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.translate(-offset)
    painter.fillPath(path, QColor(Qt.GlobalColor.white))
    painter.end()
    return image


class PreviewAlphaEffect(QGraphicsEffect):
    """Raster fallback: coverage applies to the composed child tree once."""

    def __init__(self, widget):
        super().__init__(widget)
        self.widget = widget
        self._mask_key = self._mask = None
        self._drawing = False

    def draw(self, painter):
        if self._drawing or painter is None or not painter.isActive():
            return
        self._drawing = True
        try:
            self._draw_frame(painter)
        finally:
            self._drawing = False

    def _draw_frame(self, painter):
        pixmap, offset = self.sourcePixmap(
            Qt.CoordinateSystem.LogicalCoordinates, QGraphicsEffect.PixmapPadMode.NoPad)
        if pixmap.isNull() or not painter.isActive():
            return
        offset = offset or QPointF()
        ratio = pixmap.devicePixelRatio()
        key = (self.widget.width(), self.widget.height(), self.widget.cornerRadius(),
               ratio, offset.x(), offset.y())
        if key != self._mask_key:
            self._mask = rounded_preview_alpha(self.widget.size(), self.widget.cornerRadius(),
                                               ratio, offset=QPointF(offset))
            self._mask_key = key
        composed = pixmap.copy()
        alpha = QPainter(composed)
        if not alpha.isActive():
            return
        alpha.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
        alpha.drawImage(QPointF(), self._mask)
        alpha.end()
        painter.drawPixmap(offset, composed)
