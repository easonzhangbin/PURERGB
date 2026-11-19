"""Fixed-size disclosure indicators independent of label font fallback."""

from functools import lru_cache

from PyQt6.QtCore import QEvent, QPointF, QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap, QPolygonF
from PyQt6.QtWidgets import QPushButton


@lru_cache(maxsize=8)
def _indicator(expanded, color):
    points = ((2, 3), (10, 3), (6, 9)) if expanded else ((3, 2), (9, 6), (3, 10))
    pixmap = QPixmap(36, 36)
    pixmap.setDevicePixelRatio(3)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor.fromRgba(color))
    painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
    painter.end()
    return QIcon(pixmap)


class DisclosureButton(QPushButton):
    def __init__(self, title, expanded=False, parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(expanded)
        self.setIconSize(QSize(12, 12))
        self._update_icon(expanded)
        self.toggled.connect(self._update_icon)

    def _update_icon(self, expanded):
        color = QColor("#fff5ef") if expanded else self.palette().buttonText().color()
        self.setIcon(_indicator(expanded, color.rgba()))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            self._update_icon(self.isChecked())
