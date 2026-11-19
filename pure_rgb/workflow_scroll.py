"""Geometry for passive AppKit blur at the edges of Qt scroll areas."""

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QScrollArea, QWidget


class WorkflowScrollArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        viewport = self.viewport()
        self._edges = []
        for direction in ("top", "bottom"):
            edge = QWidget(viewport)
            edge.setProperty("nativeScrollEdge", direction)
            edge.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            edge.hide()
            self._edges.append(edge)
        viewport.installEventFilter(self)
        scrollbar = self.verticalScrollBar()
        scrollbar.valueChanged.connect(self._update_edges)
        scrollbar.rangeChanged.connect(self._update_edges)

    def eventFilter(self, watched, event):
        if watched is self.viewport() and event.type() == QEvent.Type.Resize:
            self._update_edges()
        return super().eventFilter(watched, event)

    def _update_edges(self, *_):
        viewport = self.viewport()
        scrollbar = self.verticalScrollBar()
        depth = min(18, viewport.height() // 3)
        for edge, y, clipped in (
            (self._edges[0], 0, scrollbar.value() > scrollbar.minimum()),
            (self._edges[1], viewport.height() - depth, scrollbar.value() < scrollbar.maximum()),
        ):
            edge.setGeometry(0, y, viewport.width(), depth)
            edge.setVisible(clipped and depth > 0)
