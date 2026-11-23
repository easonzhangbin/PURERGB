"""Consistent content insets inside the workflow's native glass capsules."""

from PyQt6.QtCore import QPointF, QRect, QSignalBlocker, Qt
from PyQt6.QtGui import QPainter, QPainterPath, QPalette, QPen
from PyQt6.QtWidgets import QAbstractItemView, QComboBox, QStyle, QStyleOptionComboBox, QStylePainter


class WorkflowComboBox(QComboBox):
    """Keep Qt selection/accessibility while sizing text and chevron explicitly."""

    def syncItems(self, entries):
        """Update catalog rows without resetting the accessible item model."""
        entries = list(entries)
        with QSignalBlocker(self):
            for index, (title, data) in enumerate(entries):
                if index >= self.count():
                    self.addItem(title, data)
                else:
                    if self.itemText(index) != title:
                        self.setItemText(index, title)
                    if self.itemData(index) != data:
                        self.setItemData(index, data)
            if entries and self.currentIndex() >= len(entries):
                self.setCurrentIndex(0)
            for index in range(self.count() - 1, len(entries) - 1, -1):
                self.removeItem(index)

    def contentRect(self):
        return self.rect().adjusted(14, 0, -34, 0)

    def arrowRect(self):
        return QRect(self.width() - 24, (self.height() - 5) // 2, 9, 5)

    def hidePopup(self):
        view = self.findChild(QAbstractItemView)
        popup = view.parentWidget() if view is not None else None
        if popup is not None and popup.isVisible() and popup.property("nativeMaterial"):
            # Skip Qt's timed menu flash for the AppKit-backed popup. A
            # selection can immediately rebuild the workflow behind it.
            selection = view.selectionModel()
            if selection is not None:
                with QSignalBlocker(selection):
                    selection.clearSelection()
        super().hidePopup()

    def paintEvent(self, event):
        if self.isEditable():
            return super().paintEvent(event)
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        option.currentText = ""
        option.subControls = QStyle.SubControl.SC_ComboBoxFrame
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = option.palette.color(
            QPalette.ColorGroup.Active if self.isEnabled() else QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.ButtonText,
        )
        painter.setPen(color)
        rectangle = self.contentRect()
        caption = self.fontMetrics().elidedText(
            self.currentText(), Qt.TextElideMode.ElideRight, rectangle.width()
        )
        painter.drawText(rectangle, Qt.AlignmentFlag.AlignVCenter, caption)
        arrow = self.arrowRect()
        path = QPainterPath(QPointF(arrow.left(), arrow.top()))
        path.lineTo(arrow.center().x(), arrow.bottom())
        path.lineTo(arrow.right(), arrow.top())
        painter.setPen(
            QPen(color, 1.5, Qt.PenStyle.SolidLine,
                 Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        )
        painter.drawPath(path)
