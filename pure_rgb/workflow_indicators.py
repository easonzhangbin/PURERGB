"""Antialiased check indicators that retain Qt's layout and hit targets."""

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox, QStyle, QStyledItemDelegate, QStyleOptionButton, QStyleOptionViewItem,
)


def _paint_indicator(painter, rect, state, checked, accent, round_shape):
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    enabled = bool(state & QStyle.StateFlag.State_Enabled)
    hovered = enabled and bool(state & QStyle.StateFlag.State_MouseOver)
    color = QColor(accent) if enabled and checked else QColor(255, 255, 255, 60 if checked else 26)
    edge = color if checked else QColor(255, 255, 255, 145 if hovered else 95 if enabled else 40)
    painter.setPen(QPen(edge, 1))
    painter.setBrush(color)
    bounds = QRectF(rect).adjusted(0.5, 0.5, -0.5, -0.5)
    if round_shape:
        painter.drawEllipse(bounds)
    else:
        painter.drawRoundedRect(bounds, 4, 4)
    painter.restore()


class CheckIndicatorDelegate(QStyledItemDelegate):
    def __init__(self, view, accent, *, round_shape=False):
        super().__init__(view)
        self.accent = accent
        self.round_shape = round_shape

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        row = QStyleOptionViewItem(option)
        self.initStyleOption(row, index)
        if row.features & QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator:
            rect = row.widget.style().subElementRect(
                QStyle.SubElement.SE_ItemViewItemCheckIndicator, row, row.widget
            )
            _paint_indicator(
                painter, rect, row.state, row.checkState != Qt.CheckState.Unchecked,
                self.accent, self.round_shape,
            )


class ReviewCheckBox(QCheckBox):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(24)

    def paintEvent(self, event):
        super().paintEvent(event)
        option = QStyleOptionButton()
        self.initStyleOption(option)
        rect = self.style().subElementRect(QStyle.SubElement.SE_CheckBoxIndicator, option, self)
        painter = QPainter(self)
        _paint_indicator(
            painter, rect, option.state, self.checkState() != Qt.CheckState.Unchecked,
            "#f0a055", False,
        )
