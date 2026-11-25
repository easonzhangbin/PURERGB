"""Signal-level and signed-balance plots; neither is a hue vectorscope."""

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QSizePolicy, QWidget


CHANNEL_COLORS = ("#ff7777", "#72dfa0", "#83b7ff")
DEVIATION_LIMIT = 20.0
THRESHOLDS = (5.0, 10.0, 15.0)


def plot_data(analysis):
    """One testable percent-domain model shared by painting and accessibility."""
    if analysis is None:
        return dict(state="empty", signals=None, deviations=None, target=90.0,
                    thresholds=THRESHOLDS, deviation_limit=DEVIATION_LIMIT)
    signals = tuple(100 * item.signal if item.signal is not None and math.isfinite(item.signal) else None
                    for item in analysis.readings)
    deviations = None
    if analysis.usable and all(value is not None and value > 0 for value in signals):
        middle = sorted(signals)[1]
        deviations = tuple(100 * (value / middle - 1) for value in signals)
    return dict(state="valid" if deviations is not None else "unreliable", signals=signals,
                deviations=deviations, target=100 * analysis.target,
                thresholds=THRESHOLDS, deviation_limit=DEVIATION_LIMIT)


class RgbBalanceChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.data = plot_data(None)
        self.setAccessibleName("RGB 采样信号与配平偏差图")
        self.setToolTip("左图显示当前量化基准的各路信号与90%目标；右图显示相对三路中位数的偏差。\n"
                        "右图范围±20%，超出时用箭头和实际读数表示。±5/10/15%为软件分级，并非曝光补偿值。")

    def set_analysis(self, analysis):
        self.data = plot_data(analysis)
        if self.data["state"] == "empty":
            description = "尚未载入片基参考。"
        elif self.data["state"] == "unreliable":
            description = "仅显示已测得信号；配平参考待核对。"
        else:
            description = "；".join(f"{c} 信号 {level:.2f}%，相对偏差 {delta:+.2f}%"
                                   for c, level, delta in zip("RGB", self.data["signals"], self.data["deviations"]))
        self.setAccessibleDescription(description)
        self.update()

    def _panels(self):
        gap, inset = 12.0, 8.0
        width = max(1.0, (self.width() - 2 * inset - gap) / 2)
        return (QRectF(inset, 4, width, self.height() - 8),
                QRectF(inset + width + gap, 4, width, self.height() - 8))

    @staticmethod
    def _text(painter, rect, text, color="#eeeae4", *, small=False):
        font = painter.font()
        font.setPixelSize(11 if small else 13)
        painter.setFont(font)
        painter.setPen(QColor(color))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    @staticmethod
    def _track(panel):
        return QRectF(panel.left() + 29, panel.top() + 48,
                      max(1, panel.width() - 111), max(60, panel.height() - 99))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#202123"))
        left, right = self._panels()
        for panel in (left, right):
            painter.setPen(QPen(QColor("#4b4d52"), 1))
            painter.setBrush(QColor("#25272a"))
            painter.drawRoundedRect(panel, 9, 9)
        self._text(painter, QRectF(left.x(), 10, left.width(), 23), "采样信号 · 90% 安全目标")
        self._text(painter, QRectF(right.x(), 10, right.width(), 23), "配平偏差 · 三路中位数为 0%")
        if self.data["state"] == "empty":
            for panel in (left, right):
                self._text(painter, panel.adjusted(8, 35, -8, -10), "完成片基分析后显示", "#c5c6c9")
            return
        self._signal_plot(painter, left)
        if self.data["deviations"] is None:
            self._text(painter, right.adjusted(12, 35, -12, -10), "参考待核对\n核对片基与拍摄设置后重新分析", "#e7bd83")
        else:
            self._deviation_plot(painter, right)

    def _signal_plot(self, painter, panel):
        track = self._track(panel)
        for value in (0, 25, 50, 75, 100):
            x = track.x() + value / 100 * track.width()
            painter.setPen(QPen(QColor("#42454a"), 1))
            painter.drawLine(QPointF(x, track.top()), QPointF(x, track.bottom()))
            if value in (0, 50, 100):
                self._text(painter, QRectF(x - 20, track.bottom() + 3, 40, 18), f"{value}%", "#c1c4ca", small=True)
        target_x = track.x() + self.data["target"] / 100 * track.width()
        painter.setPen(QPen(QColor("#ffca70"), 1.5, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(target_x, track.top() - 3), QPointF(target_x, track.bottom()))
        self._text(painter, QRectF(target_x - 23, track.top() - 20, 46, 18), "90%", "#ffca70", small=True)
        for index, (channel, value, color) in enumerate(zip("RGB", self.data["signals"], CHANNEL_COLORS)):
            y = track.top() + (index + 0.5) / 3 * track.height()
            self._text(painter, QRectF(panel.x() + 4, y - 11, 20, 22), channel, color)
            painter.setPen(QPen(QColor("#53565d"), 2))
            painter.drawLine(QPointF(track.left(), y), QPointF(track.right(), y))
            if value is not None:
                x = track.left() + min(100, max(0, value)) / 100 * track.width()
                painter.setPen(QPen(QColor(color), 7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(QPointF(track.left(), y), QPointF(x, y))
                painter.setBrush(QColor(color))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(x, y), 4, 4)
                if value > 100:
                    painter.drawPolygon(QPolygonF((QPointF(x + 6, y), QPointF(x, y - 5), QPointF(x, y + 5))))
            text = "—" if value is None else f"{value:.2f}%"
            self._text(painter, QRectF(track.right() + 6, y - 11, 70, 22), text, color)
        self._text(painter, QRectF(panel.x() + 4, panel.bottom() - 24, panel.width() - 8, 18),
                   "当前量化基准的信号百分比", "#bfc2c8", small=True)

    def _deviation_plot(self, painter, panel):
        track = self._track(panel)

        def x_of(value):
            return track.left() + (value + DEVIATION_LIMIT) / (2 * DEVIATION_LIMIT) * track.width()

        for low, high, color in ((-20, -15, "#59393d"), (-15, -10, "#67503c"),
                                 (-10, -5, "#625c3c"), (-5, 5, "#345447"),
                                 (5, 10, "#625c3c"), (10, 15, "#67503c"), (15, 20, "#59393d")):
            painter.fillRect(QRectF(x_of(low), track.top(), x_of(high) - x_of(low), track.height()), QColor(color))
        for value in (-20, -15, -10, -5, 0, 5, 10, 15, 20):
            x = x_of(value)
            painter.setPen(QPen(QColor("#d8dee6" if value == 0 else "#77716a"), 1.5 if value == 0 else 0.6))
            painter.drawLine(QPointF(x, track.top()), QPointF(x, track.bottom()))
            self._text(painter, QRectF(x - 15, track.bottom() + 3, 30, 18),
                       f"{value:+}" if value else "0", "#c6c9ce", small=True)
        for index, (channel, delta, color) in enumerate(zip("RGB", self.data["deviations"], CHANNEL_COLORS)):
            y = track.top() + (index + 0.5) / 3 * track.height()
            self._text(painter, QRectF(panel.x() + 4, y - 11, 20, 22), channel, color)
            x = x_of(max(-DEVIATION_LIMIT, min(DEVIATION_LIMIT, delta)))
            painter.setPen(QPen(QColor(color), 2))
            painter.drawLine(QPointF(x_of(0), y), QPointF(x, y))
            painter.setBrush(QColor(color))
            if abs(delta) > DEVIATION_LIMIT:
                direction = 1 if delta > 0 else -1
                painter.drawPolygon(QPolygonF((QPointF(x, y), QPointF(x - direction * 8, y - 6), QPointF(x - direction * 8, y + 6))))
            else:
                painter.drawEllipse(QPointF(x, y), 4, 4)
            value = f"↑{delta:+.2f}%" if delta > 1e-9 else (f"↓{delta:+.2f}%" if delta < -1e-9 else "0.00%")
            self._text(painter, QRectF(track.right() + 4, y - 11, 75, 22), value, color)
        self._text(painter, QRectF(panel.x() + 4, panel.bottom() - 24, panel.width() - 8, 18),
                   "偏差（%）· ±5 最佳 / ±10 尚可 / ±15 需调整", "#d0c7b8", small=True)
