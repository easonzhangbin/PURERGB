"""Display relative film-base balance and the uncorrected RAW capture limit."""

import math

from PyQt6.QtCore import QPointF, QRect, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QSizePolicy, QWidget

from .rgb_light_balance import FULL_REGION, MeterAdvice
from .workflow_glass import obsidian_enabled


CHANNEL_COLORS = ("#ff8983", "#7addad", "#91baff")
DEVIATION_LIMIT = 20.0
HOLD_RANGE = 5.0

PANEL = QColor("#2b2927")
EDGE = QColor("#514c47")
GRID = QColor("#4d4843")
RAIL = QColor("#625d57")
OBSIDIAN_PANEL = QColor("#141517")
OBSIDIAN_EDGE = QColor("#34363c")
OBSIDIAN_GRID = QColor("#3e4148")
OBSIDIAN_RAIL = QColor("#555962")
TEXT = QColor("#f2ede4")
MUTED = QColor("#c6bfb6")
WARNING = QColor("#ffd18f")
ADVICE_COLORS = {
    "good": QColor("#93dfae"), "warning": WARNING, "danger": QColor("#ffaaa3"), "neutral": TEXT,
}
DETAIL_TOP = 61.0
CARD_PADDING = 14.0
DETAIL_FONT_SIZE = 12
CHART_BOTTOM = 61.0
MIN_TRACK_HEIGHT = 87.0
CAPTURE_CAPTION_HEIGHT = 18.0
OUTSIDE_CAPTION_HEIGHT = 15.0
RAW_OCCUPANCY = "原始 RAW 占用 · P99.9"
RAW_LIMIT = "100% 为 RAW 白电平上限"


def plot_data(analysis):
    """Prepare the shared display and accessibility model."""
    if analysis is None:
        return dict(state="empty", deviations=None, imbalance=None, captures=(),
                    full_frame=True, capture_confirmed=True, flatfield_applied=False, outside_clips=0,
                    balance_advice=MeterAdvice("等待片基参考", "载入三张片基 RAW 后分析。", "neutral"),
                    exposure_advice=MeterAdvice("等待 RAW 读数", "分析后检查保留画幅的曝光余量。", "neutral"))
    deviations = analysis.deviations_percent
    flatfield_applied = bool(analysis.flatfield and analysis.flatfield.get("applied"))
    full_frame = tuple(analysis.capture_region) == FULL_REGION
    captures = []
    outside_clips = 0
    for reading in analysis.readings:
        stats = reading.capture
        if stats is None:
            captures.append(None)
            continue
        value = stats.limit_signal
        level = 100 * value if value is not None and math.isfinite(value) else None
        outside = (max(0, reading.full_capture.clipped_count - stats.clipped_count)
                   if not full_frame and reading.full_capture is not None else 0)
        captures.append(dict(level=level, clipped=stats.clipped_count, near=stats.near_count,
                             risk=stats.risk, outside=outside))
        outside_clips += outside
    return dict(state="valid" if analysis.usable else "provisional",
                deviations=deviations, imbalance=analysis.imbalance_percent, captures=tuple(captures),
                full_frame=full_frame, capture_confirmed=getattr(analysis, "capture_confirmed", True),
                flatfield_applied=flatfield_applied, outside_clips=outside_clips,
                balance_advice=analysis.balance_advice, exposure_advice=analysis.exposure_advice)


class RgbBalanceChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("片基配平偏差和 RAW 曝光余量")
        self.set_analysis(None)

    def set_analysis(self, analysis):
        self.data = plot_data(analysis)
        description = self._description()
        self.setAccessibleDescription(description)
        self.setToolTip(
            "左图的 0% 为三路片基信号的中位数，阴影为 ±5% 保持区。"
            "\n右图测量检查范围内未经平场校正的 RAW 感光点。"
            "每条显示该次曝光各 CFA 通道中最高的 P99.9 占用。"
            f"{RAW_LIMIT}。拍摄时请留出曝光余量。近限和剪切点另行计数。"
            f"\n\n{description}"
        )
        self._fit_advice()
        self.update()

    def _description(self):
        if self.data["state"] == "empty":
            return "尚未分析片基参考。"
        deviations = self.data["deviations"]
        basis = "平场后" if self.data["flatfield_applied"] else "未套用平场"
        balance = self.data["balance_advice"]
        details = [f"片基配平，{basis}。{balance.title}。{balance.detail}"]
        if deviations is not None:
            details.append("相对三路中位数的偏差，" + "，".join(
                f"{channel} {delta:+.2f}%" for channel, delta in zip("RGB", deviations)
            ) + "。")
        else:
            details.append("暂无可计算的配平偏差。")
        exposure = self.data["exposure_advice"]
        details.append(f"原始 RAW 曝光余量，{self._capture_scope()}。{RAW_LIMIT}。"
                       f"拍摄时请留出曝光余量。{exposure.title}。{exposure.detail}")
        for channel, capture in zip("RGB", self.data["captures"]):
            if capture is None or capture["level"] is None:
                details.append(f"{channel} 无可用 RAW 高值。")
                continue
            level = capture["level"]
            headroom = max(0.0, 100.0 - level)
            outside = f"，画幅外剪切 {capture['outside']:,} 点" if capture["outside"] else ""
            details.append(f"{channel} P99.9 占用 {level:.2f}%，余量 {headroom:.2f} 个百分点，"
                           f"近限 {capture['near']:,} 点，剪切 {capture['clipped']:,} 点{outside}。")
        return " ".join(details)

    def _capture_scope(self):
        if not self.data["capture_confirmed"]:
            return "保留画幅待确认"
        return "全画幅" if self.data["full_frame"] else "保留画幅"

    def _panels(self):
        gap, inset = 10.0, 2.0
        width = max(1.0, (self.width() - 2 * inset - gap) / 2)
        return (QRectF(inset, 2, width, self.height() - 4),
                QRectF(inset + width + gap, 2, width, self.height() - 4))

    @staticmethod
    def _font(base, size, weight=QFont.Weight.Normal):
        font = QFont(base)
        font.setPixelSize(size)
        font.setWeight(weight)
        return font

    @staticmethod
    def _text(painter, rect, text, color=TEXT, *, size=12, weight=QFont.Weight.Normal,
              alignment=Qt.AlignmentFlag.AlignLeft):
        painter.setFont(RgbBalanceChart._font(painter.font(), size, weight))
        painter.setPen(color)
        text = painter.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, max(0, int(rect.width())))
        painter.drawText(rect, alignment | Qt.AlignmentFlag.AlignVCenter, text)

    def _wrapped_height(self, width, text, size=DETAIL_FONT_SIZE, weight=QFont.Weight.Normal):
        font = QFontMetrics(self._font(self.font(), size, weight))
        rect = QRect(0, 0, max(1, math.floor(width - 2 * CARD_PADDING)), 0)
        return font.boundingRect(rect, Qt.TextFlag.TextWordWrap.value, text).height()

    def _capture_title_height(self, width):
        return max(24.0, self._wrapped_height(width, self.data["exposure_advice"].title, 15,
                                             QFont.Weight.DemiBold))

    def _track_top(self, width):
        balance_detail = self._wrapped_height(width, self.data["balance_advice"].detail)
        exposure_detail = self._wrapped_height(width, self.data["exposure_advice"].detail)
        capture_extra = self._capture_title_height(width) - 24.0 + CAPTURE_CAPTION_HEIGHT
        return DETAIL_TOP + max(balance_detail, exposure_detail + capture_extra) + 18.0

    def _chart_bottom(self):
        return CHART_BOTTOM + (OUTSIDE_CAPTION_HEIGHT if self.data["outside_clips"] else 0.0)

    def _fit_advice(self):
        top = self._track_top(self._panels()[0].width())
        height = max(250, math.ceil(top + MIN_TRACK_HEIGHT + self._chart_bottom() + 4.0))
        if self.minimumHeight() != height:
            self.setMinimumHeight(height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_advice()

    def _track(self, panel, *, capture=False):
        end_margin = min(103.0, panel.width() * (0.50 if capture else 0.40))
        top = self._track_top(panel.width())
        return QRectF(panel.left() + 33.0, panel.top() + top,
                      max(1.0, panel.width() - 33.0 - end_margin),
                      max(MIN_TRACK_HEIGHT, panel.height() - top - self._chart_bottom()))

    @staticmethod
    def _line(painter, x1, y1, x2, y2, color, width=1.0):
        painter.setPen(QPen(color, width))
        painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    def _header(self, painter, panel, title, advice, track, *, capture=False):
        x, width = panel.left() + CARD_PADDING, panel.width() - 2 * CARD_PADDING
        self._text(painter, QRectF(x, panel.top() + 10, width, 20),
                   title, MUTED, size=12, weight=QFont.Weight.Medium)
        severity = "neutral" if capture and not self.data["capture_confirmed"] else advice.severity
        action_color = ADVICE_COLORS[severity]
        y = panel.top() + DETAIL_TOP
        detail_bottom = track.top() - 10.0
        if capture:
            title_height = self._capture_title_height(panel.width())
            painter.setFont(self._font(self.font(), 15, QFont.Weight.DemiBold))
            painter.setPen(action_color)
            title_rect = QRectF(x, panel.top() + 34.0, width, title_height)
            painter.drawText(title_rect, Qt.TextFlag.TextWordWrap.value, advice.title)
            y += title_height - 24.0
            detail_bottom -= CAPTURE_CAPTION_HEIGHT
        else:
            self._text(painter, QRectF(x, panel.top() + 34, width, 24),
                       advice.title, action_color, size=15, weight=QFont.Weight.DemiBold)
        painter.setFont(self._font(self.font(), DETAIL_FONT_SIZE))
        painter.setPen(MUTED)
        detail_rect = QRectF(x, y, width, detail_bottom - y)
        painter.drawText(detail_rect, Qt.TextFlag.TextWordWrap.value, advice.detail)

    def _axis_label(self, painter, track, x, label, color=MUTED):
        self._text(painter, QRectF(x - 24, track.bottom() + 5, 48, 19),
                   label, color, size=10, alignment=Qt.AlignmentFlag.AlignHCenter)

    def _rows(self, painter, panel, track):
        rail = OBSIDIAN_RAIL if obsidian_enabled() else RAIL
        for index, (channel, color) in enumerate(zip("RGB", map(QColor, CHANNEL_COLORS))):
            y = track.top() + (index + 0.5) / 3 * track.height()
            self._text(painter, QRectF(panel.left() + 13, y - 11, 18, 22),
                       channel, color, size=12, weight=QFont.Weight.DemiBold)
            self._line(painter, track.left(), y, track.right(), y, rail)
            yield index, y, color

    def _value(self, painter, panel, track, y, text, color, *, size=12):
        start = track.right() + 11
        self._text(painter, QRectF(start, y - 9, panel.right() - 10 - start, 18),
                   text, color, size=size, alignment=Qt.AlignmentFlag.AlignRight)

    def _footer(self, painter, panel, text, color=MUTED):
        self._text(painter, QRectF(panel.left() + 14, panel.bottom() - 30, panel.width() - 28, 20),
                   text, color, size=11)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        left, right = self._panels()
        fill, edge = (OBSIDIAN_PANEL, OBSIDIAN_EDGE) if obsidian_enabled() else (PANEL, EDGE)
        painter.setPen(QPen(edge, 1))
        painter.setBrush(fill)
        for panel in (left, right):
            painter.drawRoundedRect(panel, 11, 11)
        self._deviation_plot(painter, left)
        self._capture_plot(painter, right)

    def _deviation_plot(self, painter, panel):
        track = self._track(panel)

        def x_of(delta):
            return track.center().x() + delta * track.width() / (2 * DEVIATION_LIMIT)

        deviations = self.data["deviations"]
        self._header(painter, panel, "配平偏差", self.data["balance_advice"], track)
        if deviations is None:
            self._text(painter, QRectF(panel.left() + CARD_PADDING, track.top(),
                                       panel.width() - 2 * CARD_PADDING, track.height()),
                       "暂无可用偏差", MUTED, alignment=Qt.AlignmentFlag.AlignHCenter)
            return
        painter.fillRect(QRectF(x_of(-HOLD_RANGE), track.top(),
                                x_of(HOLD_RANGE) - x_of(-HOLD_RANGE), track.height()), QColor("#304c40"))
        grid = OBSIDIAN_GRID if obsidian_enabled() else GRID
        for tick in (-DEVIATION_LIMIT, -HOLD_RANGE, 0.0, HOLD_RANGE, DEVIATION_LIMIT):
            x = x_of(tick)
            grid_color = MUTED if tick == 0 else grid
            self._line(painter, x, track.top(), x, track.bottom(), grid_color)
            if abs(tick) != HOLD_RANGE:
                label = "0%" if tick == 0 else f"{tick:+.0f}%"
                self._axis_label(painter, track, x, label)

        for index, y, color in self._rows(painter, panel, track):
            delta = deviations[index]
            x = x_of(max(-DEVIATION_LIMIT, min(DEVIATION_LIMIT, delta)))
            self._line(painter, track.center().x(), y, x, y, color, 2)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            if abs(delta) > DEVIATION_LIMIT:
                direction = 1 if delta > 0 else -1
                painter.drawPolygon(QPolygonF((QPointF(x, y), QPointF(x - direction * 8, y - 5),
                                               QPointF(x - direction * 8, y + 5))))
            else:
                painter.drawEllipse(QPointF(x, y), 4, 4)
            self._value(painter, panel, track, y, f"{delta:+.2f}%", color)

        basis = "平场后" if self.data["flatfield_applied"] else "片基"
        self._footer(painter, panel, f"±5% 保持区 · 相对三路{basis}中位数")

    def _capture_plot(self, painter, panel):
        track = self._track(panel, capture=True)
        confirmed = self.data["capture_confirmed"]
        scope = self._capture_scope()
        title = f"RAW 曝光余量 · {scope}" if confirmed else f"{scope} · RAW 曝光余量"
        self._header(painter, panel, title, self.data["exposure_advice"], track, capture=True)
        captures = self.data["captures"]
        if not any(capture is not None and capture["level"] is not None for capture in captures):
            self._text(painter, QRectF(panel.left() + CARD_PADDING, track.top(),
                                       panel.width() - 2 * CARD_PADDING, track.height()),
                       "暂无可用 RAW 读数", MUTED, alignment=Qt.AlignmentFlag.AlignHCenter)
            return
        self._text(painter, QRectF(panel.left() + CARD_PADDING, track.top() - CAPTURE_CAPTION_HEIGHT - 2,
                                   panel.width() - 2 * CARD_PADDING, CAPTURE_CAPTION_HEIGHT),
                   RAW_OCCUPANCY, MUTED, size=11)
        grid = OBSIDIAN_GRID if obsidian_enabled() else GRID
        for tick in (0, 25, 50, 75, 100):
            x = track.left() + tick * track.width() / 100
            is_limit = tick == 100
            self._line(painter, x, track.top(), x, track.bottom(), WARNING if is_limit else grid,
                       1.6 if is_limit else 1.0)
            if tick in (0, 50, 100):
                self._axis_label(painter, track, x, f"{tick}%", WARNING if is_limit else MUTED)

        for index, y, color in self._rows(painter, panel, track):
            value = captures[index] if captures else None
            if value is None or value["level"] is None:
                self._value(painter, panel, track, y, "—", MUTED)
                continue
            level = value["level"]
            x = track.left() + max(0.0, min(100.0, level)) * track.width() / 100
            painter.setPen(QPen(color, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
            painter.drawLine(QPointF(track.left(), y), QPointF(x, y))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(x, y), 3.5, 3.5)
            self._value(painter, panel, track, y - 7, f"{level:.2f}%", color)
            clipped = value["clipped"]
            if clipped:
                label, label_color = f"剪切 {clipped:,} 点", ADVICE_COLORS["danger"] if confirmed else MUTED
            elif value["risk"]:
                label, label_color = f"近限 {value['near']:,} 点", WARNING if confirmed else MUTED
            else:
                label, label_color = "剪切 0 点", MUTED
            self._value(painter, panel, track, y + 8, label, label_color, size=10)

        outside = self.data["outside_clips"]
        x, width = panel.left() + CARD_PADDING, panel.width() - 2 * CARD_PADDING
        limit_y = track.bottom() + 29.0
        self._line(painter, x, limit_y + 4.0, x, limit_y + 14.0, WARNING, 1.6)
        self._text(painter, QRectF(x + 7.0, limit_y, width - 7.0, 18), RAW_LIMIT,
                   WARNING, size=11, weight=QFont.Weight.Medium)
        if outside:
            self._text(painter, QRectF(x, limit_y + OUTSIDE_CAPTION_HEIGHT, width, 18),
                       f"画幅外剪切 {outside:,} 点", WARNING if confirmed else MUTED, size=11)
