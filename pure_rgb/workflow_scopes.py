"""Bounded output-RGB signal scopes and a cached, compact Qt display.

Counts use RGB16 working/output pixels before monitor ICC conversion. Y-prime
is an encoded-signal weighted brightness, not scene luminance or exposure EV.
The NumPy computation is independent of Qt and may run in a display worker.
"""

from dataclasses import dataclass
import math

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from .preview_alpha_clip_v45 import rounded_preview_path


SCOPE_MODES = {"waveform": "波形图", "parade": "RGB分量图", "histogram": "直方图"}
_BINS = 256
_BACKGROUND = (14, 17, 20)
_CHANNELS = ((246, 91, 83), (92, 221, 135), (86, 158, 255))
_BRIGHTNESS = (223, 230, 234)


@dataclass(frozen=True)
class ScopeData:
    """Level rows increase from 0 to 255; column bins run from left to right."""

    waveform: np.ndarray
    parade: np.ndarray
    histogram: np.ndarray
    clipped_counts: np.ndarray
    source_shape: tuple
    sampled_shape: tuple
    sample_count: int

    @property
    def nbytes(self):
        return sum(array.nbytes for array in (
            self.waveform, self.parade, self.histogram, self.clipped_counts))


def _sample_grid(pixels, maximum):
    height, width = pixels.shape[:2]
    if height * width <= maximum:
        return pixels, np.arange(width, dtype=np.int64)
    scale = math.sqrt(maximum / (height * width))
    rows, columns = max(1, int(height * scale)), max(1, int(width * scale))
    rows = min(height, rows, maximum)
    columns = min(width, columns, max(1, maximum // rows))
    y = np.linspace(0, height - 1, rows, dtype=np.int64)
    x = np.linspace(0, width - 1, columns, dtype=np.int64)
    return pixels[np.ix_(y, x)], x


def compute_scopes(pixels, *, max_pixels=1_048_576):
    """Compute readonly 256-bin scope counts from a bounded spatial grid.

    Saturation counts describe exact RGB16 endpoints (0 and 65535), whereas
    histogram bins each contain 256 adjacent codes. Input pixels are never
    changed. Even a one-row/one-column image respects the sampling bound.
    """
    pixels = np.asarray(pixels)
    if (pixels.dtype != np.uint16 or pixels.ndim != 3 or pixels.shape[2] != 3
            or not pixels.shape[0] or not pixels.shape[1]):
        raise ValueError("示波器需要非空 RGB16 输出像素")
    if type(max_pixels) is not int or not 1 <= max_pixels <= 1_048_576:
        raise ValueError("示波器取样上限必须为正整数，且不超过 1,048,576 像素")
    sampled, source_columns = _sample_grid(pixels, max_pixels)
    height, width = sampled.shape[:2]
    values = sampled.astype(np.uint32)
    # Integer arithmetic makes primary and clipping bins platform independent.
    brightness = ((values[..., 0] * 2126 + values[..., 1] * 7152
                   + values[..., 2] * 722 + 5000) // 10000) >> 8
    columns = (source_columns * (_BINS - 1) // max(1, pixels.shape[1] - 1)).astype(np.uint32)
    columns = np.broadcast_to(columns, (height, width))

    def density(levels):
        indices = (levels * _BINS + columns).ravel()
        return np.bincount(indices, minlength=_BINS * _BINS).reshape(_BINS, _BINS).astype(np.uint32)

    waveform = density(brightness)
    parade = np.stack([density(values[..., channel] >> 8) for channel in range(3)])
    histogram = np.concatenate((parade.sum(axis=2, dtype=np.uint32),
                                waveform.sum(axis=1, dtype=np.uint32)[None]), axis=0)
    clipped = np.array([[np.count_nonzero(sampled[..., channel] == endpoint)
                         for endpoint in (0, 65535)]
                        for channel in range(3)], dtype=np.uint32)
    for array in (waveform, parade, histogram, clipped):
        array.setflags(write=False)
    return ScopeData(waveform=waveform, parade=parade, histogram=histogram,
                     clipped_counts=clipped, source_shape=tuple(pixels.shape),
                     sampled_shape=tuple(sampled.shape), sample_count=height * width)


def _density_image(counts, color):
    levels = np.log1p(counts.astype(np.float32))
    maximum = float(levels.max())
    intensity = np.zeros_like(levels) if maximum == 0 else np.where(counts > 0, .24 + .76 * levels / maximum, 0)
    background = np.array(_BACKGROUND, dtype=np.float32)
    rgb = background + intensity[..., None] * (np.array(color, np.float32) - background)
    rgba = np.empty((*counts.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.rint(rgb).astype(np.uint8)[::-1]
    rgba[..., 3] = 255
    return QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0],
                  QImage.Format.Format_RGBA8888).copy()


def _histogram_image(histogram):
    width, height = 512, 256
    image = QImage(width, height, QImage.Format.Format_RGBA8888)
    image.fill(QColor(*_BACKGROUND))
    levels = np.log1p(histogram.astype(np.float64))
    maximum = float(levels.max())
    if maximum == 0:
        return image
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    try:
        for values, color in zip(levels, (*_CHANNELS, _BRIGHTNESS)):
            path = QPainterPath(QPointF(0, height - 1))
            for index, value in enumerate(values):
                path.lineTo(index * (width - 1) / 255, (1 - value / maximum) * (height - 1))
            path.lineTo(width - 1, height - 1)
            painter.fillPath(path, QColor(*color, 35))
            painter.setPen(QPen(QColor(*color, 205), 1.1))
            painter.drawPath(path)
    finally:
        painter.end()
    return image


def _images(data):
    waveform = _density_image(data.waveform, _BRIGHTNESS)
    parade = QImage(770, 256, QImage.Format.Format_RGBA8888)
    parade.fill(QColor(*_BACKGROUND))
    painter = QPainter(parade)
    try:
        for channel, color in enumerate(_CHANNELS):
            painter.drawImage(channel * 257, 0, _density_image(data.parade[channel], color))
    finally:
        painter.end()
    return dict(waveform=waveform, parade=parade, histogram=_histogram_image(data.histogram))


def _count_label(value):
    return f"{value / 1000:.0f}k" if value >= 1000 else str(round(value))


class ScopePlot(QWidget):
    """Paint only cached images, a small grid and axis labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "waveform"
        self._images = {}
        self._maximum_count = 0
        self.setMinimumHeight(154)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("波形图")
        self.setAccessibleDescription("纵向为 0 至 100% 输出信号，横向为画面从左到右的位置。")

    def sizeHint(self):
        return QSize(250, 160)

    def set_scope_data(self, data):
        self._images = {} if data is None else _images(data)
        self._maximum_count = 0 if data is None else int(data.histogram.max())
        self.update()

    def set_mode(self, mode):
        self._mode = mode
        self.setAccessibleName(SCOPE_MODES[mode])
        self.setAccessibleDescription(
            "横向为 0 至 100% 输出信号，纵向为对数像素数；显示 RGB 与加权亮度 Y′。"
            if mode == "histogram" else
            "纵向为 0 至 100% 输出信号；RGB 三通道各自显示画面从左到右的位置。"
            if mode == "parade" else
            "纵向为 0 至 100% 输出信号，横向为画面从左到右的位置；显示加权亮度 Y′。")
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            surface = rounded_preview_path(self.size())
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillPath(surface, QColor(*_BACKGROUND))
            painter.setClipPath(surface)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            painter.setFont(QFont(self.font().family(), 8))
            if not self._images:
                painter.setPen(QColor("#939ba3"))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "载入照片后显示")
                return
            chart = QRectF(32, 12, max(1, self.width() - 41), max(1, self.height() - 35))
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(chart, self._images[self._mode])
            painter.setPen(QPen(QColor(146, 158, 169, 45), 1))
            for fraction in (0, .25, .5, .75, 1):
                y = chart.bottom() - chart.height() * fraction
                painter.drawLine(QPointF(chart.left(), y), QPointF(chart.right(), y))
            painter.setPen(QColor("#a7b0ba"))
            if self._mode == "histogram":
                maximum = self._maximum_count
                ticks = ((0, 0), (.5, math.sqrt(maximum + 1) - 1), (1, maximum))
            else:
                ticks = ((0, 0), (.5, 50), (1, 100))
            for fraction, value in ticks:
                y = chart.bottom() - chart.height() * fraction
                label = _count_label(value) if self._mode == "histogram" else str(int(value))
                painter.drawText(QRectF(0, y - 7, 27, 14), Qt.AlignmentFlag.AlignRight, label)
            if self._mode == "parade":
                for channel, color in enumerate(_CHANNELS):
                    left = chart.left() + chart.width() * channel / 3
                    painter.setPen(QColor(*color))
                    painter.drawText(QRectF(left, chart.bottom()+5, chart.width()/3, 15),
                                     Qt.AlignmentFlag.AlignHCenter, "RGB"[channel])
            else:
                painter.setPen(QColor("#a7b0ba"))
                if self._mode == "histogram":
                    labels = ((0, "0"), (.5, "50"), (1, "100%"))
                else:
                    labels = ((0, "左"), (.5, "画面位置"), (1, "右"))
                for fraction, label in labels:
                    x = chart.left() + chart.width() * fraction
                    alignment = (Qt.AlignmentFlag.AlignLeft if fraction == 0 else
                                 Qt.AlignmentFlag.AlignRight if fraction == 1 else Qt.AlignmentFlag.AlignHCenter)
                    painter.drawText(QRectF(x-24 if fraction == .5 else x-30 if fraction == 1 else x,
                                           chart.bottom()+5, 48 if fraction == .5 else 30, 15), alignment, label)
            if self._mode == "histogram":
                for index, (label, color) in enumerate(zip(("R", "G", "B", "Y′"), (*_CHANNELS, _BRIGHTNESS))):
                    painter.setPen(QColor(*color))
                    painter.drawText(QRectF(chart.right()-76+index*19, -1, 19, 13), Qt.AlignmentFlag.AlignCenter, label)
            elif self._mode == "waveform":
                painter.setPen(QColor(*_BRIGHTNESS))
                painter.drawText(QRectF(chart.right()-25, -1, 25, 13), Qt.AlignmentFlag.AlignRight, "Y′")
        finally:
            painter.end()


class ScopePanel(QWidget):
    """Three compact, keyboard-accessible scope modes above the photo list."""

    mode_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("scopePanel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._mode = "waveform"
        self.scope_data = None
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(5)
        self.mode_row = QWidget(self)
        self.mode_row.setObjectName("scopeModeRow")
        row = QHBoxLayout(self.mode_row)
        row.setContentsMargins(2, 2, 2, 2)
        row.setSpacing(2)
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        self.mode_buttons = {}
        for mode, title in SCOPE_MODES.items():
            control = QToolButton(self.mode_row)
            control.setText(title)
            control.setObjectName("scopeMode_" + mode)
            control.setAccessibleName(title)
            control.setToolTip(title + "；左右方向键切换")
            control.setCheckable(True)
            control.setProperty("glassCornerRadius", 6.0)
            control.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            control.setFixedHeight(26)
            control.installEventFilter(self)
            control.clicked.connect(lambda checked=False, mode=mode: self.set_mode(mode))
            self.button_group.addButton(control)
            self.mode_buttons[mode] = control
            row.addWidget(control, 1)
        self.mode_buttons[self._mode].setChecked(True)
        self.setStyleSheet("""
            QWidget#scopeModeRow {background:rgba(255,255,255,16);border-radius:8px;}
            QToolButton {font-size:11px;border:0;border-radius:6px;padding:0 3px;color:#d3d9df;background:transparent;}
            QToolButton:hover {background:rgba(255,255,255,22);}
            QToolButton:checked {color:#fff4ed;background:rgba(244,82,41,180);}
            QToolButton:focus {border:1px solid rgba(255,194,158,200);}
            QToolButton[nativeMaterial="true"] {background:transparent;}
        """)
        column.addWidget(self.mode_row)
        self.plot = ScopePlot(self)
        column.addWidget(self.plot, 1)
        self.caption = QLabel("输出 RGB · 预览取样", self)
        self.caption.setObjectName("muted")
        self.caption.setStyleSheet("font-size:10px;color:#aab1b8;")
        self.caption.setToolTip("0–100% 为输出 RGB 信号码值，取自显示器 ICC 转换前的预览像素。\n"
                                "Y′ 为 RGB 加权信号亮度；直方图纵向为对数像素数。")
        column.addWidget(self.caption)
        self.setFixedHeight(210)

    @property
    def mode(self):
        return self._mode

    def set_mode(self, mode):
        if mode not in SCOPE_MODES:
            raise ValueError("未知示波器模式")
        self.mode_buttons[mode].setChecked(True)
        if self._mode != mode:
            self._mode = mode
            self.plot.set_mode(mode)
            self.mode_changed.emit(mode)

    def set_scope_data(self, data):
        if data is not None and not isinstance(data, ScopeData):
            raise TypeError("示波器数据必须由 compute_scopes 生成")
        if data is self.scope_data:
            return
        self.scope_data = data
        self.plot.set_scope_data(data)
        self.caption.setToolTip(("尚无输出预览。\n" if data is None else f"取样 {data.sample_count:,} 像素。\n")
                                + "0–100% 为输出 RGB 信号码值，取自显示器 ICC 转换前。\n"
                                + "Y′ 为 RGB 加权信号亮度；直方图纵向为对数像素数。")

    set_data = set_scope_data

    def eventFilter(self, watched, event):
        if event.type() != QEvent.Type.KeyPress:
            return super().eventFilter(watched, event)
        buttons = getattr(self, "mode_buttons", {})
        if watched in buttons.values():
            modes = list(SCOPE_MODES)
            current = next(mode for mode, control in buttons.items() if control is watched)
            key = event.key()
            if key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Home, Qt.Key.Key_End):
                index = modes.index(current)
                target = (modes[(index + (-1 if key == Qt.Key.Key_Left else 1)) % len(modes)]
                          if key in (Qt.Key.Key_Left, Qt.Key.Key_Right) else modes[0 if key == Qt.Key.Key_Home else -1])
                self.set_mode(target)
                buttons[target].setFocus()
                event.accept()
                return True
        return super().eventFilter(watched, event)
