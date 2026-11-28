"""Independent PURE-styled desktop preview of existing verified results."""

import argparse
from collections import OrderedDict
from copy import deepcopy
from functools import lru_cache
import io
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageCms
from PyQt6.QtCore import (
    QCoreApplication,
    QEvent,
    QFileInfo,
    QLibraryInfo,
    QObject,
    QPointF,
    QRectF,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplashScreen,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .branding import (
    application_icon,
    splash_pixmap,
    wordmark_label,
)
from .workflow_about import show_about
from .workflow_glass import (
    GlassMessageBox, NATIVE_SLIDER_QSS, _set_native_material, attach_glass, detach_glass,
    install_glass_surfaces, prepare_glass,
)
from .workflow_combo import WorkflowComboBox as QComboBox
from .motion_sliders import ScrubberSlider
from .workflow_materials import attach_materials
from .preview_data import (
    DEFAULT_STUDY,
    MODES,
    PROBES,
    ROOT,
    copy_verified_tiff,
    load_samples,
    verified_pixels,
)

STYLE = """
QWidget { color: #F2EDE4; font-size: 13px; }
QMainWindow, QDialog, QWidget#root { background: #141517; }
QFrame#panel, QFrame#toolbar { background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
stop:0 #333135,stop:1 #25272B); border: 1px solid #4C4B4E; border-radius:16px; }
QFrame#toolbar { border-radius:12px; }
QLabel#muted { color:#C4C4C4; font-size:12px; }
QLabel#section { color:#F0A055; font-weight:600; }
QPushButton { background:#3B3A3E; border:1px solid #57565A; border-radius:14px;
padding:7px 13px; min-height:16px; }
QPushButton:hover { background:#4D4844; border-color:#AC8262; }
QPushButton:checked { background:#63503F; color:#FFC087; border-color:#BC8A5B; }
QPushButton#primary { background:#F0A055; color:#141517; font-weight:600; border-color:#FFC282; }
QPushButton:disabled { background:#303135; color:#77777A; border-color:#424347; }
QPushButton:focus, QComboBox:focus { border:1px solid #F0A055; }
QComboBox { background:#353438; border:1px solid #57565A; border-radius:9px;
padding:8px; min-height:18px; }
QComboBox QAbstractItemView { background:#2B2D31; selection-background-color:#63503F; }
QLineEdit, QDoubleSpinBox { background:#303135; border:1px solid #57565A;
border-radius:7px; padding:4px; min-height:20px; }
QDoubleSpinBox:focus, QLineEdit:focus { border-color:#F0A055; }
QTabWidget::pane { border:1px solid #4C4B4E; border-radius:10px; background:#25272B; }
QTabBar::tab { background:#303135; color:#C4C4C4; padding:9px 14px; margin:3px;
border:1px solid #48474B; border-radius:10px; }
QTabBar::tab:selected { background:#63503F; color:#FFC087; border-color:#BC8A5B; }
QTableWidget { background:#25272B; alternate-background-color:#303135; gridline-color:#4C4B4E; }
QHeaderView::section { background:#353438; padding:5px; border:0; }
QListWidget { background:transparent; border:0; outline:0; }
QListWidget::item { border:1px solid transparent; border-radius:10px; padding:8px; margin:3px; }
QListWidget::item:selected { background:#57483C; border-color:#88674D; color:#FFC087; }
QScrollBar:vertical { background:#25272B; width:9px; margin:0; border-radius:4px; }
QScrollBar::handle:vertical { background:#696469; min-height:28px; border-radius:4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:transparent; }
QScrollArea, QScrollArea > QWidget > QWidget { border:0; background:transparent; }
QSlider::groove:horizontal { height:4px; background:#545157; border-radius:2px; }
QSlider::handle:horizontal { background:#F0A055; width:14px; margin:-5px 0; border-radius:7px; }
QSplitter::handle { background:#141517; }
QToolTip { background:#303135; color:#F2EDE4; border:1px solid #807060; }
QMainWindow[nativeMaterial="true"], QWidget#root[nativeMaterial="true"] { background:transparent; }
QFrame#panel[nativeMaterial="true"], QFrame#toolbar[nativeMaterial="true"],
QWidget[sliderPanel="true"][nativeMaterial="true"],
QPushButton[nativeMaterial="true"], QPushButton#primary[nativeMaterial="true"],
QComboBox[nativeMaterial="true"] { background:transparent; border-color:transparent; }
QPushButton#primary[nativeMaterial="true"] { color:#fff5ef; }
""" + NATIVE_SLIDER_QSS


def label(text, role=None, wrap=False):
    widget = QLabel(text)
    if role:
        widget.setObjectName(role)
    widget.setWordWrap(wrap)
    return widget


def button(text, slot, primary=False):
    widget = QPushButton(text)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    widget.clicked.connect(slot)
    if primary:
        widget.setObjectName("primary")
    return widget


def panel(name="panel"):
    frame = QFrame()
    frame.setObjectName(name)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(15, 14, 15, 14)
    layout.setSpacing(12)
    return frame, layout


def startup_splash():
    """启动画面；资源缺失时返回 None，不阻断启动。"""
    try:
        splash = QSplashScreen(splash_pixmap())
    except RuntimeError:
        return None
    splash.show()
    QApplication.instance().processEvents()
    return splash


class Signals(QObject):
    done = pyqtSignal(object, object)
    failed = pyqtSignal(object, str)


class Work(QRunnable):
    def __init__(self, token, fn):
        super().__init__()
        self.token, self.fn = token, fn
        self.signals = Signals()

    def run(self):
        try:
            self.signals.done.emit(self.token, self.fn())
        except Exception as error:
            self.signals.failed.emit(self.token, str(error))


class Canvas(QWidget):
    zoom_changed = pyqtSignal(float)
    full_resolution_requested = pyqtSignal()
    detail_requested = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 180)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.current = self.reference = None
        self.interactive_image = None
        self.interactive_mode_text = None
        self.empty_message = "正在载入已计算样片…"
        self.zoom = 1.0
        self.sample_scale = 1
        self.origin = QPointF()
        self.fit_mode = True
        self.double_click_zoom_enabled = True
        self.split = False
        self.fraction = 0.5
        self.last = None
        self.mode_text = ""
        self.reference_name = "Native"
        # Opt in only in the staged workflow. Older workbenches keep their
        # existing whole-frame 100% request protocol.
        self.photo_lens = None
        self.detail_enabled = False
        self.detail_max_pixels = 4_000_000
        self._detail_output_size = None
        self._detail_serial = 0
        self._detail_pending = None
        self._detail_current = self._detail_reference = None
        self._detail_roi = None
        self._detail_step = 1
        self._detail_full_requested = False
        self.detail_timer = QTimer(self)
        self._detail_suspended = False
        self.detail_timer.setSingleShot(True)
        self.detail_timer.setInterval(160)
        self.detail_timer.timeout.connect(self._request_detail)

    def set_detail_enabled(self, enabled):
        self.detail_enabled = bool(enabled)
        if not self.detail_enabled:
            self.clear_detail()

    def suspend_detail(self, suspended):
        if self._detail_suspended == suspended:
            return
        self._detail_suspended = suspended
        if suspended:
            self.detail_timer.stop()
        elif not self.fit_mode or (self.photo_lens and self.photo_lens.active):
            self._schedule_detail()

    def set_output_size(self, width, height):
        """Set the exact current output grid, after crop and orientation."""
        if width is None and height is None:
            self.clear_detail()
            self._detail_output_size = None
            if not self.fit_mode:
                self._schedule_detail()
            return
        if any(type(value) is not int or value <= 0 for value in (width, height)):
            raise ValueError("高清预览需要完整输出的正整数尺寸")
        size = (width, height)
        if self._detail_output_size != size:
            self.clear_detail()
            self._detail_output_size = size
        if not self.fit_mode or (self.photo_lens and self.photo_lens.active):
            self._schedule_detail()

    def clear_detail(self):
        """Discard an old recipe's overlay and make late completions harmless."""
        self.detail_timer.stop()
        self._detail_serial += 1
        self._detail_pending = None
        self._detail_full_requested = False
        self._detail_current = self._detail_reference = None
        self._detail_roi = None
        self.update()

    def detail_failed(self, request_id):
        if self._detail_pending and self._detail_pending["request_id"] == request_id:
            self._detail_pending = None

    def _detail_view(self, *, margin=0, lens=False):
        magnification = self.photo_lens.zoom.target if lens else 1.0
        device_zoom = self.zoom * self.devicePixelRatioF() * magnification
        if (not self.detail_enabled or not self.current or self.sample_scale <= 1
                or device_zoom <= 1.02
                or not self.isVisible()):
            return None
        if self._detail_output_size is None:
            return None
        width, height = self._detail_output_size
        scale = self.sample_scale / self.zoom
        bounds = self.photo_lens.sample_view() if lens else QRectF(self.rect())
        x0 = max(0, math.floor((bounds.left() - margin - self.origin.x()) * scale))
        y0 = max(0, math.floor((bounds.top() - margin - self.origin.y()) * scale))
        x1 = min(width, math.ceil((bounds.right() + margin - self.origin.x()) * scale))
        y1 = min(height, math.ceil((bounds.bottom() + margin - self.origin.y()) * scale))
        if x1 <= x0 or y1 <= y0:
            return None
        wanted = max(1, math.floor(self.sample_scale / device_zoom))
        step = 2 ** int(math.floor(math.log2(wanted)))
        center_only = False
        if device_zoom / self.sample_scale >= 1 - 1e-9:
            # Never label a step-2 tile as actual pixels. A very large viewport
            # gets a bounded central 1:1 tile, retaining the base around it.
            if (x1 - x0) * (y1 - y0) > self.detail_max_pixels and margin:
                return self._detail_view(margin=0, lens=lens)
            if (x1 - x0) * (y1 - y0) > self.detail_max_pixels:
                ratio = math.sqrt(self.detail_max_pixels / ((x1 - x0) * (y1 - y0)))
                tile_width = max(1, math.floor((x1 - x0) * ratio))
                tile_height = max(1, min(y1 - y0, self.detail_max_pixels // tile_width))
                center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
                x0 = max(x0, math.floor(center_x - tile_width / 2))
                y0 = max(y0, math.floor(center_y - tile_height / 2))
                x1, y1 = x0 + tile_width, y0 + tile_height
                center_only = True
            step = 1
        else:
            while (math.ceil((x1 - (x0 // step) * step) / step)
                   * math.ceil((y1 - (y0 // step) * step) / step)) > self.detail_max_pixels:
                step *= 2
        if step >= self.sample_scale:
            return None
        # Stable global-grid phase avoids changing sampled source pixels while
        # panning by a fraction of a preview sample.
        x0, y0 = (x0 // step) * step, (y0 // step) * step
        return dict(roi=[x0, y0, x1, y1], sample_step=step,
                    source_size=[width, height], center_only=center_only)

    @staticmethod
    def _detail_covers(container, visible):
        if container is None or container["sample_step"] > visible["sample_step"]:
            return False
        a, b, c, d = container["roi"]
        x0, y0, x1, y1 = visible["roi"]
        return a <= x0 and b <= y0 and c >= x1 and d >= y1

    def _schedule_detail(self, delay=160):
        if self.detail_enabled and not self._detail_suspended:
            self.detail_timer.start(delay)

    def _request_detail(self):
        if self._detail_suspended:
            return
        lens = self.photo_lens is not None and self.photo_lens.active
        if (not lens and self.detail_enabled and self.current and self.sample_scale > 1
                and self._detail_output_size is None and not self._detail_full_requested
                and self.zoom * self.devicePixelRatioF() > 1.02 and self.isVisible()):
            # Unknown geometry cannot be inferred from ceil-sized samples.
            # Ask the existing full-frame route once for this base generation.
            self._detail_full_requested = True
            self.full_resolution_requested.emit()
            return
        visible = self._detail_view(lens=lens)
        if visible is None:
            return
        request = self._detail_view(margin=48, lens=lens) or deepcopy(visible)
        # The padded request may need a coarser LOD to stay under the tile
        # budget; that same accepted quality covers an unchanged viewport.
        visible["sample_step"] = request["sample_step"]
        active = (dict(roi=self._detail_roi, sample_step=self._detail_step)
                  if self._detail_roi is not None else None)
        if self._detail_covers(active, visible) or self._detail_covers(self._detail_pending, visible):
            return
        self._detail_serial += 1
        request["request_id"] = self._detail_serial
        self._detail_pending = request
        self.detail_requested.emit(deepcopy(request))

    def set_detail_images(self, current, reference, roi, sample_step=1, request_id=None):
        """Place a current-grid tile without replacing or recentering the base."""
        pending = self._detail_pending
        if (not self.detail_enabled or pending is None
                or request_id != pending["request_id"]):
            return False
        if (not isinstance(roi, (list, tuple)) or len(roi) != 4
                or any(type(value) is not int for value in roi)
                or type(sample_step) is not int or sample_step < 1):
            raise ValueError("高清预览需要原像素区域和整数采样步长")
        if list(roi) != pending["roi"] or sample_step != pending["sample_step"]:
            raise ValueError("高清预览区域或采样步长与当前请求不一致")
        x0, y0, x1, y1 = roi
        if not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ValueError("高清预览区域无效")
        expected = QSize(math.ceil((x1 - x0) / sample_step), math.ceil((y1 - y0) / sample_step))
        if current is None or current.isNull() or current.size() != expected:
            raise ValueError("高清预览图像尺寸与原像素区域不一致")
        if reference is not None and (reference.isNull() or reference.size() != expected):
            raise ValueError("高清预览对照图尺寸不一致")
        self._detail_current = QImage(current)
        self._detail_reference = QImage(reference) if reference is not None else None
        self._detail_roi, self._detail_step = list(roi), sample_step
        self._detail_pending = None
        self.update()
        return True

    def set_images(self, current, reference, text, reset=False):
        self.interactive_image = None
        self.interactive_mode_text = None
        self.clear_detail()
        self.current, self.reference, self.mode_text = current, reference, text
        if reset or self.fit_mode:
            self.fit()
        else:
            self._schedule_detail()
        self.update()

    def set_interactive_image(self, image, mode_text=None):
        if self.interactive_image is image and self.interactive_mode_text == mode_text:
            return
        self.interactive_image = image
        self.interactive_mode_text = mode_text if image is not None else None
        if image is not None:
            # This packet arrives on the GUI thread already colour-managed.
            # Present it in this event instead of waiting for a second update.
            if mode_text:
                self.repaint()
            else:
                self.repaint_interactive_image()
        else:
            self.update()

    def repaint_interactive_image(self):
        if self.current is not None and self.interactive_image is not None:
            self.repaint(self._photo_area().toAlignedRect())

    def displayed_image(self):
        if (self.current is not None and self.interactive_image is not None
                and (self.fit_mode or not self.interactive_mode_text)
                and not self.split
                and not (self.photo_lens and self.photo_lens.active)):
            return self.interactive_image
        return self.current

    def _photo_area(self):
        return QRectF(self.origin, self.current.size().toSizeF() * self.zoom)

    def _fit_scale(self, image):
        return min((self.width() - 32) / image.width(), (self.height() - 60) / image.height())

    def fit(self):
        self.clear_detail()
        self.fit_mode = True
        if self.current:
            self.zoom = self._fit_scale(self.current)
            self.center()
            if self.photo_lens and self.photo_lens.active:
                self._schedule_detail()

    def center(self):
        self.origin = QPointF(
            (self.width() - self.current.width() * self.zoom) / 2,
            (self.height() - self.current.height() * self.zoom) / 2,
        )
        self.zoom_changed.emit(self.zoom * self.devicePixelRatioF() / self.sample_scale)
        self.update()

    def actual(self, anchor=None):
        if self.detail_enabled and self.current:
            point = anchor if anchor is not None else QPointF(self.width() / 2, self.height() / 2)
            zoom = self.sample_scale / self.devicePixelRatioF()
            self.origin = point - (point - self.origin) * (zoom / self.zoom)
            self.zoom, self.fit_mode = zoom, False
            ratio = self.devicePixelRatioF()
            self.origin = QPointF(round(self.origin.x() * ratio) / ratio,
                                  round(self.origin.y() * ratio) / ratio)
            self.zoom_changed.emit(1.0)
            self.update()
            self._schedule_detail(0)
            return
        if self.sample_scale > 1:
            self.full_resolution_requested.emit()
            return
        if self.current:
            zoom = 1.0 / self.devicePixelRatioF()
            if anchor is not None:
                self.origin = anchor - (anchor - self.origin) * (zoom / self.zoom)
            self.fit_mode, self.zoom = False, zoom
            if anchor is None:
                self.center()
            else:
                self.zoom_changed.emit(1.0)
                self.update()

    def resizeEvent(self, event):
        if self.fit_mode:
            self.fit()
        else:
            self._schedule_detail()
        super().resizeEvent(event)

    def showEvent(self, event):
        if not self.fit_mode:
            self._schedule_detail()
        super().showEvent(event)

    def _zoom_maximum(self):
        return (
            max(8.0, 8.0 * self.sample_scale / self.devicePixelRatioF())
            if self.detail_enabled
            else 8.0
        )

    def _apply_zoom(self, point, zoom):
        zoom = min(self._zoom_maximum(), max(0.02, zoom))
        self.origin = point - (point - self.origin) * (zoom / self.zoom)
        self.zoom, self.fit_mode = zoom, False
        self.zoom_changed.emit(zoom * self.devicePixelRatioF() / self.sample_scale)
        self.update()
        self._schedule_detail()

    def wheelEvent(self, event):
        if not self.current:
            return
        point = event.position()
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if not delta:
            return
        self._apply_zoom(
            point, self.zoom * (1.2 if delta > 0 else 1 / 1.2)
        )

    def event(self, event):
        # macOS trackpad pinch arrives as native gesture events, not wheel;
        # every engine mode shares the same anchored zoom as the wheel.
        if (
            event.type() == QEvent.Type.NativeGesture
            and self.current is not None
            and event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture
        ):
            value = float(event.value())
            # macOS reports incremental ratios near 1.0; smaller magnitudes
            # are signed fractions. Both conventions zoom with the fingers.
            factor = value if 0.5 < abs(value) < 2.0 else 1.0 + value
            if factor > 0 and factor != 1.0:
                self._apply_zoom(event.position(), self.zoom * factor)
                return True
        return super().event(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.last is not None:
            self.origin += event.position() - self.last
            self.last, self.fit_mode = event.position(), False
            self.update()
            self._schedule_detail()

    def mouseReleaseEvent(self, event):
        self.last = None
        self.unsetCursor()
        self._schedule_detail(0)

    def mouseDoubleClickEvent(self, event):
        if (self.double_click_zoom_enabled and self.current
                and event.button() == Qt.MouseButton.LeftButton
                and QRectF(self.origin, self.current.size().toSizeF() * self.zoom).contains(event.position())):
            self.last = None
            actual_zoom = self.sample_scale / self.devicePixelRatioF()
            if not self.fit_mode and abs(self.zoom - actual_zoom) < 1e-6:
                self.fit()
            else:
                self.actual(event.position())
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _paint_detail(self, painter, image):
        if image is None or self._detail_roi is None:
            return
        x0, y0, x1, y1 = self._detail_roi
        scale = self.zoom / self.sample_scale
        visible = QRectF(self.origin.x() + x0 * scale, self.origin.y() + y0 * scale,
                         (x1 - x0) * scale, (y1 - y0) * scale)
        # A partial final stride sample still occupies sample_step source
        # pixels. Clip its tail instead of stretching the bitmap to the ROI.
        bitmap = QRectF(visible.x(), visible.y(), image.width() * self._detail_step * scale,
                        image.height() * self._detail_step * scale)
        painter.save()
        painter.setClipRect(visible, Qt.ClipOperation.IntersectClip)
        painter.drawImage(bitmap, image)
        painter.restore()

    def paint_photos(self, painter):
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        image = self.displayed_image()
        if image is self.interactive_image and self.interactive_mode_text:
            # The selected engine can use a newly applied output crop. Fit
            # its quick frame without changing the last exact image's geometry.
            size = image.size().toSizeF() * self._fit_scale(image)
            target = QRectF(QPointF((self.width() - size.width()) / 2,
                                   (self.height() - size.height()) / 2), size)
        else:
            target = self._photo_area()
        painter.drawImage(target, image)
        if image is self.current:
            self._paint_detail(painter, self._detail_current)
        if self.split and self.reference:
            painter.save()
            painter.setClipRect(
                QRectF(0, 0, self.width() * self.fraction, self.height()), Qt.ClipOperation.IntersectClip
            )
            painter.drawImage(target, self.reference)
            self._paint_detail(painter, self._detail_reference)
            painter.restore()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(event.rect(), QColor("#141517"))
        if not self.current:
            painter.setPen(QColor("#C4C4C4"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty_message
            )
            return
        self.paint_photos(painter)
        if self.split and self.reference:
            x = self.width() * self.fraction
            painter.setPen(QPen(QColor("#F0A055"), 2))
            painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        current_caption = self.mode_text
        if self.interactive_mode_text and self.displayed_image() is self.interactive_image:
            current_caption = self.interactive_mode_text
        caption = (
            (f"左 {self.reference_name}  /  右 " if self.split else "")
            + current_caption
        )
        if caption:
            # One shared badge design for every engine caption, fitted to its
            # text so short and long engine names read identically.
            metrics = painter.fontMetrics()
            available = self.width() - 16
            shown = metrics.elidedText(
                caption, Qt.TextElideMode.ElideRight, available - 28
            )
            badge = QRectF(
                8, 8, min(metrics.horizontalAdvance(shown) + 28, available), 26
            )
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(QColor(255, 255, 255, 36), 1))
            painter.setBrush(QColor(20, 21, 23, 220))
            painter.drawRoundedRect(badge, 13, 13)
            painter.setPen(QColor("#F2EDE4"))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, shown)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        if self.photo_lens is not None:
            self.photo_lens.paint(painter)


def display_image(record, display_icc):
    pixels, icc = verified_pixels(record)
    image = Image.fromarray(np.rint(pixels / 257).astype(np.uint8))
    target = (
        ImageCms.ImageCmsProfile(io.BytesIO(display_icc))
        if display_icc
        else ImageCms.createProfile("sRGB")
    )
    image = ImageCms.profileToProfile(
        image,
        ImageCms.ImageCmsProfile(io.BytesIO(icc)),
        target,
        outputMode="RGB",
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
    )
    data = image.tobytes()
    return QImage(
        data, image.width, image.height, image.width * 3, QImage.Format.Format_RGB888
    ).copy()


@lru_cache(maxsize=1)
def _plugin_mirror(source):
    """Keep visible links to the loaded Qt runtime alive for this process."""
    mirror = TemporaryDirectory(prefix="pure-rgb-qt-")
    root = Path(mirror.name)
    for plugin in source.glob("*/*.dylib"):
        target = root / plugin.relative_to(source)
        target.parent.mkdir(exist_ok=True)
        target.symlink_to(plugin)
    return mirror


def prepare_plugins():
    if sys.platform != "darwin":
        return
    source = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)).resolve()
    platforms = source / "platforms"
    if not (platforms / "libqcocoa.dylib").is_file():
        raise RuntimeError("找不到当前 Qt 运行时的 Cocoa 平台插件")
    if QFileInfo(str(platforms)).isHidden() or any(
        QFileInfo(str(p)).isHidden() for p in platforms.glob("*.dylib")
    ):
        # Source checkouts can contain Finder-hidden libraries. An app's
        # previous installation must never supply plugin links to a new one.
        QCoreApplication.addLibraryPath(_plugin_mirror(source).name)
    else:
        QCoreApplication.addLibraryPath(str(source))


class PreviewWindow(QMainWindow):
    canvas_class = Canvas

    def __init__(self, study=DEFAULT_STUDY, srgb=False, samples=None):
        super().__init__()
        prepare_glass(self)
        self.samples = load_samples(study) if samples is None else samples
        self.index, self.mode, self.probe = 0, "cool-shared-rgb", "baseline"
        self.token, self.loaded_key, self.last_error = 0, None, None
        self.display_icc, self.display_name = None, "sRGB 假定显示"
        self.force_srgb = srgb
        self.cache = OrderedDict()
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.export_pool = QThreadPool(self)
        self.export_pool.setMaxThreadCount(1)
        self.exporting = False
        self.setWindowTitle("PURE RGB · 样片适配预览")
        self.resize(1440, 920)
        selected_icon = application_icon()
        self.setWindowIcon(selected_icon)
        QApplication.instance().setWindowIcon(selected_icon)
        self.build()
        self.photo_list.setCurrentRow(min(6, len(self.samples) - 1))
        QTimer.singleShot(0, self.configure_display)

    def show_about(self):
        show_about(self)

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, "_native_glass", None) is None:
            QTimer.singleShot(0, self._install_preview_glass)

    def _install_preview_glass(self):
        if not self.isVisible() or getattr(self, "_native_glass", None) is not None:
            return
        if attach_glass(self).get("attached"):
            for widget in (self, self.centralWidget()):
                _set_native_material(widget, True)
            attach_materials(self)

    def build(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        toolbar, _ = panel("toolbar")
        old = toolbar.layout()
        row = QHBoxLayout()
        self.toolbar_row = row
        old.addLayout(row)
        self.back = button("‹ 返回灯板", self.toggle_board)
        from .workflow_attention import attach_attention

        attach_attention(self.back)
        row.addWidget(self.back)
        self.brand = wordmark_label(26)
        row.addWidget(self.brand)
        self.caption = label("样片适配预览 · 已计算结果", "muted")
        row.addWidget(self.caption)
        row.addStretch()
        self.compare = button("分割对比 Native", self.toggle_split)
        self.compare.setCheckable(True)
        row.addWidget(self.compare)
        row.addWidget(button("适应", lambda: self.canvas.fit()))
        row.addWidget(button("100%", lambda: self.canvas.actual()))
        self.zoom_label = label("—", "section")
        row.addWidget(self.zoom_label)
        row.addWidget(button("关于", self.show_about))
        layout.addWidget(toolbar)
        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)
        edit = QWidget()
        edit_layout = QHBoxLayout(edit)
        edit_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter = splitter
        edit_layout.addWidget(splitter)
        left, controls = panel()
        self.sidebar_layout = controls
        left.setMinimumWidth(250)
        left.setMaximumWidth(360)
        controls.addWidget(label("01  色彩呈现", "section"))
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = {}
        for mode, title in MODES.items():
            b = button(title, lambda checked, key=mode: self.choose_mode(key))
            b.setCheckable(True)
            b.setChecked(mode == self.mode)
            self.mode_group.addButton(b)
            self.mode_buttons[mode] = b
            controls.addWidget(b)
        self.mode_note = label("", "muted", True)
        controls.addWidget(self.mode_note)
        controls.addSpacing(12)
        controls.addWidget(label("02  输入范围对照", "section"))
        self.range_box = QComboBox()
        for key, title in PROBES.items():
            self.range_box.addItem(title, key)
        self.range_box.currentIndexChanged.connect(self.choose_probe)
        controls.addWidget(self.range_box)
        controls.addWidget(
            label(
                "固定范围实验，用于检查亮暗部变化。\n高光／阴影手动滑块仍待验证。",
                "muted",
                True,
            )
        )
        controls.addSpacing(12)
        controls.addWidget(label("03  当前样片", "section"))
        self.source_label = label("", "muted", True)
        controls.addWidget(self.source_label)
        controls.addStretch()
        controls.addWidget(label("04  输出", "section"))
        controls.addWidget(
            label(
                "复制当前已计算 TIFF，保留原始\n16-bit 像素、ICC 和文件内容。",
                "muted",
                True,
            )
        )
        self.export_button = button("导出当前 TIFF…", self.export_current, True)
        controls.addWidget(self.export_button)
        splitter.addWidget(left)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = self.canvas_class()
        self.canvas.zoom_changed.connect(
            lambda z: self.zoom_label.setText(f"{z * 100:.0f}%")
        )
        self.canvas_stack = QStackedWidget()
        self.canvas_stack.addWidget(self.canvas)
        center_layout.addWidget(self.canvas_stack, 1)
        self.split_slider = ScrubberSlider(readout=lambda: f"{self.split_slider.value()}%")
        self.split_slider.setAccessibleName("Native 分割对比位置")
        self.split_slider.setRange(5, 95)
        self.split_slider.setValue(50)
        self.split_slider.valueChanged.connect(self.move_split)
        self.split_slider.hide()
        center_layout.addWidget(self.split_slider)
        center_layout.addWidget(
            label("滚轮缩放 · 拖动平移 · 对比时共用相同位置和倍率", "muted")
        )
        splitter.addWidget(center)
        right, photos = panel()
        right.setMinimumWidth(205)
        right.setMaximumWidth(300)
        self.photo_heading = label(f"照片列表   {len(self.samples)} 组", "section")
        self.photos_layout = photos
        photos.addWidget(self.photo_heading)
        self.photo_list = QListWidget()
        self.photo_list.setIconSize(QSize(60, 48))
        self.photo_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.photo_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.photo_list.currentRowChanged.connect(self.choose_sample)
        for i, sample in enumerate(self.samples):
            item = QListWidgetItem(
                f"{i + 1:02d}  {Path(sample.inputs[0]['path']).stem}\nRGB · 已计算"
            )
            item.setIcon(self.thumbnail(sample))
            self.photo_list.addItem(item)
        photos.addWidget(self.photo_list)
        self.collection_note = label(
            "主要样本：最早 rgb2\n缩略图：Native 基准", "muted", True
        )
        photos.addWidget(self.collection_note)
        splitter.addWidget(right)
        splitter.setSizes([280, 880, 240])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, False)
        self.pages.addWidget(edit)
        board_scroll = QScrollArea()
        board_scroll.setWidgetResizable(True)
        board_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        board = QWidget()
        grid = QGridLayout(board)
        grid.setSpacing(16)
        for i, sample in enumerate(self.samples):
            b = button(
                f"{i + 1:02d} · {Path(sample.inputs[0]['path']).stem}",
                lambda checked, row=i: self.open_sample(row),
            )
            b.setIcon(self.thumbnail(sample))
            b.setIconSize(QSize(210, 165))
            b.setMinimumHeight(205)
            grid.addWidget(b, i // 3, i % 3)
        board_scroll.setWidget(board)
        self.pages.addWidget(board_scroll)
        self.status = label("准备载入…", "muted")
        layout.addWidget(self.status)

    def thumbnail(self, sample):
        candidate = sample.directory / "native-preview.png"
        if candidate.exists():
            return QIcon(str(candidate))
        image = QPixmap(str(sample.directory / "comparison.png"))
        return QIcon(image.copy(0, 35, 520, 480))

    def configure_display(self):
        # NegPy's macOS detector reads the main display; do not label it per-monitor.
        self.display_icc, self.display_name = None, "sRGB 假定显示"
        if (
            not self.force_srgb
            and sys.platform == "darwin"
            and self.screen() == QApplication.primaryScreen()
        ):
            try:
                from negpy.infrastructure.display.monitor_profile import _colorsync_icc

                data = _colorsync_icc()
                if data:
                    self.display_name = (
                        "主屏 ICC："
                        + ImageCms.getProfileDescription(
                            ImageCms.ImageCmsProfile(io.BytesIO(data))
                        ).strip()
                    )
                    self.display_icc = data
            except Exception:
                self.display_icc = None
        else:
            self.display_icc, self.display_name = None, "sRGB 假定显示"
        self.cache.clear()
        self.request()
        handle = self.windowHandle()
        if handle and not getattr(self, "screen_connected", False):
            handle.screenChanged.connect(lambda screen: self.configure_display())
            self.screen_connected = True

    def choose_sample(self, row):
        if row < 0:
            return
        changed = row != self.index
        self.index = row
        self.request(reset=changed)

    def choose_mode(self, mode):
        self.mode = mode
        self.mode_buttons[mode].setChecked(True)
        self.request()

    def choose_probe(self):
        self.probe = self.range_box.currentData()
        self.request()

    def request(self, reset=False):
        if not self.samples:
            self.export_button.setEnabled(False)
            self.status.setText("导入三张 RAW 开始，或载入已验证示例")
            return
        sample = self.samples[self.index]
        probes = [
            p for p in PROBES if all(f"{p}__{m}" in sample.results for m in MODES)
        ]
        if [
            self.range_box.itemData(i) for i in range(self.range_box.count())
        ] != probes:
            self.range_box.blockSignals(True)
            self.range_box.clear()
            for key in probes:
                self.range_box.addItem(PROBES[key], key)
            self.probe = self.probe if self.probe in probes else probes[0]
            self.range_box.setCurrentIndex(self.range_box.findData(self.probe))
            self.range_box.blockSignals(False)
        self.token += 1
        token = self.token
        key = (self.index, self.mode, self.probe)
        self.loaded_key, self.last_error = None, None
        self.export_button.setEnabled(False)
        self.canvas.current = self.canvas.reference = None
        self.canvas.update()
        self.source_label.setText(
            "\n".join(f"{r['role']}  {Path(r['path']).name}" for r in sample.inputs)
            + f"\n\n密度取样：{sample.roi[2] - sample.roi[0]} × {sample.roi[3] - sample.roi[1]}\nRGB16 · 保留原 ICC"
        )
        self.mode_note.setText(
            {
                "native": "NegPy 0.68 原生呈现。",
                "cool-shared-rgb": "公开密度算法 + 共同 RGB 曲线。\n非 Nikon 原厂精确复现。",
                "flex-endpoint": "Flex 研究适配 + 端点曲线。\n合成 profile，非 X5 原厂模式。",
            }[self.mode]
        )
        self.status.setText(f"正在核验并载入 {sample.id} · {MODES[self.mode]}…")
        display_icc = self.display_icc
        mode, probe = self.mode, self.probe
        cached = self.cache.get(key)
        if cached:
            self.loaded(token, (key, *cached, reset))
            return
        self.pool.clear()

        def read():
            current = display_image(sample.record(mode, probe), display_icc)
            reference = (
                current
                if mode == "native"
                else display_image(sample.record("native", probe), display_icc)
            )
            if current.size() != reference.size():
                raise ValueError("对比图尺寸不一致")
            return key, current, reference, reset

        work = Work(token, read)
        work.signals.done.connect(self.loaded)
        work.signals.failed.connect(self.failed)
        self.pool.start(work)

    def loaded(self, token, result):
        if token != self.token:
            return
        key, current, reference, reset = result
        self.cache[key] = current, reference
        self.cache.move_to_end(key)
        while len(self.cache) > 3:
            self.cache.popitem(last=False)
        self.loaded_key = key
        self.canvas.set_images(
            current, reference, MODES[self.mode] + " · " + PROBES[self.probe], reset
        )
        self.source_label.setText(
            self.source_label.text() + f"\n输出：{current.width()} × {current.height()}"
        )
        self.export_button.setEnabled(not self.exporting)
        self.status.setText(
            f"{self.samples[self.index].id} · {MODES[self.mode]} · {PROBES[self.probe]}  |  {self.display_name}  |  离线结果，未执行 RAW 重算"
        )
        if not getattr(self, "ready_reported", False):
            print(
                f"PURE RGB preview ready: {len(self.samples)} samples; {self.display_name}",
                flush=True,
            )
            self.ready_reported = True

    def failed(self, token, message):
        if token != self.token:
            return
        self.last_error = message
        self.status.setText("载入失败：" + message)

    def toggle_split(self):
        self.canvas.split = self.compare.isChecked()
        self.split_slider.setVisible(self.canvas.split)
        self.canvas.update()

    def move_split(self, value):
        self.canvas.fraction = value / 100
        self.canvas.update()

    def toggle_board(self):
        board = self.pages.currentIndex() == 0
        self.pages.setCurrentIndex(1 if board else 0)
        self.back.setText("‹ 返回编辑" if board else "‹ 返回灯板")
        self.compare.setEnabled(not board)

    def open_sample(self, row):
        if self.pages.currentIndex() == 1:
            self.toggle_board()
        self.photo_list.setCurrentRow(row)

    def export_current(self):
        if self.loaded_key is None or self.exporting:
            return
        sample = self.samples[self.index]
        mode, probe = self.mode, self.probe
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出已计算的 RGB16 TIFF",
            str(ROOT / "outputs" / f"{sample.id}-{mode}-{probe}.tiff"),
            "TIFF (*.tiff *.tif)",
        )
        if not path:
            return
        self.exporting = True
        self.export_button.setEnabled(False)
        work = Work(path, lambda: copy_verified_tiff(sample.record(mode, probe), path))
        work.signals.done.connect(self.export_done)
        work.signals.failed.connect(self.export_failed)
        self.export_pool.start(work)
        self.status.setText("正在核验并复制 TIFF…")

    def export_done(self, path, result):
        self.exporting = False
        self.export_button.setEnabled(self.loaded_key is not None)
        self.status.setText("已导出，文件字节和 ICC 均保留：" + result)

    def export_failed(self, path, message):
        self.exporting = False
        self.export_button.setEnabled(self.loaded_key is not None)
        GlassMessageBox.warning(
            self, "导出未完成", message + "\n已有文件不会被覆盖，请选择新的文件名。"
        )

    def closeEvent(self, event):
        detach_glass(self)
        for widget in (self, self.centralWidget()):
            if widget.property("nativeMaterial"):
                _set_native_material(widget, False)
        self.pool.clear()
        self.pool.waitForDone()
        self.export_pool.waitForDone()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="PURE RGB 独立样片界面预览")
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument(
        "--srgb", action="store_true", help="固定 sRGB 预览，适合无屏幕截图"
    )
    parser.add_argument(
        "--smoke-output", type=Path, help="执行界面烟测、保存截图并退出"
    )
    args = parser.parse_args()
    os.environ.setdefault("NEGPY_USER_DIR", str(ROOT / "outputs/ui-preview-state"))
    prepare_plugins()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("PURE RGB Preview")
    app.setFont(QFont("PingFang SC", 12))
    app.setStyleSheet(STYLE)
    install_glass_surfaces(app)
    splash = startup_splash()
    try:
        window = PreviewWindow(args.study, args.srgb)
    except Exception as error:
        if splash:
            splash.close()
        GlassMessageBox.critical(None, "无法打开样片预览", str(error))
        return 1
    window.show()
    if splash:
        splash.finish(window)
    if args.smoke_output:
        from .preview_smoke import exercise

        exercise(app, window, args.smoke_output)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
