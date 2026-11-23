"""PURE Neg review controls; detection and brush recomposition remain in its CLI.

The review window mirrors the original PURE Neg standalone layout: a glass
toolbar (back, “Powered by PURE Neg” brand, view modes, zoom), one 286 px
sidebar with the numbered 01 检测 / 02 修整 sections plus a pinned 03 导出
footer, and the photo canvas with the original charcoal/amber palette.
"""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import uuid

import numpy as np
import tifffile
from PyQt6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor, QImage, QKeySequence, QPainter, QPainterPath, QPainterPathStroker,
    QPen, QShortcut, QTransform,
)
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QColorDialog,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtGui import QFontDatabase

from . import WORKSPACE_ROOT
from .workflow_progress import WorkflowProgressBar
from .branding import pureneg_powered_label, pureneg_wordmark_label
from .native import sha256
from .preview_ui import Canvas, Work, display_image
from .tasks import Tasks
from .workflow_glass import GlassMessageBox, NATIVE_SLIDER_QSS, system_dark
from .workflow_attention import AttentionButton
from .workflow_indicators import CheckIndicatorDelegate, ReviewCheckBox
from .workflow_scroll import WorkflowScrollArea
from .motion_sliders import ScrubberControl, ScrubberSlider
from .workspace import write_json
from .workflow_notices import NoticeLabel

# PURE Neg 原版设计系统（PURENeg.swift · StudioStyle）
CHARCOAL = "#141517"
AMBER = "#F0A055"
LIGHT = "#F2EDE4"
SECONDARY = "#C4C4C4"
MASK_RED = "#FF254D"
EXCLUDED_GRAY = "#8E8E93"
REPAIR_TOOL_PREFERENCES = WORKSPACE_ROOT / "outputs/workflow-ui-state/repair-tools.json"


def _last_brush_radius():
    try:
        value = json.loads(REPAIR_TOOL_PREFERENCES.read_text()).get("brush_radius")
    except (OSError, ValueError, AttributeError):
        return 8
    return value if type(value) is int and 1 <= value <= 96 else 8


REVIEW_QSS = f"""
QDialog {{ background:{CHARCOAL}; }}
QLabel {{ background:transparent; font-size:12px; color:{LIGHT}; }}
QLabel#reviewSectionIndex {{ color:{SECONDARY}; font-size:11px; font-weight:600; }}
QLabel#reviewSectionTitle {{ color:{LIGHT}; font-size:14px; font-weight:600; }}
QLabel#reviewHeading {{ color:{LIGHT}; font-size:12px; font-weight:600; }}
QLabel#reviewCounts {{ color:{AMBER}; font-size:13px; font-weight:600; }}
QLabel#reviewHint {{ color:{SECONDARY}; font-size:11px; }}
QLabel#reviewLegend {{ color:{SECONDARY}; font-size:11px; }}
QLabel#reviewRowTitle {{ color:{SECONDARY}; font-size:11px; }}
QLabel#reviewValue {{ color:{AMBER}; font-size:11px; }}
QLabel#reviewPowered {{ color:{SECONDARY}; font-size:11px; }}
QLabel#reviewMessage {{ color:{SECONDARY}; font-size:12px; }}
QFrame#reviewToolbar, QFrame#reviewSidebar {{
    background:rgba(255,255,255,15); border:1px solid rgba(255,255,255,26);
    border-radius:12px; }}
QFrame#photoWell {{ background:transparent; border:1px solid rgba(255,255,255,22);
    border-radius:12px; }}
QFrame#reviewSeparator {{ background:rgba(255,255,255,26); max-height:1px; border:0; }}
QPushButton {{ background:rgba(255,255,255,22); border:1px solid rgba(255,255,255,30);
    border-radius:9px; padding:5px 12px; min-height:18px; color:{LIGHT};
    font-size:12px; }}
QPushButton:hover {{ background:rgba(255,255,255,30); border-color:rgba(255,255,255,60); }}
QPushButton:disabled {{ color:rgba(196,196,196,110); background:rgba(255,255,255,12); }}
QPushButton#reviewPrimary {{ background:{AMBER}; color:#24150C; font-weight:600;
    border:0; padding:6px 12px; }}
QPushButton#reviewPrimary:hover {{ background:#f7b069; }}
QPushButton#reviewPrimary:disabled {{ background:rgba(255,255,255,12);
    color:rgba(196,196,196,120); }}
QCheckBox {{ color:{LIGHT}; font-size:12px; spacing:7px; background:transparent; }}
QCheckBox::indicator {{ width:17px; height:17px; border:0; background:transparent; }}
QCheckBox:disabled {{ color:rgba(196,196,196,110); }}
QListWidget#reviewRegions {{ background:rgba(255,255,255,14);
    border:1px solid rgba(255,255,255,26); border-radius:10px; font-size:12px;
    outline:0; }}
QListWidget#reviewRegions::item {{ padding:5px 6px; margin:1px 1px;
    border-radius:8px; min-height:22px; }}
QListWidget#reviewRegions::item:selected {{ background:rgba(240,160,85,52); }}
QListWidget#reviewRegions::indicator {{ width:16px; height:16px; border:0; background:transparent; }}
QSlider::groove:horizontal {{ height:4px; background:rgba(255,255,255,42);
    border-radius:2px; }}
QSlider::sub-page:horizontal {{ background:{AMBER}; border-radius:2px; }}
QSlider::handle:horizontal {{ background:{LIGHT}; border:1px solid rgba(255,255,255,60);
    width:13px; margin:-5px 0; border-radius:7px; }}
QSlider::handle:horizontal:disabled {{ background:rgba(196,196,196,120); }}
QScrollArea {{ background:transparent; border:0; }}
QDialog[nativeMaterial="true"] {{ background:transparent; }}
QFrame#reviewToolbar[nativeMaterial="true"], QFrame#reviewSidebar[nativeMaterial="true"] {{
    background:transparent; border-color:transparent; }}
QPushButton[nativeMaterial="true"], QPushButton[nativeMaterial="true"]:hover,
QPushButton[nativeMaterial="true"]:checked, QPushButton[nativeMaterial="true"]:disabled,
QComboBox[nativeMaterial="true"], QComboBox[nativeMaterial="true"]:hover {{
    background:transparent; border-color:transparent; }}
QPushButton#reviewPrimary[nativeMaterial="true"] {{
    background:transparent; border-color:transparent; color:{LIGHT}; }}
QPushButton#reviewPrimary[nativeMaterial="true"]:disabled {{
    color:rgba(196,196,196,120); }}
""" + NATIVE_SLIDER_QSS

REVIEW_LIGHT_QSS = """
QDialog { background:#f4f2ef; }
QDialog[nativeMaterial="true"] { background:transparent; }
QLabel, QPushButton, QCheckBox { color:#272523; }
QLabel#reviewSectionIndex, QLabel#reviewHint, QLabel#reviewLegend,
QLabel#reviewRowTitle, QLabel#reviewPowered, QLabel#reviewMessage { color:#514c46; }
QLabel#reviewCounts, QLabel#reviewValue { color:#8c420d; }
QPushButton:disabled, QCheckBox:disabled { color:#74716f; }
"""


def _mono_font(point_size=11, bold=False):
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    font.setPointSize(point_size)
    font.setBold(bold)
    return font


def review_section(number, title):
    """01 检测 style header: mono index + semibold title, as in the original."""
    row = QWidget()
    row.setObjectName("reviewSectionRow")
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    index = QLabel(number)
    index.setObjectName("reviewSectionIndex")
    index.setFont(_mono_font(11, bold=True))
    name = QLabel(title)
    name.setObjectName("reviewSectionTitle")
    layout.addWidget(index)
    layout.addWidget(name)
    layout.addStretch(1)
    return row


def review_separator():
    line = QFrame()
    line.setObjectName("reviewSeparator")
    line.setFixedHeight(1)
    return line


class RepairCanvas(Canvas):
    stroke_completed = pyqtSignal(object)
    split_moved = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.brush = None
        self.radius = 8
        self.points = []
        self.stroke_tool = None
        self.stroke_radius = 0
        self.hover = None
        self._cursor_rect = QRect()
        self.show_edit_preview = True
        self.pending_erase = QPainterPath()
        self.pending_add = QPainterPath()
        self.mask = None
        self.show_mask = True
        # Detection markers mirror PURE Neg's inspection rings.
        self.markers = []
        self.excluded_ids = set()
        self.selected_id = None
        self.show_markers = True
        self.mask_opacity = 0.7
        self.mask_color = QColor(255, 37, 77)
        self._mask_alpha = None
        self._split_drag = False
        self.reference_name = "原图"

    def set_mask(self, alpha):
        """Store the repair mask and build the tinted overlay image."""
        self._mask_alpha = alpha
        self._rebuild_mask()
        self.update()

    def set_mask_color(self, color):
        color = QColor(color)
        if color == self.mask_color:
            return
        self.mask_color = color
        self._rebuild_mask()
        self.update()

    def _rebuild_mask(self):
        if self._mask_alpha is None:
            self.mask = None
            return
        alpha = self._mask_alpha
        height, width = alpha.shape
        image = QImage(width, height, QImage.Format.Format_RGBA8888)
        if image.isNull():
            self.mask = None
            return
        color = self.mask_color
        image.fill(QColor(color.red(), color.green(), color.blue(), 0))
        buffer = image.bits()
        buffer.setsize(image.sizeInBytes())
        rgba = np.ndarray((height, width, 4), dtype=np.uint8, buffer=buffer,
                          strides=(image.bytesPerLine(), 4, 1))
        coverage = rgba[..., 3]
        np.greater(alpha, 0, out=coverage)
        coverage *= 255
        self.mask = image

    def _marker_rect(self, region):
        """Image-space region rect mapped to the widget, with the original's
        3 px padding and 12 px minimum ring size."""
        image_rect = QRectF(
            self.origin
            + QPointF(region["x"], region["y"]) * self.zoom,
            QRectF(0, 0, region["width"], region["height"]).size()
            * self.zoom,
        )
        center = image_rect.center()
        width = max(12.0, image_rect.width() + 6.0)
        height = max(12.0, image_rect.height() + 6.0)
        return QRectF(
            center.x() - width / 2, center.y() - height / 2, width, height
        )

    def _draw_markers(self, painter):
        for region in self.markers:
            rect = self._marker_rect(region)
            excluded = region["id"] in self.excluded_ids
            selected = region["id"] == self.selected_id
            if excluded:
                pen = QPen(QColor(EXCLUDED_GRAY), 1.5)
                pen.setStyle(Qt.PenStyle.CustomDashLine)
                pen.setDashPattern([4, 3])
            else:
                pen = QPen(
                    QColor(
                        self.mask_color.red(),
                        self.mask_color.green(),
                        self.mask_color.blue(),
                        242,
                    ),
                    2.5 if selected else 1.5,
                )
            painter.setPen(pen)
            painter.drawEllipse(rect)
            if selected:
                painter.setPen(QPen(QColor(255, 255, 255, 242), 1.0))
                painter.drawEllipse(rect.adjusted(-3, -3, 3, 3))

    def _draw_split_grip(self, painter):
        """The original's draggable ◀ ▶ grip on the split divider."""
        x = self.width() * self.fraction
        grip = QRectF(x - 10, 16, 20, 30)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        painter.setBrush(QColor(20, 21, 23, 235))
        painter.drawRoundedRect(grip, 8, 8)
        painter.setPen(QColor(LIGHT))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(grip, Qt.AlignmentFlag.AlignCenter, "◀ ▶")
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

    def point(self, position):
        p = (position - self.origin) / self.zoom
        if (
            self.current
            and 0 <= p.x() < self.current.width()
            and 0 <= p.y() < self.current.height()
        ):
            return [
                min(self.current.width() - 1, round(p.x())),
                min(self.current.height() - 1, round(p.y())),
            ]

    def set_tool(self, tool):
        if self.brush == tool:
            return
        self.brush = tool
        self.double_click_zoom_enabled = tool is None
        self._update_cursor()

    def set_radius(self, value):
        self.radius = value
        self._update_cursor()

    def _update_cursor(self):
        previous = self._cursor_rect
        self._cursor_rect = QRect()
        if self.brush and self.show_edit_preview and self.hover is not None and self.point(self.hover):
            self.setCursor(Qt.CursorShape.BlankCursor)
            radius = max(5.0, self.radius * self.zoom, self.stroke_radius * self.zoom if self.points else 0) + 3
            self._cursor_rect = QRectF(
                self.hover.x() - radius, self.hover.y() - radius, radius * 2, radius * 2
            ).toAlignedRect()
        elif self.last is None:
            self.unsetCursor()
        self.update(previous.united(self._cursor_rect))

    def enterEvent(self, event):
        self.hover = event.position()
        self._update_cursor()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hover = None
        self._update_cursor()
        super().leaveEvent(event)

    @staticmethod
    def _stroke_path(points, radius):
        first = QPointF(*points[0]) + QPointF(0.5, 0.5)
        if len(points) == 1:
            path = QPainterPath()
            path.addEllipse(first, radius, radius)
            return path
        centerline = QPainterPath(first)
        for point in points[1:]:
            centerline.lineTo(QPointF(*point) + QPointF(0.5, 0.5))
        stroker = QPainterPathStroker()
        stroker.setWidth(2 * radius)
        stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
        stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        return stroker.createStroke(centerline)

    def set_pending_strokes(self, displayed, current):
        shared = 0
        for before, after in zip(displayed, current):
            if before != after:
                break
            shared += 1
        erase, add = QPainterPath(), QPainterPath()
        for stroke in current[shared:]:
            path = self._stroke_path(stroke["points"], stroke["radius"])
            if stroke["kind"] == "erase":
                erase = erase.united(path)
                add = add.subtracted(path)
            else:
                add = add.united(path)
                erase = erase.subtracted(path)
        self.pending_erase, self.pending_add = erase, add
        self.update()

    def _draw_brush_cursor(self, painter):
        if not (self.brush and self.show_edit_preview and self.hover is not None and self.point(self.hover)):
            return
        center = self.hover
        radius = max(5.0, self.radius * self.zoom)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for color, width in (("#141517", 3), ("#ffc282" if self.brush == "add" else "#f2ede4", 1.2)):
            painter.setPen(QPen(QColor(color), width))
            painter.drawEllipse(center, radius, radius)
            painter.drawLine(center - QPointF(3, 0), center + QPointF(3, 0))
            if self.brush == "add":
                painter.drawLine(center - QPointF(0, 3), center + QPointF(0, 3))

    def mousePressEvent(self, event):
        near_split = (
            self.split
            and event.button() == Qt.MouseButton.LeftButton
            and abs(event.position().x() - self.width() * self.fraction) <= 12
        )
        if self.brush and self.show_edit_preview and event.button() == Qt.MouseButton.LeftButton:
            self._split_drag = False
            self.hover = event.position()
            p = self.point(self.hover)
            self.points = [p] if p else []
            self.stroke_tool, self.stroke_radius = self.brush, self.radius
            self._update_cursor()
        elif near_split:
            self._split_drag = True
        else:
            self._split_drag = False
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.hover = event.position()
        if self._split_drag:
            self.fraction = min(0.95, max(0.05, event.position().x() / self.width()))
            self.split_moved.emit(self.fraction)
            self.update()
        elif self.points:
            p = self.point(self.hover)
            if p and p != self.points[-1]:
                if len(self.points) >= 2048:
                    self.points = self.points[::2]
                    self.update()
                self.points.append(p)
                self.update()
        else:
            super().mouseMoveEvent(event)
        self._update_cursor()

    def mouseReleaseEvent(self, event):
        if self._split_drag:
            self._split_drag = False
        elif self.points and event.button() == Qt.MouseButton.LeftButton:
            point = self.point(event.position())
            if point and point != self.points[-1] and len(self.points) < 2048:
                self.points.append(point)
            points, self.points = self.points, []
            self.stroke_completed.emit(
                {"kind": self.stroke_tool, "radius": int(self.stroke_radius), "points": points}
            )
        else:
            super().mouseReleaseEvent(event)
        self._update_cursor()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.current:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        erase, add = self.pending_erase, self.pending_add
        if self.points:
            path = self._stroke_path(self.points, self.stroke_radius)
            if self.stroke_tool == "erase":
                erase = erase.united(path)
                add = add.subtracted(path)
            else:
                add = add.united(path)
                erase = erase.subtracted(path)
        target = QRectF(self.origin, self.current.size().toSizeF() * self.zoom)
        transform = QTransform.fromTranslate(self.origin.x(), self.origin.y()).scale(self.zoom, self.zoom)
        painter.save()
        painter.setClipRect(target)
        if self.split:
            painter.setClipRect(
                QRectF(self.width() * self.fraction, 0, self.width(), self.height()),
                Qt.ClipOperation.IntersectClip,
            )
        if self.show_edit_preview and self.reference and not erase.isEmpty():
            restored = transform.map(erase)
            painter.save()
            painter.setClipPath(restored, Qt.ClipOperation.IntersectClip)
            painter.drawImage(target, self.reference)
            painter.restore()
            visible = QPainterPath()
            visible.addRect(target)
            painter.setClipPath(visible.subtracted(restored), Qt.ClipOperation.IntersectClip)
        if self.show_mask and self.mask:
            painter.setOpacity(self.mask_opacity)
            painter.drawImage(target, self.mask)
            painter.setOpacity(1)
        if self.show_mask and self.show_edit_preview and not add.isEmpty():
            color = QColor(self.mask_color)
            color.setAlphaF(self.mask_opacity)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillPath(transform.map(add), color)
        painter.restore()
        if self.show_markers and self.markers:
            painter.save()
            if self.split:
                painter.setClipRect(
                    QRectF(self.width() * self.fraction, 0, self.width(), self.height())
                )
            self._draw_markers(painter)
            painter.restore()
        if self.split:
            self._draw_split_grip(painter)
        self._draw_brush_cursor(painter)


class RepairPreparingDialog(QDialog):
    """Immediate feedback shell shown while dust detection runs in a subprocess.

    The review editor cannot be built until the worker writes its receipt, so
    this modal presents the PURE Neg tool identity, live progress and a cancel
    action immediately on click, removing the unresponsive wait before the
    editor appears. On completion the host closes this dialog and opens the
    real RepairDialog; on failure it calls show_failure().
    """

    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        from .workflow_glass import prepare_glass

        prepare_glass(self)
        self.setWindowTitle("PURE RGB · PURE Neg 除尘")
        self.setModal(True)
        self.setMinimumWidth(440)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 26, 28, 24)
        outer.setSpacing(14)

        heading = QHBoxLayout()
        heading.setSpacing(12)
        try:
            heading.addWidget(pureneg_wordmark_label(26))
        except (OSError, RuntimeError):
            heading.addWidget(QLabel("PURE Neg"))
        self.title = QLabel("除尘准备")
        self.title.setObjectName("section")
        heading.addWidget(self.title)
        heading.addStretch()
        outer.addLayout(heading)
        self.message = NoticeLabel("正在准备完整版本并检测尘点、划痕与纤维…")
        self.message.setWordWrap(True)
        outer.addWidget(self.message)

        self.progress = WorkflowProgressBar()
        self.progress.setRange(0, 0)  # busy/indeterminate until a fraction arrives
        self.progress.setTextVisible(False)
        self.progress.setAccessibleName("除尘准备进度")
        outer.addWidget(self.progress)

        self.detail = NoticeLabel("检测在后台独立进程运行，期间可随时取消。")
        self.detail.setWordWrap(True)
        outer.addWidget(self.detail)

        row = QHBoxLayout()
        row.addStretch()
        self.cancel_button = QPushButton("取消检测")
        self.cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_button.clicked.connect(self._request_cancel)
        row.addWidget(self.cancel_button)
        self.close_button = QPushButton("关闭")
        from .workflow_attention import attach_attention

        attach_attention(self.close_button)
        self.close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_button.clicked.connect(self.accept)
        self.close_button.hide()
        row.addWidget(self.close_button)
        outer.addLayout(row)
        self._failed = False

    def _request_cancel(self):
        self.cancel_button.setEnabled(False)
        self.message.setText("正在取消检测…")
        self.cancel_requested.emit()

    def reject(self):
        # While the subprocess runs, the window close button means cancel;
        # only the failure sheet can be dismissed outright.
        if self._failed:
            super().reject()
        else:
            self._request_cancel()

    def update_progress(self, event):
        fraction = event.get("fraction") if isinstance(event, dict) else None
        text = event.get("message") if isinstance(event, dict) else None
        if text:
            self.message.setText(text)
        if fraction is None:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(max(0, min(1000, round(float(fraction) * 1000))))

    def show_failure(self, message, cancelled=False):
        self._failed = True
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.title.setText("检测未完成")
        self.message.setText("已取消，当前调整与已有除尘结果保留。" if cancelled else (message or "检测失败"))
        self.cancel_button.hide()
        self.close_button.show()

    @property
    def failed(self):
        return self._failed


# Review region ``kinds`` use the PURE Neg detection vocabulary.
KIND_LABELS = {
    "dust": "灰尘",
    "scratches": "划痕",
    "fibers": "毛发／纤维",
    "patches": "斑块",
}
UNCLASSIFIED_KIND = "_unclassified"

# (receipt threshold key, sidebar label, detection kind that enables it)
DETECTION_SLIDER_SPECS = (
    ("dust", "灰尘", "dust"),
    ("scratches", "划痕", "scratches"),
    ("fibers", "毛发／纤维", "fibers"),
    ("patches", "斑块", "patches"),
    ("curl", "纤维弯曲度", "fibers"),
    ("softness", "纤维软度", "fibers"),
)


def region_kind_text(region):
    names = [
        KIND_LABELS[key]
        for key in region.get("kinds", [])
        if key in KIND_LABELS
    ]
    return "／".join(names) if names else "未分类"


class SegmentedRow(QWidget):
    """Glass segmented selector mirroring PURE Neg's StudioSegments.

    Exposes a QComboBox-compatible data API (``currentData``/``findData``/
    ``setCurrentIndex``) so the post-check automation needs no special casing.
    """

    def __init__(self, options, initial, parent=None):
        super().__init__(parent)
        self.setObjectName("repairSegmentedRow")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self._data = []
        self._buttons = []
        for text, data in options:
            control = QToolButton(self)
            control.setText(text)
            control.setCheckable(True)
            control.setProperty("glassCornerRadius", 8.0)
            control.setProperty("glassAccent", AMBER)
            control.setCursor(Qt.CursorShape.PointingHandCursor)
            control.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            control.setFixedHeight(24)
            self.group.addButton(control)
            layout.addWidget(control, 1)
            self._data.append(data)
            self._buttons.append(control)
        self._apply_palette()
        QApplication.styleHints().colorSchemeChanged.connect(self._apply_palette)
        self.setCurrentIndex(
            self._data.index(initial) if initial in self._data else 0
        )

    def _apply_palette(self, *_):
        text, active = (SECONDARY, AMBER) if system_dark() else ("#4c4845", "#8c420d")
        self.setStyleSheet(
            f"""
            QWidget#repairSegmentedRow {{background:rgba(255,255,255,18);
                border-radius:11px;}}
            QToolButton {{font-size:12px;border:0;border-radius:8px;padding:0 10px;
                color:{text};background:transparent;}}
            QToolButton:hover {{background:rgba(255,255,255,24);}}
            QToolButton:checked {{color:{active};background:rgba(240,160,85,36);}}
            QToolButton:disabled {{color:rgba(196,196,196,100);}}
            QToolButton[nativeMaterial="true"] {{background:transparent;}}
            QToolButton[nativeMaterial="true"]:checked:enabled {{color:{LIGHT};}}
            """
        )

    def count(self):
        return len(self._data)

    def itemData(self, index):
        return self._data[index]

    def currentIndex(self):
        checked = self.group.checkedButton()
        return self._buttons.index(checked) if checked is not None else -1

    def currentData(self):
        index = self.currentIndex()
        return self._data[index] if index >= 0 else None

    def findData(self, data):
        try:
            return self._data.index(data)
        except ValueError:
            return -1

    def setCurrentIndex(self, index):
        if not 0 <= index < len(self._buttons):
            raise ValueError("分段控件索引越界")
        self._buttons[index].setChecked(True)

    def setSegmentToolTip(self, index, tip):
        if 0 <= index < len(self._buttons):
            self._buttons[index].setToolTip(tip)


class ThresholdRow(QWidget):
    """Compact single-line slider row: label · slider · amber value."""

    valueChanged = pyqtSignal(int)

    def __init__(self, title, low, high, step, value=0, suffix="%", parent=None):
        super().__init__(parent)
        self._suffix = suffix
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.title = QLabel(title)
        self.title.setObjectName("reviewRowTitle")
        self.title.setFixedWidth(84)
        self.slider = ScrubberSlider(
            self, show_labels=False, readout=lambda: f"{self.slider.value()}{self._suffix}"
        )
        self.slider.setAccessibleName(title)
        self.slider.setRange(low, high)
        self.slider.setSingleStep(step)
        self.slider.setPageStep(step * 10)
        self.slider.setValue(value)
        self.value_label = QLabel()
        self.value_label.setObjectName("reviewValue")
        self.value_label.setFont(_mono_font(11))
        self.value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.value_label.setFixedWidth(46)
        row.addWidget(self.title)
        row.addWidget(self.slider, 1)
        row.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._moved)
        self._moved(self.slider.value())

    def _moved(self, value):
        self.value_label.setText(f"{value}{self._suffix}")
        self.valueChanged.emit(value)

    def value(self):
        return self.slider.value()

    def setValue(self, value):
        self.slider.setValue(value)
        self.value_label.setText(f"{value}{self._suffix}")

    def blockSignals(self, block):
        return self.slider.blockSignals(block)

    def setToolTip(self, tip):
        super().setToolTip(tip)
        self.title.setToolTip(tip)
        self.slider.setToolTip(tip)


class RepairDialog(QDialog):
    def __init__(self, receipt_path, requests, parent):
        super().__init__(parent, Qt.WindowType.Window)
        self.setModal(True)
        from .workflow_glass import prepare_glass

        prepare_glass(self)
        self.path = Path(receipt_path)
        self.receipt = json.loads(self.path.read_text())
        self.edits = deepcopy(self.receipt["edits"])
        self.redo_strokes = []
        self.region_index = -1
        self.dirty, self.loading = False, True
        self._preview_loading = False
        self._preview_error = False
        self._closing = False
        self._displayed_strokes = []
        self.draft_path = self.path.parent / "draft-edits.json"
        if self.draft_path.exists() and self.receipt["state"] in ("ready", "previewed"):
            draft = json.loads(self.draft_path.read_text())
            if draft.get("revision") == self.receipt["revision"]:
                self.edits = draft["edits"]
                self.dirty = self.edits != self.receipt["edits"]
        self.setWindowTitle("PURE RGB · PURE Neg 除尘审核")
        self.resize(1180, 780)
        self.setMinimumSize(980, 655)
        self._apply_system_palette()
        QApplication.styleHints().colorSchemeChanged.connect(self._apply_system_palette)
        self.tasks = Tasks(requests, self)
        self.tasks.done.connect(self.finished_task)
        self.tasks.failed.connect(self.failed_task)
        self.tasks.progress.connect(
            lambda e: self.status.setText(e.get("message", "处理中…"))
        )
        self.pool = parent.pool
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(450)
        self.preview_timer.timeout.connect(self.update_preview)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        # ---------------- 顶部玻璃工具栏 ----------------
        toolbar = QFrame()
        toolbar.setObjectName("reviewToolbar")
        bar = QHBoxLayout(toolbar)
        bar.setContentsMargins(13, 7, 13, 7)
        bar.setSpacing(11)
        back = QPushButton("‹  返回来源")
        from .workflow_attention import attach_attention

        attach_attention(back)
        back.setCursor(Qt.CursorShape.PointingHandCursor)
        back.clicked.connect(self.close)
        bar.addWidget(back)
        bar.addWidget(pureneg_powered_label(24))
        bar.addStretch(1)
        self.view = SegmentedRow(
            [("原图", "source"), ("修复后", "clean"), ("分割对比", "split")],
            "clean",
        )
        self.view.group.idClicked.connect(lambda _id: self.change_view())
        bar.addWidget(self.view)
        self.fit_button = QPushButton("适应")
        self.fit_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.fit_button.clicked.connect(lambda: self.canvas.fit())
        bar.addWidget(self.fit_button)
        self.size_button = QPushButton("100%")
        self.size_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.size_button.clicked.connect(lambda: self.canvas.actual())
        bar.addWidget(self.size_button)
        self.zoom = QLabel("—")
        self.zoom.setObjectName("reviewValue")
        self.zoom.setFont(_mono_font(11))
        self.zoom.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.zoom.setMinimumWidth(48)
        bar.addWidget(self.zoom)
        outer.addWidget(toolbar)

        body = QHBoxLayout()
        body.setSpacing(12)
        outer.addLayout(body, 1)

        # ---------------- 左侧 286px 审核侧栏 ----------------
        sidebar = QFrame()
        sidebar.setObjectName("reviewSidebar")
        sidebar.setFixedWidth(286)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(15, 14, 15, 14)
        side.setSpacing(9)
        body.addWidget(sidebar)

        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        # 01 检测
        column.addWidget(review_section("01", "检测"))
        self.counts = QLabel("")
        self.counts.setObjectName("reviewCounts")
        column.addWidget(self.counts)

        heading = QLabel("检测灵敏度")
        heading.setObjectName("reviewHeading")
        heading.setToolTip("100% 为默认灵敏度。调高会纳入更多候选。")
        column.addWidget(heading)
        self.threshold_sliders = {}
        requested = set(
            self.receipt.get("review", {}).get("stats", {}).get(
                "detection_types"
            )
            or ()
        )
        applied = self._applied_thresholds()
        for key, title, enabling_kind in DETECTION_SLIDER_SPECS:
            row = ThresholdRow(title, 50, 200, 5, round(applied[key] * 100))
            enabled = not requested or enabling_kind in requested
            row.setEnabled(enabled)
            if not enabled:
                row.setToolTip("此检测项目未启用。")
            row.valueChanged.connect(lambda _value: self._update_redetect_state())
            column.addWidget(row)
            self.threshold_sliders[key] = row
        self.keep_rebate = ReviewCheckBox("保留片边和齿孔")
        baked = bool(self.receipt.get("review", {}).get("rebateBakedIn"))
        self.keep_rebate.setChecked(
            baked or bool(self.receipt.get("review", {}).get("thresholds", {}).get("keepRebate"))
        )
        if baked:
            self.keep_rebate.setEnabled(False)
            self.keep_rebate.setToolTip("这个预览已预先保护片边；重新检测后可切换。")
        self.keep_rebate.toggled.connect(lambda _value: self._update_redetect_state())
        column.addWidget(self.keep_rebate)
        self.redetect_button = QPushButton("重新检测")
        self.redetect_button.setObjectName("reviewPrimary")
        self.redetect_button.setProperty("glassAccent", AMBER)
        self.redetect_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.redetect_button.clicked.connect(self.redetect)
        column.addWidget(self.redetect_button)

        overlay_row = QHBoxLayout()
        overlay_row.setSpacing(6)
        self.mask_toggle = ReviewCheckBox("显示修复蒙版")
        self.mask_toggle.setChecked(True)
        self.mask_toggle.toggled.connect(self.mask_changed)
        overlay_row.addWidget(self.mask_toggle)
        overlay_row.addStretch(1)
        self.color_button = QPushButton()
        self.color_button.setFixedSize(38, 24)
        self.color_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.color_button.setToolTip("更改修复蒙版和补修画笔的显示颜色")
        self.color_button.setAccessibleName("修复蒙版颜色")
        self.color_button.clicked.connect(self.choose_mask_color)
        self._style_color_button()
        overlay_row.addWidget(self.color_button)
        column.addLayout(overlay_row)

        self.opacity = ThresholdRow("不透明度", 0, 100, 1, 70)
        self.opacity.valueChanged.connect(self.opacity_changed)
        column.addWidget(self.opacity)
        display_options = QVBoxLayout()
        display_options.setSpacing(6)
        self.markers_toggle = ReviewCheckBox("显示自动检测圈")
        self.markers_toggle.setChecked(True)
        self.markers_toggle.toggled.connect(self.markers_changed)
        display_options.addWidget(self.markers_toggle)
        self.list_toggle = ReviewCheckBox("展开点位列表")
        self.list_toggle.toggled.connect(
            lambda checked: self.list_body.setVisible(checked)
        )
        display_options.addWidget(self.list_toggle)
        column.addLayout(display_options)

        self.list_body = QWidget()
        list_layout = QVBoxLayout(self.list_body)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(4)
        self.regions = QListWidget()
        self.regions.setObjectName("reviewRegions")
        self.regions.setItemDelegate(CheckIndicatorDelegate(self.regions, AMBER))
        self.regions.itemChanged.connect(self.region_changed)
        self.regions.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.regions.setFixedHeight(150)
        self._fill_region_list()
        self.regions.itemClicked.connect(self.locate)
        list_layout.addWidget(self.regions)
        nav_row = QHBoxLayout()
        self.prev_button = QPushButton("上一处")
        self.prev_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_button.clicked.connect(self.goto_prev)
        self.next_button = QPushButton("下一处")
        self.next_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_button.clicked.connect(self.goto_next)
        nav_row.addWidget(self.prev_button)
        nav_row.addWidget(self.next_button)
        list_layout.addLayout(nav_row)
        self.list_body.setVisible(False)
        column.addWidget(self.list_body)

        column.addWidget(review_separator())

        # 02 修整
        column.addWidget(review_section("02", "修整"))
        self.tool = SegmentedRow(
            [("检查", None), ("画笔", "add"), ("橡皮擦", "erase")], None
        )
        self.tool.group.idClicked.connect(self._tool_selected)
        self.tool.setSegmentToolTip(0, "V · 检查自动检测点")
        self.tool.setSegmentToolTip(1, "B · 在漏掉的灰尘上涂抹")
        self.tool.setSegmentToolTip(2, "E · 涂抹误修区域以恢复原始像素")
        column.addWidget(self.tool)
        self.tool_hint = QLabel("点击列表或画面中的检测圈定位；取消勾选可排除该处修复。")
        self.tool_hint.setObjectName("reviewHint")
        self.tool_hint.setWordWrap(True)
        column.addWidget(self.tool_hint)
        self.radius = ThresholdRow("画笔半径", 1, 96, 1, _last_brush_radius(), " px")
        self.radius.setToolTip("按 【 或 】 调整画笔半径；下次除尘沿用此半径。")
        self.radius.valueChanged.connect(self._radius_changed)
        column.addWidget(self.radius)
        history_row = QHBoxLayout()
        history_row.setSpacing(6)
        self.undo_button = QPushButton("撤销")
        self.undo_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.undo_button.setToolTip("撤销上次修整（⌘Z）")
        self.undo_button.clicked.connect(self.undo)
        self.redo_button = QPushButton("重做")
        self.redo_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.redo_button.setToolTip("重做上次撤销的修整（⇧⌘Z）")
        self.redo_button.clicked.connect(self.redo)
        history_row.addWidget(self.undo_button)
        history_row.addWidget(self.redo_button)
        column.addLayout(history_row)
        column.addStretch(1)

        scroll = WorkflowScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side.addWidget(scroll, 1)

        # 03 导出（固定在侧栏底部，随滚动区收缩）
        side.addWidget(review_separator())
        side.addWidget(review_section("03", "导出"))
        self.status = NoticeLabel("载入审核结果…")
        self.status.setObjectName("reviewMessage")
        self.status.setWordWrap(True)
        side.addWidget(self.status)
        secondary_row = QHBoxLayout()
        secondary_row.setSpacing(6)
        self.update_button = AttentionButton("重试预览")
        self.update_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update_button.clicked.connect(self.update_preview)
        secondary_row.addWidget(self.update_button)
        self.export_button = QPushButton("确认并另存修复结果…")
        from .workflow_attention import attach_attention

        attach_attention(self.export_button)
        self.export_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_button.clicked.connect(self.confirm_export)
        secondary_row.addWidget(self.export_button)
        side.addLayout(secondary_row)
        self.confirm_button = AttentionButton("确认并应用到工作台")
        self.confirm_button.setObjectName("reviewPrimary")
        self.confirm_button.setProperty("glassAccent", AMBER)
        self.confirm_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.confirm_button.clicked.connect(self.confirm)
        self.confirm_button.setDefault(True)
        side.addWidget(self.confirm_button)

        # ---------------- 中央画布 ----------------
        center = QVBoxLayout()
        center.setSpacing(9)
        well = QFrame()
        well.setObjectName("photoWell")
        well_layout = QHBoxLayout(well)
        well_layout.setContentsMargins(1, 1, 1, 1)
        self.canvas = RepairCanvas()
        self.canvas.set_radius(self.radius.value())
        well_layout.addWidget(self.canvas)
        center.addWidget(well, 1)
        bottom = QHBoxLayout()
        bottom.setSpacing(12)
        legend = QLabel("滚动缩放 · 拖动平移 · 【／】调整画笔大小")
        legend.setObjectName("reviewLegend")
        bottom.addWidget(legend)
        bottom.addStretch(1)
        self.divider = ScrubberControl("分割对比位置（%）", 5, 95, 1, 50)
        self.divider.setToolTip("移动修复后／原图的分割位置，范围为 5%–95%。")
        self.divider.slider.setToolTip(self.divider.toolTip())
        self.divider.slider.setPageStep(10)
        self.divider.valueChanged.connect(lambda _: self.divider.slider.setPageStep(10))
        self.divider.setFocusProxy(self.divider.slider)
        self.divider.valueChanged.connect(self.move_divider)
        self.divider.setVisible(False)
        self.divider.setMaximumWidth(220)
        bottom.addWidget(self.divider)
        center.addLayout(bottom)
        body.addLayout(center, 1)

        # ---------------- 画布与快捷键接线 ----------------
        self.canvas.zoom_changed.connect(
            lambda z: self.zoom.setText(f"{z * 100:.0f}%")
        )
        self.canvas.stroke_completed.connect(self.stroke)
        self.canvas.split_moved.connect(self.sync_divider)
        self.canvas.markers = self.receipt["review"]["regions"]
        self.canvas.excluded_ids = set(self.edits["excludedRegions"])
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self.undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, activated=self.redo)
        for key, target in (("[", -2), ("]", 2)):
            QShortcut(QKeySequence(key), self).activated.connect(
                lambda delta=target: self.radius.setValue(
                    max(1, min(96, self.radius.value() + delta))
                )
            )
        for index, key in enumerate(("V", "B", "E")):
            QShortcut(QKeySequence(key), self).activated.connect(
                lambda pick=index: self._pick_tool(pick)
            )
        self._sync_nav_buttons()
        self._refresh_counts()
        self._update_redetect_state()
        self.loading = False
        self.load_images()

    # ------------------------------------------------------------------ #
    # Canvas-driving controls
    # ------------------------------------------------------------------ #

    def _radius_changed(self, value):
        self.canvas.set_radius(value)
        # Tool preferences belong to the application, not the photo's edits.
        try:
            write_json(REPAIR_TOOL_PREFERENCES, {"brush_radius": int(value)})
        except OSError:
            pass

    def _apply_system_palette(self, *_):
        self.setStyleSheet(REVIEW_QSS if system_dark() else REVIEW_QSS + REVIEW_LIGHT_QSS)

    def _pick_tool(self, index):
        if self.tool.isEnabled():
            self.tool.setCurrentIndex(index)
            self._tool_selected()

    def _tool_selected(self, *_):
        if self.tool.currentData() and self.view.currentData() == "source":
            self.view.setCurrentIndex(1)
            self.change_view()
        self.set_brush()

    def _style_color_button(self):
        color = self.canvas.mask_color if hasattr(self, "canvas") else QColor(255, 37, 77)
        self.color_button.setProperty("glassTint", color.name())
        self.color_button.setProperty("glassCornerRadius", 6.0)
        self.color_button.setStyleSheet(
            f"QPushButton {{background:{color.name()};"
            "border:1px solid rgba(255,255,255,60);border-radius:6px;}"
            "QPushButton:hover {border-color:rgba(255,255,255,120);}"
            'QPushButton[nativeMaterial="true"] {background:transparent;border-color:transparent;}'
        )

    def choose_mask_color(self):
        color = QColorDialog.getColor(
            self.canvas.mask_color, self, "修复蒙版颜色"
        )
        if color.isValid():
            self.canvas.set_mask_color(color)
            self._style_color_button()

    def set_brush(self):
        self.canvas.set_tool(self.tool.currentData() if self._can_edit() else None)

    def _can_edit(self):
        return (
            (not self.loading or self._preview_loading)
            and (not self.tasks.busy or self.tasks.request.get("action") == "repair_preview")
            and self.receipt["state"] in ("ready", "previewed")
            and not self._closing
        )

    def mask_changed(self, value):
        self.change_view()

    def markers_changed(self, value):
        self.canvas.show_markers = value
        self.canvas.update()

    def opacity_changed(self, value):
        self.canvas.mask_opacity = value / 100.0
        self.canvas.update()

    def sync_divider(self, fraction):
        self.divider.slider.setValue(round(fraction * 100))

    def move_divider(self, value):
        self.canvas.fraction = value / 100
        self.canvas.update()

    def _sync_nav_buttons(self):
        count = self.regions.count()
        self.prev_button.setEnabled(self.region_index > 0)
        self.next_button.setEnabled(0 <= self.region_index < count - 1)

    def focus_region(self, index):
        if not 0 <= index < self.regions.count():
            return
        self.region_index = index
        item = self.regions.item(index)
        self.regions.setCurrentItem(item)
        self.locate(item)
        self._sync_nav_buttons()

    def goto_prev(self):
        if self.region_index > 0:
            self.focus_region(self.region_index - 1)

    def goto_next(self):
        if 0 <= self.region_index < self.regions.count() - 1:
            self.focus_region(self.region_index + 1)

    def load_images(self):
        self.loading = True
        self.refresh_actions()
        record = self.receipt["preview"]
        source = self.receipt["source"]
        display = self.parent().display_icc
        cached_original = getattr(self, "original", None)

        def read():
            clean, original = (
                display_image(record, display),
                cached_original if cached_original is not None else display_image(source, display),
            )
            if sha256(record["mask_path"]) != record["mask_sha256"]:
                raise ValueError("审核蒙版发生变化")
            mask = tifffile.imread(record["mask_path"]).squeeze()
            alpha = np.where(mask > 0, 255, 0).astype(np.uint8)
            return clean, original, alpha

        job = Work(0, read)
        job.signals.done.connect(self.images_ready)
        job.signals.failed.connect(lambda token, msg: self.load_failed(msg))
        self.pool.start(job)

    def images_ready(self, token, result):
        self.clean, self.original = result[0], result[1]
        self.canvas.set_mask(result[2])
        self.loading = False
        self._preview_loading = False
        self._displayed_strokes = self.receipt["edits"]["strokes"]
        self._refresh_live_strokes()
        self.change_view()
        self.refresh_actions()
        meta = self.receipt["preview"]
        review = self.receipt["review"]
        stats = review.get("stats", {})
        capabilities = (
            "FibreNet "
            + ("已加载" if stats.get("learned_fibre_map") else "未加载")
            + " · 除尘原生内核 "
            + ("已加载" if stats.get("native") else "未加载")
        )
        source_name = review.get("sourceName") or Path(
            self.receipt["source"]["path"]
        ).name
        detail = (
            f"{source_name} · {review.get('width', '?')}×{review.get('height', '?')}"
            f" · {review.get('depth', 16)}-bit TIFF · "
            f"{len(review['regions'])} 处候选 · 改变 {meta['changed_pixels']} 像素 · "
            f"蒙版外变化 {meta.get('outside_mask_changed_pixels', 0)} · "
            f"RGB16 / ICC 已核验 · {capabilities}"
        )
        self.status.setText(
            f"{len(review['regions'])} 处候选 · 改变 {meta['changed_pixels']} 像素 · "
            f"RGB16 / ICC 已核验"
        )
        self.status.setToolTip(detail)
        if self.dirty:
            self.status.setText("正在更新审核预览…")
            self.preview_timer.start()

    def load_failed(self, message):
        if self._preview_loading:
            self.loading = self._preview_loading = False
            self._preview_error = self.dirty = True
        self.status.setText("载入失败：" + message)
        self.refresh_actions()

    def change_view(self):
        if not hasattr(self, "clean"):
            return
        mode = self.view.currentData()
        self.canvas.split = mode == "split"
        self.canvas.show_edit_preview = mode != "source"
        self.canvas.set_images(
            self.original if mode == "source" else self.clean,
            self.original,
            "修复后",
        )
        self.canvas.show_mask = mode != "source" and self.mask_toggle.isChecked()
        self.divider.setVisible(mode == "split")
        self.canvas._update_cursor()

    # ------------------------------------------------------------------ #
    # Detection counts and the PURE Neg redetect loop
    # ------------------------------------------------------------------ #

    def _kind_counts(self):
        excluded_ids = set(self.edits["excludedRegions"])
        total, excluded = Counter(), Counter()
        for region in self.receipt["review"]["regions"]:
            keys = [
                key for key in region.get("kinds", []) if key in KIND_LABELS
            ] or [UNCLASSIFIED_KIND]
            for key in keys:
                total[key] += 1
                if region["id"] in excluded_ids:
                    excluded[key] += 1
        return total, excluded

    def _refresh_counts(self):
        regions = self.receipt["review"]["regions"]
        excluded = set(self.edits["excludedRegions"])
        active = sum(1 for region in regions if region["id"] not in excluded)
        text = f"自动 {active} 处 · 手动 {len(self.edits['strokes'])} 笔"
        self.counts.setText(text)
        self.counts.setToolTip(text)
        total, skipped = self._kind_counts()
        names = {
            **{key: KIND_LABELS[key] for key in KIND_LABELS},
            UNCLASSIFIED_KIND: "未分类",
        }
        kinds = " · ".join(
            f"{names[key]} {total[key]}" + (f"（排除 {skipped[key]}）" if skipped[key] else "")
            for key in total
        )
        if kinds:
            self.counts.setToolTip(f"{text}\n{kinds}")

    def _applied_thresholds(self):
        thresholds = self.receipt.get("review", {}).get("thresholds", {}) or {}
        return {key: float(thresholds.get(key, 1.0)) for key, _, _ in DETECTION_SLIDER_SPECS}

    def _draft_thresholds(self):
        return {
            key: self.threshold_sliders[key].value() / 100.0
            for key, _, _ in DETECTION_SLIDER_SPECS
        }

    def _update_redetect_state(self):
        if not hasattr(self, "redetect_button"):
            return
        editable = (
            not self.loading
            and not self.tasks.busy
            and not self.dirty
            and self.receipt["state"] in ("ready", "previewed")
        )
        changed = self._draft_thresholds() != self._applied_thresholds()
        baked = bool(self.receipt.get("review", {}).get("rebateBakedIn"))
        rebate_changed = self.keep_rebate.isEnabled() and (
            self.keep_rebate.isChecked()
            != bool(self.receipt.get("review", {}).get("thresholds", {}).get("keepRebate"))
        )
        self.redetect_button.setEnabled(editable and (changed or rebate_changed or baked))

    def redetect(self):
        if self.loading or self.tasks.busy or self.dirty:
            self.status.setText("正在更新审核预览，请稍后重新检测。")
            return
        thresholds = self._draft_thresholds()
        self.tasks.start(
            {
                "action": "repair_redetect",
                "receipt": str(self.path),
                "revision": self.receipt["revision"],
                "edits": deepcopy(self.edits),
                **thresholds,
                "keep_rebate": self.keep_rebate.isChecked(),
            }
        )
        self.status.setText("正在按新灵敏度重新检测；旧审核在结果核验前保留。")
        self.refresh_actions()

    def _fill_region_list(self):
        excluded = set(self.edits["excludedRegions"])
        self.regions.blockSignals(True)
        self.regions.clear()
        for region in self.receipt["review"]["regions"]:
            item = QListWidgetItem(
                f"{region['id']} · {region_kind_text(region)}  ({region['x']},{region['y']})"
            )
            item.setData(Qt.ItemDataRole.UserRole, region)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Unchecked
                if region["id"] in excluded
                else Qt.CheckState.Checked
            )
            self.regions.addItem(item)
        self.regions.blockSignals(False)

    def _rebuild_region_list(self):
        self._fill_region_list()
        self.region_index = -1
        self.canvas.selected_id = None
        self.canvas.markers = self.receipt["review"]["regions"]
        self.canvas.excluded_ids = set(self.edits["excludedRegions"])
        self.canvas.update()

    def _reset_detection_panel(self):
        applied = self._applied_thresholds()
        for key, _, _ in DETECTION_SLIDER_SPECS:
            self.threshold_sliders[key].blockSignals(True)
            self.threshold_sliders[key].setValue(round(applied[key] * 100))
            self.threshold_sliders[key].blockSignals(False)
        baked = bool(self.receipt.get("review", {}).get("rebateBakedIn"))
        self.keep_rebate.blockSignals(True)
        self.keep_rebate.setChecked(
            baked
            or bool(self.receipt.get("review", {}).get("thresholds", {}).get("keepRebate"))
        )
        self.keep_rebate.setEnabled(not baked)
        self.keep_rebate.blockSignals(False)
        self._update_redetect_state()

    def _apply_redetect(self, receipt):
        self.receipt = receipt
        self.edits = deepcopy(receipt["edits"])
        self.redo_strokes.clear()
        self.draft_path.unlink(missing_ok=True)
        self.dirty = False
        self._rebuild_region_list()
        self._reset_detection_panel()
        self.load_images()
        unmatched = receipt.get("redetection", {}).get("unmatched_exclusions", 0)
        suffix = f"；{unmatched} 处旧排除未能映射到新点位" if unmatched else ""
        self.status.setText(
            f"重新检测完成：{len(receipt['review']['regions'])} 处候选{suffix}"
        )

    def region_changed(self, item):
        if not self._can_edit():
            return
        self.edits["excludedRegions"] = [
            self.regions.item(i).data(Qt.ItemDataRole.UserRole)["id"]
            for i in range(self.regions.count())
            if self.regions.item(i).checkState() == Qt.CheckState.Unchecked
        ]
        self.canvas.excluded_ids = set(self.edits["excludedRegions"])
        self.canvas.update()
        self._refresh_counts()
        self.mark_dirty()

    def locate(self, item):
        region = item.data(Qt.ItemDataRole.UserRole)
        self.region_index = self.regions.row(item)
        self.canvas.selected_id = region["id"]
        self._sync_nav_buttons()
        if not self.canvas.current:
            return
        self.canvas.actual()
        self.canvas.origin = (
            QPointF(self.canvas.width() / 2, self.canvas.height() / 2)
            - QPointF(
                region["x"] + region["width"] / 2, region["y"] + region["height"] / 2
            )
            * self.canvas.zoom
        )
        self.canvas.update()

    def stroke(self, stroke):
        if not self._can_edit():
            return
        if (
            len(self.edits["strokes"]) >= 1024
            or sum(len(s["points"]) for s in self.edits["strokes"])
            + len(stroke["points"])
            > 200000
        ):
            self.status.setText("已达到 PURE Neg 单次审核笔画上限，请先完成当前审核。")
            return
        self.edits["strokes"].append(stroke)
        self.redo_strokes.clear()
        self.mark_dirty()

    def undo(self):
        if self._can_edit() and not self.canvas.points and self.edits["strokes"]:
            self.redo_strokes.append(self.edits["strokes"].pop())
            self.mark_dirty()

    def redo(self):
        if self._can_edit() and not self.canvas.points and self.redo_strokes:
            self.edits["strokes"].append(self.redo_strokes.pop())
            self.mark_dirty()

    def mark_dirty(self):
        self.dirty = self.edits != self.receipt["edits"] or self.tasks.busy
        self._preview_error = False
        self._sync_draft()
        self._refresh_live_strokes()
        if self.dirty:
            self.status.setText("正在更新审核预览…")
            self.preview_timer.start()
        else:
            self.status.setText("审核预览已更新")
        self.refresh_actions()

    def _sync_draft(self):
        if self.dirty:
            write_json(
                self.draft_path, {"revision": self.receipt["revision"], "edits": self.edits}
            )
        else:
            self.draft_path.unlink(missing_ok=True)

    def _refresh_live_strokes(self):
        self.canvas.set_pending_strokes(self._displayed_strokes, self.edits["strokes"])

    def refresh_actions(self):
        editable = self._can_edit()
        available = editable and not self.loading and not self.tasks.busy
        retry = available and self.dirty and self._preview_error
        self.update_button.setVisible(retry)
        self.update_button.setEnabled(retry)
        self.confirm_button.setEnabled(available and not self.dirty)
        self.update_button.set_attention(retry)
        self.confirm_button.set_attention(available and not self.dirty)
        self.export_button.setEnabled(available and not self.dirty)
        self.regions.setEnabled(editable)
        self.tool.setEnabled(editable)
        self.radius.setEnabled(editable)
        self.undo_button.setEnabled(editable and bool(self.edits["strokes"]))
        self.redo_button.setEnabled(editable and bool(self.redo_strokes))
        self._sync_nav_buttons()
        self._update_redetect_state()
        self.set_brush()

    def update_preview(self):
        if self._closing or self.loading or self.tasks.busy or not self.dirty:
            return
        if self.canvas.points:
            self.preview_timer.start()
            return
        self._preview_error = False
        try:
            self.tasks.start(
                {
                    "action": "repair_preview",
                    "receipt": str(self.path),
                    "revision": self.receipt["revision"],
                    "edits": deepcopy(self.edits),
                    "replace_live_preview": True,
                }
            )
        except Exception as error:
            self._preview_error = True
            self.status.setText("预览更新失败：" + str(error))
        self.refresh_actions()

    def confirm(self):
        destination = self.path.parent / ("confirmed-" + uuid.uuid4().hex)
        self._begin_confirm(destination)

    def confirm_export(self):
        destination = QFileDialog.getExistingDirectory(
            self, "选择独立修复结果目录", str(self.path.parent)
        )
        if not destination:
            return
        self._begin_confirm(destination)

    def _begin_confirm(self, destination):
        if self.loading or self.tasks.busy or self.dirty or self.receipt["state"] not in ("ready", "previewed"):
            return
        self.tasks.start(
            {
                "action": "repair_confirm",
                "receipt": str(self.path),
                "revision": self.receipt["revision"],
                "edits": deepcopy(self.edits),
                "output": str(destination),
            }
        )
        self.status.setText("正在确认并核验修复结果，完成后返回工作台…")
        self.refresh_actions()

    def finished_task(self, request, result):
        if self._closing:
            return
        if request.get("action") == "repair_redetect":
            self._apply_redetect(result["receipt"])
            self.refresh_actions()
            return
        self.receipt = result["receipt"]
        if request.get("action") == "repair_preview":
            self.dirty = self.edits != self.receipt["edits"]
            self._sync_draft()
            self._preview_loading = True
            self.load_images()
            return
        self.edits = deepcopy(self.receipt["edits"])
        self.draft_path.unlink(missing_ok=True)
        self.dirty = False
        if self.receipt["state"] == "confirmed":
            self.setWindowTitle("PURE RGB · 已确认除尘结果")
            self.accept()
            return
        self.load_images()

    def failed_task(self, request, message, cancelled):
        if self._closing:
            return
        if request.get("action") == "repair_preview":
            self.dirty = self.edits != self.receipt["edits"]
            self._preview_error = self.dirty
            self._sync_draft()
            self.status.setText("预览更新失败：" + message)
            self.refresh_actions()
            return
        self.status.setText(message)
        if self.path.exists():
            self.receipt = json.loads(self.path.read_text())
        self.refresh_actions()
        if not getattr(self.parent(), "automation", False):
            GlassMessageBox.warning(self, "审核未完成", message)

    def _stop_review(self):
        if self.tasks.busy and self.tasks.request["action"] == "repair_confirm":
            self.status.setText("正在确认并发布结果，请完成后再返回。")
            return False
        if self._closing:
            return True
        self._closing = True
        self.preview_timer.stop()
        self.tasks.shutdown()
        self.pool.waitForDone()
        return True

    def reject(self):
        if self._stop_review():
            super().reject()

    def closeEvent(self, event):
        if self._stop_review():
            super().closeEvent(event)
        else:
            event.ignore()
