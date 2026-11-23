"""Shared alignment and measured sizing for transient messages."""

import math

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtWidgets import QApplication, QLabel


def center_notice_text(label, *, wrap=True):
    """Center message text without changing ordinary parameter descriptions."""
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setIndent(0)
    label.setWordWrap(wrap)
    label.setProperty("noticeText", True)
    return label


def _cocoa_text(label):
    """Measure with the same AppKit font/cell that paints native glass HUDs."""
    if QApplication.platformName() != "cocoa":
        return None
    import AppKit as A

    field = A.NSTextField.labelWithString_(label.text())
    field.setBordered_(False)
    field.setDrawsBackground_(False)
    field.setAlignment_(A.NSTextAlignmentCenter)
    font = label.font()
    size = font.pixelSize() if font.pixelSize() > 0 else font.pointSizeF()
    native_font = A.NSFont.fontWithName_size_(font.family(), max(10., float(size)))
    field.setFont_(native_font or A.NSFont.systemFontOfSize_(max(10., float(size))))
    return field


class NoticeLabel(QLabel):
    """A centered message whose box follows its actual wrapped text height."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        center_notice_text(self)

    def fit_to_width(self, maximum):
        self.ensurePolished()
        maximum = max(1, int(maximum))
        metrics = self.fontMetrics()
        margins = self.contentsMargins()
        padding = margins.left() + margins.right()
        natural = max((metrics.horizontalAdvance(line) for line in self.text().splitlines()), default=0)
        native = _cocoa_text(self)
        if native is not None:
            native.cell().setUsesSingleLineMode_(True)
            native.cell().setWraps_(False)
            natural = math.ceil(native.cell().cellSize().width)
        available = max(1, maximum - padding)
        lines = max(1, math.ceil(natural / available))
        # Balance a long sentence across lines rather than leaving a single
        # trailing character on an otherwise empty second line.
        content = min(available, math.ceil(natural / lines) + (metrics.height() if lines > 1 else 2))
        width = min(maximum, max(1, content + padding))
        flags = Qt.AlignmentFlag.AlignCenter.value | Qt.TextFlag.TextWordWrap.value
        text_bounds = metrics.boundingRect(QRect(0, 0, max(1, width - padding), 100000), flags, self.text())
        height = text_bounds.height() + margins.top() + margins.bottom()
        if native is not None:
            native.cell().setUsesSingleLineMode_(False)
            native.cell().setWraps_(True)
            native.setMaximumNumberOfLines_(0)
            measured = native.cell().cellSizeForBounds_(((0, 0), (max(1, width - padding), 100000.)))
            native_height = math.ceil(measured.height) + margins.top() + margins.bottom()
            height = native_height if self.property("nativeMaterial") else max(height, native_height)
        self.setFixedSize(width, max(1, height))


def center_message_box(box):
    """Apply one text policy to errors, confirmations and informative prompts."""
    for label in box.findChildren(QLabel):
        if label.objectName() in {"qt_msgbox_label", "qt_msgbox_informativelabel"}:
            center_notice_text(label)
