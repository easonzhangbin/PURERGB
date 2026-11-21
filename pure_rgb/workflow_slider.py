"""Two-row glass sliders; retain precise programmatic recipe values."""

from decimal import Decimal
import math

from PyQt6.QtCore import QEvent, QEasingCurve, QRect, QSize, Qt, QVariantAnimation, pyqtSignal
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QSlider, QToolButton, QVBoxLayout, QWidget


class SliderControl(QWidget):
    """A numeric parameter with a slider and a noneditable numeric readout.

    Recipe values retain their full precision until the user moves the slider.
    The discrete slider position is only a representation of that value.
    """

    valueChanged = pyqtSignal(float)
    resetStarted = pyqtSignal()
    resetFinished = pyqtSignal()

    def __init__(self, name, low, high, step, value=0, parent=None):
        super().__init__(parent)
        self._minimum, self._maximum = self._bounds(low, high)
        self._step = self._positive(step)
        self._value = self._clamp(self._finite(value))
        self._default = self._value
        self._default_description = "默认值"
        self.setAccessibleName(name)
        self.setProperty("glassControl", True)
        self.setProperty("sliderPanel", True)
        self.setProperty("glassCornerRadius", 14.0)
        self.setProperty("sliderInteraction", 0.0)
        self.setProperty("sliderHighlighted", False)
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self._hovered = False
        self._interaction = QVariantAnimation(self)
        self._interaction.setDuration(120)
        self._interaction.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._interaction.valueChanged.connect(self._animate_interaction)
        column = QVBoxLayout(self)
        self._column = column
        column.setContentsMargins(12, 8, 12, 7)
        column.setSpacing(3)
        row = QHBoxLayout()
        self._header = row
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        self.label = QLabel(name, self)
        self.label.setObjectName("sliderLabel")
        self.label.setWordWrap(True)
        self.label.setMinimumWidth(0)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.label.setToolTip(name)
        self.slider = self._create_slider()
        self.slider.setAccessibleName(name)
        self.slider.setMinimumWidth(50)
        self.slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.slider.installEventFilter(self)
        self.setFocusProxy(self.slider)
        self.readout = QLabel(self)
        self.readout.setObjectName("sliderValue")
        self.readout.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop
        )
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(11)
        self.readout.setFont(font)
        self.readout.setAccessibleName(name + " 当前值")
        self.readout.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        self.reset_button = QToolButton(self)
        self.reset_button.setObjectName("sliderReset")
        self.reset_button.setProperty("parameterReset", True)
        self.reset_button.setProperty("glassCornerRadius", 6.0)
        self.reset_button.setText("复位")
        self.reset_button.setAccessibleName(name + " 复位")
        self.reset_button.setAutoRaise(True)
        self.reset_button.setFixedSize(34, 22)
        self.reset_button.setStyleSheet(
            "QToolButton#sliderReset {background:transparent;border:1px solid rgba(255,255,255,36);"
            "border-radius:6px;padding:0;font-size:11px;color:#e9e4de;}"
            "QToolButton#sliderReset:hover {background:rgba(255,255,255,24);border-color:rgba(255,255,255,100);}"
            "QToolButton#sliderReset:pressed {background:rgba(255,255,255,42);}"
            "QToolButton#sliderReset:disabled {color:rgba(225,220,214,95);border-color:rgba(255,255,255,18);}"
            'QToolButton#sliderReset[nativeMaterial="true"] {background:transparent;border-color:transparent;}'
        )
        self.reset_button.clicked.connect(self.reset)
        self._layout_controls()
        self.slider.valueChanged.connect(self._moved)
        self._sync()
        self._update_text_height()

    def _create_slider(self):
        slider = QSlider(Qt.Orientation.Horizontal, self)
        slider.setFixedHeight(22)
        return slider

    def _layout_controls(self):
        self._header.addWidget(self.label, 1)
        self._header.addWidget(self.readout)
        self._header.addWidget(self.reset_button, 0, Qt.AlignmentFlag.AlignTop)
        self._column.addLayout(self._header)
        self._column.addWidget(self.slider)

    def _header_height(self, width):
        margins = self._column.contentsMargins()
        label_width = max(1, width - margins.left() - margins.right()
                          - self._header.spacing() * 2 - self.readout.minimumWidth()
                          - self.reset_button.width())
        metrics = self.label.fontMetrics()
        text_height = metrics.boundingRect(
            QRect(0, 0, label_width, 1_000_000), int(Qt.TextFlag.TextWordWrap), self.label.text()
        ).height()
        return max(text_height, metrics.height(),
                   self.readout.sizeHint().height(), self.reset_button.height())

    def heightForWidth(self, width):
        margins = self._column.contentsMargins()
        return (margins.top() + self._header_height(width) + self._column.spacing()
                + self.slider.height() + margins.bottom())

    def sizeHint(self):
        return QSize(260, self.heightForWidth(self.width()))

    def _update_text_height(self):
        if not hasattr(self, "readout") or getattr(self, "_adjusting_height", False):
            return
        header_height = self._header_height(self.width())
        needed = self.heightForWidth(self.width())
        if self.label.minimumHeight() == header_height and self.minimumHeight() == needed:
            return
        from .workflow_materials import stable_material_layout

        self._adjusting_height = True
        try:
            with stable_material_layout(self.window()):
                self.label.setMinimumHeight(header_height)
                self.setMinimumHeight(needed)
                self.updateGeometry()
        finally:
            self._adjusting_height = False
        # Draining the outer layout can also change this control's width.
        if (self.label.minimumHeight() != self._header_height(self.width())
                or self.minimumHeight() != self.heightForWidth(self.width())):
            self._update_text_height()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_text_height()

    def event(self, event):
        result = super().event(event)
        if event.type() in (QEvent.Type.LayoutRequest, QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._update_text_height()
        return result

    def _animate_interaction(self, value):
        self.setProperty("sliderInteraction", float(value))
        if not self.property("nativeMaterial"):
            self.update()

    def _update_interaction(self):
        highlighted = self.isEnabled() and (self._hovered or self.slider.hasFocus())
        if bool(self.property("sliderHighlighted")) != highlighted:
            self.setProperty("sliderHighlighted", highlighted)
            # Cocoa observes these properties and updates just this capsule.
            # Re-polishing an already transparent Qt background would also
            # invalidate text/layout and the photograph underneath it.
            if not self.property("nativeMaterial"):
                self.style().unpolish(self)
                self.style().polish(self)
        target = 1.0 if highlighted else 0.0
        self._interaction.stop()
        self._interaction.setStartValue(float(self.property("sliderInteraction") or 0))
        self._interaction.setEndValue(target)
        self._interaction.start()

    def enterEvent(self, event):
        self._hovered = True
        self._update_interaction()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._update_interaction()
        super().leaveEvent(event)

    def eventFilter(self, watched, event):
        # A focus event can arrive while the destroyed instance's attributes
        # are already gone; swallow it instead of printing a traceback.
        if (
            getattr(self, "slider", None) is watched
            and event.type() in (QEvent.Type.FocusIn, QEvent.Type.FocusOut)
        ):
            self._update_interaction()
        return super().eventFilter(watched, event)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.EnabledChange and hasattr(self, "_interaction"):
            self._update_interaction()

    @staticmethod
    def _finite(value):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("滑杆参数必须是有限数值")
        return float(value)

    @classmethod
    def _bounds(cls, low, high):
        low, high = cls._finite(low), cls._finite(high)
        if low > high:
            raise ValueError("滑杆下限不能超过上限")
        return low, high

    @classmethod
    def _positive(cls, value):
        value = cls._finite(value)
        if value <= 0:
            raise ValueError("滑杆步长必须大于零")
        return value

    def _clamp(self, value):
        return min(self._maximum, max(self._minimum, value))

    def _sync(self):
        count = math.ceil((self._maximum - self._minimum) / self._step)
        if count > 2_147_483_647:
            raise ValueError("滑杆步数超过支持范围")
        previous = self.slider.blockSignals(True)
        try:
            self.slider.setRange(0, count)
            self.slider.setSingleStep(1)
            self.slider.setPageStep(max(1, count // 20))
            self.slider.setValue(round((self._value - self._minimum) / self._step))
        finally:
            self.slider.blockSignals(previous)
        decimals = min(8, max(0, -Decimal(str(self._step)).as_tuple().exponent))
        label = f"{self._value:.{decimals}f}"
        if label.startswith("-") and float(label) == 0:
            label = label[1:]
        # Reserve enough space for both range ends, including matrix precision.
        # Long labels can wrap above the track without taking any track width.
        candidates = [label, *(f"{v:.{decimals}f}" for v in (self._minimum, self._maximum))]
        self.readout.setMinimumWidth(max(48, max(
            self.readout.fontMetrics().horizontalAdvance(text) for text in candidates) + 2))
        self.readout.setText(label)
        self.readout.setToolTip(format(self._value, ".15g"))
        self.readout.setAccessibleDescription(format(self._value, ".15g"))
        default = self.defaultValue()
        self.reset_button.setEnabled(self._value != default)
        self.reset_button.setToolTip("复位至" + self._default_description + " " + format(default, ".15g"))
        self.reset_button.setAccessibleDescription(self.reset_button.toolTip())
        self._update_text_height()

    def _moved(self, position):
        # Decimal arithmetic prevents a nominal 0.1 slider grid from producing
        # 0.30000000000000004 and dirtying an otherwise identical recipe.
        value = float(Decimal(str(self._minimum)) + position * Decimal(str(self._step)))
        self.setValue(self._clamp(value))

    def setValue(self, value):
        value = self._clamp(self._finite(value))
        changed = value != self._value
        self._value = value
        self._sync()
        if changed:
            self.valueChanged.emit(value)

    def value(self):
        return self._value

    def setDefaultValue(self, value, *, description=None):
        """Set a parameter's neutral value without editing its current value."""
        self._default = self._clamp(self._finite(value))
        if description is not None:
            self._default_description = str(description)
        self._sync()

    def defaultValue(self):
        return self._clamp(self._default)

    def reset(self):
        """One ordinary value edit; recipe precision and other fields survive."""
        value = self.defaultValue()
        if self._value == value:
            return
        self.resetStarted.emit()
        try:
            self.setValue(value)
        finally:
            self.resetFinished.emit()

    def setRange(self, low, high):
        self._minimum, self._maximum = self._bounds(low, high)
        self.setValue(self._value)

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum

    def setSingleStep(self, step):
        self._step = self._positive(step)
        self._sync()

    def singleStep(self):
        return self._step
