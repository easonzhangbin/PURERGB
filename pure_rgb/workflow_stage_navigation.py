"""A persistent passive stage capsule; only its position animates."""

from PyQt6.QtCore import QAbstractAnimation, QEasingCurve, QEvent, QObject, QPropertyAnimation, QRect, QTimer, Qt, pyqtSlot
from PyQt6.QtWidgets import QWidget

from .workflow_glass import system_reduce_motion


class StageNavigation(QObject):
    def __init__(self, bar, buttons, *, duration=220):
        super().__init__(bar)
        self.bar, self.buttons = bar, buttons
        self.current_index = None
        self.disposed = False
        self._snap_pending = False
        self.indicator = QWidget(bar)
        self.indicator.setObjectName("stageHighlight")
        self.indicator.setProperty("glassControl", True)
        self.indicator.setProperty("stageHighlight", True)
        self.indicator.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.indicator.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.indicator.lower()
        self.indicator.hide()
        self.animation = QPropertyAnimation(self.indicator, b"geometry", self)
        self.animation.setDuration(duration)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.layout_timer = QTimer(self)
        self.layout_timer.setSingleShot(True)
        self.layout_timer.timeout.connect(self.snap_to_layout)
        bar.installEventFilter(self)
        for button in buttons.values():
            button.installEventFilter(self)
        bar.destroyed.connect(self.dispose)

    def _target(self):
        button = self.buttons.get(self.current_index)
        if button is None or button.isHidden():
            return None
        return QRect(button.geometry())

    def select(self, index, *, animate=True):
        if self.disposed or index not in self.buttons:
            return
        if self.bar.layout() is not None:
            self.bar.layout().activate()
        self.current_index = index
        target = self._target()
        if target is None or target.isEmpty():
            return
        if (not self._snap_pending and self.animation.state() == QAbstractAnimation.State.Running
                and self.animation.endValue() == target):
            return
        start = QRect(self.indicator.geometry())
        was_visible = self.indicator.isVisibleTo(self.bar)
        self.animation.stop()
        self.indicator.show()
        self.indicator.lower()
        if (not animate or system_reduce_motion() or self._snap_pending or not was_visible
                or not self.bar.isVisible() or start == target):
            self.indicator.setGeometry(target)
            return
        self.animation.setStartValue(start)
        self.animation.setEndValue(target)
        self.animation.start()

    def snap_to_layout(self):
        if getattr(self, "disposed", True):
            return
        if self.bar.layout() is not None:
            self.bar.layout().activate()
        target = self._target()
        if target is None or target.isEmpty():
            return
        # A page change can post a layout request without moving the bar.
        # Keep an active transition if its destination has not changed.
        if (not self._snap_pending and self.animation.state() == QAbstractAnimation.State.Running
                and self.animation.endValue() == target):
            return
        self._snap_pending = False
        self.animation.stop()
        self.indicator.setGeometry(target)

    def eventFilter(self, watched, event):
        if getattr(self, "disposed", True):
            return False
        kind = event.type()
        if watched is self.bar and kind == QEvent.Type.Hide:
            self.animation.stop()
        elif kind in {QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.Show, QEvent.Type.LayoutRequest}:
            if kind in {QEvent.Type.Resize, QEvent.Type.Move}:
                self._snap_pending = True
            self.layout_timer.start(0)
        elif watched is self.bar and kind == QEvent.Type.Paint:
            self.snap_to_layout()
        return False

    @pyqtSlot()
    def dispose(self):
        if not getattr(self, "disposed", True):
            self.disposed = True
            self.layout_timer.stop()
            self.animation.stop()
