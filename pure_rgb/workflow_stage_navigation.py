"""Rubber Dock springs over the existing guarded workflow stage buttons.

Two independent edges and an overlaid action card never change page layout.
Motion tokens follow the official Rubber Dock study.
"""

import math

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, QTimer, Qt, pyqtSlot
from PyQt6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QLayout, QPushButton, QVBoxLayout, QWidget

from .motion_spring import Spring
from .workflow_glass import system_dark, system_reduce_motion


def _clamp(value, low, high):
    return max(low, min(high, value))


class StageNavigation(QObject):
    def __init__(self, bar, buttons):
        super().__init__(bar)
        self.bar, self.buttons = bar, buttons
        self.current_index = None
        self.disposed = self.expanded = self._snap_pending = False
        self._drag = None
        self._press_current = None
        self._suppress_card = self._cancelled_press = False
        self._base = QRect()
        self._card_height = 0
        self._card_actions = []
        self.card = None
        self.indicator = QWidget(bar)
        self.indicator.setObjectName("stageHighlight")
        self.indicator.setProperty("glassControl", True)
        self.indicator.setProperty("stageHighlight", True)
        self.indicator.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.indicator.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.indicator.lower()
        self.indicator.hide()
        self.left = Spring(self, 0, self._render_indicator, stiffness=640, damping=38)
        self.right = Spring(self, 0, self._render_indicator, stiffness=150, damping=16)
        self.nudge = Spring(self, 0, self._render_indicator, stiffness=300, damping=18)
        self.card_width = Spring(self, 0, self._render_card, stiffness=340, damping=21)
        self.card_height = Spring(self, 0, self._render_card, stiffness=230, damping=21)
        self.layout_timer = QTimer(self)
        self.layout_timer.setSingleShot(True)
        self.layout_timer.timeout.connect(self.snap_to_layout)
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(500)
        self.status_timer.timeout.connect(self.refresh_card)
        # Ordinary navigation observes its own controls only. The application
        # filter exists only while the card needs outside-click dismissal.
        self.application = QApplication.instance()
        self._filter_installed = False
        bar.installEventFilter(self)
        for index, button in buttons.items():
            button.installEventFilter(self)
            button.setAccessibleDescription("切换到此步骤；再次按下可展开步骤操作与实时任务状态。")
            button.pressed.connect(lambda index=index: self._pressed(index))
            button.clicked.connect(lambda _checked=False, index=index: self._clicked(index))
        bar.destroyed.connect(self.dispose)

    @property
    def springs(self):
        return self.left, self.right, self.nudge, self.card_width, self.card_height

    def _target(self):
        button = self.buttons.get(self.current_index)
        return None if button is None or button.isHidden() else QRect(button.geometry())

    def _render_indicator(self, _value=0):
        if self.disposed or self._base.isEmpty():
            return
        left, right = sorted((self.left.value, self.right.value))
        width = max(right - left, 24.0)
        height = self._base.height() * _clamp(1 / math.sqrt(width / self._base.width()), .66, 1.1)
        self.indicator.setGeometry(round(left + self.nudge.value),
                                   round(self._base.center().y() + .5 - height / 2),
                                   round(width), round(height))

    def _move_to(self, target, direction, *, drag=False):
        if not direction:
            self.left.tune(600, 44)
            self.right.tune(600, 44)
        else:
            lead, trail = ((1100, 56), (320, 26)) if drag else ((640, 38), (150, 16))
            first, last = (trail, lead) if direction > 0 else (lead, trail)
            self.left.tune(*first)
            self.right.tune(*last)
        self.left.to(target.left())
        self.right.to(target.x() + target.width())

    def _jump(self, target):
        self._base = QRect(target)
        self.left.jump(target.left())
        self.right.jump(target.x() + target.width())
        self.nudge.jump(0)
        self._snap_pending = False

    def select(self, index, *, animate=True):
        if self.disposed or index not in self.buttons:
            return
        if self.bar.layout() is not None:
            self.bar.layout().activate()
        previous, self.current_index = self.current_index, index
        target = self._target()
        if target is None or target.isEmpty():
            return
        visible = self.indicator.isVisibleTo(self.bar)
        self.indicator.show()
        self.indicator.lower()
        if (not animate or system_reduce_motion() or self._snap_pending or not visible
                or not self.bar.isVisible()):
            self._jump(target)
        elif self.left.target != target.left() or self.right.target != target.x() + target.width():
            self._base = QRect(target)
            center = (self.left.value + self.right.value) / 2
            self._move_to(target, (target.center().x() > center) - (target.center().x() < center))
        if self.expanded and previous != index:
            self._populate_card()
            self.card_height.tune(260, 20)
            self.card_height.to(self._card_height)

    def snap_to_layout(self):
        if self.disposed:
            return
        if self.bar.layout() is not None:
            self.bar.layout().activate()
        target = self._target()
        if target is None or target.isEmpty():
            return
        if self._snap_pending or self._base != target:
            self._jump(target)
        self._render_card()

    def _pressed(self, _index):
        self._press_current = self.current_index

    def _clicked(self, index):
        # idToggled already traversed the real stage guard. A new selection must
        # neither navigate twice nor expand immediately after its first click.
        if (not self.disposed and not self._suppress_card
                and index == self.current_index == self._press_current):
            self.set_expanded(not self.expanded)

    def _activate(self, index, *, drag_direction=None):
        button = self.buttons.get(index)
        if button is None or not button.isEnabled() or button.isHidden():
            return
        self._suppress_card = True
        try:
            button.click()
        finally:
            self._suppress_card = False
        # A rejected transition restores checked state synchronously. Snap back
        # to that accepted stage, never a decorative false destination.
        target = self._target()
        if target is not None:
            self.nudge.to(0)
            if drag_direction is not None:
                # Release retunes held-pointer springs even when its target
                # already coincides exactly with the selected button.
                self._move_to(target, drag_direction if self.current_index == index else 0)
            elif (self.left.target != target.left()
                    or self.right.target != target.x() + target.width()):
                center = (self.left.value + self.right.value) / 2
                direction = (target.center().x() > center) - (target.center().x() < center)
                self._move_to(target, direction if self.current_index == index else 0)

    @staticmethod
    def _rubber(value, low, high, distance=46):
        if value < low:
            return low - (1 - 1 / ((low - value) * .55 / distance + 1)) * distance
        if value > high:
            return high + (1 - 1 / ((value - high) * .55 / distance + 1)) * distance
        return value

    def _pointer(self, watched, event):
        point = watched.mapTo(self.bar, event.position().toPoint())
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._cancelled_press = False
            self._drag = dict(x0=point.x(), last=point.x(), live=False, direction=0, button=watched)
        elif event.type() == QEvent.Type.MouseMove and self._drag:
            drag = self._drag
            if not drag["live"] and abs(point.x() - drag["x0"]) > 4:
                drag["live"] = True
                if isinstance(drag["button"], QPushButton):
                    drag["button"].setDown(False)
            if drag["live"]:
                delta = point.x() - drag["last"]
                drag["last"] = point.x()
                if abs(delta) > .5:
                    drag["direction"] = (delta > 0) - (delta < 0)
                centers = [b.geometry().center().x() for b in self.buttons.values() if not b.isHidden()]
                center = self._rubber(point.x(), min(centers), max(centers))
                target = QRect(self._base)
                target.moveCenter(QPoint(round(center), target.center().y()))
                self._move_to(target, drag["direction"], drag=True)
                self.nudge.to((center - _clamp(center, min(centers), max(centers))) * .35)
                return True
        elif event.type() == QEvent.Type.MouseButtonRelease:
            drag, self._drag = self._drag, None
            if self._cancelled_press:
                self._cancelled_press = False
                return True
            if drag and drag["live"]:
                centers = [(index, button.geometry().center().x()) for index, button in self.buttons.items()
                           if not button.isHidden() and button.isEnabled()]
                if centers:
                    center = (self.left.target + self.right.target) / 2
                    center += drag["direction"] * .18 * (self._base.width() + 2)
                    self._activate(min(centers, key=lambda item: abs(item[1] - center))[0],
                                   drag_direction=drag["direction"])
                return True
        return False

    def _make_card(self):
        self.card = QFrame(self.bar.window())
        self.card.setObjectName("rubberDockCard")
        if QApplication.platformName() == "cocoa":
            # This card floats over the photograph. Give its Qt controls
            # a native sibling above the canvas for the backdrop to sample.
            self.card.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors)
            self.card.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
            self.card.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.card.setProperty("glassControl", True)
        self.card.setProperty("glassCornerRadius", 24.0)
        self.card.setAccessibleName("当前步骤导航与状态")
        self.card.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        surface, text, muted = ("rgba(38,38,38,245)", "#f4f1ec", "#b4b1ab") if system_dark() else (
            "rgba(250,249,247,248)", "#201f1d", "#6e6b65")
        self.card.setStyleSheet(
            f"QFrame#rubberDockCard{{background:{surface};border:0;border-radius:24px;}}"
            'QFrame#rubberDockCard[nativeMaterial="true"]{background:transparent;}'
            f"QFrame#rubberDockCard QLabel{{background:transparent;color:{text};border:0;}}"
            f"QLabel#dockStatus{{color:{muted};font-size:12px;}}"
            "QFrame#rubberDockCard QPushButton{border:0;border-radius:12px;padding:9px 12px;"
            "background:rgba(127,127,127,28);}"
            'QFrame#rubberDockCard QPushButton[nativeMaterial="true"]{background:transparent;}'
        )
        layout = QVBoxLayout(self.card)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        layout.setContentsMargins(18, 15, 18, 17)
        layout.setSpacing(10)
        header = QHBoxLayout()
        self.card_title = QLabel()
        self.card_title.setStyleSheet("font-weight:600;font-size:15px;")
        header.addWidget(self.card_title, 1)
        close = QPushButton("×")
        close.setAccessibleName("收起步骤导航")
        close.setFixedWidth(34)
        close.clicked.connect(lambda: self.set_expanded(False))
        header.addWidget(close)
        layout.addLayout(header)
        from .workflow_notices import NoticeLabel

        self.card_status = NoticeLabel()
        self.card_status.setObjectName("dockStatus")
        self.card_status.setWordWrap(True)
        self.card_status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.card_status)
        self.card_action_layout = QVBoxLayout()
        self.card_action_layout.setSpacing(7)
        layout.addLayout(self.card_action_layout)
        self.card.hide()

    def _action_sources(self):
        owner = self.bar.window()
        names = {0: ("import_button",), 2: ("frame_inspect",), 3: ("save_button",),
                 4: ("repair_button",), 5: ("export_button",), 6: ("light_table_current", "light_table_roll", "light_table_files")}
        sources = [getattr(owner, name, None) for name in names.get(self.current_index, ())]
        # Proxy existing controls so their source/parameter guards stay intact.
        labels = {0: ("选择 RAW 文件…", "选择素材文件夹…"),
                  2: ("绘制密度测量区",), 3: ("绘制高光参考区",)}.get(self.current_index, ())
        if labels:
            sources.extend(button for button in owner.findChildren(QPushButton)
                           if button.text() in labels and (self.card is None or not self.card.isAncestorOf(button)))
        return [button for button in sources if isinstance(button, QPushButton) and not button.isHidden()]

    def _populate_card(self):
        if self.card is None:
            self._make_card()
        while self.card_action_layout.count():
            widget = self.card_action_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        self._card_actions = []
        self.card_title.setText(self.buttons[self.current_index].text())
        for source in self._action_sources():
            proxy = QPushButton(source.text(), self.card)
            proxy.setAccessibleName(source.accessibleName() or source.text())
            proxy.setToolTip(source.toolTip())
            proxy.clicked.connect(lambda _checked=False, source=source: self._invoke_action(source))
            self.card_action_layout.addWidget(proxy)
            self._card_actions.append((proxy, source))
        self.card.setFixedWidth(max(260, min(340, self.bar.window().width() - 24)))
        self.refresh_card()
        self.card.layout().activate()

    def refresh_card(self):
        if self.card is None or sip.isdeleted(self.card):
            return
        status = getattr(self.bar.window(), "status", None)
        self.card_status.setText(status.text() if isinstance(status, QLabel) else "使用顶部步骤切换工作流程。")
        for proxy, source in self._card_actions:
            if not sip.isdeleted(source):
                proxy.setEnabled(source.isEnabled())
                proxy.setText(source.text())
        layout = self.card.layout()
        width = min(340, self.bar.window().width() - 24)
        height = layout.totalHeightForWidth(width) if layout.hasHeightForWidth() else layout.sizeHint().height()
        height = max(112, height)
        if height != self._card_height:
            self._card_height = height
            if self.expanded:
                self.card_height.to(height)

    def _invoke_action(self, source):
        self.set_expanded(False)
        if not sip.isdeleted(source) and source.isEnabled():
            source.click()

    def set_expanded(self, expanded):
        if self.disposed or self.current_index is None or bool(expanded) == self.expanded:
            return
        self.expanded = bool(expanded)
        if self.expanded:
            self._populate_card()
            if not self._filter_installed:
                self.application.installEventFilter(self)
                self._filter_installed = True
            self.card.setEnabled(True)
            self.card.show()
            self.card.raise_()
            self.card_width.tune(340, 21)
            self.card_height.tune(230, 21)
            if self.card_height.value == 0:
                self.card_width.jump(self._base.width())
            self.status_timer.start()
            width, height = min(340, self.bar.window().width() - 24), self._card_height
        else:
            self.status_timer.stop()
            if self._filter_installed:
                self.application.removeEventFilter(self)
                self._filter_installed = False
            self.card.setEnabled(False)
            self.card_width.tune(420 * .72, 30 * .66)
            self.card_height.tune(420, 30)
            width, height = self._base.width(), 0
        if system_reduce_motion():
            self.card_width.jump(width)
            self.card_height.jump(height)
        else:
            self.card_width.to(width)
            self.card_height.to(height)
        self._render_card()

    def _render_card(self, _value=0):
        if self.disposed or self.card is None or sip.isdeleted(self.card):
            return
        if not self.expanded and self.card_height.value < .5:
            self.card.hide()
            return
        owner = self.bar.window()
        height = max(1, round(self.card_height.value))
        give = _clamp(-self.card_height.velocity / 9000, -.04, .04)
        width = round(_clamp(self.card_width.value * (1 + give), 120, max(120, owner.width() - 24)))
        target = self._target()
        if target is None:
            return
        center = self.bar.mapTo(owner, target.center())
        top = self.bar.mapTo(owner, QPoint(0, self.bar.height())).y() + 8
        left = round(_clamp(center.x() - width / 2, 12, max(12, owner.width() - width - 12)))
        self.card.setMinimumWidth(0)
        self.card.setMaximumWidth(16777215)
        bounds = QRect(left, top, width, min(height, max(1, owner.height() - top - 12)))
        if self.card.geometry() != bounds:
            self.card.setGeometry(bounds)
            binder = getattr(owner, "_workflow_material_binder", None)
            if binder is not None:
                binder.update_floating_card(self.card)

    def _cancel_drag(self):
        if self._drag:
            source = self._drag["button"]
            if isinstance(source, QPushButton):
                source.setDown(False)
            self._drag = None
            self._cancelled_press = True
            target = self._target()
            if target is not None:
                self.nudge.to(0)
                self._move_to(target, 0)

    def eventFilter(self, watched, event):
        if getattr(self, "disposed", True) or sip.isdeleted(self.bar):
            return False
        kind = event.type()
        relevant = watched is self.bar or watched in self.buttons.values()
        if kind == QEvent.Type.MouseButtonPress and self.expanded and isinstance(watched, QWidget):
            inside = (watched is self.bar or self.bar.isAncestorOf(watched)
                      or watched is self.card or self.card.isAncestorOf(watched))
            if not inside:
                self.set_expanded(False)
        if relevant and kind in {QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease}:
            if self._pointer(watched, event):
                return True
        if kind == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Escape and (self.expanded or self._drag):
                self.set_expanded(False)
                self._cancel_drag()
                return True
            if relevant and event.key() in {Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Home, Qt.Key.Key_End}:
                keys = [index for index, button in self.buttons.items() if button.isEnabled() and not button.isHidden()]
                if keys and self.current_index in keys:
                    position = keys.index(self.current_index)
                    if event.key() == Qt.Key.Key_Home:
                        position = 0
                    elif event.key() == Qt.Key.Key_End:
                        position = len(keys) - 1
                    else:
                        position = _clamp(position + (1 if event.key() == Qt.Key.Key_Right else -1), 0, len(keys) - 1)
                    self._activate(keys[position])
                    self.buttons[self.current_index].setFocus()
                    return True
        if watched is self.bar and kind == QEvent.Type.Hide:
            self._cancel_drag()
            self.expanded = False
            self.status_timer.stop()
            for spring in self.springs:
                spring.stop()
            if self.card is not None and not sip.isdeleted(self.card):
                self.card.hide()
            self.application.removeEventFilter(self)
            self._filter_installed = False
        elif relevant and kind in {QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.Show, QEvent.Type.LayoutRequest}:
            if kind in {QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.Show}:
                self._snap_pending = True
            self.layout_timer.start(0)
        return False

    @pyqtSlot()
    def dispose(self):
        if not getattr(self, "disposed", True):
            self.disposed = True
            self.layout_timer.stop()
            self.status_timer.stop()
            for spring in self.springs:
                spring.stop()
            if self.application is not None and not sip.isdeleted(self.application):
                self.application.removeEventFilter(self)
            if not sip.isdeleted(self.bar):
                self.bar.removeEventFilter(self)
            for button in self.buttons.values():
                if not sip.isdeleted(button):
                    button.removeEventFilter(self)
            if self.card is not None and not sip.isdeleted(self.card):
                self.card.hide()
