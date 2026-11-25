"""Photo edit commands and a preview toolbar that fits narrow workspaces."""

from pathlib import Path

from PyQt6.QtCore import QEvent, QSize, Qt, QTimer
from PyQt6.QtGui import QAction, QIcon, QKeySequence
from PyQt6.QtWidgets import QBoxLayout, QFrame, QHBoxLayout, QLayout, QPushButton, QWidget


class PreviewToolbar(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("previewToolbar")
        self.rows = QBoxLayout(QBoxLayout.Direction.LeftToRight, self)
        self.rows.setContentsMargins(12, 6, 12, 6)
        self.rows.setSpacing(9)
        self.rows.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.edit_row = QHBoxLayout()
        self.view_row = QHBoxLayout()
        for row in (self.edit_row, self.view_row):
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
        self.rows.addLayout(self.edit_row, 1)
        self.rows.addLayout(self.view_row)

    def minimumSizeHint(self):
        return QSize(330, self.sizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        direction = (QBoxLayout.Direction.TopToBottom if self.width() < 700
                     else QBoxLayout.Direction.LeftToRight)
        if self.rows.direction() != direction:
            from .workflow_materials import stable_material_layout

            with stable_material_layout(self.window()):
                self.rows.setDirection(direction)
                self.updateGeometry()


class EditHistoryUI:
    def edit_geometry(self):
        from .workflow_ui_model import RESULT_MODES

        recipe = self.project.recipe()
        values = recipe.get("post_controls", {}).get(
            RESULT_MODES[self.current_photo()["engine"]], {}).get("values", {})
        return (recipe.get("crop"), recipe.get("output_roi"), recipe.get("output_orientation"),
                {key: value for key, value in values.items() if key.startswith("geometry.")})

    def build_history_actions(self):
        self.edit_menu = self.menuBar().addMenu("编辑")
        self.undo_action = QAction("撤回", self)
        self.undo_action.setShortcuts(QKeySequence.keyBindings(QKeySequence.StandardKey.Undo))
        self.undo_action.triggered.connect(self.undo_edit)
        self.redo_action = QAction("重做", self)
        self.redo_action.setShortcuts(QKeySequence.keyBindings(QKeySequence.StandardKey.Redo))
        self.redo_action.triggered.connect(self.redo_edit)
        self.edit_menu.addAction(self.undo_action)
        self.edit_menu.addAction(self.redo_action)
        self._history_slider = None
        self._history_slider_timer = QTimer(self)
        self._history_slider_timer.setSingleShot(True)
        self._history_slider_timer.setInterval(250)
        self._history_slider_timer.timeout.connect(self.end_history_gesture)

    def history_toolbar_buttons(self, layout):
        icons = Path(__file__).resolve().parents[1] / "assets/ui"
        for name, action in (("undo", self.undo_action), ("redo", self.redo_action)):
            control = QPushButton("撤回" if name == "undo" else "重做")
            control.setObjectName(name + "Edit")
            control.setAccessibleName(control.text())
            control.setIcon(QIcon(str(icons / (name + ".svg"))))
            control.setIconSize(QSize(18, 18))
            control.clicked.connect(action.trigger)
            setattr(self, name + "_button", control)
            layout.addWidget(control)
        self.crop_button = QPushButton("裁切")
        self.crop_button.setObjectName("cropEdit")
        self.crop_button.setAccessibleName("裁切")
        self.crop_button.setIcon(QIcon(str(icons / "crop.svg")))
        self.crop_button.setIconSize(QSize(18, 18))
        self.crop_button.setToolTip("直接在画布上裁切照片")
        self.crop_button.clicked.connect(lambda: self.open_crop_tools(start_crop=True))
        layout.addWidget(self.crop_button)

    def sync_history_actions(self):
        if not hasattr(self, "undo_action"):
            return
        from .workflow_slider import SliderControl

        for control in self.findChildren(SliderControl):
            if control.window() is self and not getattr(control, "_history_reset_bound", False):
                control._history_reset_bound = True
                control.resetStarted.connect(
                    lambda control=control: self.begin_parameter_reset(control.label.text()))
                control.resetFinished.connect(self.end_parameter_reset)
        ready = bool(self.current_photo() and self.current_photo()["initial_job"]
                     and not self.tasks.busy)
        for name in ("undo", "redo"):
            action = getattr(self, name + "_action")
            enabled = ready and getattr(self.project, "can_" + name)()
            label = getattr(self.project, name + "_label")() if ready else ""
            title = "撤回" if name == "undo" else "重做"
            action.setText(title + (" · " + label if label else ""))
            action.setEnabled(enabled)
            shortcut = action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
            caption = action.text() + ("（" + shortcut + "）" if shortcut else "")
            control = getattr(self, name + "_button", None)
            if control:
                control.setEnabled(enabled)
                control.setToolTip(caption)
        if hasattr(self, "crop_button"):
            self.crop_button.setEnabled(ready)

    def begin_parameter_reset(self, name):
        self.end_history_gesture()
        self.flush_advanced()
        self.project.begin_edit("复位“" + name + "”")

    def end_parameter_reset(self):
        try:
            self.flush_advanced()
        finally:
            self.project.end_edit()
            self.sync_history_actions()

    def history_slider_event(self, watched, event):
        from PyQt6.QtWidgets import QSlider
        from .workflow_slider import SliderControl

        if (self._history_slider is not None and watched is not self._history_slider
                and isinstance(watched, QWidget) and watched.window() is self
                and event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.KeyPress)):
            self.end_history_gesture()
        if not (isinstance(watched, QSlider) and watched.window() is self
                and isinstance(watched.parentWidget(), SliderControl)):
            return
        kind = event.type()
        keys = (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down,
                Qt.Key.Key_PageUp, Qt.Key.Key_PageDown, Qt.Key.Key_Home, Qt.Key.Key_End)
        start = ((kind == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton)
                 or (kind == QEvent.Type.KeyPress and event.key() in keys)
                 or kind == QEvent.Type.Wheel)
        finish = ((kind == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton)
                  or (kind == QEvent.Type.KeyRelease and event.key() in keys and not event.isAutoRepeat())
                  or kind == QEvent.Type.FocusOut)
        if start and not self.loading and not self.tasks.busy and self.current_photo():
            if self._history_slider is not watched:
                self.end_history_gesture()
                self.project.begin_edit("调整“" + watched.parentWidget().label.text() + "”")
                self._history_slider = watched
            if kind == QEvent.Type.Wheel:
                self._history_slider_timer.start()
        elif finish and self._history_slider is watched:
            self.end_history_gesture()

    def end_history_gesture(self):
        if not getattr(self, "_history_slider", None):
            return
        self._history_slider = None
        self._history_slider_timer.stop()
        try:
            self.flush_advanced()
        finally:
            self.project.end_edit()
            self.sync_history_actions()

    def undo_edit(self):
        self.restore_edit(False)

    def redo_edit(self):
        self.restore_edit(True)

    def restore_edit(self, redo):
        if self.tasks.busy or not self.current_photo():
            return
        self.end_history_gesture()
        if not self.flush_advanced():
            self.status_message("请先修正尚未应用的参数，再撤回或重做。")
            return
        label = self.project.redo_label() if redo else self.project.undo_label()
        geometry = self.edit_geometry()
        if not (self.project.redo() if redo else self.project.undo()):
            self.sync_history_actions()
            return
        for panel in self.advanced_panels.values():
            panel.cancel_pending()
        in_crop = self.geometry_role == "crop"
        if not in_crop:
            self.finish_selection()
        self.sync_controls()
        self.refresh_photos()
        if in_crop:
            self.refresh_selection_preview()
        elif geometry != self.edit_geometry():
            self.canvas.fit()
        self.schedule_preview()
        self.status_message(("已重做：" if redo else "已撤回：") + label)

    def open_crop_tools(self, *, start_crop=False):
        if self.tasks.busy or not self.current_photo():
            return
        self.navigate_stage(4, start_crop=start_crop)
        if self.step == 4:
            panel = self.crop_panels[4]
            self.pages.widget(4).verticalScrollBar().setValue(0)
            self.pages.widget(4).ensureWidgetVisible(panel)

