"""Embedded, live advanced parameters for the staged desktop sidebar."""

from copy import deepcopy

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from .workflow_disclosure import DisclosureButton
from .workflow_combo import WorkflowComboBox as QComboBox

from .workflow_slider import SliderControl
from .workflow_parameter_reset import add_parameter_reset
from .workflow_advanced import advanced_groups, advanced_values, apply_advanced


class FieldPage(QWidget):
    """A single vertical group; scrolling belongs to the surrounding sidebar."""

    changed = pyqtSignal()
    reset_requested = pyqtSignal()

    def __init__(self, specs, note, parent=None):
        super().__init__(parent)
        self.specs = specs
        self.controls = {}
        self.parameter_resets = {}
        self.loading = True
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 8, 0, 0)
        column.setSpacing(10)
        hint = QLabel(note)
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        column.addWidget(hint)
        for spec in specs:
            if type(spec.default) is bool:
                control = QCheckBox(spec.title)
                control.toggled.connect(self.notify_changed)
            elif spec.choices:
                label = QLabel(spec.title)
                label.setWordWrap(True)
                column.addWidget(label)
                control = QComboBox()
                for value, title in spec.choices:
                    control.addItem(title, value)
                control.currentIndexChanged.connect(self.notify_changed)
            else:
                control = SliderControl(
                    spec.title,
                    spec.low,
                    spec.high,
                    spec.step,
                    spec.default,
                )
                control.valueChanged.connect(self.notify_changed)
            control.setAccessibleName(spec.title)
            control.setObjectName(spec.key)
            self.controls[spec.key] = control
            if type(spec.default) is bool or spec.choices:
                self.parameter_resets[spec.key] = add_parameter_reset(
                    column, control, spec.default, spec.title)
            else:
                column.addWidget(control)
        self.reset_button = QPushButton("复位本组")
        self.reset_button.clicked.connect(self.reset_values)
        column.addWidget(self.reset_button)
        self.set_values({})
        self.loading = False

    def notify_changed(self, *args):
        if not self.loading:
            self.changed.emit()

    def set_values(self, values):
        previous = self.loading
        self.loading = True
        try:
            for spec in self.specs:
                control = self.controls[spec.key]
                value = values.get(spec.key, spec.default)
                if type(spec.default) is bool:
                    control.setChecked(value)
                elif spec.choices:
                    control.setCurrentIndex(control.findData(value))
                else:
                    control.setValue(value)
        finally:
            self.loading = previous

    def values(self):
        values = {}
        for spec in self.specs:
            control = self.controls[spec.key]
            value = (
                control.isChecked()
                if type(spec.default) is bool
                else control.currentData()
                if spec.choices
                else control.value()
            )
            if value != spec.default:
                values[spec.key] = value
        return values

    def reset_values(self):
        self.set_values({})
        self.reset_requested.emit()
        self.changed.emit()


class AdvancedPanel(QWidget):
    """Inline independent accordions sharing the main image and sidebar scroll."""

    recipe_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original = None
        self.groups = ()
        self.pages = {}
        self.headers = {}
        self.initial_values = {}
        self.reset_groups = set()
        self.loading = False
        self.result_recipe = None
        self.engine, self.bw, self.stage = "native", False, 3
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)
        title = QLabel("高级参数")
        title.setObjectName("section")
        column.addWidget(title)
        actions = QHBoxLayout()
        self.expand_all_button = QPushButton("全部展开")
        self.expand_all_button.clicked.connect(self.expand_all)
        self.collapse_all_button = QPushButton("全部收起")
        self.collapse_all_button.clicked.connect(self.collapse_all)
        actions.addWidget(self.expand_all_button)
        actions.addWidget(self.collapse_all_button)
        column.addLayout(actions)
        self.group_host = QWidget()
        self.group_layout = QVBoxLayout(self.group_host)
        self.group_layout.setContentsMargins(0, 0, 0, 0)
        self.group_layout.setSpacing(8)
        column.addWidget(self.group_host)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setObjectName("muted")
        column.addWidget(self.error)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.apply_changes)

    def cancel_pending(self):
        self.timer.stop()

    def expand_all(self):
        for header in self.headers.values():
            header.setChecked(True)

    def collapse_all(self):
        for header in self.headers.values():
            header.setChecked(False)

    def _set_expanded(self, key, expanded):
        self.pages[key].setVisible(expanded)
        self.group_host.updateGeometry()
        self.updateGeometry()

    def configure(self, recipe, engine, bw, stage):
        self.cancel_pending()
        expanded = {key for key, header in self.headers.items() if header.isChecked()}
        self.loading = True
        try:
            self.original = deepcopy(recipe)
            self.engine, self.bw, self.stage = engine, bw, stage
            self.result_recipe = None
            self.groups = advanced_groups(recipe, engine, bw, stage)
            self.reset_groups.clear()
            self.initial_values = {}
            self.pages = {}
            self.headers = {}
            while self.group_layout.count():
                item = self.group_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()
            for group in self.groups:
                header = DisclosureButton(group.title)
                header.setAccessibleName(group.title)
                header.setObjectName("advancedGroupHeader")
                page = FieldPage(group.specs, group.note)
                page.set_values(advanced_values(recipe, group.key))
                self.pages[group.key] = page
                self.headers[group.key] = header
                self.initial_values[group.key] = page.values()
                page.changed.connect(self.schedule_changes)
                page.reset_requested.connect(
                    lambda key=group.key: self.reset_groups.add(key)
                )
                header.toggled.connect(
                    lambda checked, key=group.key: self._set_expanded(key, checked)
                )
                self.group_layout.addWidget(header)
                self.group_layout.addWidget(page)
                page.hide()
                if group.key in expanded:
                    header.setChecked(True)
            self.expand_all_button.setEnabled(bool(self.groups))
            self.collapse_all_button.setEnabled(bool(self.groups))
            self.error.clear()
        finally:
            self.loading = False
        self.group_host.updateGeometry()
        self.updateGeometry()

    def schedule_changes(self):
        if not self.loading:
            self.timer.start()

    def rebase(self, recipe):
        """Refresh a committed recipe while retaining accordion state and widgets."""
        self.cancel_pending()
        groups = advanced_groups(recipe, self.engine, self.bw, self.stage)
        if groups != self.groups:
            self.configure(recipe, self.engine, self.bw, self.stage)
            return
        self.loading = True
        try:
            self.original = deepcopy(recipe)
            self.result_recipe = None
            self.reset_groups.clear()
            for group in self.groups:
                page = self.pages[group.key]
                page.set_values(advanced_values(recipe, group.key))
                self.initial_values[group.key] = page.values()
            self.error.clear()
        finally:
            self.loading = False

    def apply_changes(self):
        """Publish a whole valid recipe; retain invalid edits for correction."""
        self.cancel_pending()
        if self.loading or self.original is None:
            return True
        try:
            changes = {}
            for group in self.groups:
                key = group.key
                current = self.pages[key].values()
                if key in self.reset_groups:
                    changes[key] = current
                    continue
                initial = self.initial_values[key]
                # Keep original precision for fields the user did not touch;
                # the displayed readout can have fewer decimals than a recipe.
                values = advanced_values(self.original, key)
                changed = False
                for spec in group.specs:
                    if current.get(spec.key, spec.default) == initial.get(
                        spec.key, spec.default
                    ):
                        continue
                    changed = True
                    values.pop(spec.key, None)
                    if spec.key in current:
                        values[spec.key] = current[spec.key]
                if changed:
                    changes[key] = values
            candidate = apply_advanced(self.original, self.engine, self.bw, changes)
        except (ValueError, TypeError, KeyError) as error:
            self.error.setText("参数尚未应用：" + str(error))
            return False
        self.error.clear()
        self.result_recipe = candidate
        if candidate == self.original:
            self.reset_groups.clear()
            return True
        if advanced_groups(candidate, self.engine, self.bw, self.stage) != self.groups:
            self.configure(candidate, self.engine, self.bw, self.stage)
            self.result_recipe = candidate
        else:
            self.original = deepcopy(candidate)
            self.initial_values = {
                key: page.values() for key, page in self.pages.items()
            }
            self.reset_groups.clear()
        self.recipe_changed.emit(candidate)
        return True
