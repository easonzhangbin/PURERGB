"""Small independent resets for nonnumeric image adjustment controls."""

from PyQt6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QPushButton


def add_parameter_reset(layout, control, default, name=None):
    """Wrap a checkbox/choice without replacing its normal change signals."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)
    row.addWidget(control, 1)
    reset = QPushButton("复位")
    reset.setObjectName("parameterReset")
    reset.setAccessibleName("复位“" + (name or control.accessibleName() or control.text()) + "”")
    reset.setFixedWidth(48)
    row.addWidget(reset)
    layout.addLayout(row)

    def desired():
        return default() if callable(default) else default

    def update(*args):
        value = desired()
        if isinstance(control, QCheckBox):
            reset.setEnabled(control.isChecked() != value)
            reset.setToolTip("复位为" + ("开启" if value else "关闭"))
        else:
            index = control.findData(value)
            reset.setEnabled(index >= 0 and control.currentData() != value)
            reset.setToolTip("复位为“" + (control.itemText(index) if index >= 0 else str(value)) + "”")

    def restore():
        value = desired()
        if isinstance(control, QCheckBox):
            control.setChecked(value)
        else:
            index = control.findData(value)
            if index >= 0:
                control.setCurrentIndex(index)

    if isinstance(control, QCheckBox):
        control.toggled.connect(update)
    elif isinstance(control, QComboBox):
        # The sidebar viewport is narrow; a combo sized to its widest item
        # pushes the fixed-width reset off the edge. Let the combo shrink and
        # elide its text instead of clipping the reset beside it.
        control.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        control.setMinimumContentsLength(10)
        control.currentIndexChanged.connect(update)
    else:
        raise TypeError("Parameter reset expects a checkbox or choice")
    reset.clicked.connect(restore)
    update()
    return reset
