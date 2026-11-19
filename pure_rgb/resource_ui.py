"""Local FlexColor profile selection with inspection and immutable hash binding."""

from pathlib import Path
from PyQt6.QtCore import pyqtSignal
from .workflow_glass import GlassMessageBox  # noqa: E402
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QLabel,
    QPushButton,
    QFileDialog,
)

from .engine_resources import (
    checked_resource,
    inspect_flex_profile,
    require_flex_profile,
)
from .professional_controls import ENGINE_SPECS


class FlexResourceSelector(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.resource, self.report = None, None
        column = QVBoxLayout(self)
        column.setContentsMargins(2, 2, 2, 2)
        row = QHBoxLayout()
        self.name = QLineEdit()
        self.name.setReadOnly(True)
        self.name.setPlaceholderText("合成中性配置（当前基准）")
        choose = QPushButton("导入胶片配置…")
        choose.clicked.connect(self.choose)
        clear = QPushButton("移除")
        clear.clicked.connect(self.clear)
        row.addWidget(choose)
        row.addWidget(clear)
        column.addWidget(QLabel("FlexColor 胶片配置"))
        column.addWidget(self.name)
        column.addLayout(row)
        self.status = QLabel("未载入原厂配置；默认合成中性曲线。")
        self.status.setWordWrap(True)
        column.addWidget(self.status)

    def choose(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入 FlexColor 胶片配置",
            "",
            "Plist/XML (*.plist *.xml *.xml_*);;所有文件 (*)",
        )
        if path:
            try:
                self.select_path(path)
            except Exception as error:
                GlassMessageBox.warning(self, "胶片配置未载入", str(error))

    def select_path(self, path):
        _, report = inspect_flex_profile(path)
        require_flex_profile(report["resource"])
        for spec in ENGINE_SPECS["flex-endpoint"]:
            if (
                spec.key in report["defaults"]
                and not spec.low <= report["defaults"][spec.key] <= spec.high
            ):
                raise ValueError("配置参数超出当前工作台控件范围：" + spec.title)
        self.set_resource(report["resource"])
        self.changed.emit()

    def set_resource(self, resource):
        self.resource = checked_resource(resource) if resource else None
        self.report = None
        self.name.setText(Path(resource["path"]).name if resource else "")
        self.name.setToolTip(resource["path"] if resource else "")
        if resource:
            try:
                _, self.report = require_flex_profile(resource)
                self.status.setText(
                    f"{self.report['curve_count']} 条曲线已检查 · Gamma {self.report['defaults']['film_gamma']:g}\n执行公开重建的曲线／参数层；未执行原厂设备和 CMS。"
                )
            except Exception as error:
                self.status.setText(str(error))
        else:
            self.status.setText("未载入原厂配置；默认合成中性曲线。")

    def clear(self):
        self.set_resource(None)
        self.changed.emit()
