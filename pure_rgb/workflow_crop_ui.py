"""Crop, lossless output rotation and positive-domain straightening controls."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .motion_sliders import ScrubberControl
from .workflow_attention import AttentionButton
from .workflow_parameter_reset import add_parameter_reset
from .workflow_ui_model import RESULT_MODES, with_straighten
from .positive_post import POST_SPECS
from .professional_controls import checked_controls
from .workspace import checked_recipe


def straighten_recipe(recipe, engine, angle, crop_to_valid):
    """Change only the active output's fine angle and blank-corner policy.

    Density masks, manually selected crop coordinates and every other engine's
    finishing settings remain in their original source-coordinate domain.
    """
    changed = checked_controls(dict(version=1, values={
        "geometry.fine_rotation": angle, "geometry.crop_to_valid": crop_to_valid,
    }), POST_SPECS)["values"]
    return checked_recipe(with_straighten(recipe, engine, changed))


class CropGeometryPanel(QWidget):
    """One shared layout for the measurement and retouching workspaces."""

    manual_crop_requested = pyqtSignal()
    clear_crop_requested = pyqtSignal()
    auto_crop_requested = pyqtSignal(str)
    rotation_requested = pyqtSignal(str)
    straighten_changed = pyqtSignal(float, bool)
    straighten_draw_requested = pyqtSignal()
    straighten_draw_label = "画线拉直 · 沿参照线拖动"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.loading = False
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)
        self.manual = QPushButton("绘制独立输出裁切")
        self.manual.clicked.connect(self.manual_crop_requested)
        column.addWidget(self.manual)
        self.auto_film = QPushButton("自动裁切 · 保留片基文字")
        self.auto_film.clicked.connect(lambda: self.auto_crop_requested.emit("film_border"))
        column.addWidget(self.auto_film)
        self.auto_image = QPushButton("自动裁切 · 仅有效图像")
        self.auto_image.clicked.connect(lambda: self.auto_crop_requested.emit("image"))
        column.addWidget(self.auto_image)
        self.clear = QPushButton("取消输出裁切")
        self.clear.clicked.connect(self.clear_crop_requested)
        column.addWidget(self.clear)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setObjectName("muted")
        column.addWidget(self.summary)
        rotation = QHBoxLayout()
        rotation.setSpacing(10)
        self.rotate_left = QPushButton("向左旋转 90°")
        self.rotate_right = QPushButton("向右旋转 90°")
        for control, action in ((self.rotate_left, "left"), (self.rotate_right, "right")):
            control.clicked.connect(lambda checked=False, action=action: self.rotation_requested.emit(action))
            rotation.addWidget(control)
        column.addLayout(rotation)
        self.straighten = ScrubberControl("拉直角度（°）", -45, 45, 0.05, 0)
        self.straighten.setAccessibleName("裁切拉直角度")
        self.straighten.valueChanged.connect(self._changed)
        column.addWidget(self.straighten)
        self.draw_straighten = AttentionButton(self.straighten_draw_label)
        self.draw_straighten.setObjectName("drawStraighten")
        self.draw_straighten.setCheckable(True)
        self.draw_straighten.setToolTip(
            "在画面上沿需要拉平的线拖一条直线，松开后自动填入拉直角度。"
            "画竖直参照线时自动换算成对应角度；激活时按钮高亮，"
            "再次点击可退出画线模式。"
        )
        self.draw_straighten.clicked.connect(self.straighten_draw_requested)
        column.addWidget(self.draw_straighten)
        self.trim_corners = QCheckBox("裁去拉直后的空白角")
        self.trim_corners.toggled.connect(self._changed)
        self.trim_reset = add_parameter_reset(column, self.trim_corners, False, "裁去空白角")
        self.reset = QPushButton("复位拉直")
        self.reset.clicked.connect(self.reset_straighten)
        column.addWidget(self.reset)
        self.note = QLabel("裁切与密度测量分别保存。同一张照片的裁切、拉直和旋转在各引擎间共用。")
        self.note.setWordWrap(True)
        self.note.setObjectName("muted")
        column.addWidget(self.note)

    def _changed(self, *args):
        if not self.loading:
            self.straighten_changed.emit(self.straighten.value(), self.trim_corners.isChecked())

    def reset_straighten(self):
        self.loading = True
        try:
            self.straighten.setValue(0)
            self.trim_corners.setChecked(False)
        finally:
            self.loading = False
        self._changed()

    def set_recipe(self, recipe, engine):
        self.loading = True
        try:
            values = recipe.get("post_controls", {}).get(RESULT_MODES[engine], {}).get("values", {})
            self.straighten.setValue(values.get("geometry.fine_rotation", 0))
            self.trim_corners.setChecked(values.get("geometry.crop_to_valid", False))
            roi = recipe.get("output_roi")
            self.summary.setText("输出裁切：" + (" × ".join(map(str, (roi[2] - roi[0], roi[3] - roi[1])))
                                     + " 源像素" if roi else "跟随测量画幅" if recipe.get("crop") else "完整源画幅"))
        finally:
            self.loading = False


def install_crop_panel(window, layout, stage):
    panel = CropGeometryPanel()
    panels = getattr(window, "crop_panels", None)
    if panels is None:
        window.crop_panels = panels = {}
    panels[stage] = panel
    panel.manual_crop_requested.connect(lambda: window.begin_selection("crop"))
    panel.clear_crop_requested.connect(window.clear_crop)
    panel.rotation_requested.connect(window.change_output_orientation)
    panel.straighten_changed.connect(window.crop_post_changed)
    panel.straighten_draw_requested.connect(
        lambda panel=panel: window.begin_straighten_draw(panel)
    )
    panel.auto_crop_requested.connect(window.apply_auto_crop)
    layout.addWidget(panel)
    return panel
