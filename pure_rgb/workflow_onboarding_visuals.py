"""Offline photo and diagram panels for the first-use introduction."""

from functools import lru_cache
import math
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from .workflow_onboarding_theme import _luminance, theme_tokens


ASSETS = Path(__file__).resolve().parents[1] / "assets/ui/onboarding"
PAGE_KEYS = ("welcome", "science", "prepare", "workflow")
DESCRIPTIONS = {
    "welcome": ("实拍三色与正像", "同一格底片的红光、绿光、蓝光 RAW 灰度预览，以及 PURE RGB 转换后的正像。"),
    "science": ("染料吸收与窄波段", "概念示意，非实测曲线。黄色、品红、青色染料对不同波长的吸收有重叠。"),
    "prepare": ("橙色片基框和蓝色画幅框", "橙框在两帧之间并避开齿孔，比较三路片基信号。蓝框圈定保留画幅，检查原始 RAW 白位。"),
    "workflow": ("有效画幅中心测量", "白框是有效画幅，绿色内框是居中且保持长宽比的约百分之八十的密度测量面积。"),
}


def _readable(value, background):
    """Retain a channel hue while keeping annotation text legible."""
    color = QColor(value)
    target = _luminance(background)
    hue, saturation, lightness, alpha = color.getHslF()
    for _ in range(40):
        low, high = sorted((_luminance(color.name()), target))
        if (high + .05) / (low + .05) >= 4.5:
            break
        lightness = min(1, max(0, lightness + (.025 if target < .4 else -.025)))
        color.setHslF(hue, saturation, lightness, alpha)
    return color.name()


def visual_theme(dark, colors=None):
    if type(dark) is not bool:
        raise ValueError("图示外观需要深色或浅色布尔值")
    colors = colors or theme_tokens(dark)
    card = colors["card"]
    return dict(
        background=colors["surface"], card=card, divider=colors["edge"],
        text=colors["text"], secondary=colors["secondary"],
        channels=tuple(_readable(colors[key], card) for key in ("red", "green", "blue")),
        curves=("#ebca6b", "#ed9bd3", "#82d6e1") if dark else ("#876300", "#943277", "#006e83"),
        film_base="#cba57e", orange="#ffad42", blue="#76bcff", mint="#72e1af",
    )


@lru_cache(maxsize=4)
def _image(name):
    return QImage(str(ASSETS / f"{name}.jpg"))


class _Illustration(QWidget):
    WIDTH = 900
    HEIGHT = 326

    def __init__(self, parent=None):
        super().__init__(parent)
        self.page_key = "welcome"
        self.colors = visual_theme(True)
        self.setMinimumSize(560, 222)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def sizeHint(self):
        return QSize(870, 316)

    def set_theme(self, dark, *, colors=None):
        self.colors = visual_theme(dark, colors)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(self.colors["background"]))
        scale = min(self.width() / self.WIDTH, self.height() / self.HEIGHT)
        painter.translate((self.width() - self.WIDTH * scale) / 2, (self.height() - self.HEIGHT * scale) / 2)
        painter.scale(scale, scale)
        self._panel(painter, (5, 5, 890, 316))
        getattr(self, "_" + self.page_key)(painter)

    def _text(self, p, box, text, *, color=None, size=21, bold=False, center=False):
        font = p.font()
        font.setPixelSize(size)
        font.setWeight(QFont.Weight.DemiBold if bold else QFont.Weight.Normal)
        p.setFont(font)
        p.setPen(QColor(color or self.colors["text"]))
        alignment = Qt.AlignmentFlag.AlignVCenter | (
            Qt.AlignmentFlag.AlignHCenter if center else Qt.AlignmentFlag.AlignLeft)
        p.drawText(QRectF(*box), alignment, text)

    def _line(self, p, x0, y0, x1, y1, color, width=1):
        p.setPen(QPen(QColor(color), width))
        p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

    def _panel(self, p, box, *, fill=None, border=None, radius=18):
        p.setPen(QPen(QColor(border or self.colors["divider"]), 1))
        p.setBrush(QColor(fill or self.colors["card"]))
        p.drawRoundedRect(QRectF(*box), radius, radius)

    def _photo(self, p, box, name, *, radius=9):
        image = _image(name)
        rect = QRectF(*box)
        if image.isNull():
            self._panel(p, box)
            self._text(p, box, "样片暂不可用", size=14, center=True)
            return
        scale = max(rect.width() / image.width(), rect.height() / image.height())
        w, h = rect.width() / scale, rect.height() / scale
        source = QRectF((image.width() - w) / 2, (image.height() - h) / 2, w, h)
        clip = QPainterPath()
        clip.addRoundedRect(rect, radius, radius)
        p.save()
        p.setClipPath(clip)
        p.drawImage(rect, image, source)
        p.restore()

    def _outline(self, p, box, color, *, width=2, dashed=False):
        style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
        p.setPen(QPen(QColor(color), width, style))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(*box))

    def _welcome(self, p):
        muted = self.colors["secondary"]
        for index, (name, cn, letter, color) in enumerate(
            zip(("red", "green", "blue"), ("红光", "绿光", "蓝光"), "RGB", self.colors["channels"])
        ):
            y = 25 + index * 94
            self._photo(p, (28, y, 112, 86), name)
            self._text(p, (164, y + 5, 184, 39), cn, color=color, size=25, bold=True)
            self._text(p, (164, y + 43, 184, 34), letter + " 路 RAW", color=muted, size=18)
        self._line(p, 319, 163, 387, 163, muted, 1.5)
        self._line(p, 378, 155, 387, 163, muted, 1.5)
        self._line(p, 378, 171, 387, 163, muted, 1.5)
        self._photo(p, (420, 25, 449, 232), "positive")
        self._text(p, (420, 268, 220, 35), "合成并转换后的正像", size=20, bold=True)
        self._text(p, (687, 268, 180, 35), "同一格实拍底片", color=muted, size=16)

    def _science(self, p):
        muted = self.colors["secondary"]
        self._text(p, (31, 17, 280, 37), "相对吸收", size=20, bold=True)
        self._text(p, (678, 17, 195, 37), "概念示意", color=muted, size=16)
        x, y, w, h = 69, 63, 758, 154
        for line in range(4):
            level = y + h * line / 3
            self._line(p, x, level, x + w, level, self.colors["divider"])
        bands = (.22, .52, .79)
        for center, color in zip(bands, reversed(self.colors["channels"])):
            tint = QColor(color)
            tint.setAlpha(38)
            p.fillRect(QRectF(x + (center - .027) * w, y, .054 * w, h), tint)
        for center, width, shoulder, color in zip(bands, (.15, .13, .16), (.58, .81, .48), self.colors["curves"]):
            curve = QPainterPath()
            for i in range(145):
                u = i / 144
                strength = .08 + .78 * math.exp(-.5 * ((u - center) / width) ** 2)
                strength += .13 * math.exp(-.5 * ((u - shoulder) / .17) ** 2)
                point = QPointF(x + w * u, y + h * (1 - strength))
                curve.moveTo(point) if i == 0 else curve.lineTo(point)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(color), 3))
            p.drawPath(curve)
        self._text(p, (69, 222, 170, 31), "短波", color=muted, size=17)
        self._text(p, (364, 222, 173, 31), "波长方向", color=muted, size=17, center=True)
        self._text(p, (776, 222, 51, 31), "长波", color=muted, size=17)
        for x0, name, color in zip((132, 383, 644), ("黄染料", "品红染料", "青染料"), self.colors["curves"]):
            self._line(p, x0, 283, x0 + 30, 283, color, 3)
            self._text(p, (x0 + 41, 263, 155, 40), name, size=19)

    def _prepare(self, p):
        muted = self.colors["secondary"]
        self._text(p, (30, 18, 437, 36), "片间空隙取样示意", size=22, bold=True)
        self._panel(p, (30, 66, 458, 175), fill=self.colors["film_base"], border="#987358", radius=9)
        for index in range(12):
            for y in (75, 221):
                p.setBrush(QColor("#302b28"))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(QRectF(44 + index * 36, y, 18, 11), 2, 2)
        self._photo(p, (41, 94, 192, 117), "red", radius=2)
        self._photo(p, (256, 94, 221, 117), "green", radius=2)
        self._outline(p, (239, 118, 11, 69), self.colors["orange"], width=2.6)
        self._outline(p, (265, 102, 203, 100), self.colors["blue"], dashed=True)
        for x, hue, name in ((32, "orange", "片基量区"), (251, "blue", "保留画幅")):
            color = self.colors[hue]
            self._line(p, x, 277, x + 25, 277, color, 3)
            self._text(p, (x + 39, 258, 159, 38), name, size=20)
        self._line(p, 515, 31, 515, 291, self.colors["divider"])
        self._text(p, (549, 27, 288, 34), "灯光配平", size=22, bold=True)
        self._text(p, (549, 61, 314, 29), "比较三路片基信号", color=muted, size=17)
        self._line(p, 724, 99, 724, 169, self.colors["divider"])
        for i, (letter, color, end) in enumerate(zip("RGB", self.colors["channels"], (703, 724, 741))):
            y = 109 + 25 * i
            self._text(p, (549, y - 12, 37, 24), letter, color=color, size=17, bold=True)
            self._line(p, 589, y, 855, y, self.colors["divider"])
            self._line(p, 724, y, end, y, color, 4)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(color))
            p.drawEllipse(QPointF(end, y), 4, 4)
        self._text(p, (549, 183, 288, 34), "RAW 曝光余量", size=22, bold=True)
        self._line(p, 549, 232, 855, 232, self.colors["divider"], 3)
        self._line(p, 549, 232, 772, 232, self.colors["blue"], 4)
        self._line(p, 855, 222, 855, 242, muted, 2)
        self._text(p, (549, 250, 317, 36), "100% 是原始 RAW 白电平上限", color=muted, size=17)

    def _workflow(self, p):
        muted = self.colors["secondary"]
        self._photo(p, (36, 25, 348, 276), "positive")
        x, y, w, h = 51, 48, 318, 230
        self._outline(p, (x, y, w, h), "#ffffff", dashed=True)
        inset = (1 - math.sqrt(.8)) / 2
        self._outline(p, (x + w * inset, y + h * inset, w * (1 - 2 * inset), h * (1 - 2 * inset)),
                      self.colors["mint"], width=3)
        self._text(p, (439, 29, 412, 51), "中心约 80%", size=40, bold=True)
        self._text(p, (439, 83, 412, 37), "按有效画幅面积计算", color=muted, size=21)
        for y, color, label in ((147, muted, "有效画幅"), (186, self.colors["mint"], "默认密度测量区")):
            self._line(p, 439, y, 467, y, color, 3)
            self._text(p, (484, y - 19, 372, 38), label, size=19)
        self._line(p, 439, 219, 856, 219, self.colors["divider"])
        self._text(p, (439, 230, 417, 28), "导入 → 测量 → 转换调色", size=19)
        self._text(p, (439, 266, 417, 27), "修整 → 审阅 → 导出", color=muted, size=19)


class OnboardingVisual(QWidget):
    """A scalable, accessible illustration for an introduction page."""

    def __init__(self, parent=None, page_key="welcome"):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = _Illustration(self)
        layout.addWidget(self.canvas)
        self.set_page(page_key)

    @property
    def page_key(self):
        return self.canvas.page_key

    def set_theme(self, dark, *, colors=None):
        self.canvas.set_theme(dark, colors=colors)
        self.setProperty("onboardingDark", dark)

    def set_page(self, key):
        if key not in PAGE_KEYS:
            raise ValueError("未知首次介绍图示")
        self.canvas.page_key = key
        name, description = DESCRIPTIONS[key]
        self.setAccessibleName("首次使用图示：" + name)
        self.setAccessibleDescription(description)
        self.canvas.update()


def create_page_visual(page_key, parent=None):
    return OnboardingVisual(parent, page_key)
