"""Offline vector teaching illustrations; sample curves/data are not measurements."""

import math

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF, QRadialGradient
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget


PAGE_KEYS = ("welcome", "science", "prepare", "workflow", "sources")
PAGE_TITLES = dict(welcome="三色采集流程", science="染料吸收与窄波段示意", prepare="片基和平场参考",
                   workflow="线性信号与相对密度", sources="参考资料的不同用途")
COLORS = ("#ff453a", "#32d74b", "#0a84ff")


def _luminance(value):
    color = QColor(value)
    linear = [part / 12.92 if part <= .04045 else ((part + .055) / 1.055) ** 2.4
              for part in (color.redF(), color.greenF(), color.blueF())]
    return sum(part * weight for part, weight in zip(linear, (.2126, .7152, .0722)))


def _readable_channel(value, background):
    """Keep the system hue while making small chart annotations readable."""
    color = QColor(value)
    background_luma = _luminance(background)
    hue, saturation, lightness, alpha = color.getHslF()
    for _ in range(40):
        low, high = sorted((_luminance(color), background_luma))
        if (high + .05) / (low + .05) >= 4.5:
            break
        lightness = max(0., min(1., lightness + (.025 if background_luma < .4 else -.025)))
        color.setHslF(hue, saturation, lightness, alpha)
    return color.name()


def visual_theme(dark, colors=None):
    """Semantic surfaces and accessible annotation colours for both appearances."""
    if type(dark) is not bool:
        raise ValueError("图示外观需要深色或浅色布尔值")
    result = (dict(background="#1c1c1e", card="#2c2c2e", divider="#48484a",
                 text="#f5f5f7", secondary="#a1a1a6", accent="#ffd089",
                 channels=COLORS, curves=("#e8c867", "#e292cb", "#78cad5"),
                 lens="#121214", lens_edge="#98989d") if dark else
            dict(background="#f5f5f7", card="#ffffff", divider="#d2d2d7",
                 text="#1d1d1f", secondary="#6e6e73", accent="#8a5700",
                 channels=("#d70015", "#248a3d", "#0066cc"),
                 curves=("#967000", "#a72e79", "#007785"),
                 lens="#dce3eb", lens_edge="#818995"))
    if colors is None:
        try:
            from .workflow_onboarding_theme import theme_tokens

            colors = theme_tokens(dark)
        except ImportError:
            colors = None
    if colors is not None:
        for target, source in (("background", "surface"), ("card", "card"), ("divider", "edge"),
                               ("text", "text"), ("secondary", "secondary")):
            if source in colors:
                result[target] = colors[source]
        result["channels"] = tuple(colors.get(key, value) for key, value in
                                    zip(("red", "green", "blue"), result["channels"]))
    result["channels"] = tuple(_readable_channel(color, result["card"]) for color in result["channels"])
    return result


class _Illustration(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.page_key = "welcome"
        self.dark = True
        self.colors = visual_theme(True)
        self.setMinimumSize(560, 250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self):
        return QSize(840, 310)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(self.colors["background"]))
        scale = min(self.width() / 960, self.height() / 340)
        painter.translate((self.width() - 960 * scale) / 2, (self.height() - 340 * scale) / 2)
        painter.scale(scale, scale)
        getattr(self, "_" + self.page_key)(painter)

    def set_theme(self, dark, *, colors=None):
        self.colors = visual_theme(dark, colors)
        self.dark = dark
        self.update()

    @property
    def channel_colors(self):
        return self.colors["channels"]

    def _color(self, value, default):
        if isinstance(value, QColor):
            return value
        secondary = {"#d1c8bb", "#bfc6d2", "#b7beca", "#c4c9d2", "#e1c6a5",
                     "#d3dbe6", "#bdc5d1", "#c6cbd5", "#c4cad3", "#bfc6d1"}
        cards = {"#2c2f35", "#303239", "#32363b", "#4f5662", "#303740", "#3b3430"}
        accents = {"#ffca79", "#ffbd65"}
        role = ("secondary" if value in secondary else "card" if value in cards
                else "accent" if value in accents else "divider" if value in {"#555961", "#515660", "#b5bdc9"}
                else "text" if value == "#f2ede4" else None)
        return QColor(self.colors[role or default] if value is None or role else value)

    def _text(self, painter, box, text, *, color=None, size=20, bold=False):
        font = painter.font()
        font.setPixelSize(size)
        font.setWeight(QFont.Weight.DemiBold if bold else QFont.Weight.Normal)
        painter.setFont(font)
        painter.setPen(self._color(color, "text"))
        painter.drawText(QRectF(*box), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)

    def _panel(self, painter, box, color=None, edge=None, radius=12):
        painter.setPen(QPen(self._color(edge, "divider"), .7) if edge is not False else Qt.PenStyle.NoPen)
        painter.setBrush(self._color(color, "card"))
        painter.drawRoundedRect(QRectF(*box), radius, radius)

    def _arrow(self, painter, x0, y, x1, color=None):
        painter.setPen(QPen(self._color(color, "secondary"), 2))
        painter.drawLine(QPointF(x0, y), QPointF(x1 - 7, y))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color(color, "secondary"))
        painter.drawPolygon(QPolygonF((QPointF(x1, y), QPointF(x1 - 10, y - 7), QPointF(x1 - 10, y + 7))))

    def _film(self, painter, box, *, image=True):
        x, y, w, h = box
        self._panel(painter, box, "#624133", "#ba8061", 5)
        for i in range(6):
            for py in (y + 5, y + h - 13):
                painter.setBrush(QColor("#1b1e24"))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(QRectF(x + 9 + i * (w - 27) / 5, py, 10, 8), 2, 2)
        if image:
            painter.setBrush(QColor("#997156"))
            painter.drawRect(QRectF(x + 13, y + 21, w - 26, h - 42))
            painter.setBrush(QColor("#4c6472"))
            painter.drawEllipse(QPointF(x + w * .7, y + h * .4), 11, 11)
            painter.setBrush(QColor("#4f7165"))
            painter.drawPolygon(QPolygonF((QPointF(x + 15, y + h - 21), QPointF(x + w * .4, y + h * .42),
                                           QPointF(x + w - 13, y + h - 21))))

    def _welcome(self, p):
        self._text(p, (18, 8, 924, 26), "依次照明  ·  固定底片和相机  ·  保留三张 RAW", color="#bfc6d2", size=17)
        for i, (channel, color) in enumerate(zip("RGB", self.channel_colors)):
            y = 71 + i * 61
            tint = QColor(color)
            tint.setAlpha(28 if self.dark else 18)
            self._panel(p, (32, y, 95, 44), tint, False)
            self._text(p, (32, y, 95, 44), channel + " 灯", color=color, bold=True)
            self._arrow(p, 143, y + 22, 192, color)
        self._film(p, (207, 76, 152, 170))
        self._text(p, (200, 254, 166, 30), "固定底片", color="#d1c8bb")
        self._arrow(p, 378, 160, 419)
        self._panel(p, (433, 107, 127, 106))
        self._panel(p, (461, 92, 48, 20), "#4f5662")
        p.setPen(QPen(QColor(self.colors["lens_edge"]), 3))
        p.setBrush(QColor(self.colors["lens"]))
        p.drawEllipse(QPointF(496, 160), 31, 31)
        p.setPen(QPen(QColor(self.colors["lens_edge"]), 1))
        p.drawEllipse(QPointF(496, 160), 22, 22)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self.colors["card"]))
        p.drawEllipse(QPointF(487, 151), 5, 5)
        self._text(p, (418, 254, 158, 30), "相机保持不动", color="#d1c8bb")
        self._arrow(p, 578, 160, 620)
        for i, (channel, color) in enumerate(zip("RGB", self.channel_colors)):
            self._panel(p, (638, 63 + i * 67, 98, 49))
            self._text(p, (638, 63 + i * 67, 98, 49), channel + " RAW", color=color)
        self._arrow(p, 756, 160, 800)
        self._panel(p, (815, 103, 115, 114), "#32363b")
        p.fillRect(QRectF(831, 119, 82, 56), QColor("#b6c5cf"))
        p.setBrush(QColor("#ecd5a1"))
        p.drawEllipse(QPointF(892, 134), 8, 8)
        p.setBrush(QColor("#728b70"))
        p.drawPolygon(QPolygonF((QPointF(831, 175), QPointF(857, 139), QPointF(886, 175))))
        p.setBrush(QColor("#526d65"))
        p.drawPolygon(QPolygonF((QPointF(857, 175), QPointF(886, 148), QPointF(913, 175))))
        self._text(p, (818, 180, 109, 28), "合成影像", size=18)
        self._text(p, (126, 297, 708, 28), "三个光谱观测，进入同一套转换流程", color="#bfc6d2", size=18)

    def _science(self, p):
        self._text(p, (28, 2, 904, 28), "概念示意 · 非实测曲线", color="#bfc6d2", size=17)
        x, y, w, h = 90, 72, 792, 150
        for center, color in ((.23, self.channel_colors[2]), (.51, self.channel_colors[1]), (.79, self.channel_colors[0])):
            band_color = QColor(color)
            band_color.setAlpha(35 if self.dark else 22)
            p.fillRect(QRectF(x + (center - .018) * w, y, .036 * w, h), band_color)
        curves = ((.23, .14, .11, .56, self.colors["curves"][0], "黄染料"),
                  (.51, .13, .14, .78, self.colors["curves"][1], "品红染料"),
                  (.79, .15, .13, .47, self.colors["curves"][2], "青染料"))
        # Smooth invented side-absorption curves teach overlap only. They do
        # not encode any stock's wavelength, spectral power, or measured data.
        for center, sigma, side, side_center, color, _ in curves:
            path = QPainterPath()
            for step in range(161):
                u = step / 160
                value = .08 + .78 * math.exp(-.5 * ((u - center) / sigma) ** 2)
                value += side * math.exp(-.5 * ((u - side_center) / .19) ** 2)
                point = QPointF(x + u * w, y + h * (1 - value))
                path.moveTo(point) if step == 0 else path.lineTo(point)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(color), 3))
            p.drawPath(path)
        p.setPen(QPen(QColor(self.colors["divider"]), 1))
        p.drawLine(QPointF(x, y + h), QPointF(x + w, y + h))
        p.drawLine(QPointF(x, y), QPointF(x, y + h))
        self._text(p, (14, 102, 65, 80), "相对\n吸收", size=18)
        self._text(p, (x - 10, y + h + 3, 140, 28), "短波", color=self.channel_colors[2], size=18)
        self._text(p, (x + w - 130, y + h + 3, 140, 28), "长波", color=self.channel_colors[0], size=18)
        self._text(p, (288, y + h + 3, 390, 28), "波长方向（不标定具体波长）", color="#b7beca", size=18)
        for index, item in enumerate(curves):
            self._text(p, (156 + index * 240, 37, 205, 30), item[-1], color=item[-2], size=18)
        self._text(p, (70, 267, 820, 28), "光源光谱  ×  胶片透射  ×  传感器响应", bold=True, size=21)
        self._text(p, (60, 306, 840, 24), "共同决定信号；窄波段有助于分离，但不会使侧吸收归零。", color="#c4c9d2", size=17)

    def _prepare(self, p):
        for box in ((24, 10, 438, 315), (498, 10, 438, 315)):
            self._panel(p, box)
        self._text(p, (36, 22, 414, 35), "片基参考 · 放入片基", bold=True, size=22)
        self._text(p, (510, 22, 414, 35), "平场参考 · 移开底片", bold=True, size=23)
        p.fillRect(QRectF(56, 77, 374, 164), QColor("#c5c9cb"))
        self._film(p, (145, 108, 195, 104), image=False)
        p.setPen(QPen(QColor("#ff9f0a"), 3))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(203, 136, 80, 47))
        for x in (203, 283):
            for y in (136, 183):
                p.fillRect(QRectF(x - 4, y - 4, 8, 8), QColor("#ff9f0a"))
        gradient = QRadialGradient(QPointF(717, 159), 245)
        gradient.setColorAt(0, QColor("#eef0ec"))
        gradient.setColorAt(1, QColor("#a3acb9"))
        p.fillRect(QRectF(530, 77, 374, 164), gradient)
        self._text(p, (72, 251, 342, 62), "R / G / B 各拍一次\n小块片基也可局部采样", color="#e1c6a5", size=20)
        self._text(p, (544, 251, 346, 62), "均匀白光参考\n用于修正空间亮度不均", color="#d3dbe6", size=20)

    def _workflow(self, p):
        self._text(p, (20, 4, 920, 27), "保留采集信息，再决定正像怎样呈现", color="#bfc6d2", size=17)
        names = ("线性 RAW", "平场校正", "密度测量", "印相／显示", "16 位输出")
        for index, label in enumerate(names):
            x = 23 + index * 188
            self._panel(p, (x, 55, 162, 69), "#303740" if index < 3 else "#3b3430")
            self._text(p, (x + 5, 60, 152, 59), label, size=21, bold=True)
            if index < 4:
                self._arrow(p, x + 169, 89, x + 181)
        self._panel(p, (25, 148, 910, 172))
        self._text(p, (46, 157, 870, 29), "相对片基密度 · 扣除暗信号后", bold=True, size=20)
        self._text(p, (43, 199, 422, 48), "T_rel = (I − I_dark) / (I_base − I_dark)", size=21)
        self._text(p, (43, 257, 422, 39), "D_rel = −log₁₀(T_rel)", size=25)
        for index, (transmission, density) in enumerate(((1., 0.), (.5, .301), (.1, 1.))):
            x = 490 + index * 140
            level = round(45 + 200 * transmission)
            self._panel(p, (x + 28, 196, 74, 26), QColor(level, level, level), radius=4)
            self._text(p, (x, 230, 130, 30), f"T_rel = {transmission:g}", size=19)
            self._text(p, (x, 268, 130, 33), f"D_rel = {density:g}", size=20, bold=True)

    def _sources(self, p):
        self._text(p, (20, 4, 920, 28), "参考资料各有适用范围", color="#bfc6d2", size=17)
        cards = (("片基", "采集曝光与透射参考", "没有画面的底片\n不是中灰目标", self.colors["accent"]),
                 ("LAD", "印相／流程质控", "规定条件下的中间调\n不等同于片基", self.colors["text"]),
                 ("编码", "数字文件容纳多少信息", "16 位能更细地记录\n不能补回采集丢失", self.channel_colors[2]))
        for index, (name, purpose, caption, color) in enumerate(cards):
            x = 25 + index * 313
            self._panel(p, (x, 62, 283, 234))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(color))
            if index == 0:
                p.drawRoundedRect(QRectF(x + 112, 91, 59, 47), 4, 4)
            elif index == 1:
                for cell in range(5):
                    value = 48 + cell * 42
                    p.fillRect(QRectF(x + 42 + cell * 40, 98, 38, 33), QColor(value, value, value))
            else:
                for cell in range(12):
                    p.fillRect(QRectF(x + 41 + cell * 17, 98, 16, 33), QColor(43 + cell * 17, 55 + cell * 16, 72 + cell * 15))
            self._text(p, (x + 10, 150, 263, 35), name, color=color, bold=True, size=25)
            self._text(p, (x + 10, 190, 263, 32), purpose, size=19)
            self._text(p, (x + 10, 236, 263, 48), caption, color="#c4cad3", size=18)
        self._text(p, (60, 310, 840, 25), "资料出处可展开阅读；示意图与演示数字不是设备标定结果。", color="#bfc6d1", size=18)


class OnboardingVisual(QWidget):
    """A responsive illustration for an introduction page."""
    def __init__(self, parent=None, page_key="welcome"):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = _Illustration(self)
        layout.addWidget(self.canvas, 1)
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
        self.setAccessibleName("首次使用图示：" + PAGE_TITLES[key])
        self._update_description()
        self.canvas.update()

    def _update_description(self):
        if self.page_key == "science":
            self.setAccessibleDescription("概念示意，非实测曲线：三种染料主吸收与侧吸收，三个可选窄波段。")
        elif self.page_key == "workflow":
            self.setAccessibleDescription("线性RAW经平场、密度、印相显示到16位输出；以片基归一化透射率0.5对应相对密度约0.301。")
        elif self.page_key == "sources":
            self.setAccessibleDescription("片基用于采集曝光与透射参考，LAD用于印相流程质控，编码用于记录已采得信息；三者不能互相替代。")
        else:
            self.setAccessibleDescription("离线矢量示意；不使用照片、光谱实测或设备标定数据。")


def create_page_visual(page_key, parent=None):
    return OnboardingVisual(parent, page_key)
