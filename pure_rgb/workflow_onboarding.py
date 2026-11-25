"""Skippable first-use introduction, shared as structured text/HTML/Markdown.

The application owns when this window is shown and remembers that decision.
This module never opens a project, changes preferences or starts processing.
"""

from dataclasses import dataclass
from html import escape
import math

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QTextBrowser, QToolButton, QVBoxLayout, QWidget,
)

from .workflow_onboarding_theme import body_font, theme_tokens


@dataclass(frozen=True)
class IntroductionSource:
    key: str
    title: str
    url: str
    scope: str


SOURCES = (
    IntroductionSource(
        "kodak", "Kodak VISION3 500T 5219/7219 技术资料，第 4 页",
        "https://www.kodak.com/content/pdfs/motion/KODAK-VISION3-500T-5219-7219-technical-information.pdf",
        "区分曝光时的光谱感光度与冲洗后的染料光谱密度；染料曲线按峰值归一化。",
    ),
    IntroductionSource(
        "scanner", "Flueckiger 等，2018，Film Material–Scanner Interaction，第 2.1–2.3 节",
        "https://scan2screen.com/wp-content/uploads/2026/06/flueckigeretal_investigationfilmmaterialscannerinteraction_2018_v_1-1c.pdf",
        "说明染料侧吸收、扫描系统与材料的相互作用，以及窄波段用于改善色分离的条件。",
    ),
    IntroductionSource(
        "printing", "Trumpy 与 Anderson，2025，From Negatives to Positives",
        "https://doi.org/10.2352/issn.2168-3204.2025.22.1.33",
        "以未曝光片基为线性透射参考，并将采集波段与后续印相模型联系；示例不等于普适还原证明。",
    ),
    IntroductionSource(
        "range", "Kodak：扩展动态范围胶片扫描建议，第 3 页",
        "https://www.kodak.com/content/products-brochures/Film/Scanning-Recommendations-for-Extended-Dynamic-Range-Camera-Films-EN.pdf",
        "16 位编码需配合足够的光电采集范围，不能恢复原始采集已丢失的信息。",
    ),
    IntroductionSource(
        "arri", "ARRI ARRISCAN XT 原厂规格（工程参考）",
        "https://www.arri.com/en/cine-systems/archive-technologies/arriscan-xt",
        "工业实现结合 RGB/IR、闭环灯光与均匀照明；不构成本软件的性能认证。",
    ),
    IntroductionSource(
        "negpy", "NegPy：相机采集与片基校准",
        "https://github.com/marcinz606/NegPy/blob/d8123a9b2a18fdaab5941347a91e402fa936d3dc/docs/CAMERA_SCANNING.md",
        "记录 NegPy 的相机采集实践，包括共享拍摄设置、逐通道测量片基和检查曝光上限。",
    ),
    IntroductionSource(
        "lad_negative", "Kodak H-61A：彩色负片 LAD",
        "https://www.kodak.com/content/products-brochures/Film/LAD-for-KODAK-Color-Negative-Film-H-61a.pdf",
        "LAD 是规定的灰块与流程控制目标，不是未曝光片基或相机 RAW 的曝光上限。",
    ),
    IntroductionSource(
        "lad_print", "Kodak H-61B：VISION 彩色印片 LAD",
        "https://www.kodak.com/content/products-brochures/Film/LAD-for-KODAK-VISION-Color-Print-Film-H-61b.pdf",
        "印片的视觉密度与 Status A 控制目标用于印相质控。",
    ),
)


@dataclass(frozen=True)
class IntroductionPage:
    key: str
    title: str
    paragraphs: tuple[str, ...]
    sources: tuple[str, ...] = ()


PAGES = (
    IntroductionPage("welcome", "一帧底片，三次曝光", (
        "PURE RGB 将红、绿、蓝三次独立 RAW 采集合成为一张照片，再进行密度测量、负片转换、调色与导出。",
        "每次只点亮一个照明通道，底片、相机与构图保持固定。三色采集提供三次不同照明下的观测；普通白光照片里的三个 RGB 数值不能替代这三次采集。",
        "这份介绍可以随时跳过，之后仍可从应用菜单再次打开。",
    )),
    IntroductionPage("science", "窄带光与色彩分离", (
        "冲洗后的彩色胶片主要由青、品红、黄色染料形成影像。染料除主吸收波段外，还会吸收其他波段；这种侧吸收使各次颜色观测产生混合。",
        "选择合适的窄波段，可以减少某些侧吸收与波段重叠的影响，改善色分离。实际效果仍取决于胶片、光源光谱与传感器响应；窄带不意味着零串扰或自动获得准确色彩。",
        "线性信号的简化关系：s_c − b_c ∝ t_c ∫ L_c(λ) T(λ) Q_c(λ) dλ。光源 L、胶片透射 T 与传感器响应 Q 共同决定测得的信号。",
        "Kodak 的光谱感光曲线描述曝光前乳剂的响应；冲洗后的染料光谱密度曲线才用于理解扫描吸收。两组曲线不能互换，也不能直接当作本机灯光标定。",
    ), ("kodak", "scanner")),
    IntroductionPage("prepare", "片基与平场参考", (
        "固定相机、底片、镜头焦距、对焦和光圈；使用线性 RAW，明确记录每张对应的红、绿、蓝照明。三次拍摄之间避免移动与振动。",
        "观察片基透射时，使用同卷已冲洗、无画面的片基。固定 ISO、快门与光圈，各拍一张 R、G、B RAW。小片基可放在画面中央，避开齿孔、字样和阴影。",
        "拍平场参考时移开底片，用均匀白光拍摄。平场用于校正照明与镜头造成的空间不均。",
        "准备好后选择 RAW 文件或素材文件夹，软件自动归组和排队；无需逐张再次确认导入。",
    ), ("printing", "negpy")),
    IntroductionPage("workflow", "从 RAW 到正像", (
        "采集信号、光学密度和显示亮度属于不同阶段。图中以片基为参考解释相对密度；软件通过自动测量与印相映射决定正像呈现，16 位输出负责记录已有的处理结果。",
        "「图片库…」用于集中管理图片目录与项目入口。保留原始 RAW；在工作台卸载照片不会删除原文件。",
        "同型号相机的平场可匹配不同分辨率，按对应画幅缩放；文件的 MB 大小不作为兼容依据。镜头、光圈、对焦或灯位变化后仍要核对参考是否适用。",
        "「自动镜头色差矫正」针对高反差边缘的红绿、蓝紫彩边；可用「自动匹配」或「选择 LCP…」指定镜头配置，也可「清除」配置。「镜头畸变矫正」另行开关，照明不均使用平场。",
        "按照六步工作台完成测量、调色与修整；最后在照片列表单选、多选或全选，再导出所选照片。",
    ), ("printing", "range")),
    IntroductionPage("sources", "扫描与印相的参考资料", (
        "以下资料用于说明胶片数字化的科学原理与实现参考；Kodak 或其他研究、机构没有因此为 PURE RGB 背书。",
        "已冲洗片基可以作为线性透射参考。要评估整套装置的色彩准确度，还需测量该光源、胶片和传感器组合的实际响应。",
        "Kodak 的 16 位扫描建议同时要求前端能采集所需动态范围。ARRI 的工业实例结合了灯光闭环与均匀照明。",
        "更多操作与处理范围，可从应用菜单打开「使用指南」。",
    ), ("kodak", "scanner", "printing", "range", "arri", "negpy", "lad_negative", "lad_print")),
)


CALCULATION_NOTES = {
    "prepare": (
        "有效 RAW 信号 u = (RAW − 黑位) / (白位 − 黑位)；平场后信号 s = 增益 g × u。",
        "平场增益随像素位置变化，用于减轻画面边缘变暗和照明不均。",
    ),
    "workflow": (
        "以同类已冲洗片基为参考时：T_rel = (I − I_dark) / (I_base − I_dark)，D_rel = −log₁₀(T_rel)。",
        "例：校正后样本信号为片基参考的一半，T_rel = 0.5，则相对片基密度 D_rel ≈ 0.301。",
        "若改用无胶片的白参考，结果包含片基贡献，不能把两种密度混为一谈；这些关系也不能单独给出绝对染料浓度。",
    ),
}


def _sources_for(page):
    lookup = {source.key: source for source in SOURCES}
    return tuple(lookup[key] for key in page.sources)


def onboarding_html(index, *, colors=None, show_title=True):
    page = PAGES[index]
    colors = colors or theme_tokens(True)
    paragraphs = "".join(f"<p>{escape(text)}</p>" for text in page.paragraphs)
    notes = CALCULATION_NOTES.get(page.key, ())
    if notes:
        paragraphs += '<h2>算一次看看</h2>' + "".join(f"<p>{escape(text)}</p>" for text in notes)
    citations = "".join(
        f'<p class="source"><a href="{escape(source.url, quote=True)}">{escape(source.title)}</a>'
        + (f'<br>{escape(source.scope)}' if page.key != "sources" else "") + "</p>"
        for source in _sources_for(page)
    )
    heading = f"<h1>{escape(page.title)}</h1>" if show_title else ""
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
        body {{ color:{colors['text']}; background:{colors['surface']}; text-align:center; font-size:14px; }}
        h1 {{ font-size:25px; font-weight:600; margin-bottom:22px; }}
        p {{ line-height:1.65; margin:16px 12px; }}
        a {{ color:{colors['link']}; }} .source {{ font-size:12px; color:{colors['secondary']}; }}
        </style></head><body>{heading}{paragraphs}{citations}</body></html>"""


def onboarding_markdown():
    """Text-only reuse for help content; scientific and implementation sources stay labeled."""
    parts = ["# PURE RGB 首次使用介绍"]
    for page in PAGES:
        parts.append(f"## {page.title}")
        parts.extend(page.paragraphs)
        parts.extend(CALCULATION_NOTES.get(page.key, ()))
        for source in _sources_for(page):
            parts.append(f"[{source.title}]({source.url})：{source.scope}")
    return "\n\n".join(parts) + "\n"


QUICK_READS = {
    "welcome": "固定相机和底片，依次点亮红、绿、蓝灯，各拍一张 RAW。PURE RGB 将三张 RAW 对齐合成。",
    "science": "每种染料都会吸收多个波段。选择合适的窄带光，可以减轻染料吸收重叠对色彩分离的影响。",
    "prepare": "片基参考用同卷已冲洗、无画面的胶片，可只放一小块在中央。拍平场时移开底片，记录均匀白光。",
    "workflow": "和片基相比，透过的光越少，相对密度越高。下面的例子展示透射率怎么换算成密度。",
    "sources": "展开原理与来源，查看 Kodak、ARRI 和胶片数字化研究各自回答了什么问题。",
}


def quick_read_html(index, *, colors=None):
    page = PAGES[index]
    colors = colors or theme_tokens(True)
    summary = escape(QUICK_READS[page.key])
    return f'''<html><head><style>
        body {{ color:{colors['text']}; background:{colors['surface']}; text-align:center; font-size:14px; }}
        h1 {{ font-size:29px; font-weight:600; margin:2px 0 10px; }}
        p {{ color:{colors['secondary']}; line-height:1.4; margin:4px 8px; }}
        </style></head><body><h1>{escape(page.title)}</h1><p>{summary}</p></body></html>'''


class IntroductionText(QTextBrowser):
    """Fit the document to its width so that the page only has one scrollbar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fitting = False
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.document().documentLayout().documentSizeChanged.connect(self.fit_height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_height()

    def fit_height(self, *args):
        if self._fitting:
            return
        self._fitting = True
        try:
            chrome = self.height() - self.viewport().height()
            height = math.ceil(self.document().size().height()) + chrome + 4
            if self.minimumHeight() != height or self.maximumHeight() != height:
                self.setFixedHeight(height)
        finally:
            self._fitting = False


class OnboardingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PURE RGB · 首次使用介绍")
        self.setObjectName("onboardingWindow")
        self.setProperty("useSystemWindowBackground", True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.resize(860, 565)
        self.setMinimumSize(650, 550)
        self.setFont(body_font())
        self.colors = theme_tokens()
        self.page_index = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 16)
        layout.setSpacing(10)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        body.setObjectName("onboardingBody")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(4, 0, 4, 4)
        column = QWidget()
        column.setMaximumWidth(1000)
        body_layout.addStretch()
        body_layout.addWidget(column, 100)
        body_layout.addStretch()
        content = QVBoxLayout(column)
        content.setContentsMargins(0, 0, 0, 4)
        content.setSpacing(8)
        self.scroll.setWidget(body)
        layout.addWidget(self.scroll, 1)
        self.viewer = IntroductionText()
        self.viewer.setObjectName("onboardingSummary")
        self.viewer.document().setDefaultFont(self.font())
        self.viewer.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        content.addWidget(self.viewer)
        self.visual_host = QWidget()
        self.visual_layout = QVBoxLayout(self.visual_host)
        self.visual_layout.setContentsMargins(0, 0, 0, 0)
        self.visual_layout.setSpacing(0)
        content.addWidget(self.visual_host)
        self.visual = None
        self.details_button = QToolButton()
        self.details_button.setText("原理与来源")
        self.details_button.setObjectName("onboardingDisclosure")
        self.details_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.details_button.setArrowType(Qt.ArrowType.RightArrow)
        self.details_button.setCheckable(True)
        self.details_button.toggled.connect(self.toggle_details)
        content.addWidget(self.details_button, 0, Qt.AlignmentFlag.AlignHCenter)
        self.detail_viewer = IntroductionText()
        self.detail_viewer.setObjectName("onboardingDetails")
        self.detail_viewer.document().setDefaultFont(self.font())
        self.detail_viewer.setOpenExternalLinks(True)
        self.detail_viewer.hide()
        content.addWidget(self.detail_viewer)
        content.addStretch(1)
        divider = QFrame()
        divider.setObjectName("onboardingDivider")
        divider.setFixedHeight(1)
        layout.addWidget(divider)
        self.page_indicator = QLabel()
        self.page_indicator.setObjectName("onboardingPageIndicator")
        self.page_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer = QGridLayout()
        footer.setColumnStretch(0, 1)
        footer.setColumnStretch(2, 1)
        self.skip_button = QPushButton("稍后再看")
        self.skip_button.setObjectName("onboardingSkip")
        self.skip_button.setAutoDefault(False)
        self.skip_button.clicked.connect(self.reject)
        footer.addWidget(self.skip_button, 0, 0, Qt.AlignmentFlag.AlignLeft)
        footer.addWidget(self.page_indicator, 0, 1)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.previous_button = QPushButton("上一步")
        self.previous_button.setAutoDefault(False)
        self.previous_button.clicked.connect(self.previous_page)
        buttons.addWidget(self.previous_button)
        self.next_button = QPushButton("下一步")
        self.next_button.setObjectName("onboardingNext")
        self.next_button.setDefault(True)
        self.next_button.clicked.connect(self.next_page)
        buttons.addWidget(self.next_button)
        footer.addLayout(buttons, 0, 2, Qt.AlignmentFlag.AlignRight)
        layout.addLayout(footer)
        self.apply_appearance()
        app = QApplication.instance()
        if app is not None:
            app.styleHints().colorSchemeChanged.connect(self._appearance_changed)
        self._show_page()

    def _appearance_changed(self, *args):
        self.apply_appearance()

    def apply_appearance(self, dark=None):
        self.colors = c = theme_tokens(dark)
        hover = "rgba(255,255,255,18)" if c["dark"] else "rgba(0,0,0,12)"
        pressed = "rgba(255,255,255,32)" if c["dark"] else "rgba(0,0,0,24)"
        self.setStyleSheet(f"""
            QDialog#onboardingWindow, QWidget#onboardingBody {{ background:{c['surface']}; }}
            QDialog#onboardingWindow QWidget {{ color:{c['text']}; }}
            QFrame#onboardingDivider {{ background:{c['edge']}; border:0; }}
            QDialog#onboardingWindow QLabel#onboardingPageIndicator {{
                color:{c['secondary']}; font-size:12px; background:transparent; }}
            QDialog#onboardingWindow QPushButton {{ background:{c['card']}; color:{c['text']};
                border:1px solid {c['edge']}; border-radius:6px; padding:3px 11px; min-height:22px;
                font-size:13px; }}
            QDialog#onboardingWindow QPushButton:hover {{ background:{hover}; }}
            QDialog#onboardingWindow QPushButton:pressed {{ background:{pressed}; }}
            QDialog#onboardingWindow QPushButton:disabled {{ color:{c['secondary']}; background:transparent; }}
            QDialog#onboardingWindow QPushButton#onboardingSkip {{ background:transparent; border-color:transparent; color:{c['secondary']}; }}
            QDialog#onboardingWindow QPushButton#onboardingSkip:hover {{ background:{hover}; color:{c['text']}; }}
            QDialog#onboardingWindow QPushButton#onboardingNext {{
                background:{c['button_fill']}; color:{c['on_accent']}; border-color:{c['button_fill']}; font-weight:600; }}
            QDialog#onboardingWindow QPushButton#onboardingNext:hover {{
                background:{c['button_hover']}; }}
            QDialog#onboardingWindow QPushButton:focus,
            QDialog#onboardingWindow QPushButton#onboardingSkip:focus,
            QDialog#onboardingWindow QPushButton#onboardingNext:focus {{ border-color:{c['link']}; }}
            QDialog#onboardingWindow QToolButton#onboardingDisclosure {{ color:{c['link']};
                background:transparent; border:1px solid transparent; border-radius:5px; padding:5px 8px; font-size:13px; }}
            QDialog#onboardingWindow QToolButton#onboardingDisclosure:hover {{ background:{hover}; }}
            QDialog#onboardingWindow QToolButton#onboardingDisclosure:focus {{ border-color:{c['link']}; }}
            QDialog#onboardingWindow QScrollArea {{ background:transparent; border:0; }}
            QTextBrowser#onboardingSummary, QTextBrowser#onboardingDetails {{
                background:{c['surface']}; color:{c['text']};
                border:0; padding:0; }}
        """)
        self._update_text()
        if self.visual is not None:
            self.visual.set_theme(c["dark"], colors=c)

    def _update_text(self):
        self.viewer.setHtml(quick_read_html(self.page_index, colors=self.colors))
        self.detail_viewer.setHtml(onboarding_html(self.page_index, colors=self.colors, show_title=False))

    def _show_page(self):
        from .workflow_onboarding_visuals import create_page_visual

        self._update_text()
        if self.visual is not None:
            self.visual_layout.removeWidget(self.visual)
            self.visual.hide()
            self.visual.deleteLater()
        self.visual = create_page_visual(PAGES[self.page_index].key, self.visual_host)
        self.visual.set_theme(self.colors["dark"], colors=self.colors)
        self.visual_layout.addWidget(self.visual)
        self.visual_host.updateGeometry()
        self.scroll.verticalScrollBar().setValue(0)
        self.page_indicator.setText(f"{self.page_index + 1} / {len(PAGES)}")
        self.page_indicator.setAccessibleName(f"第 {self.page_index + 1} 页，共 {len(PAGES)} 页")
        self.previous_button.setEnabled(self.page_index > 0)
        self.next_button.setText("开始使用" if self.page_index == len(PAGES) - 1 else "下一步")

    def toggle_details(self, expanded):
        self.detail_viewer.setVisible(expanded)
        self.details_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        if expanded:
            QTimer.singleShot(0, self._reveal_details)

    def _reveal_details(self):
        if self.details_button.isChecked():
            top = self.details_button.mapTo(self.scroll.widget(), self.details_button.rect().topLeft()).y()
            self.scroll.verticalScrollBar().setValue(max(0, top - 24))

    def previous_page(self):
        if self.page_index > 0:
            self.page_index -= 1
            self._show_page()

    def next_page(self):
        if self.page_index == len(PAGES) - 1:
            self.accept()
        else:
            self.page_index += 1
            self._show_page()


def show_onboarding(parent=None, on_finished=None):
    """Show optional introduction; on_finished receives QDialog's result int."""
    dialog = OnboardingDialog(parent)
    if on_finished is not None:
        dialog.finished.connect(on_finished)
    dialog.show()
    return dialog
