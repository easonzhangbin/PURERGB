"""Skippable first-use introduction, shared as structured text/HTML/Markdown.

The application owns when this window is shown and remembers that decision.
This module never opens a project, changes preferences or starts processing.
"""

from dataclasses import dataclass
from html import escape
import math

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QTextBrowser, QToolButton, QVBoxLayout, QWidget,
)

from .workflow_branding import header_logo
from .workflow_glass import observe_appearance
from .workflow_onboarding_theme import body_font, theme_tokens
from .workflow_onboarding_visuals import create_page_visual


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
        "分别记录曝光时的乳剂光谱感光度与冲洗后的染料光谱密度。",
    ),
    IntroductionSource(
        "scanner", "Flueckiger 等，2018，Film Material–Scanner Interaction，第 2.1–2.3 节",
        "https://scan2screen.com/wp-content/uploads/2026/06/flueckigeretal_investigationfilmmaterialscannerinteraction_2018_v_1-1c.pdf",
        "说明染料侧吸收、扫描系统与材料的相互作用，以及窄波段用于改善色分离的条件。",
    ),
    IntroductionSource(
        "printing", "Trumpy 与 Anderson，2025，From Negatives to Positives",
        "https://doi.org/10.2352/issn.2168-3204.2025.22.1.33",
        "以未曝光片基为线性透射参考，并将采集波段与后续印相模型联系。",
    ),
    IntroductionSource(
        "range", "Kodak：扩展动态范围胶片扫描建议，第 3 页",
        "https://www.kodak.com/content/products-brochures/Film/Scanning-Recommendations-for-Extended-Dynamic-Range-Camera-Films-EN.pdf",
        "说明如何采集较宽的胶片密度范围，并用足够的位深记录测得的信息。",
    ),
    IntroductionSource(
        "arri", "ARRI ARRISCAN XT 原厂规格（工程参考）",
        "https://www.arri.com/en/cine-systems/archive-technologies/arriscan-xt",
        "工业实现结合 RGB/IR、闭环灯光与均匀照明。",
    ),
    IntroductionSource(
        "negpy", "NegPy：相机采集与片基校准",
        "https://github.com/marcinz606/NegPy/blob/d8123a9b2a18fdaab5941347a91e402fa936d3dc/docs/CAMERA_SCANNING.md",
        "记录 NegPy 的相机采集实践，包括共享拍摄设置、逐通道测量片基和检查曝光上限。",
    ),
    IntroductionSource(
        "lad_negative", "Kodak H-61A：彩色负片 LAD",
        "https://www.kodak.com/content/products-brochures/Film/LAD-for-KODAK-Color-Negative-Film-H-61a.pdf",
        "说明彩负 D-min 片基参照与 LAD 灰块各自的用途。",
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
        "固定底片、相机、构图和对焦，依次单独点亮红、绿、蓝灯，每次拍一张 RAW。"
        "三张使用相同 ISO、快门与光圈，并记录各自的照明颜色。",
        "PURE RGB 对齐并合成同一帧的三张 RAW，随后进行负片转换和调色。",
        "可随时跳过这份介绍，也可从应用菜单重新打开。",
    ), ("negpy",)),
    IntroductionPage("science", "窄带光与色彩分离", (
        "彩色胶片冲洗后，青、品红、黄染料的吸收波段会重叠。"
        "合适的窄带照明可以减轻这种混合。各路仍受到染料侧吸收影响，色分离由胶片、灯光光谱和传感器共同决定。",
        "Kodak 分别记录乳剂的光谱感光度与染料的光谱密度。前者用于理解拍摄时的感光响应，后者用于理解翻拍时冲洗后染料的吸收。",
    ), ("kodak", "scanner", "arri")),
    IntroductionPage("prepare", "检查片基与曝光余量", (
        "「RGB 灯光配平」可按需使用。优先用同卷已冲洗、未曝光的干净片基作为 D-min 最低密度参照，"
        "在相同设置下分别拍摄 R、G、B 三张 RAW。",
        "没有单独片基时，可拍下同一处片间空隙，选择「片间空隙」并点击「识别片间空隙」。"
        "用橙色建议框定位，手工避开帧线、字样、齿孔和片外。找不到候选时可手绘。分析后勾选「确认是同卷未曝光片间空隙」。",
        "橙框比较三路片基的中位信号。套用平场后，偏差以校正后信号计算。制作平场时移开底片，拍摄均匀白光。",
        "蓝框圈住实际保留的整个画幅，原始 RAW 的白电平 100% 是上限。确认蓝框后查看曝光建议，"
        "画幅内出现明显削顶时，缩短三路共同的快门时间再复拍。",
    ), ("lad_negative", "range")),
    IntroductionPage("workflow", "确认画幅，再测密度", (
        "在「导入归组」中使用「选择 RAW 文件…」或「选择素材文件夹…」。素材会自动归组和排队，"
        "选中照片后可核对红、绿、蓝是否对应实际照明。",
        "有效画幅确定后，初始密度测量取中心约 80% 的面积，保持原长宽比。"
        "自动定位不确定时，用「框选有效画幅并测量中心 80%」或直接绘制密度测量区。",
        "可独立调整密度测量区与输出裁切。确定测量范围后，在工作台完成转换、调色与修整，再导出所选照片。",
    ), ("printing", "lad_print")),
)


CALCULATION_NOTES = {
    "science": (
        "简化的线性采集关系为 s_c − b_c ∝ t_c ∫ L_c(λ) T(λ) Q_c(λ) dλ。光源 L、胶片透射 T 与传感器响应 Q 一起决定测得的信号。",
    ),
    "prepare": (
        "有效 RAW 信号 u = (RAW − 黑位) / (白位 − 黑位)。平场后信号 s = 增益 g × u。",
        "平场增益随位置变化，用于减轻边缘变暗和照明不均。蓝框的 RAW 上限使用原始感光点。",
    ),
    "workflow": (
        "以同类已冲洗片基为参考，相对透射率 T_rel = (I − I_dark) / (I_base − I_dark)，相对密度 D_rel = −log₁₀(T_rel)。",
        "扣除暗信号后，若样本信号是片基参考的一半，T_rel = 0.5，相对片基密度 D_rel ≈ 0.301。",
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
        paragraphs += '<h2>信号与密度怎么算</h2>' + "".join(f"<p>{escape(text)}</p>" for text in notes)
    citations = "".join(
        f'<p class="source"><a href="{escape(source.url, quote=True)}">{escape(source.title)}</a>'
        f'<br>{escape(source.scope)}</p>'
        for source in _sources_for(page)
    )
    heading = f"<h1>{escape(page.title)}</h1>" if show_title else ""
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
        body {{ color:{colors['text']}; background:{colors['surface']}; text-align:left; font-size:14px; }}
        h1 {{ font-size:25px; font-weight:600; margin-bottom:18px; }}
        h2 {{ font-size:17px; font-weight:600; margin:20px 12px 6px; }}
        p {{ line-height:1.55; margin:12px; }}
        a {{ color:{colors['link']}; }} .source {{ font-size:13px; color:{colors['secondary']}; }}
        </style></head><body>{heading}{paragraphs}<h2>参考资料</h2>{citations}</body></html>"""


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
    "welcome": "固定相机和底片，依次单独点亮红、绿、蓝灯，各拍一张 RAW，再由 PURE RGB 对齐并合成。",
    "science": "合适的窄带照明能减轻青、品红、黄染料吸收重叠对色彩分离的影响。",
    "prepare": "配平可选。片间模式可给出橙色建议框，调整后人工确认。蓝框检查保留画幅的 RAW 曝光。",
    "workflow": "选择 RAW 文件或素材文件夹后自动归组，有效画幅确定后，初始密度测量取中心约 80% 的面积。",
}


def quick_read_html(index, *, colors=None):
    page = PAGES[index]
    colors = colors or theme_tokens(True)
    summary = escape(QUICK_READS[page.key])
    return f'''<html><head><style>
        body {{ color:{colors['text']}; background:{colors['surface']}; text-align:left; font-size:15px; }}
        h1 {{ font-size:33px; font-weight:600; margin:2px 0 8px; }}
        p {{ color:{colors['secondary']}; line-height:1.45; margin:0 0 4px; }}
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
        self.resize(946, 590)
        self.setMinimumSize(680, 540)
        self.setFont(body_font())
        self.colors = theme_tokens()
        self.page_index = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 19, 26, 15)
        layout.setSpacing(12)
        header = QHBoxLayout()
        header.setSpacing(14)
        wordmark = QLabel()
        wordmark.setObjectName("onboardingWordmark")
        wordmark.setAccessibleName("PURE RGB")
        wordmark.setPixmap(header_logo(32))
        header.addWidget(wordmark)
        header.addStretch(1)
        chapters = QFrame()
        chapters.setObjectName("onboardingChapters")
        chapter_layout = QHBoxLayout(chapters)
        chapter_layout.setContentsMargins(3, 3, 3, 3)
        chapter_layout.setSpacing(2)
        self.chapter_buttons = QButtonGroup(self)
        for index, title in enumerate(("三色拍摄", "光谱原理", "片基与曝光", "密度与转换")):
            chapter = QPushButton(title)
            chapter.setObjectName("onboardingChapter")
            chapter.setCheckable(True)
            chapter.setAutoDefault(False)
            self.chapter_buttons.addButton(chapter, index)
            chapter_layout.addWidget(chapter)
        self.chapter_buttons.idClicked.connect(self._select_page)
        header.addWidget(chapters)
        layout.addLayout(header)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("onboardingBody")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(2, 0, 2, 4)
        column = QWidget()
        column.setMaximumWidth(1000)
        body_layout.addStretch()
        body_layout.addWidget(column, 100)
        body_layout.addStretch()
        content = QVBoxLayout(column)
        content.setContentsMargins(0, 9, 0, 4)
        content.setSpacing(9)
        self.scroll.setWidget(body)
        layout.addWidget(self.scroll, 1)
        self.viewer = IntroductionText()
        self.viewer.setObjectName("onboardingSummary")
        self.viewer.document().setDefaultFont(self.font())
        self.viewer.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        content.addWidget(self.viewer)
        self.visual = create_page_visual(PAGES[0].key, column)
        content.addWidget(self.visual)
        self.details_button = QToolButton()
        self.details_button.setText("原理与参考资料")
        self.details_button.setObjectName("onboardingDisclosure")
        self.details_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.details_button.setArrowType(Qt.ArrowType.RightArrow)
        self.details_button.setCheckable(True)
        self.details_button.toggled.connect(self.toggle_details)
        content.addWidget(self.details_button, 0, Qt.AlignmentFlag.AlignLeft)
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
        self.skip_button.setToolTip("可从应用菜单重新打开首次使用介绍。")
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
        observe_appearance(self, self._appearance_changed)
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
            QLabel#onboardingWordmark {{ background:#19191b; border-radius:8px; padding:0 4px; }}
            QFrame#onboardingChapters {{ background:{c['card']}; border:1px solid {c['edge']}; border-radius:10px; }}
            QFrame#onboardingDivider {{ background:{c['edge']}; border:0; }}
            QDialog#onboardingWindow QLabel#onboardingPageIndicator {{
                color:{c['secondary']}; font-size:12px; background:transparent; }}
            QDialog#onboardingWindow QPushButton {{ background:{c['card']}; color:{c['text']};
                border:1px solid {c['edge']}; border-radius:7px; padding:4px 12px; min-height:23px;
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
            QDialog#onboardingWindow QPushButton#onboardingChapter {{ background:transparent;
                border-color:transparent; border-radius:7px; min-height:22px; padding:2px 12px; color:{c['secondary']}; }}
            QDialog#onboardingWindow QPushButton#onboardingChapter:hover {{ background:{hover}; color:{c['text']}; }}
            QDialog#onboardingWindow QPushButton#onboardingChapter:checked {{ background:{c['selected']};
                color:{c['text']}; font-weight:600; }}
            QDialog#onboardingWindow QPushButton:focus,
            QDialog#onboardingWindow QPushButton#onboardingChapter:focus,
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
        self.visual.set_theme(c["dark"], colors=c)

    def _update_text(self):
        self.viewer.setHtml(quick_read_html(self.page_index, colors=self.colors))
        self.detail_viewer.setHtml(onboarding_html(self.page_index, colors=self.colors, show_title=False))

    def _show_page(self):
        self.details_button.setChecked(False)
        self._update_text()
        self.visual.set_page(PAGES[self.page_index].key)
        self.chapter_buttons.button(self.page_index).setChecked(True)
        self.scroll.verticalScrollBar().setValue(0)
        self.page_indicator.setText(f"{self.page_index + 1} / {len(PAGES)}")
        self.page_indicator.setAccessibleName(f"第 {self.page_index + 1} 页，共 {len(PAGES)} 页")
        self.previous_button.setEnabled(self.page_index > 0)
        self.next_button.setText("开始使用" if self.page_index == len(PAGES) - 1 else "下一步")

    def _select_page(self, index):
        if self.page_index != index:
            self.page_index = index
            self._show_page()

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
            self._select_page(self.page_index - 1)

    def next_page(self):
        if self.page_index == len(PAGES) - 1:
            self.accept()
        else:
            self._select_page(self.page_index + 1)


def show_onboarding(parent=None, on_finished=None):
    """Show optional introduction; on_finished receives QDialog's result int."""
    dialog = OnboardingDialog(parent)
    if on_finished is not None:
        dialog.finished.connect(on_finished)
    dialog.show()
    return dialog
