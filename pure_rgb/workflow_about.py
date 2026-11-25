"""The shared About window for PURE RGB."""

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QDialog, QFrame, QLabel, QPushButton, QVBoxLayout

from .branding import application_icon
from .release import version_description
from .workflow_branding import header_logo
from .workflow_onboarding_theme import body_font, theme_tokens
from .workflow_support import DONORS


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("aboutWindow")
        self.setProperty("useSystemWindowBackground", True)
        self.setWindowTitle("关于 PURE RGB")
        self.setFixedWidth(408)
        self.setFont(body_font())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 24, 30, 18)
        layout.setSpacing(4)
        layout.setSizeConstraints(
            QVBoxLayout.SizeConstraint.SetNoConstraint, QVBoxLayout.SizeConstraint.SetFixedSize,
        )

        icon = QLabel()
        icon.setAccessibleName("PURE RGB 应用图标")
        ratio = self.devicePixelRatioF()
        icon_pixels = round(80 * ratio)
        pixmap = application_icon().pixmap(QSize(icon_pixels, icon_pixels), 1.0)
        pixmap.setDevicePixelRatio(ratio)
        icon.setPixmap(pixmap)
        layout.addWidget(icon, alignment=Qt.AlignmentFlag.AlignHCenter)

        wordmark = QLabel()
        wordmark.setObjectName("aboutWordmark")
        wordmark.setAccessibleName("PURE RGB")
        wordmark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wordmark.setPixmap(header_logo(44))
        layout.addWidget(wordmark, alignment=Qt.AlignmentFlag.AlignHCenter)

        version = self._text(f"版本 {version_description()}", "aboutVersion")
        version.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(version)

        layout.addSpacing(10)
        layout.addWidget(self._text("StudioPURE 出品", "aboutCredit"))
        layout.addWidget(self._text("三色负片与黑白影像工作流。"))
        layout.addWidget(self._text("导入归组、画幅与测量、转换调色、\n"
                                    "修整审核、观片台与 RGB16／ICC 导出。", "aboutDetails"))

        layout.addSpacing(10)
        divider = QFrame()
        divider.setObjectName("aboutDivider")
        divider.setFixedHeight(1)
        layout.addWidget(divider)
        layout.addSpacing(8)
        layout.addWidget(self._text("感谢以下捐助者", "aboutCredit"))
        layout.addWidget(self._text(DONORS, "aboutDetails"))

        layout.addSpacing(12)
        close = QPushButton("关闭")
        close.setDefault(True)
        close.clicked.connect(self.accept)
        layout.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)

        self.apply_appearance()
        QApplication.instance().styleHints().colorSchemeChanged.connect(self._appearance_changed)

    @staticmethod
    def _text(value, name=""):
        label = QLabel(value)
        label.setObjectName(name)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        return label

    def _appearance_changed(self, _scheme):
        self.apply_appearance()

    def apply_appearance(self, dark=None):
        colors = theme_tokens(dark)
        wordmark_background = "transparent" if colors["dark"] else "#1c1c1e"
        self.setStyleSheet(f"""
            QDialog#aboutWindow {{ background: {colors['surface']}; }}
            QDialog#aboutWindow QLabel {{ background: transparent; color: {colors['text']};
                font-size: 13px; font-weight: 400; }}
            QDialog#aboutWindow QLabel#aboutWordmark {{
                background: {wordmark_background}; border-radius: 8px; padding: 0 8px;
            }}
            QDialog#aboutWindow QLabel#aboutVersion,
            QDialog#aboutWindow QLabel#aboutDetails {{ color: {colors['secondary']}; font-size: 12px; }}
            QDialog#aboutWindow QLabel#aboutCredit {{ font-weight: 600; }}
            QDialog#aboutWindow QFrame#aboutDivider {{ background: {colors['edge']}; border: 0; }}
            QDialog#aboutWindow QPushButton {{
                background: {colors['card']}; color: {colors['text']};
                font-size: 13px; font-weight: 400; border: 1px solid {colors['edge']};
                border-radius: 6px; padding: 3px 12px; min-width: 48px; min-height: 22px;
            }}
            QDialog#aboutWindow QPushButton:hover,
            QDialog#aboutWindow QPushButton:pressed {{ background: {colors['selected']}; }}
            QDialog#aboutWindow QPushButton:focus {{ border-color: {colors['link']}; }}
        """)


def about_dialog(parent=None):
    return AboutDialog(parent)


def show_about(parent):
    if not hasattr(parent, "_about_dialog"):
        parent._about_dialog = about_dialog(parent)
    dialog = parent._about_dialog
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
