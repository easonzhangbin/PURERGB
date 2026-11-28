"""Credits and the offline StudioPURE donation page."""

from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QPixmap
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from .release import DONATION_URL
from .workflow_glass import observe_appearance, prepare_glass, stylesheet


DONORS = "nothing老庄、J3RRYK33PF1T、熊小默、基岩、一笑而过"
DONATION_QR = Path(__file__).resolve().parents[1] / "docs/images/SupportAlipay.jpg"


def donation_dialog(parent):
    dialog = QDialog(parent)
    prepare_glass(dialog)
    dialog.setObjectName("workflowRoot")
    dialog.setStyleSheet(stylesheet(False))
    observe_appearance(dialog, lambda: dialog.setStyleSheet(stylesheet(False)))
    dialog.setWindowTitle("捐赠 StudioPURE 项目")
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(24, 20, 24, 18)
    layout.setSpacing(12)
    layout.addWidget(QLabel("使用支付宝扫码，收款方为 StudioPURE。捐赠金额自定。"))
    qr = QLabel()
    qr.setAccessibleName("StudioPURE 支付宝捐赠二维码")
    image = QPixmap(str(DONATION_QR))
    if image.isNull():
        qr.setText("请打开下方的 GitHub 捐赠页查看收款码。")
    else:
        ratio = dialog.devicePixelRatioF()
        # Preserve the original artwork and quiet zone at Retina density.
        image = image.scaledToHeight(round(340 * ratio), Qt.TransformationMode.SmoothTransformation)
        image.setDevicePixelRatio(ratio)
        qr.setPixmap(image)
    layout.addWidget(qr, alignment=Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(QLabel("感谢以下捐助者"))
    credits = QLabel(DONORS)
    credits.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    credits.setWordWrap(True)
    layout.addWidget(credits)
    actions = QHBoxLayout()
    online = QPushButton("在 GitHub 查看")
    online.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(DONATION_URL)))
    actions.addWidget(online)
    actions.addStretch()
    close = QPushButton("关闭")
    close.setDefault(True)
    close.clicked.connect(dialog.accept)
    actions.addWidget(close)
    layout.addLayout(actions)
    return dialog
