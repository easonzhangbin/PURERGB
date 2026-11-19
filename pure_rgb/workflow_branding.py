"""The user-supplied horizontal header logo, independent of the Dock icon."""

import hashlib
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication, QPixmap
from PyQt6.QtWidgets import QSplashScreen


HEADER_LOGO_PATH = (
    Path(__file__).resolve().parents[1] / "assets/branding/pure-rgb-header.png"
)
HEADER_LOGO_SHA256 = "5943158193415d4bd7c094ea6474c645c5c7e9d87c877f155f9660d25e5a610f"
SPLASH_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets/branding/deliveries/2026-11-13-png-final/Asset-1@2x.png"
)
SPLASH_SHA256 = "a0d996f13c6129db53e437708f1241511e4f01310aa541e85898d7325bfdab9d"


def header_logo(height=40) -> QPixmap:
    """Display the exact local transparent PNG at Retina density, full canvas.

    The user's latest wPURERGBLOGO2.png supersedes the previous header SVG.
    Its original transparent margins and PURE RGB wordmark remain intact;
    only smooth display scaling is applied.  Application icons are separate.
    """
    if type(height) is not int or not 1 <= height <= 1024:
        raise ValueError("顶部标识高度须为1至1024的整数")
    payload = HEADER_LOGO_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != HEADER_LOGO_SHA256:
        raise RuntimeError("PURE RGB 顶部标识与指定原图不一致")
    original = QPixmap()
    if not original.loadFromData(payload, "PNG"):
        raise RuntimeError("PURE RGB 顶部PNG无法读取")
    screen = QGuiApplication.primaryScreen()
    ratio = max(2.0, screen.devicePixelRatio() if screen else 2.0)
    pixmap = original.scaledToHeight(
        round(height * ratio), Qt.TransformationMode.SmoothTransformation
    )
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


def create_splash() -> QSplashScreen:
    """Display the exact archived PNG within 720x400, retaining its alpha.

    This archive has been byte-checked against the user's specified
    PURERGBUI/png/Asset-1@2x.png.  Scaling is for display only: there is no
    crop, added text, artwork substitution or rewritten source image.
    """
    payload = SPLASH_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SPLASH_SHA256:
        raise RuntimeError("PURE RGB 启动画面与指定原图不一致")
    original = QPixmap()
    if not original.loadFromData(payload, "PNG"):
        raise RuntimeError("PURE RGB 启动PNG无法读取")
    screen = QGuiApplication.primaryScreen()
    ratio = max(2.0, screen.devicePixelRatio() if screen else 2.0)
    width, height = 720, 400
    if screen:
        available = screen.availableGeometry()
        width = max(1, min(width, int(available.width() * 0.8)))
        height = max(1, min(height, int(available.height() * 0.8)))
    pixmap = original.scaled(
        round(width * ratio),
        round(height * ratio),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    pixmap.setDevicePixelRatio(ratio)
    splash = QSplashScreen(pixmap)
    splash.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    splash.setAccessibleName("PURE RGB 启动画面")
    return splash
