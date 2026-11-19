"""PURE RGB 品牌资产，供工作台与预览界面共用。

2026-11-17 版本把 SVG 展平为纯 path（无 mix-blend-mode、无 filter、无内嵌位图），
QtSvg 可稳定渲染，字形在任意尺寸下保持矢量锐利。
"""

from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QGuiApplication, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

ROOT = Path(__file__).resolve().parents[1]
FLAT = ROOT / "assets/branding/deliveries/2026-11-17-flat"
WORDMARK = FLAT / "PURERGBLOGO2.flat.svg"
WORDMARK_DARK = FLAT / "PURERGBLOGO2.dark.svg"
MARK = FLAT / "mark.flat.svg"
ICON_FALLBACK = FLAT / "app_1024.png"
LOCKUP = FLAT / "lockup_about@2x.png"
LOCKUP_DARK = FLAT / "lockup_about_dark@2x.png"
SPLASH = FLAT / "splash@2x.png"
# PURE Neg 原版字标（含辉光分隔线），除尘审核窗品牌位使用。
PURENEG_WORDMARK = ROOT / "assets/branding/originals/pureneg-wordmark.png"

# 逻辑尺寸 -> iconset 文件；@2x 按实际像素登记，Retina 下取原生清晰度。
ICON_FILES = (
    (16, "icon_16x16.png"),
    (32, "icon_16x16@2x.png"),
    (32, "icon_32x32.png"),
    (64, "icon_32x32@2x.png"),
    (128, "icon_128x128.png"),
    (256, "icon_128x128@2x.png"),
    (256, "icon_256x256.png"),
    (512, "icon_256x256@2x.png"),
    (512, "icon_512x512.png"),
    (1024, "icon_512x512@2x.png"),
)


def application_icon():
    """多尺寸应用图标（Apple 网格版），缺失时回退到 1024 主图。"""
    icon = QIcon()
    for size, name in ICON_FILES:
        path = FLAT / "app.iconset" / name
        if path.is_file():
            icon.addFile(str(path), QSize(size, size))
    if icon.isNull() and ICON_FALLBACK.is_file():
        icon.addFile(str(ICON_FALLBACK))
    if icon.isNull() or icon.pixmap(32, 32).isNull():
        raise RuntimeError("PURE RGB 应用图标资源缺失或无法读取")
    return icon


def _ratio() -> float:
    screen = QGuiApplication.primaryScreen()
    return screen.devicePixelRatio() if screen else 1.0


def _render(svg: Path, height: int, ratio: float) -> QPixmap:
    if not svg.is_file():
        raise RuntimeError(f"PURE RGB 品牌矢量缺失：{svg}")
    renderer = QSvgRenderer(str(svg))
    if not renderer.isValid():
        raise RuntimeError(f"PURE RGB 品牌矢量无法解析：{svg}")
    default = renderer.defaultSize()
    width = round(default.width() * height / max(1, default.height()))
    pixmap = QPixmap(round(width * ratio), round(height * ratio))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


def wordmark_pixmap(height=22, *, dark=False):
    """水平字标；dark=True 使用深色 PURE，适合浅色玻璃底。"""
    return _render(WORDMARK_DARK if dark else WORDMARK, height, _ratio())


def mark_pixmap(height=20, *, dark=False):
    """仅图标（不含 PURE RGB 字标）。"""
    return _render(MARK, height, _ratio())


def wordmark_label(height=22, *, dark=False):
    """工具栏字标 QLabel：按设备像素比渲染，避免位图拉伸。"""
    from PyQt6.QtWidgets import QLabel

    widget = QLabel()
    pixmap = wordmark_pixmap(height, dark=dark)
    widget.setPixmap(pixmap)
    widget.setFixedSize(pixmap.deviceIndependentSize().toSize())
    widget.setAccessibleName("PURE RGB")
    widget.setToolTip("PURE RGB")
    return widget


def lockup_pixmap(*, dark=False):
    """关于页横向组合标（图标 + 字标），2x 位图按逻辑尺寸显示。"""
    path = LOCKUP_DARK if dark else LOCKUP
    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        raise RuntimeError(f"PURE RGB 关于页组合标缺失：{path}")
    pixmap.setDevicePixelRatio(2.0)
    return pixmap


def pureneg_wordmark_pixmap(height=24):
    """PURE Neg 原版字标位图，深色工具栏上的品牌位使用。"""
    source = QPixmap(str(PURENEG_WORDMARK))
    if source.isNull():
        raise RuntimeError(f"PURE Neg 字标缺失：{PURENEG_WORDMARK}")
    ratio = _ratio()
    scaled = source.scaled(
        round(source.width() * height * ratio / source.height()),
        round(height * ratio),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    scaled.setDevicePixelRatio(ratio)
    return scaled


def pureneg_wordmark_label(height=24):
    """除尘审核窗顶栏的字标 QLabel；调用方自带“Powered by”前缀文案。"""
    from PyQt6.QtWidgets import QLabel

    widget = QLabel()
    pixmap = pureneg_wordmark_pixmap(height)
    widget.setPixmap(pixmap)
    widget.setFixedSize(pixmap.deviceIndependentSize().toSize())
    widget.setAccessibleName("PURE Neg")
    widget.setToolTip("Powered by PURE Neg")
    return widget


def pureneg_powered_label(height=24):
    """除尘审核窗顶栏“Powered by PURE Neg”品牌位。

    字标资源缺失时退化为纯文字，绝不因品牌资产问题阻断审核窗口。
    """
    from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

    widget = QWidget()
    row = QHBoxLayout(widget)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(5)
    prefix = QLabel("Powered by")
    prefix.setObjectName("reviewPowered")
    row.addWidget(prefix)
    try:
        row.addWidget(pureneg_wordmark_label(height))
    except (OSError, RuntimeError):
        name = QLabel("PURE Neg")
        name.setObjectName("reviewPowered")
        row.addWidget(name)
    return widget


def splash_pixmap(width=360):
    """启动画面：深色底板图标，按逻辑宽度缩放并保持方形。"""
    source = QPixmap(str(SPLASH))
    if source.isNull():
        raise RuntimeError(f"PURE RGB 启动画面缺失：{SPLASH}")
    ratio = _ratio()
    scaled = source.scaled(
        round(width * ratio),
        round(width * ratio),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    scaled.setDevicePixelRatio(ratio)
    return scaled
