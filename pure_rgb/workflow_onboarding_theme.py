"""System typography and semantic colors for the first-use content surface."""

import sys

from PyQt6.QtGui import QColor, QFontDatabase
from PyQt6.QtWidgets import QApplication

from .workflow_glass import system_dark


def body_font():
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)


def _luminance(value):
    color = QColor(value)
    parts = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4
             for v in (color.redF(), color.greenF(), color.blueF())]
    return sum(part * weight for part, weight in zip(parts, (.2126, .7152, .0722)))


def accent_text(accent):
    """Select readable text even for a user's yellow or custom accent."""
    level = _luminance(accent)
    return "#ffffff" if 1.05 / (level + .05) >= (level + .05) / .05 else "#000000"


def theme_tokens(dark=None):
    """Explicit modes support previews; normal use resolves macOS NSColor roles."""
    explicit = dark is not None
    dark = system_dark() if dark is None else bool(dark)
    result = dict(
        dark=dark,
        background="#1c1c1e" if dark else "#f5f5f7",
        surface="#1c1c1e" if dark else "#f5f5f7",
        card="#2c2c2e" if dark else "#ffffff",
        edge="#424245" if dark else "#dedee3",
        text="#f5f5f7" if dark else "#1d1d1f",
        secondary="#a1a1a6" if dark else "#6e6e73",
        accent="#0a84ff" if dark else "#007aff",
        link="#419cff" if dark else "#0066cc",
        selected="#505054" if dark else "#e6e6ea",
        red="#ff453a" if dark else "#ff3b30",
        green="#30d158" if dark else "#34c759",
        blue="#0a84ff" if dark else "#007aff",
    )
    app = QApplication.instance()
    if not explicit and sys.platform == "darwin" and app is not None and app.platformName() == "cocoa":
        try:
            from AppKit import NSColor, NSColorSpace

            def color(name, backdrop=None):
                value = getattr(NSColor, name)().colorUsingColorSpace_(NSColorSpace.sRGBColorSpace())
                if value is None:
                    raise ValueError("Unavailable semantic color")
                channels = [value.redComponent(), value.greenComponent(), value.blueComponent()]
                if backdrop is not None and value.alphaComponent() < 1:
                    base = QColor(backdrop)
                    alpha = value.alphaComponent()
                    channels = [v * alpha + b * (1 - alpha)
                                for v, b in zip(channels, (base.redF(), base.greenF(), base.blueF()))]
                return QColor.fromRgbF(*(max(0, min(1, v)) for v in channels)).name()

            background = color("windowBackgroundColor")
            result.update(background=background,
                          surface=background, card=color("controlBackgroundColor", background),
                          text=color("labelColor", background), secondary=color("secondaryLabelColor", background),
                          accent=color("controlAccentColor"), link=color("linkColor"), red=color("systemRedColor"),
                          green=color("systemGreenColor"), blue=color("systemBlueColor"),
                          edge=color("separatorColor", background),
                          selected=color("unemphasizedSelectedContentBackgroundColor", background))
        except (ImportError, AttributeError, ValueError, TypeError):
            pass
    fill = QColor(result["accent"])
    # Keep the familiar white label on the standard blue control, while light
    # custom accents (for example yellow) retain their color and get dark text.
    if _luminance(fill) > (1.05 / 4.5 - .05):
        muted_fill = fill.darker(114)
        if _luminance(muted_fill) <= (1.05 / 4.5 - .05):
            fill = muted_fill
    result["button_fill"] = fill.name()
    result["on_accent"] = accent_text(result["button_fill"])
    result["button_hover"] = (
        fill.darker(110) if result["on_accent"] == "#ffffff" else fill.lighter(110)
    ).name()
    return result
