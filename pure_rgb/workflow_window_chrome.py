"""Extend the existing native backdrop through the real macOS title bar.

Qt's QNSView excludes the native title-bar height.  Its NSThemeFrame parent
includes that area, so a backdrop sized only to QNSView leaves transparent
traffic lights over the desktop when titlebarAppearsTransparent is enabled.
This helper changes only the passive background, preserving native window
chrome, traffic lights, dragging, resize handles and Qt content coordinates.
"""


def _rectangle(rect):
    return [
        float(rect.origin.x),
        float(rect.origin.y),
        float(rect.size.width),
        float(rect.size.height),
    ]


def attach_chrome(ns_window, view, container, baseeffect):
    """Fill the existing theme frame, called after attaching the base effect."""
    report = dict(mode="native-titlebar-continuity", attached=False)
    if any(item is None for item in (ns_window, view, container, baseeffect)):
        return {**report, "reason": "native-chrome-not-attached"}
    try:
        # Refuse another window's objects rather than relocating native controls.
        if (
            view.window() != ns_window
            or view.superview() != container
            or baseeffect.superview() != container
        ):
            return {**report, "reason": "native-chrome-hierarchy-mismatch"}
        bounds = container.bounds()
        if bounds.size.width <= 0 or bounds.size.height <= 0:
            return {**report, "reason": "empty-native-theme-frame"}
        from AppKit import NSColor, NSViewWidthSizable, NSViewHeightSizable

        old = _rectangle(baseeffect.frame())
        content = _rectangle(view.frame())
        baseeffect.setFrame_(bounds)
        baseeffect.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        # The native effect supplies both title bar and content material.
        # An opaque NSWindow background blocks its behind-window sampling.
        ns_window.setBackgroundColor_(NSColor.clearColor())
        ns_window.setTitlebarAppearsTransparent_(True)
        return dict(
            report,
            attached=True,
            previous_backdrop_frame=old,
            backdrop_frame=_rectangle(baseeffect.frame()),
            theme_frame=_rectangle(bounds),
            qt_content_frame=content,
            titlebar_extension_points=max(0.0, float(bounds.size.height) - content[3]),
            native_traffic_lights_preserved=True,
            qt_content_geometry_changed=False,
        )
    except Exception as error:
        return {**report, "reason": str(error)}
