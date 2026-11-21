"""Select one native glass surface for windows, panels, controls and popups."""

import ctypes
from pathlib import Path


LIBRARY_PATH = Path(__file__).resolve().parent / "libPUREUltraThinMaterial.dylib"
_library = None


def make_swiftui_view(*, thick=False):
    """Return a passive NSView or a diagnostic reason for the AppKit fallback."""
    global _library
    if not LIBRARY_PATH.is_file():
        return None, "swiftui-material-library-unavailable"
    try:
        import objc

        if _library is None:
            _library = ctypes.CDLL(str(LIBRARY_PATH), mode=ctypes.RTLD_GLOBAL)
        bridge = objc.lookUpClass("PUREUltraThinBridge")
        view = bridge.makeThickView() if thick else bridge.makeView()
        if view is None:
            return None, "swiftui-material-view-unavailable"
        return view, None
    except Exception as error:
        return None, f"swiftui-material-error: {error}"


def make_material(api, frame, fallback, *, behind=False, window_base=False):
    """Reserve the thicker material for window backgrounds, including title bars."""
    if not window_base and hasattr(api, "NSGlassEffectViewStyleClear") and hasattr(api, "NSGlassEffectView"):
        effect = api.NSGlassEffectView.alloc().initWithFrame_(frame)
        effect.setStyle_(api.NSGlassEffectViewStyleClear)
        return effect, "clear", None
    effect, reason = make_swiftui_view(thick=window_base)
    if effect is not None:
        effect.setFrame_(frame)
        return effect, "thickMaterial" if window_base else "ultraThinMaterial", None
    effect = api.NSVisualEffectView.alloc().initWithFrame_(frame)
    effect.setMaterial_(fallback)
    effect.setBlendingMode_(api.NSVisualEffectBlendingModeBehindWindow if behind
                            else api.NSVisualEffectBlendingModeWithinWindow)
    effect.setState_(api.NSVisualEffectStateActive if behind else api.NSVisualEffectStateFollowsWindowActiveState)
    return effect, "visualEffect", reason


def set_material_radius(effect, material, radius):
    if material == "clear":
        effect.setCornerRadius_(radius)
    else:
        effect.setWantsLayer_(True)
        effect.layer().setCornerRadius_(radius)
        effect.layer().setMasksToBounds_(True)
