"""Optional SwiftUI ultra-thin material bridge for the native window backdrop."""

import ctypes
from pathlib import Path


LIBRARY_PATH = Path(__file__).resolve().parent / "libPUREUltraThinMaterial.dylib"
_library = None


def make_ultrathin_view():
    """Return a passive NSView or a diagnostic reason for the AppKit fallback."""
    global _library
    if not LIBRARY_PATH.is_file():
        return None, "swiftui-material-library-unavailable"
    try:
        import objc

        if _library is None:
            _library = ctypes.CDLL(str(LIBRARY_PATH), mode=ctypes.RTLD_GLOBAL)
        view = objc.lookUpClass("PUREUltraThinBridge").makeView()
        if view is None:
            return None, "swiftui-material-view-unavailable"
        return view, None
    except Exception as error:
        return None, f"swiftui-material-error: {error}"
