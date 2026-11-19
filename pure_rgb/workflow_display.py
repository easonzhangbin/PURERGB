"""Window-specific display profiles, kept separate from output/proof spaces.

Qt 6.11 Cocoa paints raster widgets into the QNSView's color space and tags
the IOSurface with it. Convert image pixels to that actual surface profile;
ColorSync then composites to the physical monitor. A QImage color-space tag
documents the converted pixels; it is not an extra conversion by QPainter.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import io
import sys
import weakref

from PIL import ImageCms
from PyQt6.QtCore import QEvent, QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QColorSpace
from PyQt6.QtWidgets import QApplication


REFERENCE_SPACES = {
    "srgb": "sRGB",
    "adobe-rgb": "Adobe RGB (1998)",
    "display-p3": "Display P3",
}


def profile_description(icc):
    if not isinstance(icc, bytes) or not 128 <= len(icc) <= 4 * 1024 * 1024:
        raise ValueError("显示ICC内容无效")
    profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
    if profile.profile.xcolor_space.strip() != "RGB":
        raise ValueError("显示ICC必须描述RGB像素")
    return ImageCms.getProfileDescription(profile).strip() or "未命名RGB ICC"


@lru_cache(maxsize=3)
def named_reference_profile(name):
    """Named OUTPUT/proof profiles; these do not identify a physical monitor."""
    from negpy.domain.models import ColorSpace
    from negpy.infrastructure.display.color_mgmt import icc_bytes_for_space

    mapping = {
        "srgb": ColorSpace.SRGB.value,
        "adobe-rgb": ColorSpace.ADOBE_RGB.value,
        "display-p3": ColorSpace.P3_D65.value,
    }
    if name not in mapping:
        raise ValueError("未知输出／软打样色域")
    icc = icc_bytes_for_space(mapping[name])
    if not icc:
        raise ValueError("输出／软打样ICC资源不可用：" + name)
    profile_description(icc)
    return icc


@dataclass(frozen=True)
class DisplayTarget:
    icc: bytes
    description: str
    profile_name: str
    monitor_icc: bytes | None
    monitor_description: str
    screen_name: str
    screen_id: int | None
    source: str
    assumed: bool
    reason: str = ""

    @property
    def signature(self):
        return (
            hashlib.sha256(self.icc).hexdigest(),
            hashlib.sha256(self.monitor_icc).hexdigest() if self.monitor_icc else None,
            self.screen_id,
            self.screen_name,
            self.assumed,
        )


def _cocoa_window_profiles(window):
    """Read only this window; never substitute CGMainDisplayID for its screen."""
    import objc

    view = objc.objc_object(c_void_p=int(window.winId()))
    native_window = view.window()
    screen = native_window.screen() if native_window else None
    if screen is None:
        raise ValueError("窗口尚未关联实际屏幕")
    monitor = screen.colorSpace()
    surface = view.colorSpace()
    return dict(
        monitor_icc=bytes(monitor.ICCProfileData()) if monitor else None,
        surface_icc=bytes(surface.ICCProfileData()) if surface else None,
        screen_name=str(screen.localizedName()),
        screen_id=int(screen.deviceDescription()["NSScreenNumber"]),
    )


def resolve_display_target(window, *, native_reader=None):
    """Read the real paint target. Unknown surfaces are clearly labelled sRGB.

    The injected reader supports deterministic tests without opening or changing
    any user's native window. A detected monitor profile alone does not prove
    that the window's raster backing store uses that same encoding.
    """
    values = {}
    reason = "当前平台未提供窗口绘图ICC"
    try:
        if native_reader is not None:
            values = native_reader(window) or {}
        elif sys.platform == "darwin" and QApplication.platformName() == "cocoa":
            values = _cocoa_window_profiles(window)
        monitor = values.get("monitor_icc")
        monitor_name = "未取得屏幕ICC"
        if monitor:
            try:
                monitor_name = profile_description(monitor)
            except Exception:
                monitor, monitor_name = None, "屏幕ICC无效"
        surface = values.get("surface_icc")
        if surface:
            name = profile_description(surface)
            screen_name = values.get("screen_name", "当前屏幕")
            return DisplayTarget(
                icc=surface,
                description=f"自动显示 · {screen_name} · 绘图：{name}",
                profile_name=name,
                monitor_icc=monitor,
                monitor_description=monitor_name,
                screen_name=screen_name,
                screen_id=values.get("screen_id"),
                source="cocoa-window-surface",
                assumed=False,
            )
        if values:
            reason = "已取得屏幕信息，但窗口绘图ICC尚不可用"
    except Exception as error:
        reason = str(error)
    icc = named_reference_profile("srgb")
    return DisplayTarget(
        icc=icc,
        description="sRGB 显示回退 · 未确认当前绘图ICC",
        profile_name=profile_description(icc),
        monitor_icc=None,
        monitor_description="未确认屏幕ICC",
        screen_name=values.get("screen_name", "当前屏幕"),
        screen_id=values.get("screen_id"),
        source="srgb-fallback",
        assumed=True,
        reason=reason,
    )


def tag_display_image(image, target):
    """Tag pixels already converted to target.icc; never transform them again."""
    color_space = QColorSpace.fromIccProfile(target.icc)
    if not color_space.isValid():
        raise ValueError("Qt无法标记实际显示ICC；不静默改标为P3或sRGB")
    result = image.copy()
    result.setColorSpace(color_space)
    return result


def managed_display_image(record, target):
    """Existing source-ICC→surface conversion, with an accurate QImage tag."""
    from .preview_ui import display_image

    return tag_display_image(display_image(record, target.icc), target)


class DisplayTracker(QObject):
    """Coalesce screen/profile changes and notify the owner to reload pixels."""

    changed = pyqtSignal(object)
    _native_changed = pyqtSignal()

    def __init__(self, window, *, resolver=resolve_display_target):
        super().__init__(window)
        self.window = window
        self.resolver = resolver
        self.target = None
        self.handle = None
        self.observers = []
        self.disposed = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.refresh)
        self._native_changed.connect(self.schedule)
        window.installEventFilter(self)
        window.destroyed.connect(self.dispose)
        self.schedule()

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.Show, QEvent.Type.WinIdChange):
            self.schedule()
        return False

    def schedule(self, *args):
        if not self.disposed:
            self.timer.start()

    def _connect_screen(self):
        handle = self.window.windowHandle()
        if handle is not None and handle is not self.handle:
            if self.handle is not None:
                try:
                    self.handle.screenChanged.disconnect(self.schedule)
                except (RuntimeError, TypeError):
                    pass
            self.handle = handle
            self.handle.screenChanged.connect(self.schedule)
        if (
            self.observers
            or sys.platform != "darwin"
            or QApplication.platformName() != "cocoa"
        ):
            return
        from Foundation import NSNotificationCenter
        import AppKit

        reference = weakref.ref(self)

        def changed(notification):
            tracker = reference()
            if tracker is not None and not tracker.disposed:
                try:
                    tracker._native_changed.emit()
                except RuntimeError:
                    pass

        center = NSNotificationCenter.defaultCenter()
        for name in (
            AppKit.NSScreenColorSpaceDidChangeNotification,
            AppKit.NSApplicationDidChangeScreenParametersNotification,
        ):
            self.observers.append(
                center.addObserverForName_object_queue_usingBlock_(
                    name, None, None, changed
                )
            )

    def refresh(self):
        if self.disposed:
            return
        self.timer.stop()
        self._connect_screen()
        target = self.resolver(self.window)
        if self.target is None or target.signature != self.target.signature:
            self.target = target
            self.changed.emit(target)

    def dispose(self, *args):
        if self.disposed:
            return
        self.disposed = True
        try:
            self.timer.stop()
        except RuntimeError:
            pass
        if self.observers:
            from Foundation import NSNotificationCenter

            center = NSNotificationCenter.defaultCenter()
            for observer in self.observers:
                center.removeObserver_(observer)
            self.observers.clear()
        try:
            self.window.removeEventFilter(self)
            if self.handle is not None:
                self.handle.screenChanged.disconnect(self.schedule)
        except (RuntimeError, TypeError):
            pass
        self.handle = None
