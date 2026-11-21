"""macOS NSVisualEffectView behind Qt's content, with an opaque fallback."""

import sys
import weakref

from PyQt6.QtCore import QEvent, QObject, Qt, QTimer
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QColorDialog,
    QDialog,
    QFileDialog,
    QMenu,
    QMessageBox,
    QWidget,
)

WORKSPACE_INSET = 16
WORKSPACE_GAP = 12
MENU_RADIUS = 12
WORKFLOW_ACCENT = "#d93722"

# Attribute-guarded transparency appended to every tracked surface so the
# frosted native material shows through regardless of the active stylesheet.
_GLASS_SURFACE_QSS = """
QDialog#workflowRoot[nativeMaterial="true"], QDialog[nativeMaterial="true"], QMessageBox[nativeMaterial="true"] { background:transparent; }
QMenu[nativeMaterial="true"] { background:transparent; border-color:transparent; }
QComboBoxPrivateContainer[nativeMaterial="true"] { background:transparent; border-color:transparent; }
QComboBox QAbstractItemView[nativeMaterial="true"] { background:transparent; border-color:transparent; }
QComboBox QAbstractItemView[nativeMaterial="true"] QWidget { background:transparent; }
"""

NATIVE_SLIDER_QSS = """
QSlider[nativeMaterial="true"]::handle:horizontal,
QSlider[nativeMaterial="true"]::handle:vertical,
QSlider[nativeMaterial="true"]::handle:horizontal:disabled,
QSlider[nativeMaterial="true"]::handle:vertical:disabled {
    background:transparent; border-color:transparent;
}
"""


def native_red_gradient(frame, *, backdrop=False):
    """A passive Core Animation tint; the underlying AppKit glass stays native."""
    from Quartz import CAGradientLayer, CGColorCreateGenericRGB

    gradient = CAGradientLayer.layer()
    gradient.setFrame_(frame)
    gradient.setAutoresizingMask_(18)
    if backdrop:
        # A faint warm edge light over PURE Neg's existing black 0.26 veil.
        # The photograph canvas paints its own opaque neutral background.
        gradient.setType_("radial")
        gradient.setStartPoint_((0.08, 0.96))
        gradient.setEndPoint_((0.94, 0.12))
        colors = ((1.0, 0.36, 0.10, 0.10), (1.0, 0.16, 0.06, 0.035), (0.95, 0.10, 0.05, 0.0))
        gradient.setLocations_((0.0, 0.45, 1.0))
    else:
        gradient.setStartPoint_((0.0, 1.0))
        gradient.setEndPoint_((1.0, 0.0))
        colors = ((1.0, 0.40, 0.12, 0.20), (1.0, 0.20, 0.08, 0.12), (0.98, 0.13, 0.08, 0.18))
        gradient.setLocations_((0.0, 0.48, 1.0))
    gradient.setColors_([CGColorCreateGenericRGB(*color) for color in colors])
    return gradient


def native_slider_highlight(frame):
    """A faint passive edge light over the slider's actual AppKit glass."""
    from Quartz import CAGradientLayer, CGColorCreateGenericRGB

    gradient = CAGradientLayer.layer()
    gradient.setFrame_(frame)
    gradient.setAutoresizingMask_(18)
    gradient.setStartPoint_((0.0, 1.0))
    gradient.setEndPoint_((0.0, 0.0))
    gradient.setLocations_((0.0, 0.42, 1.0))
    gradient.setColors_([CGColorCreateGenericRGB(1.0, 1.0, 1.0, alpha)
                         for alpha in (0.12, 0.045, 0.012)])
    gradient.setBorderColor_(CGColorCreateGenericRGB(1.0, 1.0, 1.0, 0.20))
    gradient.setBorderWidth_(0.75)
    gradient.setOpacity_(0.0)
    return gradient


def prepare_glass(window):
    if sys.platform == "darwin" and QApplication.platformName() == "cocoa":
        window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)


def attach_glass(window, *, base="under", veil=None):
    """Place the high-frost background behind the Qt window and its title bar."""
    result = dict(mode="solid", attached=False)
    if getattr(window, "closing", False):
        return {**result, "reason": "window-closing"}
    if sys.platform != "darwin" or QApplication.platformName() != "cocoa":
        return result
    try:
        import objc
        import AppKit
        from AppKit import (
            NSColor,
            NSVisualEffectMaterialHUDWindow,
            NSVisualEffectMaterialUnderWindowBackground,
            NSVisualEffectMaterialMenu,
            NSVisualEffectMaterialSidebar,
            NSView,
            NSWindowBelow,
        )

        view = objc.objc_object(c_void_p=int(window.winId()))
        ns_window = view.window()
        container = view.superview()
        if ns_window is None or container is None:
            raise RuntimeError("Qt native content view is not attached")
        materials = {
            "hud": NSVisualEffectMaterialHUDWindow,
            "under": NSVisualEffectMaterialUnderWindowBackground,
            "menu": NSVisualEffectMaterialMenu,
            "sidebar": NSVisualEffectMaterialSidebar,
        }
        from .workflow_ultrathin import make_material, set_material_radius
        effect, material, reason = make_material(
            AppKit, view.frame(), materials.get(base, NSVisualEffectMaterialUnderWindowBackground),
            behind=True, window_base=True)
        effect.setAutoresizingMask_(18)
        set_material_radius(effect, material, 0)
        brand_veil = None
        if veil:
            brand_veil = NSView.alloc().initWithFrame_(effect.bounds())
            brand_veil.setAutoresizingMask_(18)
            brand_veil.setWantsLayer_(True)
            brand_veil.setLayer_(native_red_gradient(brand_veil.bounds(), backdrop=True))
            brand_veil.layer().setBackgroundColor_(
                NSColor.blackColor().colorWithAlphaComponent_(veil).CGColor()
            )
            effect.addSubview_(brand_veil)
        # QWidget.winId() is an NSView, not an NSWindow. Keep the Qt view on top.
        container.addSubview_positioned_relativeTo_(effect, NSWindowBelow, view)
        ns_window.setOpaque_(False)
        ns_window.setBackgroundColor_(NSColor.clearColor())
        ns_window.setTitlebarAppearsTransparent_(True)
        from .workflow_window_chrome import attach_chrome

        chrome = attach_chrome(ns_window, view, container, effect)
        window._native_glass = effect
        window._native_glass_veil = brand_veil
        return dict(
            mode="native-vibrancy",
            attached=True,
            window_number=int(ns_window.windowNumber()),
            native_class=str(effect.className()),
            accent_theme="orange-red" if veil else "system",
            subtle_native_gradient=bool(veil),
            base_material=base,
            material=material,
            swiftui_material=material if material == "thickMaterial" else None,
            swiftui_reason=reason,
            veil_alpha=veil,
            chrome=chrome,
        )
    except Exception as error:
        return {**result, "reason": str(error)}


def attach_popup_glass(popup, *, base="menu"):
    """Frost a borderless Qt popup (combo container, menu) behind its content.

    Popup windows host the QNSView as the window's content view, so there is no
    sibling container below it: the effect takes over the content view and keeps
    the original QNSView on top at the same frame. detach_popup_glass restores
    the previous hierarchy while the native window still exists.
    """
    result = dict(mode="solid", attached=False)
    if getattr(popup, "closing", False):
        return {**result, "reason": "window-closing"}
    if sys.platform != "darwin" or QApplication.platformName() != "cocoa":
        return result
    try:
        import objc
        import AppKit
        from AppKit import (
            NSColor,
            NSVisualEffectMaterialMenu,
            NSVisualEffectMaterialPopover,
            NSWindowAbove,
            NSWindowBelow,
        )

        view = objc.objc_object(c_void_p=int(popup.winId()))
        ns_window = view.window()
        if ns_window is None:
            return {**result, "reason": "native-window-not-attached"}
        materials = {
            "menu": NSVisualEffectMaterialMenu,
            "popover": NSVisualEffectMaterialPopover,
        }
        frame = view.frame()
        from .workflow_ultrathin import make_material, set_material_radius
        local_frame = ((0, 0), (frame.size.width, frame.size.height))
        # The Qt view stays a sibling of the effect. AppKit only guarantees
        # drawing order for NSGlassEffectView.contentView, not arbitrary children.
        backdrop = AppKit.NSView.alloc().initWithFrame_(local_frame)
        backdrop.setAutoresizingMask_(18)
        effect, material, reason = make_material(
            AppKit, local_frame, materials.get(base, NSVisualEffectMaterialMenu), behind=True)
        effect.setAutoresizingMask_(18)
        set_material_radius(effect, material, MENU_RADIUS)
        backdrop.addSubview_(effect)
        surfaces = [effect]
        if hasattr(popup, "glass_regions"):
            second, _, _ = make_material(AppKit, local_frame, AppKit.NSVisualEffectMaterialMenu, behind=True)
            backdrop.addSubview_(second)
            surfaces.append(second)
        workspace = AppKit.NSWorkspace.sharedWorkspace()
        solid = (workspace.accessibilityDisplayShouldReduceTransparency()
                 or workspace.accessibilityDisplayShouldIncreaseContrast())
        if solid:
            for surface in surfaces:
                surface.setWantsLayer_(True)
                surface.layer().setBackgroundColor_(NSColor.windowBackgroundColor().CGColor())
        container = view.superview()
        swapped = container is None
        if swapped:
            backdrop.addSubview_positioned_relativeTo_(view, NSWindowAbove, None)
            ns_window.setContentView_(backdrop)
        else:
            container.addSubview_positioned_relativeTo_(backdrop, NSWindowBelow, view)
            backdrop.setFrame_(frame)
        ns_window.setOpaque_(False)
        ns_window.setBackgroundColor_(NSColor.clearColor())
        popup._popup_glass = dict(
            effect=backdrop, surfaces=surfaces, material=material, view=view, window=ns_window, swapped=swapped
        )
        sync_popup_materials(popup)
        return dict(
            mode="native-popup-vibrancy",
            attached=True,
            window_number=int(ns_window.windowNumber()),
            native_class=str(effect.className()),
            base_material=base,
            material=material,
            swiftui_reason=reason,
            content_view_swapped=swapped,
            corner_radius=MENU_RADIUS,
        )
    except Exception as error:
        return {**result, "reason": str(error)}


def sync_popup_materials(popup):
    """Follow the spring trigger and list without frosting their transparent gutters."""
    state = getattr(popup, "_popup_glass", None)
    if not state or not hasattr(popup, "glass_regions"):
        return
    regions = popup.glass_regions()
    if state.get("regions") == regions:
        return
    state["regions"] = regions
    from .workflow_ultrathin import set_material_radius
    from Quartz import CATransaction
    CATransaction.begin()
    CATransaction.setDisableActions_(True)
    try:
        for effect, (rect, radius, alpha) in zip(state["surfaces"], regions):
            y = rect.y() if state["effect"].isFlipped() else popup.height() - rect.bottom()
            effect.setFrame_(((rect.x(), y), (rect.width(), rect.height())))
            set_material_radius(effect, state["material"], radius)
            effect.setAlphaValue_(alpha)
    finally:
        CATransaction.commit()


def detach_popup_glass(popup):
    """Restore a popup's original content view before its window disappears."""
    state = getattr(popup, "_popup_glass", None)
    if state is None:
        return
    popup._popup_glass = None
    try:
        effect, view, ns_window = state["effect"], state["view"], state["window"]
        if state.get("swapped"):
            if ns_window.contentView() is effect:
                ns_window.setContentView_(view)
        else:
            effect.removeFromSuperview()
    except Exception:
        # A closed popup already released its native window and views.
        pass


def detach_glass(window):
    """Remove passive Cocoa siblings before Qt destroys its content view."""
    binder = getattr(window, "_workflow_material_binder", None)
    if binder is not None:
        binder._disable()
        binder._dispose_native()
        del window._workflow_material_binder
        binder.deleteLater()
    effect = getattr(window, "_native_glass", None)
    if effect is not None:
        effect.removeFromSuperview()
        window._native_glass = None
        window._native_glass_veil = None


def _set_native_material(widget, active):
    if widget.property("nativeMaterial") != active:
        widget.setProperty("nativeMaterial", active)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


class GlassSurfaces(QObject):
    """Frost every Qt-owned surface the window shells do not manage themselves.

    ChildAdded preparation runs before the native window exists, so dialogs,
    alerts, menus and combo popups can become translucent at creation; the
    first Show attaches vibrancy and every Hide detaches it while the native
    window is still alive.  Surface keys are held strongly: sip drops unrefer
    -enced wrappers of C++-owned widgets, which would silently untrack popups.
    """

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.surfaces = {}
        app.installEventFilter(self)

    @staticmethod
    def _classify(surface):
        if isinstance(surface, (QFileDialog, QColorDialog)):
            return None  # Native file/colour panels own their materials.
        if isinstance(surface, (QMessageBox, QDialog)):
            return "dialog"
        if isinstance(surface, QMenu) or surface.property("glassPopup"):
            return "popup"
        if surface.metaObject().className() == "QComboBoxPrivateContainer":
            return "popup"
        return None

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind == QEvent.Type.ChildAdded:
            surface = event.child()
            if isinstance(surface, QWidget):
                role = self._classify(surface)
                if role is not None:
                    self._track(surface, role)
                elif (
                    isinstance(watched, QWidget)
                    and watched.metaObject().className() == "QComboBoxPrivateContainer"
                ):
                    # Popup containers are Qt windows: they never arrive as a
                    # child event themselves, but each view they adopt does.
                    self._track(watched, "popup")
        elif kind == QEvent.Type.Show and isinstance(watched, QWidget) and watched.isWindow():
            # QDialog/QMenu window-children never emit ChildAdded either; catch
            # them at Show. Translucency only frosts when their own constructor
            # called prepare_glass (GlassMessageBox/GlassMenu/our dialogs).
            role = self._classify(watched)
            if role is not None:
                if watched not in self.surfaces:
                    self._track(watched, role)
                reference = weakref.ref(watched)
                QTimer.singleShot(0, lambda: self._shown(reference()))
        elif kind == QEvent.Type.Hide and watched in self.surfaces:
            self._hidden(watched)
        return False

    def _track(self, surface, role):
        if surface in self.surfaces:
            return
        self.surfaces[surface] = dict(role=role, attached=False)
        surface.destroyed.connect(self._forget)
        prepare_glass(surface)
        if role == "popup" and not surface.testAttribute(Qt.WidgetAttribute.WA_WState_Created):
            surface.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        surface.setStyleSheet((surface.styleSheet() or "") + _GLASS_SURFACE_QSS)

    def _forget(self, surface=None):
        """destroyed() delivers the dying QObject wrapper; drop its entry."""
        if surface is not None:
            self.surfaces.pop(surface, None)

    def _shown(self, surface):
        if surface is None:
            return
        state = self.surfaces.get(surface)
        if (
            state is None
            or state["attached"]
            or not surface.isWindow()
            or not surface.isVisible()
        ):
            return
        if state["role"] == "popup":
            report = attach_popup_glass(surface)
        else:
            report = attach_glass(surface, base="under", veil=None)
            if report.get("attached"):
                from .workflow_materials import attach_materials

                report["controls"] = attach_materials(surface)
        state["attached"] = bool(report.get("attached"))
        if state["attached"]:
            state["report"] = report
            _set_native_material(surface, True)
            if state["role"] == "popup":
                self._set_popup_content(surface, True)

    @staticmethod
    def _set_popup_content(surface, active):
        for view in surface.findChildren(QAbstractItemView):
            _set_native_material(view, active)
            viewport = view.viewport()
            # The view and viewport each cache their resolved style. Repolish
            # both so the popup's Base brush does not cover the AppKit layer.
            _set_native_material(viewport, active)

    def _hidden(self, surface):
        state = self.surfaces.get(surface)
        if state is None or not state["attached"]:
            return
        state["attached"] = False
        _set_native_material(surface, False)
        if state["role"] == "popup":
            self._set_popup_content(surface, False)
            detach_popup_glass(surface)
        else:
            detach_glass(surface)

    def report(self):
        states = list(self.surfaces.values())
        return dict(
            mode="glass-surfaces",
            dialog_count=sum(state["role"] == "dialog" for state in states),
            popup_count=sum(state["role"] == "popup" for state in states),
            attached_count=sum(bool(state["attached"]) for state in states),
        )


def install_glass_surfaces(app):
    """Install the app-wide surface froster once; no-op outside Cocoa."""
    existing = getattr(app, "_glass_surfaces", None)
    if existing is not None and not sip_deleted(existing):
        return existing
    if sys.platform != "darwin" or app.platformName() != "cocoa":
        return None
    surfaces = GlassSurfaces(app)
    app._glass_surfaces = surfaces
    return surfaces


def sip_deleted(obj):
    try:
        from PyQt6 import sip

        return sip.isdeleted(obj)
    except Exception:
        return False


class GlassMenu(QMenu):
    """A QMenu that turns translucent before its popup window ever exists."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prepare_glass(self)


class GlassMessageBox(QMessageBox):
    """A QMessageBox that turns translucent before its native window exists.

    Drop-in statics mirror QMessageBox.warning/critical/information/question/
    about so existing call sites keep their arguments and StandardButton API.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prepare_glass(self)

    @staticmethod
    def _present(icon, parent, title, text, buttons=None, default=None, button_labels=None,
                 *, use_widget_dialog=False):
        box = GlassMessageBox(parent)
        if use_widget_dialog:
            box.setOption(QMessageBox.Option.DontUseNativeDialog)
        box.setWindowTitle(title)
        box.setText(text)
        box.setIcon(icon)
        box.setStandardButtons(buttons if buttons is not None else QMessageBox.StandardButton.Ok)
        for standard, label in (button_labels or {}).items():
            box.button(standard).setText(label)
        if default is not None:
            box.setDefaultButton(default)
        from .workflow_attention import (
            ATTENTION_TEXT_FLAGS, ATTENTION_TEXT_INSET, attach_attention, attention_font,
        )

        for control in box.buttons():
            if box.buttonRole(control) in (QMessageBox.ButtonRole.AcceptRole, QMessageBox.ButtonRole.YesRole):
                control.ensurePolished()
                width = QFontMetrics(attention_font(control)).size(ATTENTION_TEXT_FLAGS, control.text()).width()
                control.setMinimumWidth(max(control.minimumWidth(), width + 2 * (ATTENTION_TEXT_INSET + 1)))
                attach_attention(control)
        box.exec()
        clicked = box.clickedButton()
        if clicked is None:
            return QMessageBox.StandardButton.NoButton
        return box.standardButton(clicked)

    @classmethod
    def warning(cls, parent, title, text, buttons=None, default=None):
        return cls._present(QMessageBox.Icon.Warning, parent, title, text,
                            buttons or QMessageBox.StandardButton.Ok, default)

    @classmethod
    def critical(cls, parent, title, text, buttons=None, default=None):
        return cls._present(QMessageBox.Icon.Critical, parent, title, text,
                            buttons or QMessageBox.StandardButton.Ok, default)

    @classmethod
    def information(cls, parent, title, text, buttons=None, default=None):
        return cls._present(QMessageBox.Icon.Information, parent, title, text,
                            buttons or QMessageBox.StandardButton.Ok, default)

    @classmethod
    def question(cls, parent, title, text, buttons=None, default=None, *, button_labels=None,
                 use_widget_dialog=False):
        return cls._present(
            QMessageBox.Icon.Question, parent, title, text,
            buttons or (QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No),
            default,
            button_labels,
            use_widget_dialog=use_widget_dialog,
        )

    @classmethod
    def about(cls, parent, title, text):
        return cls._present(QMessageBox.Icon.Information, parent, title, text)


def system_dark():
    app = QApplication.instance()
    return app is None or app.platformName() != "cocoa" or app.styleHints().colorScheme() != Qt.ColorScheme.Light


def system_reduce_motion():
    if QApplication.platformName() == "cocoa":
        from AppKit import NSWorkspace

        return bool(NSWorkspace.sharedWorkspace().accessibilityDisplayShouldReduceMotion())
    return False


def stylesheet(glass=False, dark=None):
    if dark is None:
        dark = system_dark()
    text, muted = ("#f2ede4", "#c4c4c4") if dark else ("#1d1d1f", "#60646c")
    background = "transparent" if glass else ("#242322" if dark else "#e8eaee")
    panel = (
        "rgba(38,37,36,30)"
        if glass and dark
        else "rgba(249,250,253,48)"
        if glass
        else "#2b2927"
        if dark
        else "#f7f8fa"
    )
    field = "rgba(255,255,255,22)" if dark else "#ffffff"
    accent = "#ff8052" if dark else "#c93a18"
    edge = "rgba(255,255,255,36)" if dark else "rgba(32,38,48,32)"
    selected = "rgba(255,72,34,48)" if dark else "rgba(222,72,34,40)"
    primary = f"qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #df361a,stop:0.48 {WORKFLOW_ACCENT},stop:1 #ca2415)"
    popup = "#302f2e" if dark else "#f7f8fa"
    slider_edge = "rgba(255,255,255,74)" if dark else "rgba(32,38,48,72)"
    slider_hover = "rgba(255,255,255,30)" if dark else "rgba(255,255,255,235)"
    return f"""
QWidget {{ color:{text}; font-size:13px; }}
QMainWindow, QWidget#workflowRoot {{ background:{background}; }}
QFrame#glassPanel {{ background:{panel}; border:1px solid {edge}; border-radius:16px; }}
QFrame#workflowHeader, QFrame#workflowFooter {{ background:{panel}; border:1px solid {edge}; border-radius:12px; }}
QFrame#previewToolbar {{ background:{panel}; border:1px solid {edge}; border-radius:12px; }}
QTabWidget::pane {{ background:transparent; border:0; }}
QTabBar {{ background:transparent; }}
QTabBar::tab {{ background:{field}; color:{muted}; border:1px solid {edge}; border-radius:8px; padding:5px 8px; margin:1px; }}
QTabBar::tab:selected {{ background:{selected}; color:{accent}; }}
QLabel {{ background:transparent; }}
QLabel#muted {{ color:{muted}; font-size:12px; }}
QLabel#section {{ color:{text}; font-weight:600; }}
QLabel#sliderLabel {{ color:{muted}; font-size:12px; }}
QLabel#sliderValue {{ color:{accent}; font-size:12px; }}
QPushButton {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 rgba(255,255,255,30),stop:1 rgba(255,255,255,12)); border:1px solid {edge}; border-radius:14px; padding:4px 8px; min-height:20px; }}
QPushButton[orientationControl="true"] {{ padding:0; min-height:28px; }}
QPushButton:hover {{ border-color:{accent}; }}
QPushButton:checked {{ background:{WORKFLOW_ACCENT}; color:#fff;
    border:1px solid {accent}; }}
QPushButton#primary {{ background:{primary}; color:#fff; font-weight:600; }}
QPushButton:disabled {{ color:{muted}; background:rgba(255,255,255,10); }}
QPushButton#primary:disabled {{ background:rgba(255,255,255,10); color:{muted}; }}
QFrame#workflowStages {{ background:{field}; border:1px solid {edge}; border-radius:17px; }}
QFrame#workflowStages QPushButton {{ border:0; background:transparent; }}
QFrame#workflowStages QPushButton:checked {{ background:transparent; color:#fff; }}
QWidget[glassControl="true"] {{ background:{field}; border:1px solid {edge}; border-radius:14px; }}
QWidget[sliderPanel="true"]:hover, QWidget[sliderPanel="true"][sliderHighlighted="true"] {{ background:{slider_hover}; border-color:{slider_edge}; }}
QWidget[stageHighlight="true"] {{ background:{primary}; border:1px solid {edge}; border-radius:14px; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background:{field}; border:1px solid {edge}; border-radius:9px; padding:4px; min-height:20px; selection-background-color:{selected}; }}
QComboBoxPrivateContainer {{ background:{popup}; border:1px solid {edge}; border-radius:{MENU_RADIUS}px; padding:5px; }}
QComboBox QAbstractItemView {{ background:{popup}; color:{text}; border:0; selection-background-color:{selected}; }}
QComboBox QAbstractItemView QWidget {{ background:{popup}; }}
QCheckBox {{ spacing:6px; background:transparent; }}
QCheckBox::indicator:checked {{ background:{accent}; border:1px solid {accent}; border-radius:3px; }}
QSlider::groove:horizontal {{ height:4px; background:rgba(196,196,196,60); border-radius:2px; }}
QSlider::sub-page:horizontal {{ background:rgba(196,196,196,60); border-radius:2px; }}
QSlider::handle:horizontal {{ background:{text}; border:1px solid {edge}; width:14px; margin:-5px 0; border-radius:7px; }}
QSlider::handle:horizontal:hover {{ border-color:{accent}; }}
QListWidget {{ background:transparent; border:0; outline:0; }}
QListWidget::item {{ padding:4px; margin:2px 0; border-radius:8px; border:1px solid transparent; font-size:12px; }}
QListWidget::item:selected {{ background:{selected}; color:{accent}; }}
QListWidget#workflowPhotos::indicator {{ width:18px; height:18px; border:0; background:transparent; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background:transparent; border:0; }}
QScrollBar:vertical {{ background:transparent; width:6px; margin:0; }}
QScrollBar::handle:vertical {{ background:rgba(196,196,196,85); min-height:24px; border-radius:3px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background:transparent; }}
QSplitter::handle {{ background:transparent; width:{WORKSPACE_GAP}px; }}
QWidget[nativeMaterial="true"], QFrame[nativeMaterial="true"],
QFrame#glassPanel[nativeMaterial="true"],
QFrame#workflowHeader[nativeMaterial="true"],
QFrame#workflowFooter[nativeMaterial="true"],
QFrame#workflowStages[nativeMaterial="true"],
QFrame#previewToolbar[nativeMaterial="true"],
QPushButton[nativeMaterial="true"], QPushButton[nativeMaterial="true"]:checked,
QPushButton[nativeMaterial="true"]:disabled, QPushButton#primary[nativeMaterial="true"],
QPushButton#primary[nativeMaterial="true"]:disabled,
QComboBox[nativeMaterial="true"], QToolButton[nativeMaterial="true"] {{
    background:transparent; border-color:transparent;
}}
QWidget[sliderPanel="true"][nativeMaterial="true"]:hover,
QWidget[sliderPanel="true"][nativeMaterial="true"][sliderHighlighted="true"] {{ background:transparent; border-color:transparent; }}
QFrame#workflowStages QPushButton[nativeMaterial="true"]:checked {{ background:transparent; border:0; color:#fff; }}
QPushButton#primary[nativeMaterial="true"] {{ color:#fff; }}
QPushButton[nativeMaterial="true"]:checked {{ color:#fff; }}
QPushButton[nativeMaterial="true"]:checked:disabled, QPushButton#primary[nativeMaterial="true"]:disabled {{ color:{muted}; }}
QMenu {{ background:{popup}; border:1px solid {edge}; border-radius:{MENU_RADIUS}px; padding:6px; }}
QMenu::item {{ padding:5px 16px; border-radius:5px; }}
QMenu::item:selected {{ background:{selected}; color:{accent}; }}
QMenu::item:disabled {{ color:{muted}; }}
QMenu#meterZoneMenu {{ border-radius:12px; padding:6px; }}
QMenu#meterZoneMenu::item {{ padding:7px 14px 7px 28px; border-radius:8px; }}
QMenu::separator {{ height:1px; background:{edge}; margin:6px 8px; }}
QDialog#workflowRoot[nativeMaterial="true"], QDialog[nativeMaterial="true"], QMessageBox[nativeMaterial="true"] {{ background:transparent; }}
QMenu[nativeMaterial="true"], QMenu#meterZoneMenu[nativeMaterial="true"] {{ background:transparent; border-color:transparent; }}
QComboBoxPrivateContainer[nativeMaterial="true"] {{ background:transparent; border-color:transparent; }}
QComboBox QAbstractItemView[nativeMaterial="true"] {{ background:transparent; border-color:transparent; }}
QComboBox QAbstractItemView[nativeMaterial="true"] QWidget {{ background:transparent; }}
QToolTip {{ background:{field}; color:{text}; border:1px solid {edge}; }}
""" + NATIVE_SLIDER_QSS
