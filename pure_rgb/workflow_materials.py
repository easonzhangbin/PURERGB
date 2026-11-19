"""Passive native glass backgrounds beneath Qt's existing interactive widgets.

Controls and panels are siblings below the Qt content NSView. Passive HUDs
use a foreground sibling with native text, so their real glass can sample the
photo painted by Qt. No native effect receives input. A clipped wrapper keeps
each full capsule's geometry while its viewport clips the visible material.
"""

from contextlib import contextmanager
import math
import sys
import weakref

from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, QTimer, pyqtSlot
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFrame, QLabel, QPushButton, QSlider, QStyle,
    QStyleOptionSlider, QToolButton, QWidget,
)


_PANEL_NAMES = {"glassPanel", "workflowHeader", "workflowFooter", "workflowStages", "previewToolbar",
                "reviewToolbar", "reviewSidebar", "panel", "toolbar"}
_HEADER_NAMES = {"workflowHeader", "workflowFooter", "workflowStages", "previewToolbar",
                 "reviewToolbar", "toolbar"}
_EVENTS = {
    QEvent.Type.Resize,
    QEvent.Type.Move,
    QEvent.Type.Show,
    QEvent.Type.Hide,
    QEvent.Type.LayoutRequest,
    QEvent.Type.ParentChange,
    QEvent.Type.EnabledChange,
    QEvent.Type.DynamicPropertyChange,
    QEvent.Type.Wheel,
    QEvent.Type.WindowStateChange,
    QEvent.Type.ChildAdded,
    QEvent.Type.ChildRemoved,
    QEvent.Type.ZOrderChange,
}


def _role(widget):
    if widget.property("glassMenuStrip"):
        return "menu-strip"
    if widget.property("nativeScrollEdge"):
        return "scroll-edge"
    if isinstance(widget, QLabel) and widget.property("nativeHUD"):
        return "hud"
    if isinstance(widget, QFrame) and widget.objectName() in _PANEL_NAMES:
        return "panel"
    if isinstance(widget, QSlider):
        return "slider-thumb"
    if isinstance(widget, QPushButton):
        ancestor = widget.parentWidget()
        while ancestor is not None:
            if ancestor.objectName() == "workflowStages":
                return None
            ancestor = ancestor.parentWidget()
    if isinstance(widget, (QPushButton, QToolButton, QComboBox)) or widget.property("glassControl"):
        return "control"
    return None


def _visible_rect(widget, window, part=None):
    """Return full and ancestor-clipped geometry in window logical points."""
    if not window.isVisible() or not widget.isVisibleTo(window):
        return None
    if widget.window() is not window:
        return None
    local = QRect(QPoint(), widget.size()) if part is None else part
    full = QRect(widget.mapTo(window, local.topLeft()), local.size())
    clipped = QRect(full)
    ancestor = widget.parentWidget()
    while ancestor is not None:
        bounds = QRect(ancestor.mapTo(window, QPoint(0, 0)), ancestor.size())
        clipped = clipped.intersected(bounds)
        if ancestor is window:
            break
        ancestor = ancestor.parentWidget()
    return (full, clipped) if not clipped.isEmpty() else None


def _finite_property(widget, name, default):
    value = widget.property(name)
    return float(value) if type(value) in (int, float) and math.isfinite(value) else default


def _native_types(AppKit):
    # AppKit registers subclass names process-wide.  Reuse them on module reload.
    import objc

    try:
        wrapper = objc.lookUpClass("PUREWorkflowPassiveMaterialClipV36")
    except objc.nosuchclass_error:

        class PUREWorkflowPassiveMaterialClipV36(AppKit.NSView):
            def hitTest_(self, point):
                return None

            def isFlipped(self):
                return getattr(self, "workflow_flipped", False)

        wrapper = PUREWorkflowPassiveMaterialClipV36
    return wrapper


class MaterialBinder(QObject):
    """Track native backgrounds without changing Qt focus or event delivery."""

    def __init__(self, window, AppKit, view, container, report):
        super().__init__(window)
        self.window, self.api = window, AppKit
        self.view, self.container = view, container
        self.report = report
        self.entries = {}
        self.watched = set()
        self.disposed = False
        self._syncing = False
        self._dirty = True
        self._appearance_dirty = set()
        self._layout_depth = 0
        self._settling_layout = False
        self._frame_depth = 0
        self.clip_class = _native_types(AppKit)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(0)
        self.timer.timeout.connect(self.request_frame)
        # Qt's resize event may precede Cocoa's content-view resize during a
        # live window drag.  A zero-delay Qt pass alone can leave every sibling
        # material at the previous native frame until the next interaction.
        self.settle_timer = QTimer(self)
        self.settle_timer.setSingleShot(True)
        self.settle_timer.setInterval(32)
        self.settle_timer.timeout.connect(self.request_frame)
        self.native_observers = []
        self._observe_native_geometry()
        window.destroyed.connect(self._dispose_native)
        self.sync()

    def schedule(self, *args):
        if self.disposed:
            return
        self._dirty = True
        if not self.timer.isActive():
            self.timer.start()

    def request_frame(self):
        if not self.disposed:
            if not self._dirty:
                self._sync_appearance()
            elif self.window.isVisible():
                self.window.update()
            else:
                self.sync()

    def _appearance_changed(self, widget):
        key = id(widget)
        if key not in self.entries:
            self.schedule()
            return
        self._appearance_dirty.add(key)
        if not self.timer.isActive():
            self.timer.start()

    @pyqtSlot(bool)
    def _button_toggled(self, checked):
        widget = self.sender()
        if widget is not None:
            self._appearance_changed(widget)

    def _sync_appearance(self):
        """Apply only cosmetic changes; they cannot move native siblings."""
        if self.disposed or self._syncing or self._layout_depth or self._dirty:
            return
        keys, self._appearance_dirty = self._appearance_dirty, set()
        if not keys or not self.report.get("attached"):
            return
        from Quartz import CATransaction

        self._syncing = True
        CATransaction.begin()
        CATransaction.setDisableActions_(True)
        try:
            for key in keys:
                entry = self.entries.get(key)
                if entry is not None:
                    self._update(entry, appearance_only=True)
        except Exception as error:
            self._disable()
            self.report.update(attached=False, reason=str(error))
        finally:
            CATransaction.commit()
            self._syncing = False

    @contextmanager
    def frame_transaction(self):
        """Submit Qt's backing store and sibling materials in one transaction."""
        if self._frame_depth:
            yield
            return
        from Quartz import CATransaction

        self._frame_depth += 1
        CATransaction.begin()
        CATransaction.setDisableActions_(True)
        try:
            yield
            if not self.disposed and self.view is not None:
                self.view.displayIfNeeded()
                native_window = self.view.window()
                if native_window is not None:
                    native_window.displayIfNeeded()
        finally:
            CATransaction.commit()
            CATransaction.flush()
            self._frame_depth -= 1

    def eventFilter(self, watched, event):
        if self.disposed:
            return False
        kind = event.type()
        if (kind == QEvent.Type.Paint and not self._syncing and not self._layout_depth):
            # Qt has settled its layouts before painting. Commit the passive
            # native geometry in this same frame, not a later zero/32ms timer.
            if self._dirty:
                self.sync()
            else:
                if watched.property("nativeHUD") or isinstance(watched, QSlider):
                    self._appearance_dirty.add(id(watched))
                self._sync_appearance()
        elif (kind == QEvent.Type.DynamicPropertyChange
              and bytes(event.propertyName()) in {
                  b"sliderInteraction", b"sliderHighlighted", b"glassTint",
              }):
            self._appearance_changed(watched)
        elif kind == QEvent.Type.EnabledChange and id(watched) in self.entries:
            self._appearance_changed(watched)
        elif kind in _EVENTS:
            self.schedule()
            self.settle_timer.start()
        elif (watched.property("nativeHUD")
              and kind in {QEvent.Type.FontChange, QEvent.Type.StyleChange, QEvent.Type.PaletteChange}):
            self._appearance_changed(watched)
        return False

    def begin_layout(self):
        if self._layout_depth == 0 and not self._settling_layout:
            self._restore_updates = self.window.updatesEnabled()
            self.window.setUpdatesEnabled(False)
        self._layout_depth += 1

    def end_layout(self):
        self._layout_depth -= 1
        if self._layout_depth or self._settling_layout:
            return
        restore_updates = self._restore_updates
        self._settling_layout = True
        try:
            QApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)
            for widget in (self.window, *self.window.findChildren(QWidget)):
                if widget.window() is self.window and widget.isVisibleTo(self.window) and widget.layout():
                    widget.layout().activate()
            QApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)
            self.sync()
        finally:
            self.window.setUpdatesEnabled(restore_updates)
            if restore_updates:
                # This can run inside a child's resize/show event. Re-entering
                # the backing-store painter here invalidates QGraphicsEffect's
                # source painter on Cocoa. Let the normal UpdateRequest own
                # the next Qt/native frame transaction instead.
                self.window.update()
            self._settling_layout = False

    def _observe_native_geometry(self):
        from Foundation import NSNotificationCenter, NSOperationQueue

        center = NSNotificationCenter.defaultCenter()
        reference = weakref.ref(self)

        def changed(notification):
            binder = reference()
            if binder is not None and not getattr(binder, "disposed", False):
                binder.schedule()
                binder.settle_timer.start()

        for native in (self.view, self.container):
            native.setPostsFrameChangedNotifications_(True)
            native.setPostsBoundsChangedNotifications_(True)
            for name in (
                self.api.NSViewFrameDidChangeNotification,
                self.api.NSViewBoundsDidChangeNotification,
            ):
                self.native_observers.append(
                    center.addObserverForName_object_queue_usingBlock_(
                        name, native, None, changed
                    )
                )
        self.native_observers.append(
            center.addObserverForName_object_queue_usingBlock_(
                self.api.NSWindowDidResizeNotification,
                self.view.window(),
                None,
                changed,
            )
        )
        workspace_center = self.api.NSWorkspace.sharedWorkspace().notificationCenter()
        self.workspace_observer = (
            workspace_center,
            workspace_center.addObserverForName_object_queue_usingBlock_(
                self.api.NSWorkspaceAccessibilityDisplayOptionsDidChangeNotification,
                None, NSOperationQueue.mainQueue(), changed,
            ),
        )

    def _dispose_native(self, *args):
        if getattr(self, "disposed", False):
            return
        self.disposed = True
        from Foundation import NSNotificationCenter

        self.timer.stop()
        self.settle_timer.stop()
        center = NSNotificationCenter.defaultCenter()
        for observer in self.native_observers:
            center.removeObserver_(observer)
        self.native_observers.clear()
        if hasattr(self, "workspace_observer"):
            workspace_center, observer = self.workspace_observer
            workspace_center.removeObserver_(observer)
            self.workspace_observer = None
        for key in tuple(self.entries):
            self._remove(key)
        # Detach while Qt's content NSView and widget tree still exist, rather
        # than waiting for SIP's interpreter-exit object walk.
        if hasattr(self, "window"):
            try:
                for widget in (self.window, *self.window.findChildren(QWidget)):
                    widget.removeEventFilter(self)
            except RuntimeError:
                # The destroyed signal is a fallback for callers which did
                # not explicitly close the window before deleting its wrapper.
                pass
        self.view = self.container = None

    def _remove(self, key, *, widget_destroyed=False):
        self._appearance_dirty.discard(key)
        entry = self.entries.pop(key, None)
        if entry is not None:
            entry["wrapper"].removeFromSuperview()
            if not widget_destroyed and not self.disposed:
                try:
                    self._mark(entry["widget"], False)
                except RuntimeError:
                    pass
        if widget_destroyed:
            self.watched.discard(key)

    @pyqtSlot(QObject)
    def _widget_destroyed(self, widget):
        # QObject still owns its dynamic properties while destroyed is emitted.
        # A receiver-bound slot also disconnects automatically with this binder;
        # a per-widget lambda can be GC-cleared before Qt emits the signal.
        key = widget.property("_pureRgbMaterialKey") if widget is not None else None
        if type(key) is int:
            self._remove(key, widget_destroyed=True)

    @staticmethod
    def _mark(widget, active):
        if widget.property("nativeMaterial") != active:
            widget.setProperty("nativeMaterial", active)
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def _disable(self):
        for entry in self.entries.values():
            entry["wrapper"].setHidden_(True)
            entry["hidden"] = True
            for state in ("geometry_state", "appearance_state", "text_state"):
                entry.pop(state, None)
            try:
                self._mark(entry["widget"], False)
            except RuntimeError:
                pass

    def _create(self, widget, role):
        A = self.api
        wrapper = self.clip_class.alloc().initWithFrame_(((0, 0), (0, 0)))
        wrapper.workflow_flipped = bool(self.container.isFlipped())
        # Cocoa's unflipped parent anchors children from the bottom.  Preserve
        # their top edge while Cocoa and Qt settle on the new content height.
        wrapper.setAutoresizingMask_(
            0 if self.container.isFlipped() else A.NSViewMinYMargin
        )
        wrapper.setWantsLayer_(True)
        wrapper.layer().setMasksToBounds_(True)
        use_glass = (role in {"control", "slider-thumb"}
                     or widget.objectName() in {"workflowStages", "previewToolbar"}) and hasattr(A, "NSGlassEffectView")
        if use_glass:
            effect = A.NSGlassEffectView.alloc().initWithFrame_(((0, 0), (0, 0)))
            effect.setStyle_(A.NSGlassEffectViewStyleRegular)
        else:
            effect = A.NSVisualEffectView.alloc().initWithFrame_(((0, 0), (0, 0)))
            if role == "hud":
                material = A.NSVisualEffectMaterialHUDWindow
            elif role == "menu-strip":
                material = A.NSVisualEffectMaterialMenu
            elif role in {"control", "slider-thumb"}:
                material = A.NSVisualEffectMaterialPopover
            elif role == "scroll-edge" or widget.objectName() in _HEADER_NAMES:
                material = A.NSVisualEffectMaterialHeaderView
            else:
                material = A.NSVisualEffectMaterialSidebar
            effect.setMaterial_(material)
            effect.setBlendingMode_(A.NSVisualEffectBlendingModeWithinWindow)
            effect.setState_(A.NSVisualEffectStateActive if role == "hud"
                             else A.NSVisualEffectStateFollowsWindowActiveState)
            effect.setWantsLayer_(True)
            effect.layer().setMasksToBounds_(True)
            if role != "scroll-edge":
                effect.layer().setBorderWidth_(1.0)
        wrapper.addSubview_(effect)
        scroll_mask = None
        if role == "scroll-edge":
            from Quartz import CAGradientLayer, CGColorCreateGenericRGB

            scroll_mask = CAGradientLayer.layer()
            top = widget.property("nativeScrollEdge") == "top"
            leading = float(top != bool(effect.layer().isGeometryFlipped()))
            scroll_mask.setStartPoint_((0.5, leading))
            scroll_mask.setEndPoint_((0.5, 1 - leading))
            scroll_mask.setColors_([CGColorCreateGenericRGB(0, 0, 0, alpha) for alpha in (1, 0.65, 0)])
            scroll_mask.setLocations_((0, 0.40, 1))
            effect.layer().setMask_(scroll_mask)
        tint_view = gradient = None
        if role == "control" and not use_glass:
            tint_view = self.clip_class.alloc().initWithFrame_(((0, 0), (0, 0)))
            tint_view.setWantsLayer_(True)
            from .workflow_glass import native_red_gradient, native_slider_highlight

            gradient = (native_slider_highlight(tint_view.bounds()) if widget.property("sliderPanel")
                        else native_red_gradient(tint_view.bounds()))
            tint_view.setLayer_(gradient)
            tint_view.layer().setMasksToBounds_(True)
            effect.addSubview_positioned_relativeTo_(tint_view, A.NSWindowAbove, None)
        native_text = None
        if role == "hud":
            native_text = A.NSTextField.labelWithString_(widget.text())
            native_text.setEditable_(False)
            native_text.setSelectable_(False)
            native_text.setBordered_(False)
            native_text.setDrawsBackground_(False)
            native_text.setAlignment_(A.NSTextAlignmentCenter)
            native_text.setTextColor_(A.NSColor.colorWithSRGBRed_green_blue_alpha_(0.98, 0.965, 0.94, 1.0))
            native_text.cell().setWraps_(widget.wordWrap())
            native_text.cell().setUsesSingleLineMode_(not widget.wordWrap())
            if hasattr(native_text, "setMaximumNumberOfLines_"):
                native_text.setMaximumNumberOfLines_(0 if widget.wordWrap() else 1)
            if hasattr(native_text, "setAccessibilityElement_"):
                native_text.setAccessibilityElement_(False)
            effect.addSubview_(native_text)
        self.container.addSubview_positioned_relativeTo_(
            wrapper, A.NSWindowAbove if role in {"hud", "scroll-edge"} else A.NSWindowBelow, self.view
        )
        # Dynamic QSS selectors need one polish after this property is installed.
        if role != "scroll-edge":
            self._mark(widget, True)
        return dict(
            widget=widget,
            role=role,
            wrapper=wrapper,
            effect=effect,
            glass=use_glass,
            tint=tint_view,
            gradient=gradient,
            native_text=native_text,
            scroll_mask=scroll_mask,
        )

    def _to_native(self, rectangle):
        bounds = self.view.bounds()
        x = float(bounds.origin.x + rectangle.x())
        y = float(rectangle.y())
        if not self.view.isFlipped():
            y = float(bounds.size.height - rectangle.y() - rectangle.height())
        y += float(bounds.origin.y)
        return self.view.convertRect_toView_(
            ((x, y), (rectangle.width(), rectangle.height())), self.container
        )

    def _update(self, entry, *, appearance_only=False):
        widget, wrapper, effect = entry["widget"], entry["wrapper"], entry["effect"]
        role = entry["role"]
        if role != "scroll-edge":
            self._mark(widget, True)
        if appearance_only and entry.get("hidden", True):
            return
        part = None
        if entry["role"] == "slider-thumb":
            option = QStyleOptionSlider()
            widget.initStyleOption(option)
            part = widget.style().subControlRect(
                QStyle.ComplexControl.CC_Slider, option,
                QStyle.SubControl.SC_SliderHandle, widget,
            )
        geometry = (
            None if role == "scroll-edge" and self._reduce_transparency
            else _visible_rect(widget, self.window, part)
        )
        if geometry is None:
            if entry.get("hidden") is not True:
                wrapper.setHidden_(True)
                entry["hidden"] = True
            return
        full, clip = (self._to_native(rect) for rect in geometry)
        frame = ((full.origin.x - clip.origin.x, full.origin.y - clip.origin.y),
                 (full.size.width, full.size.height))
        radius = (
            full.size.height / 2
            if entry["role"] in {"control", "slider-thumb"}
            else (0 if role == "scroll-edge" else 10 if role == "hud" else
                  17 if widget.objectName() == "workflowStages" else
                  12 if role == "menu-strip" or widget.objectName() in _HEADER_NAMES else 16)
        )
        if entry["role"] == "control":
            override = _finite_property(widget, "glassCornerRadius", None)
            if override is not None and override >= 0:
                radius = min(override, max(0.0, full.size.width / 2), max(0.0, full.size.height / 2))
        geometry_state = (float(clip.origin.x), float(clip.origin.y), float(clip.size.width),
                          float(clip.size.height), frame, radius)
        geometry_changed = geometry_state != entry.get("geometry_state")
        if not geometry_changed:
            # Cocoa can autoresize siblings between Qt's layout and paint.
            # A cached desired frame alone cannot establish their actual frame.
            actual_clip, actual_full = wrapper.frame(), effect.frame()
            geometry_changed = (
                actual_clip != clip
                or ((actual_full.origin.x, actual_full.origin.y),
                    (actual_full.size.width, actual_full.size.height)) != frame
            )
        if geometry_changed:
            wrapper.setFrame_(clip)
            effect.setFrame_(frame)
            if entry["glass"]:
                effect.setCornerRadius_(radius)
            else:
                effect.layer().setCornerRadius_(radius)
            if entry["tint"] is not None:
                entry["tint"].setFrame_(effect.bounds())
                entry["tint"].layer().setCornerRadius_(radius)
                entry["gradient"].setFrame_(effect.bounds())
            if entry["scroll_mask"] is not None:
                entry["scroll_mask"].setFrame_(effect.bounds())
            entry["geometry_state"] = geometry_state
        slider_panel = bool(widget.property("sliderPanel"))
        interaction = (min(1.0, max(0.0, _finite_property(widget, "sliderInteraction", 0.0)))
                       if slider_panel and widget.isEnabled() else 0.0)
        name = widget.objectName()
        selected = bool(widget.property("stageHighlight")) or (
            isinstance(widget, (QPushButton, QToolButton))
            and (name in {"primary", "reviewPrimary"} or widget.isChecked())
        )
        tint_name = widget.property("glassTint")
        appearance = (slider_panel, interaction, selected, widget.isEnabled(), tint_name, self._increase_contrast)
        appearance_changed = appearance != entry.get("appearance_state")
        if appearance_changed:
            tint = None
            if widget.isEnabled():
                if tint_name or selected:
                    color = QColor(tint_name or widget.property("glassAccent") or "#ff3c1d")
                    red, green, blue, alpha = color.getRgbF()
                    tint = self.api.NSColor.colorWithSRGBRed_green_blue_alpha_(
                        red, green, blue, alpha if tint_name else 0.68,
                    )
                elif slider_panel and interaction > 0:
                    tint = self.api.NSColor.whiteColor().colorWithAlphaComponent_(0.045 * interaction)
            if entry["glass"] and entry["role"] != "hud":
                effect.setTintColor_(tint)
            elif entry["role"] == "slider-thumb":
                fill = self.api.NSColor.whiteColor().colorWithAlphaComponent_(
                    0.35 if widget.isEnabled() else 0.15
                )
                effect.layer().setBackgroundColor_(fill.CGColor())
            if not entry["glass"] and role != "scroll-edge":
                edge = self.api.NSColor.separatorColor().colorWithAlphaComponent_(
                    0.65 if self._increase_contrast else 0.28
                )
                effect.layer().setBorderColor_(edge.CGColor())
            if entry["tint"] is not None:
                entry["tint"].layer().setBackgroundColor_(
                    (tint if tint and not entry["glass"] else self.api.NSColor.clearColor()).CGColor())
                if slider_panel:
                    entry["gradient"].setOpacity_(interaction)
                    entry["tint"].setHidden_(interaction <= 0)
                else:
                    entry["tint"].setHidden_(tint is None)
            entry["appearance_state"] = appearance
        if entry["native_text"] is not None:
            text = entry["native_text"]
            font = widget.font()
            size = font.pixelSize() if font.pixelSize() > 0 else font.pointSizeF()
            text_state = (widget.text(), font.family(), max(10.0, float(size)))
            if text_state != entry.get("text_state"):
                text.setStringValue_(text_state[0])
                native_font = self.api.NSFont.fontWithName_size_(text_state[1], text_state[2])
                text.setFont_(native_font or self.api.NSFont.systemFontOfSize_(text_state[2]))
                entry["text_state"] = text_state
            inset_x, inset_y = 14.0, 7.0
            bounds = effect.bounds()
            if geometry_changed:
                text.setFrame_(((inset_x, inset_y),
                                (max(0.0, bounds.size.width - 2 * inset_x),
                                 max(0.0, bounds.size.height - 2 * inset_y))))
        if entry.get("hidden") is not False:
            wrapper.setHidden_(False)
            entry["hidden"] = False

    def sync(self):
        if self.disposed:
            return
        if self._syncing or self._layout_depth:
            self.schedule()
            return
        self._syncing, self._dirty = True, False
        self._appearance_dirty.clear()
        transaction = animation = None
        try:
            A = self.api
            from Quartz import CATransaction

            transaction = CATransaction
            transaction.begin()
            transaction.setDisableActions_(True)
            animation = A.NSAnimationContext
            animation.beginGrouping()
            animation.currentContext().setDuration_(0.0)
            animation.currentContext().setAllowsImplicitAnimation_(False)
            workspace = A.NSWorkspace.sharedWorkspace()
            self._reduce_transparency = bool(workspace.accessibilityDisplayShouldReduceTransparency())
            self._increase_contrast = bool(workspace.accessibilityDisplayShouldIncreaseContrast())
            widgets = [widget for widget in (self.window, *self.window.findChildren(QWidget))
                       if widget.window() is self.window]
            active = set()
            for widget in widgets:
                key = id(widget)
                if key not in self.watched:
                    widget.installEventFilter(self)
                    widget.setProperty("_pureRgbMaterialKey", key)
                    widget.destroyed.connect(self._widget_destroyed)
                    if isinstance(widget, (QPushButton, QToolButton)):
                        widget.toggled.connect(self._button_toggled)
                    self.watched.add(key)
                role = _role(widget)
                if role is None:
                    continue
                active.add(key)
                if key not in self.entries:
                    self.entries[key] = self._create(widget, role)
                self._update(self.entries[key])
            for key in set(self.entries) - active:
                self._remove(key)
            self.report.update(
                attached=True,
                target_count=len(self.entries),
                glass_capsules=sum(entry["glass"] for entry in self.entries.values()),
                visual_effect_panels=sum(
                    not entry["glass"] for entry in self.entries.values()
                ),
                input_capture=False,
                native_resize_observers=len(self.native_observers),
                clipping="Qt ancestor and viewport intersections; full capsule geometry retained",
                foreground_native_HUDs=sum(entry["role"] == "hud" for entry in self.entries.values()),
            )
        except Exception as error:
            # A failed native association must not leave orphan materials visible.
            self._disable()
            self.report.update(attached=False, reason=str(error))
        finally:
            if animation is not None:
                animation.endGrouping()
            if transaction is not None:
                transaction.commit()
            self._syncing = False


@contextmanager
def stable_material_layout(window):
    """Commit a page's Qt layout and passive glass as one visible update."""
    binder = getattr(window, "_workflow_material_binder", None)
    if binder is None or binder.disposed:
        yield
        return
    # A layout callback may itself originate in an active paint/resize stack.
    # Flushing Cocoa's display here can recursively enter that same painter.
    # material_frame_event owns the real frame transaction at UpdateRequest.
    binder.begin_layout()
    try:
        yield
    finally:
        binder.end_layout()


def material_frame_event(window, event, dispatch):
    """Keep native materials aligned with the pixels flushed by Qt's update."""
    binder = getattr(window, "_workflow_material_binder", None)
    if (binder is None or binder.disposed or binder._frame_depth
            or event.type() != QEvent.Type.UpdateRequest):
        return dispatch(event)
    with binder.frame_transaction():
        return dispatch(event)


def attach_materials(window):
    """Attach native materials when Cocoa is active; every other case is a no-op."""
    if getattr(window, "closing", False):
        return dict(mode="solid", attached=False, target_count=0, reason="window-closing")
    previous = getattr(window, "_workflow_material_binder", None)
    if previous is not None:
        previous.schedule()
        return previous.report
    report = dict(mode="solid", attached=False, target_count=0)
    if sys.platform != "darwin" or QApplication.platformName() != "cocoa":
        return {**report, "reason": "non-cocoa"}
    try:
        import objc
        import AppKit

        view = objc.objc_object(c_void_p=int(window.winId()))
        container = view.superview()
        if container is None or view.window() is None:
            return {**report, "reason": "native-view-not-attached"}
        report["mode"] = "native-per-control-materials"
        binder = MaterialBinder(window, AppKit, view, container, report)
        window._workflow_material_binder = binder
        return report
    except Exception as error:
        return {**report, "reason": str(error)}
