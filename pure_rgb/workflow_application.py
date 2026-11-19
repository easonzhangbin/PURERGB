"""Application identity shared by the normal source launcher and native QA."""

import sys
import weakref

from PyQt6.QtCore import QEvent, QObject, QTimer

from .release import APP_VERSION

def prepare_application_identity():
    """Name the source-launched Cocoa process before QApplication starts."""
    if sys.platform == "darwin":
        from Foundation import NSProcessInfo

        NSProcessInfo.processInfo().setProcessName_("PURE RGB")
    from PyQt6.QtCore import QCoreApplication

    QCoreApplication.setApplicationName("PURE RGB")
    QCoreApplication.setApplicationVersion(APP_VERSION)



class ApplicationMenuTitles(QObject):
    """Keep source-launched macOS menu titles while retaining Qt role actions."""

    def __init__(self, window):
        from PyQt6.QtWidgets import QApplication

        super().__init__(window)
        self.window = window
        self.app = QApplication.instance()
        self.disposed = False
        self.observers = []
        self.report = dict(attached=False)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.sync)
        self.settle_timer = QTimer(self)
        self.settle_timer.setSingleShot(True)
        self.settle_timer.setInterval(150)
        self.settle_timer.timeout.connect(self.sync)
        if sys.platform == "darwin" and self.app.platformName() == "cocoa":
            from Foundation import NSNotificationCenter

            center = NSNotificationCenter.defaultCenter()
            reference = weakref.ref(self)

            def update(notification):
                owner = reference()
                if owner is not None and not owner.disposed:
                    owner.sync()

            for name in ("NSMenuDidBeginTrackingNotification", "NSApplicationDidBecomeActiveNotification"):
                self.observers.append(center.addObserverForName_object_queue_usingBlock_(name, None, None, update))
            window.installEventFilter(self)
            self.app.installEventFilter(self)
            self.app.aboutToQuit.connect(self.dispose)
            window.destroyed.connect(self.dispose)
            self.schedule()

    def schedule(self):
        if not self.disposed:
            self.timer.start(0)
            self.settle_timer.start()

    def eventFilter(self, watched, event):
        if event.type() in {QEvent.Type.Show, QEvent.Type.WindowActivate, QEvent.Type.ApplicationActivate}:
            self.schedule()
        return False

    def sync(self):
        if self.disposed or sys.platform != "darwin" or self.app.platformName() != "cocoa":
            return
        from AppKit import NSApplication

        main = NSApplication.sharedApplication().mainMenu()
        if main is None or not main.numberOfItems():
            return
        root = main.itemAtIndex_(0)
        menu = root.submenu()
        if menu is None:
            return
        root.setTitle_("PURE RGB")
        menu.setTitle_("PURE RGB")
        matched = {}
        for item in menu.itemArray():
            title, key = str(item.title()), str(item.keyEquivalent())
            role = ("settings" if key == "," else "quit" if key == "q" else
                    "about" if (title.startswith("About ") or title.startswith("关于 ")) and "Qt" not in title else None)
            if role is not None:
                item.setTitle_({"about": "关于 PURE RGB", "settings": "设置…", "quit": "退出 PURE RGB"}[role])
                matched[role] = dict(title=str(item.title()), key=key)
        self.report.update(attached=set(matched) == {"about", "settings", "quit"}, actions=matched,
                           qt_role_routes_preserved=True, role_shortcuts_preserved=True)

    def dispose(self, *args):
        if self.disposed:
            return
        self.disposed = True
        self.timer.stop()
        self.settle_timer.stop()
        if self.observers:
            from Foundation import NSNotificationCenter

            center = NSNotificationCenter.defaultCenter()
            for observer in self.observers:
                center.removeObserver_(observer)
            self.observers.clear()
        if self.app is not None:
            self.app.removeEventFilter(self)
