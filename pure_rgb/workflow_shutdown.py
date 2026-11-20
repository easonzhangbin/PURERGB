"""Finish asynchronous work and destroy Qt objects before interpreter teardown."""

import gc

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QApplication


def shutdown_window(window):
    """Idempotent for window-close, Cmd-Q and the application finally block."""
    if getattr(window, "_shutdown_complete", False):
        return
    window.closing = True
    for panel in window.advanced_panels.values():
        panel.cancel_pending()
    window.maintenance_timer.stop()
    window.idle_scheduler.stop()
    QApplication.instance().removeEventFilter(window)
    window.preview_timer.stop()
    import_preview_timer = getattr(window, "import_preview_timer", None)
    if import_preview_timer is not None:
        import_preview_timer.stop()
    activity = getattr(window, "preview_activity", None)
    if activity is not None:
        activity.timer.stop()
    settings = getattr(window, "settings_window", None)
    if settings is not None:
        settings.dispose()
    update_notice = getattr(window, "update_notice", None)
    if update_notice is not None:
        update_notice.dispose()
    # Invalidate queued display completions before joining their pools.
    window.token += 1
    window.live.invalidate(window.token)
    window.live.shutdown()
    window.tasks.shutdown()
    window.pool.waitForDone()
    window.idle_pool.waitForDone()
    window.snapshot_store.close(force=True)
    tracker = getattr(window, "display_tracker", None)
    if tracker is not None:
        tracker.dispose()
    from .workflow_glass import detach_glass

    detach_glass(window)
    try:
        window.project.save()
    finally:
        window.lock.unlock()
    window._shutdown_complete = True


def dispose_application(application, window):
    """Destroy owned QObjects while Python slots and Cocoa bridges remain valid.

    Returning from QApplication.exec() alone defers some wrapper destruction to
    PyQt's atexit hook, which cannot enforce our worker/native-view dependency
    order. Explicit deletion makes that order part of the desktop lifecycle.
    """
    if window is not None and not sip.isdeleted(window):
        shutdown_window(window)
    for widget in list(application.topLevelWidgets()):
        if not sip.isdeleted(widget):
            sip.delete(widget)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    gc.collect()
    sip.delete(application)
