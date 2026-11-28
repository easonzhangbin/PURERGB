"""Finish asynchronous work and destroy Qt objects before interpreter teardown."""

import gc

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QApplication


def shutdown_window(window):
    """Idempotent for window-close, Cmd-Q and the application finally block."""
    if getattr(window, "_shutdown_complete", False):
        return
    window._event_filters_active = False
    window.closing = True
    end_tone_drag = getattr(window, "end_tone_drag", None)
    if end_tone_drag is not None:
        end_tone_drag()
    window._discard_pending_tone_frame()
    auto_import = getattr(window, "auto_import", None)
    if auto_import is not None:
        auto_import.close()
    for cancel in list(getattr(window, "_deferred_display_work", {}).values()):
        try:
            cancel()
        except OSError as error:
            from .diagnostics import failure

            failure("deferred_display_cleanup_failed", error)
    for panel in window.advanced_panels.values():
        panel.cancel_pending()
    window.maintenance_timer.stop()
    window.idle_scheduler.stop()
    QApplication.instance().removeEventFilter(window)
    window.canvas.removeEventFilter(window)
    window.stage_navigation.dispose()
    window.preview_timer.stop()
    window.tone_preview_timer.stop()
    window._recipe_autosave_timer.stop()
    window.preview_scopes.close()
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
    thumbnails = getattr(window, "photo_thumbnails", None)
    if thumbnails is not None:
        thumbnails.close()
    library = getattr(window, "image_library_dialog", None)
    if library is not None:
        library.dispose()
    balance = getattr(window, "rgb_balance_dialog", None)
    if balance is not None:
        balance.dispose()
        balance.close()
    window.pool.waitForDone()
    window.tone_display_pool.waitForDone()
    window.idle_pool.waitForDone()
    window.light_table.close()
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
