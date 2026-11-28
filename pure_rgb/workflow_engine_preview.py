"""Install a temporary engine frame without publishing it as the exact output."""

from pathlib import Path
import time

from PyQt6.QtCore import QTimer

from .diagnostics import event as diagnostic_event
from .preview_ui import Work
from .workflow_ui_model import ENGINES


def receive_engine_preview(window, event, context):
    token = event["token"]
    target, proof = window.display_target, window.proof_space
    binding = {**context, "display_revision": window.display_revision}
    discard = window.live.discard

    def read():
        try:
            return window.display_cache.get_interactive_image(event["records"][0], target, proof=proof)
        finally:
            discard(event)

    def exact_or_present(result_token, image=None):
        if (window.closing or result_token != window.token
                or not window._preview_recipe_matches(binding)):
            return
        if (image is not None and not image.isNull()
                and binding["display_revision"] == window.display_revision
                and proof == window.proof_space and window.canvas.fit_mode and window._can_show_tone_frame()
                and not window.compare.isChecked()):
            window.canvas.set_interactive_image(image, ENGINES[context["engine"]] + " · 快速预览")
        window.request_preview()

    if token != window.token or not window._preview_recipe_matches(binding):
        discard(event)
        return
    if target is None or not window.canvas.fit_mode or not window._can_show_tone_frame():
        discard(event)
        exact_or_present(token)
        return
    work = Work(token, read)
    work.signals.done.connect(exact_or_present)
    work.signals.failed.connect(lambda failed_token, _: exact_or_present(failed_token))
    window.tone_display_pool.start(work)


def restore_visible_engine_preview(window):
    """Reuse the exact native frame still on the canvas after a rapid round trip."""
    shown, canvas = window.shown_context, window.canvas
    binding = (shown or {}).get("visible_engine_binding")
    if (window.closing or not binding or not canvas.fit_mode or canvas.current is None
            or window.canvas_stack.currentWidget() is not canvas
            or (canvas.photo_lens and canvas.photo_lens.active)
            or (window.compare.isChecked() and canvas.reference is None)
            or getattr(window, "shown_display", None) != (window.display_revision, window.proof_space)
            or not window._preview_recipe_matches(shown)):
        return False
    from .draft_preview import _identity

    try:
        manifest = window.project.preview_base().resolve(strict=True)
        if (manifest != Path(binding["manifest"]) or _identity(manifest) != tuple(binding["identity"])
                or any(_identity(path) != tuple(identity) for path, identity in binding["sources"].items())):
            return False
    except (OSError, KeyError, RuntimeError, TypeError, ValueError):
        return False

    token = window.token
    window.shown_context = {**shown, "generation": token}
    window.preview_activity.reset()
    window.set_preview_updating(False)
    window.show_preview_notice()
    window._sync_comparison()
    scopes = getattr(window, "shown_scopes", None)
    window.scope_panel.set_scope_data(scopes)
    window.scope_panel.caption.setText("输出 RGB · 预览取样" if scopes is not None else "输出 RGB · 预览取样更新中")
    if scopes is None:
        QTimer.singleShot(0, lambda: window.token == token and not window.closing and window.refresh_display_snapshot())
    window.last_activity = time.monotonic()
    window.status_message("已复用当前引擎的精确预览")
    diagnostic_event("preview_visible_engine_reused", token=token, engine=shown["engine"])
    window.canvas.repaint()
    window.image_presented.emit()
    return True
