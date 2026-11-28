"""Full-source positive crop editor, separate from the final cropped canvas.

The locator is rendered by the selected real color route with its current tone
settings. Crop and geometric resampling are omitted only from this private
positioning view; source-coordinate rectangles stay exact and can expand.
"""

from copy import deepcopy
import hashlib
import json
import math

from PyQt6.QtGui import QTransform

from .native import sha256
from .workflow_ui_model import ENGINES, RESULT_MODES
from .workspace import checked_recipe, output_rectangle

HIGHLIGHT_ROLES = ("highlight", "highlight_effect", "highlight_protect")


def _locator_role(window):
    if window.geometry_role == "crop":
        return "crop"
    if (window.geometry_role in HIGHLIGHT_ROLES
            and getattr(window, "highlight_full_source", False)):
        return "highlight"
    return None


def full_source_recipe(recipe, engine):
    """Keep current measurement/color, exposing the complete source canvas."""
    result = deepcopy(recipe)
    result["crop"] = False
    result.pop("output_roi", None)
    mode = RESULT_MODES[engine]
    post = result.get("post_controls", {})
    if mode in post:
        values = post[mode]["values"]
        for key in list(values):
            if key.startswith("geometry."):
                values.pop(key)
        if not values:
            post.pop(mode)
        if not post:
            result.pop("post_controls", None)
    return checked_recipe(result)


def locator_context(window):
    photo = window.current_photo()
    if not photo or not photo.get("initial_job"):
        return None
    recipe = full_source_recipe(window.project.recipe(), photo["engine"])
    repair = window.project.repair_context(recipe=recipe)
    source_view = bool(not repair and window.project.repair_context())
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, allow_nan=False).encode()).hexdigest()
    role = _locator_role(window)
    return dict(photo=photo["id"], engine=photo["engine"], recipe=recipe,
                key=(photo["id"], photo["engine"], digest, window.display_revision, window.proof_space, role,
                     repair["receipt_sha256"] if repair else None),
                locator_role=role, crop_locator=True, repair_plan=repair, repair_source_view=source_view)


def _matches(window, context):
    desired = locator_context(window)
    return bool(not window.closing and _locator_role(window) is not None and desired
                and desired["key"] == context["key"])


def _configure_crop_handles(window):
    photo = window.current_photo()
    recipe = window.project.recipe()
    values = recipe.get("post_controls", {}).get(RESULT_MODES[photo["engine"]], {}).get("values", {})
    bounds = output_rectangle(recipe, photo["analysis_roi"], photo["source_shape"])
    window.selection.crop_handles.configure(bounds, values.get("geometry.fine_rotation", 0))


def _install(window, image, context, *, reference=False, scope_data=None, source_grid=None):
    if not _matches(window, context):
        return False
    photo = window.current_photo()
    direction = context["recipe"].get("output_orientation")
    height, width = photo["source_shape"][:2]
    if direction and direction["quarter_turns"] % 2:
        width, height = height, width
    if (image.isNull() or abs(image.width() / image.height() - width / height)
            > 2 / min(image.width(), image.height())):
        raise ValueError("完整正像裁切预览比例与源画幅不符")
    overlays, grid = window.selection_overlays()
    is_crop = context["locator_role"] == "crop"
    preview_kind = "crop-positive" if is_crop else "positive"
    preserve = (window.selection.source_photo_id == photo["id"]
                and window.selection.geometry_info
                and window.selection.geometry_info.get("preview_kind") == preview_kind)
    if not preserve:
        window.selection.crop_handles.configure(None, 0)
    window.selection.set_preview(image, photo["source_shape"], output_orientation=direction,
                                 overlays=overlays, grid=grid, preserve_view=bool(preserve),
                                 crop_locator=is_crop)
    window.selection.source_photo_id = photo["id"]
    window.selection._crop_locator_key = context["key"]
    window.selection.drawing_enabled = True
    window.selection.shape_kind = (window.shape_choice.currentData()
                                   if window.geometry_role == "highlight_protect" else "rectangle")
    if is_crop:
        _configure_crop_handles(window)
    window.crop_locator_cache = dict(key=context["key"], image=image, context=deepcopy(context),
                                    reference=reference, scope_data=scope_data, source_grid=source_grid)
    window.crop_locator_context = deepcopy(context)
    window.canvas_stack.setCurrentWidget(window.selection)
    window.show_region_exit()
    if is_crop and hasattr(window, "scope_panel"):
        window.scope_panel.set_scope_data(scope_data)
        window.scope_panel.caption.setText("参考 RGB · 正在更新" if reference else "输出 RGB · 预览取样")
    return True


def refresh_crop_frame(window):
    """Refresh only exact source overlays; never substitute a cropped frame."""
    cache = getattr(window, "crop_locator_cache", None)
    if not cache or not _matches(window, cache["context"]):
        return False
    view = window.selection
    if (cache["context"].get("locator_role") == "crop"
            and getattr(view, "_crop_locator_key", None) == cache["key"]
            and view.geometry_info and view.geometry_info.get("preview_kind") == "crop-positive"):
        overlays, grid = window.selection_overlays()
        view.refresh_source_overlays(overlays, grid)
        _configure_crop_handles(window)
        return True
    return _install(window, cache["image"], cache["context"],
                    reference=cache.get("reference", False), scope_data=cache.get("scope_data"),
                    source_grid=cache.get("source_grid"))


def confirmed_crop_preview(window):
    """Project the selected source rectangle onto a matching fit image."""
    from .preview_grid import post_requires_full_grid
    from .selection_mapping_v43 import SelectionMapping

    canvas = window.canvas
    if (window.tasks.busy or not canvas.current or not canvas.fit_mode or canvas.split or window.compare.isChecked()
            or (canvas.photo_lens is not None and canvas.photo_lens.active)):
        return None
    context = locator_context(window)
    if context is None or context["repair_plan"] or context["repair_source_view"]:
        return None
    photo, recipe = window.current_photo(), window.project.recipe()
    if post_requires_full_grid(recipe.get("post_controls", {}).get(RESULT_MODES[context["engine"]])):
        return None
    cache = getattr(window, "crop_locator_cache", None)
    shown = window.shown_context or {}
    if (shown.get("photo") == context["photo"] and shown.get("engine") == context["engine"]
            and shown.get("recipe") == recipe):
        return None
    if (cache and not cache["reference"] and cache["key"] == context["key"]):
        image = cache["image"]
    elif (shown.get("photo") == context["photo"] and shown.get("engine") == context["engine"]
          and shown.get("recipe") == context["recipe"] and not shown.get("repair_plan_sha256")
          and not shown.get("preview_quality", {}).get("approximate")
          and getattr(window, "shown_display", None) == (window.display_revision, window.proof_space)):
        image = canvas.current
    else:
        return None
    if image.isNull():
        return None

    bounds = output_rectangle(recipe, photo["analysis_roi"], photo["source_shape"])
    mapping = SelectionMapping(photo["source_shape"], output_orientation=recipe.get("output_orientation"))
    size = image.width(), image.height()
    width, height = mapping.output_size
    if abs(size[0] / size[1] - width / height) > 2 / min(size):
        return None
    first = mapping.source_to_preview(bounds[:2], size)
    last = mapping.source_to_preview(bounds[2:], size)
    x, x1 = sorted(round(value) for value in (first[0], last[0]))
    y, y1 = sorted(round(value) for value in (first[1], last[1]))
    # Narrow crops can collapse to almost nothing on the sampled locator.
    if min(x1 - x, y1 - y) < 2:
        return None
    return image.copy(x, y, x1 - x, y1 - y)


def _sample_crop_grid(pixels, source, target, step):
    """Map the existing locator grid into exact final-output coordinates."""
    import cv2
    import numpy as np

    from .repair_geometry import _transform

    grid_shape = tuple((extent + step - 1) // step for extent in source["output_size"][::-1])
    if pixels.shape != (*grid_shape, 3) or source["source_shape"] != target["source_shape"]:
        raise ValueError("裁切定位取样与当前源画幅不一致")
    source_matrix, _ = _transform(source)
    target_matrix, (width, height) = _transform(target)
    # Map final sample centres through the source before interpolation. Fill
    # stretched corners outside the actual crop with the renderer's black.
    inverse = np.linalg.inv(target_matrix)
    x = np.arange(0, width, step, dtype=np.float64)[None, :]
    y = np.arange(0, height, step, dtype=np.float64)[:, None]

    def coords(transform):
        return tuple(a * x + b * y + c for a, b, c in transform[:2])

    sx, sy = coords(inverse)
    x0, y0, x1, y1 = target["rectangle"]
    inside = (sx >= x0 - 1e-8) & (sx <= x1 - 1 + 1e-8) & (sy >= y0 - 1e-8) & (sy <= y1 - 1 + 1e-8)
    map_x, map_y = coords((source_matrix @ inverse) / step)
    sampled = cv2.remap(pixels, map_x.astype(np.float32), map_y.astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    sampled[~inside] = 0
    return sampled


def submit_crop_fit_preview(window):
    """Submit a small, colour-matched crop/straighten preview without RAW processing."""
    from types import SimpleNamespace

    from .diagnostics import event
    from .output_orientation import sample_oriented
    from .preview_data import verified_pixels
    from .preview_fit_transport import write_fit_records
    from .preview_grid import post_requires_full_grid
    from .preview_ui import Work
    from .repair_geometry import frame_descriptor
    from .repair_pipeline import preview_repair

    photo, canvas = window.current_photo(), window.canvas
    cache = getattr(window, "crop_locator_cache", None) or {}
    cached_context = cache.get("context", {})
    grid = cache.get("source_grid")
    if (not grid or not photo or window.tasks.busy or window.geometry_role is not None
            or not canvas.fit_mode or not canvas.current or canvas.split or window.compare.isChecked()
            or (canvas.photo_lens and canvas.photo_lens.active)
            or cached_context.get("photo") != photo["id"] or cached_context.get("engine") != photo["engine"]):
        return False
    engine, recipe = photo["engine"], window.project.recipe()
    mode = RESULT_MODES[engine]
    try:
        locator = full_source_recipe(recipe, engine)
        if (locator != cached_context.get("recipe")
                or post_requires_full_grid(locator.get("post_controls", {}).get(mode))):
            return False
        manifest = json.loads(window.project.preview_base().read_text())
        source, target = frame_descriptor(locator, manifest, mode), frame_descriptor(recipe, manifest, mode)
        repair = window.project.repair_context(recipe=recipe)
    except (OSError, ValueError):
        return False
    size, step = target["output_size"], grid["sample_step"]
    if min(size) < step * 2:
        return False
    window.preview_timer.stop()
    window.preview_scopes.cancel()
    window.token += 1
    token = window.token
    window.live.invalidate(token)
    window.preview_activity.start(token, "正在更新裁切与拉直预览…", stage="display", quiet_for=.18)
    context = window.preview_context(recipe=recipe, repair_plan=repair)
    context.update(output_size=size, reference_deferred=True, preview_quality=dict(
        approximate=True, crop_proxy=True, label="已更新裁切与拉直预览",
    ))
    directory = window.live.directory

    def check(_event=None):
        if window.closing or token != window.token:
            raise ValueError("裁切预览已被新的调整替换")

    def render():
        check()
        source_grid = grid
        if "record" in grid:
            pixels, icc = verified_pixels(grid["record"])
            pixels = sample_oriented(pixels, grid["direction"], locator.get("output_orientation"), step)
            pixels.setflags(write=False)
            source_grid = dict(pixels=pixels, icc=icc, sample_step=step)
        pixels = _sample_crop_grid(source_grid["pixels"], source, target, step)
        icc = source_grid["icc"]
        provenance = None
        if repair:
            request = dict(draft_recipe=recipe, mode=mode, repair_plan=repair, directory=str(directory))
            session = SimpleNamespace(manifest=manifest, output_size=size, sample_step=step)
            pixels, _, provenance = preview_repair(request, session, pixels, pixels, icc, on_progress=check)
        check()
        records = write_fit_records(directory, token, pixels, pixels, icc, current_only=True)
        if provenance is not None:
            records[0].update(outside_mask_changed_pixels=0, repair_replay=provenance)
        return records, source_grid

    def shown(_, result):
        records, source_grid = result
        packet = dict(token=token, records=records, reference_deferred=True)
        if window.closing or token != window.token or not window._preview_recipe_matches(context):
            window.live.discard(packet)
            return
        cache["source_grid"] = source_grid
        event("preview_crop_grid_reused", token=token, sample_step=step, repaired=bool(repair))
        window.load_records(*records, sample_step=step, temporary=packet, context=context)

    def failed(_, message):
        if not window.closing and token == window.token and window._preview_recipe_matches(context):
            event("preview_crop_grid_reuse_failed", token=token, message=message)
            window.request_preview()

    work = Work(token, render)
    work.signals.done.connect(shown)
    work.signals.failed.connect(failed)
    window.pool.start(work)
    return True


def restore_crop_scopes(window):
    backup = window.__dict__.pop("_crop_canvas_scopes", None)
    if backup is None:
        return
    shown, scopes = backup
    photo = window.current_photo()
    if photo and photo["id"] == shown["photo"] and photo["engine"] == shown["engine"]:
        current = window.shown_context or {}
        if all(shown.get(key) == current.get(key) for key in (
            "photo", "engine", "recipe_sha256", "repair_plan_sha256",
        )):
            window.scope_panel.set_scope_data(scopes)
            window.scope_panel.caption.setText("输出 RGB · 预览取样")


def _finished(window, token):
    window.crop_locator_pending = None
    if window.preview_activity.active and window.preview_activity.token == token:
        window.preview_activity.finish(token, message="完整正像预览已更新")
    window.set_preview_updating(False)
    recipe = window.project.recipe()
    values = recipe.get("post_controls", {}).get(RESULT_MODES[window.current_photo()["engine"]], {}).get("values", {})
    has_geometry = any(key.startswith("geometry.") for key in values)
    name = "完整正像" if window.current_photo()["source_mode"] == "bw" else "完整彩色正像"
    if window.geometry_role == "crop":
        window.show_preview_notice(name + (" · 未除尘源视图" if window.crop_locator_context.get("repair_source_view") else "")
                                   + (" · 拉直前裁切定位" if has_geometry else " · 源画幅裁切定位"))
        window.status_message("拖动四角调整画幅，拖动圆形把手旋转。")
    else:
        window.show_preview_notice(name + (" · 未除尘源视图" if window.crop_locator_context.get("repair_source_view") else "")
                                   + (" · 拉直前完整画面" if has_geometry else " · 完整画面"))
        window.status_message("在完整正像上绘制高光区域。" + (" 当前显示拉直前的画面。" if has_geometry else ""))


def begin_crop_selection(window):
    if (window.geometry_role != "crop" and window.shown_context
            and window.scope_panel.caption.text() == "输出 RGB · 预览取样"):
        window._crop_canvas_scopes = window.shown_context, window.scope_panel.scope_data
        window.preview_scopes.cancel()
    window.geometry_role = "crop"
    if refresh_crop_frame(window):
        window.canvas_stack.setCurrentWidget(window.selection)
        window.show_region_exit()
        if window.crop_locator_cache.get("reference"):
            request_crop_frame(window)
        return
    # Keep the existing positive visible while preparing the uncropped color
    # view. The gray source locator is reserved for automatic edge detection.
    window.canvas_stack.setCurrentWidget(window.canvas)
    window.show_region_exit()
    request_crop_frame(window)


def request_crop_frame(window):
    """Handle one crop-only preview request, using the existing live process."""
    if _locator_role(window) is None:
        return False
    refreshed = refresh_crop_frame(window)
    cache = getattr(window, "crop_locator_cache", {})
    if refreshed and not cache.get("reference"):
        _finished(window, window.token)
        return True
    if window.tasks.busy:
        return True
    context = locator_context(window)
    if context is None:
        return True
    shown = window.shown_context or {}
    repair = context["repair_plan"]
    if (window.canvas.current is not None
            and shown.get("photo") == context["photo"]
            and shown.get("engine") == context["engine"]
            and getattr(window, "shown_display", None) == (window.display_revision, window.proof_space)
            and shown.get("recipe") == context["recipe"]
            and not shown.get("preview_quality", {}).get("approximate")
            and shown.get("repair_plan_sha256") == (repair["receipt_sha256"] if repair else None)):
        window.preview_timer.stop()
        scopes = getattr(getattr(window, "scope_panel", None), "scope_data", None)
        _install(window, window.canvas.current, context, scope_data=scopes)
        _finished(window, window.token)
        return True
    pending = getattr(window, "crop_locator_pending", None)
    if pending and pending["key"] == context["key"] and pending["token"] == window.token:
        return True
    from .workflow_crop_saved import saved_full_source_record

    # A historic render with the same recipe can predate the highlight mask
    # algorithm. Recompute highlights rather than silently displaying it.
    saved = (saved_full_source_record(window, context["recipe"])
             if context["locator_role"] == "crop" and not context["repair_plan"] else None)
    window.preview_timer.stop()
    window._failed_preview = False
    window.token += 1
    token = window.token
    window.live.invalidate(token)
    context["generation"] = token
    window.preview_contexts[token] = context
    window.preview_contexts = {key: value for key, value in window.preview_contexts.items()
                               if key >= token - 16}
    window.begin_preview_activity(token, "正在准备完整正像预览…")
    window.crop_locator_pending = dict(key=context["key"], token=token)
    if saved:
        display_crop_record(window, token, saved["record"], context,
                            saved_direction=saved.get("output_orientation"))
        return True
    reference = (saved_full_source_record(window, context["recipe"], allow_reference=True)
                 if context["locator_role"] == "crop" else None)
    if reference and not refreshed:
        display_crop_record(window, token, reference["record"], context,
                            saved_direction=reference.get("output_orientation"), reference=True)
    manifest = window.project.preview_base()
    from .draft_preview import _controls

    window.live.submit(dict(token=token, manifest=str(manifest), manifest_sha256=sha256(manifest),
                            mode=RESULT_MODES[context["engine"]], draft_recipe=context["recipe"],
                            controls=_controls(context["recipe"], RESULT_MODES[context["engine"]]),
                            full_resolution=False, crop_locator=True, repair_plan=repair,
                            **({"crop_source_grid": True} if context["locator_role"] == "crop" else {})))
    window.status_message("正在按当前 " + ENGINES[context["engine"]] + " 调色准备完整正像…")
    return True


def receive_crop_frame(window, event, context):
    if event.get("token") != window.token or not _matches(window, context):
        window.live.discard(event)
        return
    display_crop_record(window, event["token"], event["records"][0], context,
                        sample_step=event.get("sample_step", 1), temporary=event,
                        saved_direction=context["recipe"].get("output_orientation"))


def display_crop_record(window, token, record, context, *, sample_step=1, temporary=None,
                        saved_direction=None, reference=False):
    """ICC conversion is asynchronous and bound to photo, recipe and monitor."""
    from .preview_ui import Work
    from .output_orientation import _relative_document, sample_oriented
    from .preview_fit_transport import FIT_TRANSPORT, MAX_FIT_EDGE, read_fit_record
    from .preview_requests_v44 import read_preview_metering
    from .workflow_display import DisplayTarget, named_reference_profile, profile_description

    record = deepcopy(record)
    shape = record.get("shape")
    source_step = 1 if temporary or not shape else max(1, math.ceil(max(shape[:2]) / 2048))
    target, proof = window.display_target, window.proof_space
    if target is None:
        icc = window.display_icc or named_reference_profile("srgb")
        target = DisplayTarget(icc=icc, description="完整正像显示回退", profile_name=profile_description(icc),
                               monitor_icc=None, monitor_description="未确认屏幕ICC", screen_name="显示回退",
                               screen_id=None, source="explicit-display-fallback", assumed=True)
    direction = _relative_document(saved_direction, context["recipe"].get("output_orientation"))
    want_scopes = context["locator_role"] == "crop" and hasattr(window, "scope_panel")

    def read():
        try:
            metering = read_preview_metering(temporary) if temporary else None
            result = window.display_cache.get_image(record, target, proof=proof,
                sample_step=sample_step, source_step=source_step, include_scopes=want_scopes,
                scope_orientation=direction if want_scopes else None)
            image, scopes = result if want_scopes else (result, None)
            if direction["quarter_turns"]:
                image = image.transformed(QTransform().rotate(direction["quarter_turns"] * 90))
            if direction["flip_horizontal"] or direction["flip_vertical"]:
                image = image.mirrored(direction["flip_horizontal"], direction["flip_vertical"])
            source_grid = None
            if not reference and context["locator_role"] == "crop":
                source = (temporary or {}).get("crop_source") or (None if context["repair_plan"] else record)
                if source is not None and source.get("transport") == FIT_TRANSPORT:
                    try:
                        pixels, icc = read_fit_record(source)
                        pixels = sample_oriented(pixels, saved_direction, context["recipe"].get("output_orientation"))
                        pixels.setflags(write=False)
                        source_grid = dict(pixels=pixels, icc=icc, sample_step=sample_step)
                    except (OSError, ValueError) as error:
                        from .diagnostics import event

                        event("crop_source_grid_invalid", token=token, message=str(error))
                elif source is not None and not temporary and shape:
                    # A saved whole-frame TIFF remains an authenticated source.
                    # Read its bounded display grid on demand, once per locator.
                    step = max(1, math.ceil(max(shape[:2]) / MAX_FIT_EDGE))
                    source_grid = dict(record=source, direction=saved_direction, sample_step=step)
            return image, scopes, metering, source_grid
        finally:
            if temporary:
                window.live.discard(temporary)

    work = Work(token, read)

    def shown(result_token, result):
        if result_token != window.token or not _matches(window, context):
            return
        try:
            image, scopes, metering, source_grid = result
            # A slower saved-reference conversion must never replace a fresh
            # current-color frame delivered by the same generation.
            existing = getattr(window, "crop_locator_cache", {})
            if reference and existing.get("key") == context["key"] and not existing.get("reference", True):
                return
            if _install(window, image, context, reference=reference, scope_data=scopes, source_grid=source_grid):
                if metering is not None:
                    window.project.remember_preview_metering(context["recipe"], metering)
                    window.refresh_metering_status()
                if reference:
                    caption = "已保存正像参考" if window.current_photo()["source_mode"] == "bw" else "已保存彩色参考"
                    window.show_preview_notice(caption + " · 正在更新当前调色")
                else:
                    _finished(window, result_token)
                window.image_presented.emit()
        except Exception as error:
            window.preview_failed(result_token, str(error))

    work.signals.done.connect(shown)
    if not reference:
        work.signals.failed.connect(window.preview_failed)
    window.pool.start(work)
