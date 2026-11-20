"""Recover a failed first picture detection using its existing source locator."""

from copy import deepcopy
import json
from pathlib import Path

from .workspace import source_recipe_identity


def offer_import_roi(window, request, message):
    if "No reliable picture rectangle" not in str(message):
        return False
    photo = window.current_photo()
    binding = window.project.data["jobs"].get(request.get("id"), {})
    if not photo or photo["initial_job"] or binding.get("photo") != photo["id"]:
        return False
    active = window.project.recipe()
    source = source_recipe_identity(active)
    if active["roi"] is not None or source != source_recipe_identity(request["recipe"]):
        return False
    directory = Path(request["directory"]).resolve()
    geometry = json.loads((directory / "geometry.json").read_text())
    if (Path(geometry.get("positive_preview", "")).resolve() != directory / "geometry-positive.png"
            or any(type(geometry.get(key)) is not int or geometry[key] < 16
                   for key in ("width", "height"))):
        raise ValueError("首次导入正像定位图不可用，请重试导入")
    # Detection failure is recoverable in the same frame-and-measurement
    # stage where the user expects to inspect the picture.  Enter it before
    # activating the locator because set_stage() clears the previous tool.
    if window.step != 2:
        window.set_stage(2)
        if window.step != 2:
            return False
    window.selection.set_geometry(geometry, source_id=photo["id"], positive=True)
    # Keep a source-bound retry so exiting the tool does not strand a photo
    # when the user later clicks the next stage.
    window._import_roi_retry = deepcopy(request)
    window._import_roi_context = dict(photo=photo["id"], geometry=geometry,
                                      source_key=source)
    window.geometry_role = "import_roi"
    window.selection.set_hover_model(None)
    window.selection.shape_kind = "rectangle"
    window.selection.drawing_enabled = True
    window.canvas_stack.setCurrentWidget(window.selection)
    window.show_region_exit()
    window.sync_controls()
    window.status_message("自动画面定位未通过。请在正像上拖框圈出有效画面，松开后自动继续导入与密度测量。")
    return True


def accept_import_roi(window, bounds):
    context = getattr(window, "_import_roi_context", None)
    photo = window.current_photo()
    if (not context or not photo or context["photo"] != photo["id"]
            or context["source_key"] != source_recipe_identity(window.project.recipe())):
        raise ValueError("来源已改变，请重新核对导入")
    x0, y0, x1, y1 = bounds
    geometry = context["geometry"]
    if not (0 <= x0 < x1 <= geometry["width"] and 0 <= y0 < y1 <= geometry["height"]
            and x1 - x0 >= 16 and y1 - y0 >= 16):
        raise ValueError("请框选至少 16 × 16 源像素的有效图像区域")
    recipe = window.project.recipe()
    recipe["roi"] = list(bounds)
    window.project.set_recipe(recipe, label="首次导入手动画面范围")
    window.__dict__.pop("_import_roi_retry", None)
    window.finish_selection()
    window.sync_controls()
    window.confirm_import()
