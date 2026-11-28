"""Screen-grid spatial finishing for fit previews, with explicit quality receipts.

Full-source density statistics and color point rendering retain their existing
contracts. Only this fit-only positive post layer uses the smaller screen grid;
saved output, full-resolution requests and detail viewports never enter here.
"""

from copy import deepcopy

import numpy as np

from .output_orientation import checked_document, orient_codes
from .positive_post import post_codes
from .preview_grid import post_requires_full_grid
from .workspace import checked_recipe, output_rectangle


def exact_output_size(recipe, source_shape, analysis_roi, mode):
    """Use the SDK's own canvas/crop rules without allocating a full raster."""
    from negpy.features.geometry.logic import compute_geometry_crop_rect, get_manual_rect_coords

    x0, y0, x1, y1 = output_rectangle(recipe, analysis_roi, source_shape)
    width, height = x1 - x0, y1 - y0
    geometry = {key.split(".", 1)[1]: value
                for key, value in recipe.get("post_controls", {}).get(mode, {}).get("values", {}).items()
                if key.startswith("geometry.")}
    if int(geometry.get("rotation", 0)) // 90 % 2:
        width, height = height, width
    if geometry.get("crop_to_valid", False):
        rect = compute_geometry_crop_rect(geometry.get("fine_rotation", 0),
            geometry.get("converge_v", 0), geometry.get("converge_h", 0), width, height)
        a, b, c, d = get_manual_rect_coords((height, width), rect)
        width, height = d - c, b - a
    if min(width, height) < 1:
        raise ValueError("几何变换后的有效画幅为空")
    direction = checked_document(recipe.get("output_orientation"))
    if direction and direction["quarter_turns"] % 2:
        width, height = height, width
    return [int(width), int(height)]


def _screen_shape(pixels, size):
    """Keep the proven final aspect despite odd source/crop rounding at a LOD."""
    if pixels.shape[:2] == (size[1], size[0]):
        return np.array(pixels, copy=True)
    import negpy_core

    resized = negpy_core.resize_area(np.ascontiguousarray(pixels, dtype=np.float32), tuple(size))
    return np.rint(np.clip(resized, 0, 65535)).astype(np.uint16)


def render_interactive_post(owner, request, recipe, mode):
    document = recipe.get("post_controls", {}).get(mode)
    if (request.get("full_resolution", False) or "viewport" in request
            or not post_requires_full_grid(document)):
        return None
    from .draft_preview import _controls, _geometry_document
    from .live_worker import preview_checkpoint

    checkpoint = preview_checkpoint()
    checkpoint()
    neutral = deepcopy(recipe)
    post = neutral.get("post_controls", {})
    post.pop(mode, None)
    if not post:
        neutral.pop("post_controls", None)
    neutral.pop("output_orientation", None)
    neutral = checked_recipe(neutral)
    neutral_request = {**request, "draft_recipe": neutral, "controls": _controls(neutral, mode)}
    neutral_request.pop("interactive_edge", None)
    # Unsupported Native film/spatial combinations keep the original full-grid
    # fallback. Their settings are never removed to manufacture eligibility.
    baseline = owner.density_preview.try_render(neutral_request, neutral, mode)
    if baseline is None and neutral.get("measurement_method") == "negpy":
        from .native_original_preview import try_original_density_screen

        baseline = try_original_density_screen(owner, neutral_request, neutral, mode)
    if baseline is None:
        return None
    checkpoint()
    step = owner.sample_step
    owner.density_preview._progress("interactive_post", 0.96, "正在更新交互预览；完整源密度测量保持不变…")
    current, reference, icc = baseline
    current, current_receipt = post_codes(current, document, sample_step=step, collect_ranges=False)
    checkpoint()
    reference, _ = post_codes(reference, _geometry_document(recipe, mode),
                              sample_step=step, collect_ranges=False)
    checkpoint()
    direction = recipe.get("output_orientation")
    current, reference = orient_codes(current, direction), orient_codes(reference, direction)
    size = exact_output_size(recipe, owner.manifest["source_shape"], owner.manifest["roi"], mode)
    sampled_size = [(value + step - 1) // step for value in size]
    current, reference = _screen_shape(current, sampled_size), _screen_shape(reference, sampled_size)
    checkpoint()
    owner._validate(request)
    owner.output_size = size
    owner.viewport_result = None
    owner.preview_quality = dict(
        mode="interactive", approximate=True, label="交互预览 · 100%查看精细效果",
        full_support_measurement=True, sample_step=step,
        full_output_size=size, screen_shape=list(current.shape),
        sampling_scope="Spatial positive finishing runs on the screen grid; not an exact saved-image subset",
        full_resolution_and_saved_output_unchanged=True, reference_geometry_synchronized=True,
        post_stages=current_receipt.get("stages", []),
    )
    if owner.density_preview.last_receipt is not None:
        receipt = deepcopy(owner.density_preview.last_receipt)
        receipt.update(color_grid_route=receipt["route"], route="full-support-screen-post-interactive", approximate=True)
        owner.density_preview.last_receipt = receipt
    return current, reference, icc
