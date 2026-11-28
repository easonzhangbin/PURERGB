"""Original-pixel viewport for straightening with point-only finishing."""

from copy import deepcopy

import numpy as np

from .output_orientation import checked_document, orient_codes
from .preview_grid import post_requires_full_grid
from .repair_geometry import frame_descriptor
from .viewport_preview_v43 import MAX_TILE_PIXELS, source_rectangle_for_viewport


def _affine_coordinates(rect, size, angle):
    """Use the SDK's global affine rounding before rebasing a source tile."""
    import negpy_core

    x0, y0, x1, y1 = rect
    width, height = size
    matrix = negpy_core.rotation_matrix((width / 2, height / 2), angle, 1.0)
    a, b, tx, d, e, ty = matrix.ravel()
    reciprocal = 1.0 / (a * e - b * d)
    a, b, d, e = e * reciprocal, b * -reciprocal, d * -reciprocal, a * reciprocal
    xs, ys = np.arange(x0, x1), np.arange(y0, y1)

    def quantize(column, row, bias):
        # NegPy warp_affine rounds the 10-bit row and column contributions
        # separately, then interpolates on a 5-bit fixed-point grid.
        columns = np.rint(column * xs * 1024).astype(np.int32)
        rows = np.rint((row * ys + bias) * 1024).astype(np.int32)
        return (rows[:, None] + columns + 16) >> 5

    return quantize(a, b, -a * tx - b * ty), quantize(d, e, -d * tx - e * ty)


def render_straightened_viewport(owner, request, recipe, mode, size):
    """Run only the source pixels and interpolation footprint of a 1:1 tile."""
    if size is None or request["viewport"]["sample_step"] != 1:
        return None
    post = recipe.get("post_controls", {}).get(mode)
    if post is None or not any(key.startswith("geometry.") for key in post["values"]):
        return None
    remaining = {"version": 1, "values": {
        key: value for key, value in post["values"].items() if not key.startswith("geometry.")
    }}
    if post_requires_full_grid(remaining):
        return None

    import negpy_core
    from negpy.features.geometry.logic import compute_geometry_crop_rect, get_manual_rect_coords

    from .diagnostics import span
    from .draft_preview import _anchor_recipe, _controls, _geometry_document
    from .editing_render import encode_positive_rows
    from .live_worker import preview_checkpoint
    from .positive_post import GAMMA, post_linear
    from .viewport_preview_v43 import checked_viewport, read_saved_tile
    from .workspace import output_rectangle

    frame = frame_descriptor(recipe, owner.manifest, mode)
    geometry = frame["geometry"]
    viewport = checked_viewport(request["viewport"], size)
    direction = frame["orientation"]
    finished_size = size[::-1] if direction and direction["quarter_turns"] % 2 else size
    x0, y0, x1, y1 = source_rectangle_for_viewport([0, 0, *finished_size], direction, viewport)
    a, b, c, d = frame["rectangle"]
    pre_direction = checked_document(dict(
        version=1, quarter_turns=(-geometry["rotation"] // 90) % 4,
        flip_horizontal=geometry["flip_horizontal"], flip_vertical=geometry["flip_vertical"],
    ))
    width, height = c - a, d - b
    if pre_direction and pre_direction["quarter_turns"] % 2:
        width, height = height, width
    before = deepcopy(recipe)
    before.pop("output_orientation", None)
    if pre_direction:
        before["output_orientation"] = pre_direction
    before["post_controls"].pop(mode)
    if not before["post_controls"]:
        before.pop("post_controls")
    want_reference = not request.get("defer_reference")
    record = owner.manifest["results"].get(mode) if want_reference else None
    saved_roi = output_rectangle(owner.recipe, owner.manifest["roi"], owner.manifest["source_shape"])
    sx0, sy0, sx1, sy1 = saved_roi
    if record is not None and (
        _geometry_document(owner.recipe, mode) or not (sx0 <= a < c <= sx1 and sy0 <= b < d <= sy1)
    ):
        record = None
    neutral = None
    if want_reference and record is None:
        neutral = _anchor_recipe(before, mode)
        for key in ("crop", "output_roi", "output_orientation"):
            if key in before:
                neutral[key] = before[key]
    angle = geometry["fine_rotation"]
    left = top = 0
    if geometry["crop_to_valid"]:
        crop = compute_geometry_crop_rect(angle, 0., 0., width, height)
        top, _, left, _ = get_manual_rect_coords((height, width), crop)

    def tile(rect):
        if angle:
            map_x, map_y = _affine_coordinates(rect, (width, height), angle)
            # Keep fractional maps even outside the source. The SDK applies
            # replicated borders with its original bilinear arithmetic.
            ax = int(np.clip(map_x.min() // 32, 0, width - 1))
            ay = int(np.clip(map_y.min() // 32, 0, height - 1))
            bx = int(np.clip(map_x.max() // 32 + 1, 0, width - 1)) + 1
            by = int(np.clip(map_y.max() // 32 + 1, 0, height - 1)) + 1
        else:
            ax, ay, bx, by = rect
        if (bx - ax) * (by - ay) > MAX_TILE_PIXELS:
            return None
        preview_checkpoint()()
        source_view = dict(x=ax, y=ay, width=bx - ax, height=by - ay,
                           output_size=[width, height], sample_step=1)

        def read_source(document):
            return owner.render({
                **request, "draft_recipe": document, "controls": _controls(document, mode),
                "repair_plan": None, "viewport": source_view, "defer_reference": True,
            }, allow_full_viewport=False)

        source = read_source(before)
        if source is None:
            return None
        pixels, _, icc = source
        compare = None
        if want_reference:
            if record is not None:
                raw_roi = source_rectangle_for_viewport(frame["rectangle"], pre_direction, source_view)
                compare = read_saved_tile(owner, record, saved_roi, raw_roi, pre_direction, 1)
            else:
                other_source = read_source(neutral)
                if other_source is None:
                    return None
                compare = other_source[0], other_source[2]
            if compare[1] != icc:
                raise ValueError("拉直视口与对照画面的工作 ICC 不一致")

        def finish(codes, document):
            linear = (codes.astype(np.float32) / 65535) ** GAMMA
            if angle:
                linear = negpy_core.remap(
                    linear, (map_x - ax * 32).astype(np.float32) / 32,
                    (map_y - ay * 32).astype(np.float32) / 32,
                )
            if document is not None:
                linear, _ = post_linear(linear, document, collect_ranges=False)
            return encode_positive_rows(np.maximum(linear, 0))

        current = finish(pixels, remaining)
        return current, current if compare is None else finish(compare[0], None), icc

    with span("preview_straightened_viewport", pixels=(x1 - x0) * (y1 - y0)):
        first = tile((x0 + left, y0 + top, x1 + left, y1 + top))
        if first is not None:
            pixels, reference, icc = first
        else:
            # One nearly screen-sized rotated bounding box can exceed the
            # source pixel budget. Split into bounded tiles, with no seams.
            pixels = np.empty((y1 - y0, x1 - x0, 3), dtype=np.uint16)
            reference = np.empty_like(pixels) if want_reference else pixels
            icc = None
            for y in range(y0, y1, 1200):
                for x in range(x0, x1, 1200):
                    right, bottom = min(x + 1200, x1), min(y + 1200, y1)
                    part = tile((x + left, y + top, right + left, bottom + top))
                    if part is None:
                        return None
                    current_tile, reference_tile, profile = part
                    if icc is not None and icc != profile:
                        raise ValueError("拉直视口的工作 ICC 不一致")
                    icc = profile
                    block = np.s_[y-y0:bottom-y0, x-x0:right-x0]
                    pixels[block] = current_tile
                    if want_reference:
                        reference[block] = reference_tile
    current = orient_codes(pixels, direction)
    compared = orient_codes(reference, direction) if want_reference else current
    owner.sample_step, owner.viewport_result, owner.output_size = 1, viewport, size
    owner.reference_deferred = not want_reference
    if want_reference:
        owner.reference_label = "已保存版本" if record is not None else "调整前（当前草稿）"
    return current, compared, icc
