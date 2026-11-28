"""Exact detail tiles in the final output grid, without remeasuring a viewport.

The parent DraftPreview owns source identity and full-grid frozen analysis.
Only point operators run on a tile. Spatial finishing uses the existing full
render fallback, then takes the requested final-output rectangle.
"""

from copy import deepcopy
import hashlib
import os
from pathlib import Path

import numpy as np

from .output_orientation import _matrix, _oriented_view, _relative_document, checked_document, sample_oriented
from .output_orientation_cache import _color_recipe
from .preview_data import deflate_rgb_strips, verified_pixels
from .preview_grid import post_requires_full_grid
from .workspace import checked_recipe, output_rectangle


MAX_TILE_PIXELS = 4_194_304
_VIEWPORT_KEYS = {"x", "y", "width", "height", "output_size", "sample_step"}


def output_size_for_recipe(recipe, source_shape, analysis_roi, mode):
    """Return a proven final size, or None for general geometry finishing."""
    values = recipe.get("post_controls", {}).get(mode, {}).get("values", {})
    if any(key.startswith("geometry.") for key in values):
        return None
    x0, y0, x1, y1 = output_rectangle(recipe, analysis_roi, source_shape)
    width, height = x1 - x0, y1 - y0
    direction = checked_document(recipe.get("output_orientation"))
    if direction and direction["quarter_turns"] % 2:
        width, height = height, width
    return [width, height]


def checked_viewport(document, output_size):
    """Validate one bounded tile; coordinates are complete final-output pixels."""
    if type(document) is not dict or set(document) != _VIEWPORT_KEYS:
        raise ValueError("高清视口需要完整的输出像素矩形、尺寸和采样步长")
    if (not isinstance(output_size, (list, tuple)) or len(output_size) != 2
            or any(type(value) is not int or value <= 0 for value in output_size)):
        raise ValueError("高清视口缺少已确认的完整输出尺寸")
    actual = document["output_size"]
    if (type(actual) is not list or len(actual) != 2
            or any(type(value) is not int or value <= 0 for value in actual)
            or actual != list(output_size)):
        raise ValueError("高清视口的完整尺寸与当前输出不一致")
    if any(type(document[key]) is not int for key in (
        "x", "y", "width", "height", "sample_step",
    )):
        raise ValueError("高清视口坐标与采样步长必须为整数")
    x, y, width, height, step = (
        document["x"], document["y"], document["width"],
        document["height"], document["sample_step"],
    )
    if (x < 0 or y < 0 or width < 1 or height < 1 or step < 1
            or x + width > output_size[0] or y + height > output_size[1]):
        raise ValueError("高清视口超出当前完整输出画面")
    if ((width + step - 1) // step) * ((height + step - 1) // step) > MAX_TILE_PIXELS:
        raise ValueError("单次高清视口超过像素预算")
    return deepcopy(document)


def source_rectangle_for_viewport(source_rectangle, orientation, viewport):
    """Invert an exact orthogonal output rectangle back to full source axes."""
    if (not isinstance(source_rectangle, (list, tuple)) or len(source_rectangle) != 4
            or any(type(value) is not int for value in source_rectangle)):
        raise ValueError("高清视口需要完整源坐标裁切")
    sx0, sy0, sx1, sy1 = source_rectangle
    width, height = sx1 - sx0, sy1 - sy0
    if sx0 < 0 or sy0 < 0 or width < 1 or height < 1:
        raise ValueError("高清视口的源坐标裁切无效")
    matrix = _matrix(orientation).astype(np.int64)
    source_corners = np.array([[0, 0], [width - 1, 0],
                               [0, height - 1], [width - 1, height - 1]], dtype=np.int64)
    mapped = source_corners @ matrix.T
    translation = -mapped.min(axis=0)
    output_size = (mapped.max(axis=0) - mapped.min(axis=0) + 1).tolist()
    view = checked_viewport(viewport, output_size)
    x0, y0 = view["x"], view["y"]
    x1, y1 = x0 + view["width"] - 1, y0 + view["height"] - 1
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=np.int64)
    original = (corners - translation) @ matrix
    low, high = original.min(axis=0), original.max(axis=0) + 1
    return [int(low[0]) + sx0, int(low[1]) + sy0,
            int(high[0]) + sx0, int(high[1]) + sy0]


def _stat_identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _identity(path):
    return _stat_identity(Path(path).stat())


def _saved_strip_tile(owner, record, identity, saved_roi, source_roi, direction, step):
    """Read only the needed strips from an authenticated saved RGB16 TIFF."""
    from tifffile import TiffFile

    from .diagnostics import span

    path = Path(record["path"])
    a, b, c, d = saved_roi
    x0, y0, x1, y1 = source_roi
    if (x1 - x0) * (y1 - y0) > MAX_TILE_PIXELS:
        return None
    saved_direction = owner.recipe.get("output_orientation")
    key = (path, record["sha256"], identity)
    with path.open("rb") as stream, TiffFile(stream) as tif:
        if _stat_identity(os.fstat(stream.fileno())) != identity:
            raise ValueError("高清参考成片读取期间发生变化")
        layout = deflate_rgb_strips(tif, identity[2])
        if layout is None:
            return None
        shape, icc, _ = layout
        turns = (saved_direction or {}).get("quarter_turns", 0)
        expected_shape = (c - a, d - b, 3) if turns % 2 else (d - b, c - a, 3)
        if shape != expected_shape:
            raise ValueError("高清参考成片与保存网格不一致")
        if getattr(owner, "_saved_viewport_identity", None) != key:
            with span("verify_saved_viewport_file"):
                stream.seek(0)
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != record["sha256"]:
                raise ValueError("样片 TIFF 已变化，哈希核验失败")
        viewport = dict(x=x0 - a, y=y0 - b, width=x1 - x0, height=y1 - y0,
                        output_size=[c - a, d - b], sample_step=1)
        inverse = _relative_document(saved_direction, None)
        x0, y0, x1, y1 = source_rectangle_for_viewport([0, 0, shape[1], shape[0]], inverse, viewport)
        tile = np.empty((y1 - y0, x1 - x0, 3), dtype=np.uint16)
        page = tif.pages[0]
        rows = page.rowsperstrip
        decode = page.decode
        with span("decode_saved_viewport_strips"):
            for index in range(y0 // rows, (y1 - 1) // rows + 1):
                start = index * rows
                count = page.databytecounts[index]
                stream.seek(page.dataoffsets[index])
                data = stream.read(count)
                if len(data) != count:
                    raise ValueError("高清参考成片的压缩条带不完整")
                pixels, position, _ = decode(data, index)
                if (pixels is None or position != (0, 0, start, 0, 0)
                        or pixels.shape != (1, min(rows, shape[0] - start), shape[1], 3)
                        or pixels.dtype != np.uint16):
                    raise ValueError("高清参考成片的压缩条带不完整")
                top, bottom = max(start, y0), min(start + rows, y1)
                tile[top - y0:bottom - y0] = pixels[0, top - start:bottom - start, x0:x1]
        if _stat_identity(os.fstat(stream.fileno())) != identity:
            raise ValueError("高清参考成片读取期间发生变化")
    if _identity(path) != identity:
        raise ValueError("高清参考成片读取期间发生变化")
    owner._saved_viewport_identity = key
    return sample_oriented(tile, saved_direction, direction, step), icc


def _saved_tile(owner, recipe, mode, source_roi, step):
    """An unchanged saved color result can supply an exact viewport directly."""
    from .draft_preview import _saved_color_valid

    if mode not in owner.manifest["results"] or not _saved_color_valid(owner.manifest, mode):
        return None
    old, new = _color_recipe(owner.recipe, mode), _color_recipe(recipe, mode)
    for doc in (old, new):
        doc.pop("crop", None)
        doc.pop("output_roi", None)
    if old != new:
        return None
    saved_roi = output_rectangle(owner.recipe, owner.manifest["roi"], owner.manifest["source_shape"])
    a, b, c, d = saved_roi
    x0, y0, x1, y1 = source_roi
    if not (a <= x0 < x1 <= c and b <= y0 < y1 <= d):
        return None
    record = owner.manifest["results"][mode]
    identity = _identity(record["path"])
    direction = recipe.get("output_orientation")
    tile = _saved_strip_tile(owner, record, identity, saved_roi, source_roi, direction, step)
    if tile is None:
        pixels, icc = verified_pixels(record)
        inverse = _relative_document(owner.recipe.get("output_orientation"), None)
        pixels = _oriented_view(pixels, inverse)
        if pixels.shape[:2] != (d - b, c - a):
            raise ValueError("高清参考成片与保存网格不一致")
        current = sample_oriented(pixels[y0-b:y1-b, x0-a:x1-a], None, direction, step).copy()
    else:
        current, icc = tile
    if (hashlib.sha256(icc).hexdigest() != owner.manifest["icc_sha256"]
            or _identity(record["path"]) != identity):
        raise ValueError("高清参考成片与保存网格或身份不一致")
    owner.reference_label = "已保存版本"
    owner.metering_receipt = owner._metering_receipt(owner.manifest, mode)
    return current, current, icc


def render_viewport(owner, request, recipe=None, mode=None):
    """Return a point-exact final-grid detail tile, or None for full fallback."""
    if recipe is None or mode is None:
        recipe, mode, _ = owner._validate(request)
    size = output_size_for_recipe(recipe, owner.manifest["source_shape"], owner.manifest["roi"], mode)
    if size is None:
        return None
    viewport = checked_viewport(request["viewport"], size)
    if post_requires_full_grid(recipe.get("post_controls", {}).get(mode)):
        return None
    base_roi = output_rectangle(recipe, owner.manifest["roi"], owner.manifest["source_shape"])
    source_roi = source_rectangle_for_viewport(base_roi, recipe.get("output_orientation"), viewport)
    step = viewport["sample_step"]
    tile_recipe = checked_recipe({**deepcopy(recipe), "crop": True, "output_roi": source_roi})
    result = _saved_tile(owner, recipe, mode, source_roi, step)
    if result is None and mode == "cool-shared-rgb" and recipe.get("nikon_research_controls"):
        from .nikon_draft_preview_v40 import NikonDraftPreview

        if owner.nikon_preview is None:
            owner.nikon_preview = NikonDraftPreview(owner.manifest, owner.digest)
        converted = owner.nikon_preview.render(tile_recipe, sample_step_override=step)
        if converted is not None:
            owner.reference_label = converted.reference_label
            owner.metering_receipt = {}
            result = converted.pixels, converted.reference, converted.icc
    if result is None and mode == "flex-endpoint" and recipe.get("flex_research_controls"):
        from .flex_draft_preview_v40 import FlexDraftPreview

        if owner.flex_preview is None:
            owner.flex_preview = FlexDraftPreview(owner)
        converted = owner.flex_preview.try_render(tile_recipe, sample_step_override=step)
        if converted is not None:
            current, reference, icc, _, receipt = converted
            owner.reference_label = receipt["preview"]["reference_label"]
            owner.metering_receipt = {}
            result = current, reference, icc
    if result is None:
        # Shared measurement may require one complete color anchor. Once
        # prepared, every point-only engine can finish exact viewport samples
        # from it without reusing an older neutral TIFF or density window.
        result = _color_anchor_tile(owner, tile_recipe, mode, source_roi, step)
    if result is None:
        return None
    current, reference, icc = result
    expected = ((viewport["height"] + step - 1) // step,
                (viewport["width"] + step - 1) // step, 3)
    if current.shape != expected or reference.shape != expected:
        raise ValueError("高清视口结果与完整输出坐标网格不一致")
    owner.sample_step = step
    owner.viewport_result = viewport
    owner.output_size = size
    return current, reference, icc


def _color_anchor_tile(owner, recipe, mode, source_roi, step):
    from .draft_preview import _anchor_recipe, _canonical, _verified_anchor_pixels

    anchor_recipe = _anchor_recipe(recipe, mode)
    key = hashlib.sha256(_canonical([mode, anchor_recipe]).encode()).hexdigest()
    if owner.anchor is None or owner.anchor["key"] != key:
        if not owner._saved_neutral_eligible(anchor_recipe, mode):
            return None
        owner._use_saved_anchor(mode, key)
    anchor = owner.anchor
    pixels, icc = _verified_anchor_pixels(anchor)
    if hashlib.sha256(icc).hexdigest() != anchor["manifest"]["icc_sha256"]:
        raise ValueError("高清正像锚点的工作ICC与保存记录不一致")
    x0, y0, x1, y1 = source_roi
    before = pixels[y0:y1, x0:x1]
    sampled = sample_oriented(before, None, recipe.get("output_orientation"), step)
    current = owner._finish(sampled, recipe, mode)
    reference = owner._reference(before, source_roi, recipe, mode, icc)
    reference = sample_oriented(reference, None, recipe.get("output_orientation"), step)
    _verified_anchor_pixels(anchor, check_only=True)
    owner.metering_receipt = owner._metering_receipt(anchor["manifest"], mode)
    return np.array(current, copy=True), np.array(reference, copy=True), icc


def sample_full_viewport(result, document):
    """The exact fallback: run existing full finishing, then select its pixels."""
    current, reference, icc = result
    if current.shape != reference.shape:
        raise ValueError("完整精细预览与参考网格不一致")
    viewport = checked_viewport(document, [current.shape[1], current.shape[0]])
    x, y, width, height, step = (viewport[k] for k in (
        "x", "y", "width", "height", "sample_step",
    ))
    return (np.array(current[y:y+height:step, x:x+width:step], copy=True),
            np.array(reference[y:y+height:step, x:x+width:step], copy=True), icc), viewport
