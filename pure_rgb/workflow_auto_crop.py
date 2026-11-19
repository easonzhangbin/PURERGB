"""Conservative, reversible film-frame crops from a full-source locator.

Detection uses coherent straight edges and the common film-base band. It does
not use metering exclusions, face detection, a cropped display, or RAW pixels.
An uncertain frame is left untouched for manual cropping.
"""

from copy import deepcopy
import json
import math
from pathlib import Path
import uuid

import cv2
import numpy as np
from PIL import Image


MODES = {"film_border": "保留片基文字", "image": "仅裁切有效图像"}


def _gray(image):
    values = np.asarray(image)
    if values.ndim not in (2, 3) or min(values.shape[:2]) < 96:
        raise ValueError("定位图过小，无法可靠识别片边，请手动绘制裁切框。")
    if values.ndim == 3 and values.shape[2] not in (3, 4):
        raise ValueError("自动裁切需要灰度或 RGB 定位图。")
    if values.dtype.kind not in "uif" or not np.isfinite(values).all():
        raise ValueError("定位图包含无效像素。")
    values = values.astype(np.float32)
    if values.ndim == 3:
        values = values[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    low, high = np.quantile(values, [0.005, 0.995])
    if high - low <= max(1e-8, float(np.max(np.abs(values))) * 0.015):
        raise ValueError("片边对比不足，未改变裁切范围；请手动绘制裁切框。")
    values = np.clip((values - low) / (high - low), 0, 1).astype(np.float32)
    height, width = values.shape
    if max(height, width) > 1600:
        ratio = 1600 / max(height, width)
        values = cv2.resize(values, (round(width * ratio), round(height * ratio)),
                            interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(values, (5, 5), 0.8)


def _profiles(gray, bounds):
    x0, y0, x1, y1 = bounds
    width, height = x1 - x0, y1 - y0
    a, b = x0 + int(width * .18), x1 - int(width * .18)
    c, d = y0 + int(height * .18), y1 - int(height * .18)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)) / 4
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)) / 4
    # At least three quarters of each line must support the edge; isolated
    # lettering, dust and scene details cannot define a frame by themselves.
    return np.quantile(gx[c:d], .25, axis=0), np.quantile(gy[:, a:b], .25, axis=1)


def _outer_peak(profile, start, stop, reverse=False):
    start, stop = max(1, start), min(len(profile) - 1, stop)
    part = profile[start:stop]
    if (not len(part) or float(part.max()) < .035
            or float(part.max()) < float(np.median(part)) * 3):
        return None
    threshold = max(.025, float(part.max()) * .32)
    candidates = np.flatnonzero(part >= threshold)
    # Keep the outer cut edge, including the film base and printed lettering.
    return int(start + (candidates[-1] if reverse else candidates[0]))


def _film_bounds(gray):
    height, width = gray.shape
    px, py = _profiles(gray, (0, 0, width, height))
    edges = [
        _outer_peak(px, 1, int(width * .28)),
        _outer_peak(py, 1, int(height * .28)),
        _outer_peak(px, int(width * .72), width - 1, True),
        _outer_peak(py, int(height * .72), height - 1, True),
    ]
    # A film strip can reach a capture edge. Missing margins stay uncropped;
    # the independent inner-frame check below must still pass on all sides.
    result = [0 if edges[0] is None else max(0, edges[0] - 1),
              0 if edges[1] is None else max(0, edges[1] - 1),
              width if edges[2] is None else min(width, edges[2] + 2),
              height if edges[3] is None else min(height, edges[3] + 2)]
    return result, sum(value is not None for value in edges)


def _base_level(gray, bounds):
    x0, y0, x1, y1 = bounds
    width, height = x1 - x0, y1 - y0
    dx, dy = max(2, round(width * .025)), max(2, round(height * .025))
    a, b = x0 + round(width * .2), x1 - round(width * .2)
    c, d = y0 + round(height * .2), y1 - round(height * .2)
    bands = [gray[c:d, x0 + dx:x0 + dx * 2],
             gray[c:d, x1 - dx * 2:x1 - dx],
             gray[y0 + dy:y0 + dy * 3, a:b],
             gray[y1 - dy * 3:y1 - dy, a:b]]
    levels = [float(np.median(band)) for band in bands]
    base = float(np.median(levels))
    if max(levels) - min(levels) > .20:
        raise ValueError("未找到一致的片基边带，已保留当前裁切；请手动绘制。")
    return base


def _inner_peak(gray, profile, bounds, side, base):
    x0, y0, x1, y1 = bounds
    width, height = x1 - x0, y1 - y0
    vertical = side in ("left", "right")
    reverse = side in ("right", "bottom")
    extent = width if vertical else height
    edge = (x1 if reverse else x0) if vertical else (y1 if reverse else y0)
    distance = max(3, round(extent * .012))
    far = round(extent * .23)
    start, stop = (edge - far, edge - distance) if reverse else (edge + distance, edge + far)
    start, stop = max(3, start), min(len(profile) - 3, stop)
    part = profile[start:stop]
    if (not len(part) or float(part.max()) < .012
            or float(part.max()) < float(np.median(part)) * 3):
        raise ValueError("有效画面边缘不够明确，已保留当前裁切；请手动绘制。")
    cross0 = y0 + round(height * .18) if vertical else x0 + round(width * .18)
    cross1 = y1 - round(height * .18) if vertical else x1 - round(width * .18)
    candidates = []
    radius = max(2, round(extent * .004))
    for position in range(start, stop):
        strength = float(profile[position])
        if strength < .012:
            continue
        inner = position - radius if reverse else position + radius
        outer = position + radius if reverse else position - radius
        if vertical:
            inside = gray[cross0:cross1, inner]
            outside = gray[cross0:cross1, outer]
        else:
            inside = gray[inner, cross0:cross1]
            outside = gray[outer, cross0:cross1]
        inner_distance = float(np.median(np.abs(inside - base)))
        outer_distance = float(np.median(np.abs(outside - base)))
        # Reject the cut edge / light path transition: its inner side is the
        # film base. The picture edge must leave a quieter base on the outside.
        improvement = inner_distance - outer_distance
        if improvement < .012 or outer_distance > max(.08, inner_distance * .65):
            continue
        candidates.append((strength * min(1., improvement / .08), position, strength))
    if not candidates:
        raise ValueError("有效画面边缘不够明确，已保留当前裁切；请手动绘制。")
    _, position, strength = max(candidates)
    return position, strength


def detect_crop(image, source_shape, mode="image"):
    """Return full-source integer bounds or fail without proposing a guess."""
    if mode not in MODES:
        raise ValueError("未知自动裁切方式")
    if (not isinstance(source_shape, (list, tuple)) or len(source_shape) not in (2, 3)
            or any(type(value) is not int or value <= 0 for value in source_shape)
            or (len(source_shape) == 3 and source_shape[2] != 3)):
        raise ValueError("自动裁切需要完整源图像尺寸。")
    gray = _gray(image)
    height, width = gray.shape
    source_height, source_width = source_shape[:2]
    source_aspect = source_width / source_height
    if abs(width / height - source_aspect) > source_aspect * .01:
        raise ValueError("定位图与完整源比例不符，未改变裁切范围；请重新生成定位图。")
    film, detected_margins = _film_bounds(gray)
    base = _base_level(gray, film)
    px, py = _profiles(gray, film)
    positions, strengths = [], []
    for side, profile in (("left", px), ("top", py), ("right", px), ("bottom", py)):
        value, strength = _inner_peak(gray, profile, film, side, base)
        positions.append(value)
        strengths.append(strength)
    x0, y0, x1, y1 = positions
    if x1 - x0 < width * .4 or y1 - y0 < height * .4:
        raise ValueError("识别框过小，已保留当前裁切；请手动绘制。")
    # Step just inside the coherent transition so film-base lines do not remain
    # in the image-only proposal. The visible proposal can be redrawn freely.
    image_bounds = [x0 + 2, y0 + 2, x1 - 1, y1 - 1]
    selected = film if mode == "film_border" else image_bounds
    bounds = [max(0, math.floor(selected[0] * source_width / width)),
              max(0, math.floor(selected[1] * source_height / height)),
              min(source_width, math.ceil(selected[2] * source_width / width)),
              min(source_height, math.ceil(selected[3] * source_height / height))]
    return dict(version=1, mode=mode, bounds=bounds, preview_bounds=selected,
                film_bounds=film, image_bounds=image_bounds,
                preview_size=[width, height], source_shape=list(source_shape),
                confidence="clear-frame" if min(strengths) >= .025 else "review-frame",
                detected_margins=detected_margins,
                source="full-source-locator", algorithm="coherent-film-base-v1")


def source_geometry(window):
    photo = window.project.photo()
    source_job = window.project.source_job()
    if source_job is None:
        return None
    if photo.get("geometry") and Path(photo["geometry"]["preview"]).is_file():
        return photo["geometry"]
    path = window.project.workspace.root / "runs" / source_job["id"] / "geometry.json"
    if path.is_file():
        geometry = json.loads(path.read_text())
        if Path(geometry["preview"]).is_file():
            return geometry
    return None


def apply_auto_crop(window, mode):
    """Apply an undoable proposal and leave its source outline open to edit."""
    if mode not in MODES:
        window.error("未知自动裁切方式")
        return
    photo = window.current_photo()
    if not photo or not photo["initial_job"] or window.tasks.busy:
        window.status_message("请先完成源准备，并等待当前处理结束。")
        return
    if not window.flush_advanced():
        window.status_message("请先修正左侧高级参数组合。")
        return
    try:
        geometry = source_geometry(window)
        if geometry is None:
            # Reuse the existing asynchronous geometry worker; no RAW decoding
            # on the UI thread. Do not call begin_selection here: its positive
            # fast path may skip preparation despite the missing full locator.
            from .native import sha256

            source_job = window.project.source_job()
            if source_job is None:
                raise ValueError("当前平场／源配置尚未重新处理完成。")
            manifest = (window.project.workspace.root / "runs"
                        / source_job["id"] / "manifest.json")
            manifest_hash = sha256(manifest)
            if window.live.process is not None:
                window.stop_preview()
            request = dict(
                action="geometry_prepare", manifest=str(manifest),
                manifest_sha256=manifest_hash,
                directory=str(window.project.workspace.root / "geometry-cache" / uuid.uuid4().hex),
            )
            window.pending_geometry = dict(photo=photo["id"], role="crop",
                                           stage=window.step, auto_crop=mode)
            try:
                window.tasks.start(request)
            except Exception:
                window.pending_geometry = None
                raise
            window.status_message("正在准备完整源定位图，随后自动识别裁切范围…")
            window.sync_controls()
            return
        with Image.open(geometry["preview"]) as image:
            report = detect_crop(image, photo["source_shape"], mode)
        recipe = deepcopy(window.project.recipe())
        recipe.update(crop=True, output_roi=report["bounds"])
        window.project.set_recipe(recipe, label="自动裁切 · " + MODES[mode])
        window.last_auto_crop = {**report, "photo": photo["id"]}
        # The gray transmission locator is analysis-only. Open the ordinary
        # complete positive crop editor so auto and manual crops look alike.
        window.begin_selection("crop")
        window.sync_controls()
        window.schedule_preview()
        window.status_message(f"{MODES[mode]}：已识别裁切框，可重新绘制调整；支持撤回。")
    except (ValueError, OSError, KeyError, TypeError, cv2.error) as error:
        window.error(error)
