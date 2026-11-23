"""Carry authenticated dust coverage through reversible output geometry.

Descriptors contain geometry only; the caller must authenticate RAW/calibration
identity and the original review. Pixel centres, not selection-box boundaries,
are mapped here. The original mask is never changed. Areas outside the original
reviewed source rectangle receive no invented repair coverage.
"""

from copy import deepcopy
import math

import cv2
import numpy as np

from .output_orientation import checked_document
from .workspace import output_rectangle


_MODES = {"native", "cool-shared-rgb", "flex-endpoint"}
_DEFAULTS = dict(rotation=0, flip_horizontal=False, flip_vertical=False,
                 fine_rotation=0.0, converge_v=0.0, converge_h=0.0,
                 distortion_k1=0.0, crop_to_valid=False)
_FRAME_KEYS = {"version", "source_shape", "rectangle", "geometry", "orientation", "output_size"}


def _integers(values, length, name):
    if (not isinstance(values, (list, tuple)) or len(values) != length
            or any(type(value) is not int for value in values)):
        raise ValueError(name)
    return list(values)


def _geometry(values):
    if not isinstance(values, dict) or set(values) - set(_DEFAULTS):
        raise ValueError("除尘继承遇到未知输出几何参数")
    result = {**_DEFAULTS, **values}
    for key in ("flip_horizontal", "flip_vertical", "crop_to_valid"):
        if type(result[key]) is not bool:
            raise ValueError("除尘继承的输出几何开关无效")
    for key in ("rotation", "fine_rotation", "converge_v", "converge_h", "distortion_k1"):
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("除尘继承的输出几何数值无效")
    if result["rotation"] not in (0, 90, 180, 270) or not -45 <= result["fine_rotation"] <= 45:
        raise ValueError("除尘继承的旋转角度超出支持范围")
    if any(result[key] != 0 for key in ("converge_v", "converge_h", "distortion_k1")):
        raise ValueError("透视或径向畸变改变后暂不能继承除尘，请保留原几何或重新审核")
    result["rotation"] = int(result["rotation"])
    return result


def _translate(x, y):
    return np.array([[1., 0., x], [0., 1., y], [0., 0., 1.]])


def _orientation(width, height, quarter_turns, horizontal, vertical):
    """Pixel-index D4 matrix; positive quarter turns are counter-clockwise."""
    k = quarter_turns % 4
    matrices = (
        np.eye(3),
        np.array([[0., 1., 0.], [-1., 0., width - 1.], [0., 0., 1.]]),
        np.array([[-1., 0., width - 1.], [0., -1., height - 1.], [0., 0., 1.]]),
        np.array([[0., -1., height - 1.], [1., 0., 0.], [0., 0., 1.]]),
    )
    matrix = matrices[k]
    if k % 2:
        width, height = height, width
    if horizontal:
        matrix = np.array([[-1., 0., width - 1.], [0., 1., 0.], [0., 0., 1.]]) @ matrix
    if vertical:
        matrix = np.array([[1., 0., 0.], [0., -1., height - 1.], [0., 0., 1.]]) @ matrix
    return matrix, width, height


def _transform(frame):
    import negpy_core
    from negpy.features.geometry.logic import compute_geometry_crop_rect, get_manual_rect_coords

    x0, y0, x1, y1 = frame["rectangle"]
    width, height = x1 - x0, y1 - y0
    geometry = frame["geometry"]
    matrix = _translate(-x0, -y0)
    orthogonal, width, height = _orientation(
        width, height, geometry["rotation"] // 90,
        geometry["flip_horizontal"], geometry["flip_vertical"],
    )
    matrix = orthogonal @ matrix
    angle = geometry["fine_rotation"]
    if angle:
        rotation = np.vstack((negpy_core.rotation_matrix((width / 2., height / 2.), angle, 1.),
                              [0., 0., 1.]))
        matrix = rotation @ matrix
    if geometry["crop_to_valid"]:
        rectangle = compute_geometry_crop_rect(angle, 0., 0., width, height)
        top, bottom, left, right = get_manual_rect_coords((height, width), rectangle)
        if bottom <= top or right <= left:
            raise ValueError("拉直后没有可继承除尘的有效画幅")
        matrix = _translate(-left, -top) @ matrix
        width, height = right - left, bottom - top
    direction = frame["orientation"]
    if direction:
        orthogonal, width, height = _orientation(
            width, height, -direction["quarter_turns"],
            direction["flip_horizontal"], direction["flip_vertical"],
        )
        matrix = orthogonal @ matrix
    return matrix, [width, height]


def _checked_frame(frame):
    if (not isinstance(frame, dict) or set(frame) != _FRAME_KEYS
            or type(frame.get("version")) is not int or frame["version"] != 1):
        raise ValueError("除尘坐标描述不完整")
    result = deepcopy(frame)
    shape = _integers(result["source_shape"], 3, "除尘坐标需要完整源RGB尺寸")
    if min(shape) <= 0 or shape[2] != 3:
        raise ValueError("除尘坐标需要完整源RGB尺寸")
    rectangle = _integers(result["rectangle"], 4, "除尘坐标需要整数源裁切范围")
    x0, y0, x1, y1 = rectangle
    if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
        raise ValueError("除尘坐标的裁切超出完整源画面")
    result.update(source_shape=shape, rectangle=rectangle,
                  geometry=_geometry(result["geometry"]),
                  orientation=checked_document(result["orientation"]))
    size = _integers(result["output_size"], 2, "除尘坐标需要有效输出尺寸")
    if min(size) <= 0 or _transform(result)[1] != size:
        raise ValueError("除尘坐标描述与实际输出尺寸不符")
    return result


def frame_descriptor(recipe, manifest, mode):
    """Describe this recipe's final pixel grid using the renderer's exact rules."""
    try:
        if mode not in _MODES or not isinstance(recipe, dict) or not isinstance(manifest, dict):
            raise ValueError("除尘坐标需要有效配方及色彩模式")
        shape = _integers(manifest["source_shape"], 3, "除尘坐标需要完整源RGB尺寸")
        if min(shape) <= 0 or shape[2] != 3 or type(recipe.get("crop")) is not bool:
            raise ValueError("除尘坐标的源尺寸或裁切开关无效")
        rectangle = output_rectangle(recipe, recipe.get("roi") or manifest.get("roi"), shape)
        values = recipe.get("post_controls", {}).get(mode, {}).get("values", {})
        geometry = _geometry({key[9:]: value for key, value in values.items()
                              if key.startswith("geometry.")})
        result = dict(version=1, source_shape=shape, rectangle=rectangle,
                      geometry=geometry, orientation=checked_document(recipe.get("output_orientation")),
                      output_size=[])
        result["output_size"] = _transform(result)[1]
        return _checked_frame(result)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("当前画面缺少完整除尘坐标证据") from error


def sample_repair_mask(mask, source_frame, target_frame, *, output_size=None,
                       sample_step=1, viewport=None):
    """Map an original reviewed mask to the requested current output grid.

    Orthogonal changes preserve exact mask bytes. Interpolated geometry retains
    every nonzero bilinear footprint as binary coverage, so a one-pixel dust
    selection cannot disappear through rounding to an 8-bit mask. Sampling uses
    global final-output coordinates, including a viewport's exact x/y phase.
    """
    from .viewport_preview_v43 import checked_viewport

    source, target = _checked_frame(source_frame), _checked_frame(target_frame)
    if source["source_shape"] != target["source_shape"]:
        raise ValueError("除尘坐标的原始源尺寸已改变")
    if (not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8
            or list(mask.shape[::-1]) != source["output_size"]):
        raise ValueError("除尘蒙版尺寸与原审核坐标不一致")
    if type(sample_step) is not int or sample_step < 1:
        raise ValueError("除尘蒙版采样步长必须为正整数")
    size = target["output_size"]
    if output_size is not None and (_integers(output_size, 2, "除尘输出尺寸无效") != size):
        raise ValueError("除尘坐标映射与当前输出尺寸不符")
    if viewport is None:
        grid = dict(x=0, y=0, width=size[0], height=size[1],
                    sample_step=sample_step, output_size=list(size))
    else:
        grid = checked_viewport(viewport, size)
        if grid["sample_step"] != sample_step:
            raise ValueError("除尘视口采样步长与当前预览不一致")
    old_matrix, _ = _transform(source)
    new_matrix, _ = _transform(target)
    inverse_new = np.linalg.inv(new_matrix)
    lookup = old_matrix @ inverse_new
    exact = np.allclose(lookup, np.rint(lookup), rtol=0, atol=1e-10)
    if exact:
        lookup = np.rint(lookup).astype(np.int64)
    height = (grid["height"] + sample_step - 1) // sample_step
    width = (grid["width"] + sample_step - 1) // sample_step
    result = np.zeros((height, width), dtype=np.uint8)
    # Bound both coordinate maps and float source tiles, even for 96MP masks.
    tile = max(1, 512 // sample_step)
    old_height, old_width = mask.shape
    sx0, sy0, sx1, sy1 = source["rectangle"]
    unchanged_support = (source["rectangle"] == target["rectangle"]
                         and source["geometry"] == target["geometry"])
    for row in range(0, height, tile):
        ys = grid["y"] + np.arange(row, min(height, row + tile), dtype=np.int64) * sample_step
        for column in range(0, width, tile):
            xs = grid["x"] + np.arange(column, min(width, column + tile), dtype=np.int64) * sample_step
            x, y = xs[None, :], ys[:, None]
            old_x = lookup[0, 0] * x + lookup[0, 1] * y + lookup[0, 2]
            old_y = lookup[1, 0] * x + lookup[1, 1] * y + lookup[1, 2]
            source_x = inverse_new[0, 0] * x + inverse_new[0, 1] * y + inverse_new[0, 2]
            source_y = inverse_new[1, 0] * x + inverse_new[1, 1] * y + inverse_new[1, 2]
            valid = ((old_x >= 0) & (old_x <= old_width - 1)
                     & (old_y >= 0) & (old_y <= old_height - 1))
            if not unchanged_support:
                valid &= ((source_x >= sx0 - 1e-8) & (source_x <= sx1 - 1 + 1e-8)
                          & (source_y >= sy0 - 1e-8) & (source_y <= sy1 - 1 + 1e-8))
            if not np.any(valid):
                continue
            if exact:
                sampled = mask[np.clip(old_y, 0, old_height - 1), np.clip(old_x, 0, old_width - 1)]
            else:
                left = max(0, int(np.floor(old_x[valid].min())) - 1)
                top = max(0, int(np.floor(old_y[valid].min())) - 1)
                right = min(old_width, int(np.ceil(old_x[valid].max())) + 2)
                bottom = min(old_height, int(np.ceil(old_y[valid].max())) + 2)
                # OpenCV uses a 1/32 interpolation grid. Quantize globally before
                # moving to a source tile, making preview/viewport phases stable.
                map_x = (np.rint(old_x * 32) / 32 - left).astype(np.float32)
                map_y = (np.rint(old_y * 32) / 32 - top).astype(np.float32)
                map_x, map_y = np.broadcast_arrays(map_x, map_y)
                coverage = cv2.remap(mask[top:bottom, left:right].astype(np.float32),
                                     map_x, map_y, cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                sampled = np.where(coverage > 0, 255, 0).astype(np.uint8)
            destination = result[row:row + len(ys), column:column + len(xs)]
            destination[:] = np.where(valid, sampled, 0)
    grid["geometry_mapping"] = "exact-orthogonal" if exact else "affine-bilinear-coverage"
    return result, grid
