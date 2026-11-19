"""Lossless output-only rotations and reflections, after all source metering.

The original source canvas, density masks, highlight references and SDK
geometry stay unchanged.  Only final RGB16 pixel positions are permuted.
"""

from copy import deepcopy

import numpy as np


FIELDS = {"version", "quarter_turns", "flip_horizontal", "flip_vertical"}
IDENTITY = dict(version=1, quarter_turns=0, flip_horizontal=False, flip_vertical=False)


def checked_document(document=None):
    """Return a strict versioned document, or None for the identity mapping."""
    if document is None:
        return None
    if type(document) is not dict or set(document) != FIELDS:
        raise ValueError("输出方向需要完整的版本化旋转／翻转文档")
    if type(document["version"]) is not int or document["version"] != 1:
        raise ValueError("未知输出方向版本")
    if (
        type(document["quarter_turns"]) is not int
        or not 0 <= document["quarter_turns"] <= 3
    ):
        raise ValueError("输出旋转必须为0至3个顺时针直角")
    if (
        type(document["flip_horizontal"]) is not bool
        or type(document["flip_vertical"]) is not bool
    ):
        raise ValueError("输出翻转开关必须为布尔值")
    q, h, v = (
        document["quarter_turns"],
        document["flip_horizontal"],
        document["flip_vertical"],
    )
    if (q == 0 and not h and not v) or (q == 2 and h and v):
        return None
    return dict(document)


def orient_codes(pixels, document=None):
    """Permute finite RGB16 code positions, without interpolation or encoding."""
    return np.ascontiguousarray(_oriented_view(pixels, document))


def _oriented_view(pixels, document=None):
    """An exact stride-only view; no full-frame pixel allocation."""
    document = checked_document(document)
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.uint16
        or pixels.ndim != 3
        or pixels.shape[2] != 3
        or not pixels.size
    ):
        raise ValueError("输出方向仅接受非空RGB16像素")
    if document is None:
        return pixels
    result = np.rot90(pixels, -document["quarter_turns"])
    if document["flip_horizontal"]:
        result = result[:, ::-1]
    if document["flip_vertical"]:
        result = result[::-1]
    return result


def change_orientation(recipe, action):
    """Apply one visual action to the current displayed output, atomically."""
    result = deepcopy(recipe)
    document = checked_document(result.get("output_orientation")) or dict(IDENTITY)
    if action in ("left", "right"):
        document["quarter_turns"] = (
            document["quarter_turns"] + (1 if action == "right" else -1)
        ) % 4
        document["flip_horizontal"], document["flip_vertical"] = (
            document["flip_vertical"],
            document["flip_horizontal"],
        )
    elif action in ("flip_horizontal", "flip_vertical"):
        document[action] = not document[action]
    else:
        raise ValueError("未知输出方向操作")
    document = checked_document(document)
    if document is None:
        result.pop("output_orientation", None)
    else:
        result["output_orientation"] = document
    return result


def _matrix(document):
    doc = checked_document(document) or IDENTITY
    rotations = (
        ((1, 0), (0, 1)),
        ((0, -1), (1, 0)),
        ((-1, 0), (0, -1)),
        ((0, 1), (-1, 0)),
    )
    rotation = np.array(rotations[doc["quarter_turns"]], dtype=np.int8)
    flips = np.diag(
        (-1 if doc["flip_horizontal"] else 1, -1 if doc["flip_vertical"] else 1)
    )
    return flips @ rotation


def _relative_document(saved_document, current_document):
    target = _matrix(current_document) @ _matrix(saved_document).T
    for q in range(4):
        for horizontal in (False, True):
            candidate = dict(
                version=1,
                quarter_turns=q,
                flip_horizontal=horizontal,
                flip_vertical=False,
            )
            if np.array_equal(_matrix(candidate), target):
                return candidate
    raise ValueError("无法组合输出方向")


def orient_reference(pixels, saved_document, current_document):
    """Align saved output axes in one exact, contiguous pixel permutation."""
    return orient_codes(pixels, _relative_document(saved_document, current_document))


def sample_oriented(pixels, saved_document, current_document, step=1):
    """Exactly full orientation then stride, allocating only the final sample.

    Negative-stride rotation/reflection views retain the correct corner phase
    for odd dimensions.  Subsampling an already reduced image cannot do that.
    """
    if type(step) is not int or step < 1:
        raise ValueError("方向预览采样步长必须为正整数")
    view = _oriented_view(pixels, _relative_document(saved_document, current_document))
    return np.ascontiguousarray(view[::step, ::step])
