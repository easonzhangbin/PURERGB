"""Bounded final-color anchors for orientation-only preview requests.

An authenticated completed TIFF can answer a changed direction without RAW
decoding.  A previous full-grid color render can do the same while edits are
unsaved.  Every other recipe change follows the existing processing route.
"""

from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path

import numpy as np

from .output_orientation import sample_oriented


# Match the existing live-worker branches. Inactive engine documents are
# deliberately removed by its request merger, so their absence is not a color
# edit in the selected result. Shared input/density/crop/exposure fields remain.
_ENGINE_DOCUMENTS = {
    "native": {
        "native_controls",
        "native_float_controls",
        "native_spatial_controls",
        "highlight_controls",
        "native_film",
    },
    "cool-shared-rgb": {"cool_headroom_controls", "nikon_research_controls"},
    "flex-endpoint": {"flex_research_controls", "flex_profile"},
}
_ALL_ENGINE_DOCUMENTS = set().union(*_ENGINE_DOCUMENTS.values())
_MODE_MAPS = {"output_controls", "engine_controls", "post_controls"}


def _color_recipe(recipe, mode):
    # density_metering is a shared upstream analysis input. Keep it in every
    # mode's color/cache identity, independently of creative engine documents.
    result = {
        key: value
        for key, value in recipe.items()
        if key != "output_orientation"
        and key not in (_ALL_ENGINE_DOCUMENTS - _ENGINE_DOCUMENTS[mode])
        and key not in _MODE_MAPS
    }
    for key in _MODE_MAPS:
        if mode in recipe.get(key, {}):
            result[key] = {mode: recipe[key][mode]}
    return result


def _immutable_codes(pixels):
    if pixels.dtype != np.uint16 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("方向缓存需RGB16完整成片")
    # A bytes-backed array cannot be made writable by a returned frame's caller.
    return np.frombuffer(pixels.tobytes(), dtype=np.uint16).reshape(pixels.shape)


def _identity(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def write_oriented_display_records(directory, token, lease, before, after, *, reference_deferred=False):
    """Reorient a verified preview grid, preserving working ICC and sample scale."""
    from .preview_data import verified_pixels
    from .preview_fit_transport import FIT_TRANSPORT, MAX_FIT_EDGE, read_fit_record, write_fit_records

    def read(record):
        decode = read_fit_record if record.get("transport") == FIT_TRANSPORT else verified_pixels
        pixels, icc = decode(record)
        source_step = record.get("display_source_step", 1)
        return sample_oriented(pixels, before, after, source_step), icc, source_step

    current, reference = lease.current, lease.reference
    pixels, icc, source_step = read(current)
    if max(pixels.shape[:2]) > MAX_FIT_EDGE:
        raise ValueError("方向预览画面超出适应窗口尺寸")
    compared = pixels
    if not reference_deferred and current["sha256"] != reference["sha256"]:
        compared, other_icc, other_step = read(reference)
        if compared.shape != pixels.shape or other_icc != icc or other_step != source_step:
            raise ValueError("方向预览与对照画面的尺寸或ICC不一致")
    records = write_fit_records(directory, token, pixels, compared, icc)
    if current.get("outside_mask_changed_pixels") == 0:
        # The old repaired pixels and its unchanged exterior are permuted alike.
        records[0]["outside_mask_changed_pixels"] = 0
    return records, lease.sample_step * source_step


_CACHE_NAMES = ("_orientation_saved", "_orientation_completed")
_DEFAULT_BUDGET_BYTES = 2 * 1024**3


def cache_budget_bytes():
    """Worker retention budget, not a hard cap on RAW/SDK/render working RAM."""
    try:
        return max(
            0,
            int(os.environ.get("PURE_RGB_PREVIEW_CACHE_BYTES", _DEFAULT_BUDGET_BYTES)),
        )
    except (TypeError, ValueError):
        return _DEFAULT_BUDGET_BYTES


def _retained_bytes(owner):
    return sum(
        value["pixels"].nbytes
        for name in _CACHE_NAMES
        if (value := getattr(owner, name, None)) is not None
    )


def _drop(owner, name):
    setattr(owner, name, None)
    owner._orientation_lru = [
        key for key in getattr(owner, "_orientation_lru", []) if key != name
    ]


def _touch(owner, name):
    owner._orientation_lru = [
        key for key in getattr(owner, "_orientation_lru", []) if key != name
    ] + [name]


def _trim(owner, reserve=0):
    budget = cache_budget_bytes()
    while _retained_bytes(owner) + reserve > budget:
        order = getattr(owner, "_orientation_lru", [])
        active = [name for name in order if getattr(owner, name, None) is not None]
        if not active:
            active = [
                name for name in _CACHE_NAMES if getattr(owner, name, None) is not None
            ]
        if not active:
            break
        _drop(owner, active[0])
    owner.orientation_cache_bytes = _retained_bytes(owner)
    return budget


def _retain(owner, name, entry, pixels):
    _drop(owner, name)
    budget = _trim(owner)
    if pixels.nbytes > budget:
        return None
    _trim(owner, reserve=pixels.nbytes)
    entry["pixels"] = _immutable_codes(pixels)
    setattr(owner, name, entry)
    _touch(owner, name)
    owner.orientation_cache_bytes = _retained_bytes(owner)
    return entry


def remember_completed(owner, recipe, mode, pixels, icc):
    """Retain one post-color/pre-direction frame if it fits the shared LRU."""
    return (
        _retain(
            owner,
            "_orientation_completed",
            dict(
                recipe=deepcopy(_color_recipe(recipe, mode)), mode=mode, icc=bytes(icc)
            ),
            pixels,
        )
        is not None
    )


def _saved_frame(owner, mode, *, protect_completed=False):
    from .preview_data import verified_pixels

    _trim(owner)
    record = owner.manifest["results"][mode]
    cached = getattr(owner, "_orientation_saved", None)
    if cached is not None and cached["record"] == record:
        if _identity(record["path"]) != cached["identity"]:
            raise ValueError("已保存方向参考文件发生变化")
        _touch(owner, "_orientation_saved")
        return cached["pixels"], cached["icc"]
    before = _identity(record["path"])
    pixels, icc = verified_pixels(record)
    if (
        before != _identity(record["path"])
        or list(pixels.shape) != record["shape"]
        or hashlib.sha256(icc).hexdigest() != record["icc_sha256"]
    ):
        raise ValueError("已保存方向参考的像素／ICC身份不一致")
    completed = getattr(owner, "_orientation_completed", None)
    keep_temporary = bool(
        protect_completed
        and completed is not None
        and completed["pixels"].nbytes + pixels.nbytes > cache_budget_bytes()
    )
    cached = (
        None
        if keep_temporary
        else _retain(
            owner,
            "_orientation_saved",
            dict(record=deepcopy(record), identity=before, icc=bytes(icc)),
            pixels,
        )
    )
    if cached is not None:
        return cached["pixels"], cached["icc"]
    # A low budget still reuses the authenticated TIFF without decoding RAW.
    # The temporary full frame is released after its display sample is copied.
    pixels.setflags(write=False)
    return pixels, bytes(icc)


def cached_preview(owner, recipe, mode, full_resolution):
    """Return orientation-only (current, reference, ICC), or None to process.

    Call only after the normal full recipe/resource and RAW identity checks.
    A direction request never authorizes different active color, masks, crop,
    profile, exposure, source selection or density to reuse an old frame. Only
    documents not read by the selected existing engine branch are ignored.
    """
    from .draft_preview import _saved_color_valid

    _trim(owner)
    if not (
        recipe.get("output_orientation")
        or owner.recipe.get("output_orientation")
        or getattr(owner, "_orientation_used", False)
    ):
        return None
    same_saved = (_color_recipe(recipe, mode) == _color_recipe(owner.recipe, mode)
                  and _saved_color_valid(owner.manifest, mode))
    completed = getattr(owner, "_orientation_completed", None)
    same_completed = bool(
        completed is not None
        and completed["mode"] == mode
        and completed["recipe"] == _color_recipe(recipe, mode)
    )
    if not same_saved and not same_completed:
        return None
    if getattr(owner, "spatial_snapshot", None) != owner.manifest.get(
        "native_spatial_snapshot"
    ):
        raise ValueError("冻结的分区空间快照发生变化")
    if same_completed and not same_saved:
        _touch(owner, "_orientation_completed")
    saved, icc = _saved_frame(
        owner, mode, protect_completed=same_completed and not same_saved
    )
    step = 1 if full_resolution else max(1, math.ceil(max(saved.shape[:2]) / 1400))
    reference = sample_oriented(
        saved,
        owner.recipe.get("output_orientation"),
        recipe.get("output_orientation"),
        step,
    )
    if same_saved:
        current = reference
        route = "verified-saved-final-color"
    else:
        if completed["icc"] != icc or completed["pixels"].flags.writeable:
            raise ValueError("已完成色彩缓存的ICC或只读状态改变")
        current = sample_oriented(
            completed["pixels"], None, recipe.get("output_orientation"), step
        )
        route = "completed-full-grid-final-color"
    current.setflags(write=False)
    reference.setflags(write=False)
    owner.sample_step = step
    owner._orientation_used = True
    owner.orientation_cache_report = dict(
        route=route,
        sample_step=step,
        whole_color_pipeline_reused=True,
        retained_cache_bytes=_retained_bytes(owner),
        retention_budget_bytes=cache_budget_bytes(),
        budget_scope="Orientation anchors only; RAW/SDK working buffers are not capped",
        full_orientation_allocation_before_sampling=False,
    )
    return current, reference, icc
