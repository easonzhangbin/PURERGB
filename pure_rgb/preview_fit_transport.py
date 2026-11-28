"""Bounded RGB16 frames for the desktop worker's fit-preview handoff."""

import hashlib
import os
from pathlib import Path
import struct

import numpy as np


FIT_TRANSPORT = "rgb16-v1"
MAX_FIT_EDGE = 1600
_HEADER = struct.Struct("<8sIII")
_MAGIC = b"PRGB16V1"
_MAX_ICC = 4 * 1024**2
_REFERENCE_ANCHOR = ".fit-reference-anchor.rgb16"


def can_transport_fit(request, current, reference, icc, *, repair_ready=False):
    return (
        request.get("preview_transport") == FIT_TRANSPORT
        and not any(request.get(key) for key in ("full_resolution", "viewport"))
        and (not request.get("crop_locator") or current is reference)
        and (request.get("repair_plan") is None or repair_ready)
        and isinstance(icc, bytes) and 128 <= len(icc) <= _MAX_ICC
        and all(
            image.dtype == np.dtype(np.uint16) and image.ndim == 3 and image.shape[2] == 3
            and 0 < min(image.shape[:2]) <= max(image.shape[:2]) <= MAX_FIT_EDGE
            for image in (current, reference)
        )
    )


def fit_record_paths(directory, token):
    directory = Path(directory)
    return [directory / f"{token}-{role}.rgb16" for role in ("current", "reference")]


def _write_frame(path, image, icc):
    height, width = image.shape[:2]
    header = _HEADER.pack(_MAGIC, width, height, len(icc))
    pixels = memoryview(np.ascontiguousarray(image, dtype="<u2")).cast("B")
    digest = hashlib.sha256()
    partial = path.with_suffix(".rgb16.part")
    stream = partial.open("xb")
    try:
        with stream:
            for block in (header, icc, pixels):
                stream.write(block)
                digest.update(block)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {"path": str(path), "sha256": digest.hexdigest(), "transport": FIT_TRANSPORT}


def _immutable_pixels(image):
    owner = image
    while isinstance(owner, np.ndarray) and not owner.flags.writeable:
        owner = owner.base
    return isinstance(owner, bytes)


def _payload_identity(path):
    stat = path.stat()
    # Allocating and adopting a hard link changes ctime. This is only a
    # worker-side reuse hint. The desktop authenticates the published packet.
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def discard_fit_reference_anchor(directory):
    (Path(directory) / _REFERENCE_ANCHOR).unlink(missing_ok=True)


class FitReferenceCache:
    """Keep one private encoded reference for immutable renderer-owned pixels."""

    def __init__(self):
        self._entry = None

    def clear(self):
        entry, self._entry = self._entry, None
        if entry is not None:
            try:
                entry[0].unlink(missing_ok=True)
            except OSError:
                pass

    def write(self, path, image, icc):
        if not _immutable_pixels(image):
            return _write_frame(path, image, icc)
        anchor = path.parent / _REFERENCE_ANCHOR
        if self._entry is not None:
            previous, pixels, profile, digest, identity = self._entry
            if previous == anchor and pixels is image and profile == icc:
                try:
                    if not anchor.is_symlink() and _payload_identity(anchor) == identity:
                        os.link(anchor, path)
                        return {"path": str(path), "sha256": digest, "transport": FIT_TRANSPORT}
                except OSError:
                    pass
        self.clear()
        record = _write_frame(path, image, icc)
        try:
            os.link(path, anchor)
            self._entry = (anchor, image, icc, record["sha256"], _payload_identity(anchor))
        except OSError:
            # A filesystem may not support links. The unique packet is complete.
            pass
        return record


def write_fit_records(directory, token, current, reference, icc, *, current_only=False, reference_cache=None):
    paths = fit_record_paths(directory, token)
    first = _write_frame(paths[0], current, icc)
    try:
        same_frame = current_only or current is reference or (
            current.shape == reference.shape
            and np.array_equal(current[::64, ::64], reference[::64, ::64])
            and np.array_equal(current, reference)
        )
        if same_frame:
            second = first.copy()
        elif reference_cache is not None:
            second = reference_cache.write(paths[1], reference, icc)
        else:
            second = _write_frame(paths[1], reference, icc)
    except BaseException:
        paths[0].unlink(missing_ok=True)
        raise
    return [first, second]


def discard_undelivered_fit(directory, token):
    for path in (*fit_record_paths(directory, token), *fit_record_paths(directory, f"{token}-crop-source")):
        for temporary in (path, path.with_suffix(".rgb16.part"), path.with_suffix(".metering.json")):
            temporary.unlink(missing_ok=True)


def read_fit_record(record):
    path = Path(record["path"])
    with path.open("rb") as stream:
        header = stream.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise ValueError("预览帧不完整")
        marker, width, height, icc_size = _HEADER.unpack(header)
        if (marker != _MAGIC or not 0 < min(width, height) <= max(width, height) <= MAX_FIT_EDGE
                or not 128 <= icc_size <= _MAX_ICC):
            raise ValueError("预览帧格式或尺寸无效")
        icc = stream.read(icc_size)
        size = width * height * 6
        pixels = stream.read(size + 1)
    if len(icc) != icc_size or len(pixels) != size:
        raise ValueError("预览帧长度无效")
    digest = hashlib.sha256(header)
    digest.update(icc)
    digest.update(pixels)
    if digest.hexdigest() != record["sha256"]:
        raise ValueError("预览帧哈希核验失败")
    array = np.frombuffer(pixels, dtype="<u2").reshape(height, width, 3).astype(np.uint16, copy=False)
    array.setflags(write=False)
    return array, icc
