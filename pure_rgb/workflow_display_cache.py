"""Bounded, shared display images after verified TIFF and ICC/proof conversion.

Finished QImages and small authenticated RGB16 display grids share one budget.
This does not open RAW sources, change saved images, or bound render memory.
"""

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import re
import sys
import threading

import numpy as np
from PIL import Image
from PyQt6.QtGui import QColorSpace, QImage

from .preview_data import verified_pixels
from .preview_fit_transport import FIT_TRANSPORT, read_fit_record
from .workflow_display import named_reference_profile, profile_description
from .workflow_color_parallel import transform_packed_rgb, transform_rgb_image


@dataclass(frozen=True)
class _Entry:
    image: QImage
    source_icc_sha256: str
    size_bytes: int
    scope_data: object = None


@dataclass(frozen=True)
class _SourceEntry:
    pixels: np.ndarray
    icc: bytes
    scope_data: object = None

    @property
    def size_bytes(self):
        return (self.pixels.nbytes + len(self.icc)
                + (self.scope_data.nbytes if self.scope_data is not None else 0))


_MAX_SOURCE_ITEM_BYTES = 32 * 1024**2
_DISPLAY_CODE_CHUNK_ELEMENTS = 2_000_000
_RGB16_TO_RGB8 = ((np.arange(65536, dtype=np.uint32) + 128) // 257).astype(np.uint8)
_RGB16_TO_RGB8.setflags(write=False)


def _scope_data(pixels, orientation):
    from .workflow_scopes import compute_scopes

    if orientation:
        from .output_orientation import orient_codes

        step = max(1, math.ceil(math.sqrt(pixels.shape[0] * pixels.shape[1] / 1_048_576)))
        pixels = orient_codes(pixels[::step, ::step], orientation)
    return compute_scopes(pixels)


def _file_identity(path):
    value = path.stat()
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _display_codes(pixels):
    # With odd divisor 257 no uint16 value lies on a half-code tie. This is
    # identical to rint(pixels / 257), without a full-frame float64 temporary.
    # Bound the uint32 working copy for a full-resolution display image. This
    # keeps the common small preview on the direct path and works with strided
    # source grids without first making a full contiguous copy.
    if pixels.ndim == 0 or pixels.size <= _DISPLAY_CODE_CHUNK_ELEMENTS:
        codes = pixels.astype(np.uint32)
        codes += 128
        codes //= 257
        return codes.astype(np.uint8)
    rows = max(1, _DISPLAY_CODE_CHUNK_ELEMENTS // (pixels.size // pixels.shape[0]))
    result = np.empty(pixels.shape, dtype=np.uint8)
    for start in range(0, pixels.shape[0], rows):
        codes = pixels[start:start + rows].astype(np.uint32)
        codes += 128
        codes //= 257
        result[start:start + rows] = codes
    return result


def _converted_image(pixels, source_icc, target, proof, full, max_edge, workers=1):
    image = Image.fromarray(_display_codes(pixels))
    if not full and max(image.size) > max_edge:
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    profile_description(source_icc)
    profile_description(target.icc)
    if proof == "off":
        image = transform_rgb_image(image, source_icc, target.icc, workers=workers)
    else:
        # Matrix RGB profiles do not necessarily clip an intermediate gamut in
        # a LittleCMS proof transform.  Materialize its bounded display encoding
        # to show the selected RGB space, then convert those values to the real
        # window surface.  This is an 8-bit display preview, never an export.
        reference = named_reference_profile(proof)
        image = transform_rgb_image(image, source_icc, reference, workers=workers)
        image = transform_rgb_image(image, reference, target.icc, workers=workers)
    space = QColorSpace.fromIccProfile(target.icc)
    if not space.isValid():
        raise ValueError("Qt无法标记实际显示ICC")
    data = image.tobytes()
    # Own the Python byte buffer before the local PIL image is released.
    result = QImage(
        data, image.width, image.height, image.width * 3, QImage.Format.Format_RGB888
    ).copy()
    result.setColorSpace(space)
    return result


@lru_cache(maxsize=12)
def _qt_display_color_space(icc):
    profile_description(icc)
    space = QColorSpace.fromIccProfile(icc)
    if not space.isValid():
        raise ValueError("Qt无法标记实际显示ICC")
    return space


@lru_cache(maxsize=12)
def _validate_source_profile(icc):
    profile_description(icc)


def _interactive_converted_image(pixels, source_icc, target, proof, workers):
    """Convert a bounded grid directly into the canvas's opaque RGB32 raster."""
    if sys.byteorder == "little":
        _validate_source_profile(source_icc)
        target_space = _qt_display_color_space(target.icc)
        try:
            from negpy_core import lut_u16
        except ImportError:
            codes = _display_codes(pixels)
        else:
            codes = lut_u16(np.ascontiguousarray(pixels), _RGB16_TO_RGB8)
        packed_icc = source_icc
        if proof != "off":
            reference = named_reference_profile(proof)
            bounded = np.empty_like(codes)
            if not transform_packed_rgb(codes, bounded, source_icc, reference, workers=workers):
                return _converted_image(pixels, source_icc, target, proof, True, 2048, workers)
            codes, packed_icc = bounded, reference
        height, width = codes.shape[:2]
        image = QImage(width, height, QImage.Format.Format_RGB32)
        if not image.isNull() and image.bytesPerLine() == width * 4:
            image.fill(0xff000000)
            address = image.bits()
            address.setsize(image.sizeInBytes())
            output = np.frombuffer(address, dtype=np.uint8).reshape(height, width, 4)
            if transform_packed_rgb(codes, output, packed_icc, target.icc, workers=workers):
                image.setColorSpace(target_space)
                return image
    return _converted_image(pixels, source_icc, target, proof, True, 2048, workers)


class DisplayImageCache:
    """Thread-safe LRU for foreground or idle conversion of saved TIFF records.

    Returned QImages share storage through Qt's copy-on-write semantics.  A
    caller can safely mutate its handle; it will detach from the cached copy.
    Authenticated source grids may outlive one monitor/proof image, but both
    layers share the byte budget. Shared summaries are conservatively charged
    to each retained entry, so actual retained data does not exceed that budget.
    """

    def __init__(self, budget_bytes=128 * 1024**2, workers=1):
        self._lock = threading.RLock()
        self._entries = OrderedDict()
        self._sources = OrderedDict()
        self._recency = OrderedDict()
        self._bytes = self._hits = self._misses = self._conversions = 0
        self._source_bytes = self._source_hits = self._source_misses = self._verifications = 0
        self._settle_image = None
        self._epoch = 0
        self._budget = 0
        self.set_workers(workers)
        self.set_budget(budget_bytes)

    def set_workers(self, workers):
        if type(workers) is not int or not 1 <= workers <= 256:
            raise ValueError("显示变换线程数必须为1至256")
        with self._lock:
            self._workers = min(8, workers)

    def set_budget(self, budget_bytes):
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError("显示缓存预算必须为非负字节数")
        with self._lock:
            self._budget = budget_bytes
            self._evict()

    def _evict(self):
        if self._settle_image is not None and self._settle_bytes > self._budget // 8:
            self._settle_image = None
        while self._recency and self._bytes + self._source_bytes + self._settle_bytes > self._budget:
            (kind, key), _ = self._recency.popitem(last=False)
            entries = self._entries if kind == "image" else self._sources
            removed = entries.pop(key, None)
            if removed is not None:
                if kind == "image":
                    self._bytes -= removed.size_bytes
                else:
                    self._source_bytes -= removed.size_bytes

    @property
    def _settle_bytes(self):
        return self._settle_image[1].sizeInBytes() if self._settle_image is not None else 0

    def _touch(self, kind, key):
        self._recency[(kind, key)] = None
        self._recency.move_to_end((kind, key))

    def _remember_source(self, key, source, epoch):
        # A retained grid owns bounded bytes, never a slice whose base pins a
        # full camera image. Tiny image-only budgets keep their former policy.
        limit = min(_MAX_SOURCE_ITEM_BYTES, self._budget // 2)
        previous = self._sources.get(key)
        if previous is not None and source.scope_data is None and previous.scope_data is not None:
            source = _SourceEntry(source.pixels, source.icc, previous.scope_data)
        if epoch == self._epoch and source.size_bytes <= limit:
            old = self._sources.pop(key, None)
            if old is not None:
                self._source_bytes -= old.size_bytes
            self._sources[key] = source
            self._source_bytes += source.size_bytes
            self._touch("source", key)

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._sources.clear()
            self._recency.clear()
            self._bytes = 0
            self._source_bytes = 0
            self._settle_image = None
            self._epoch += 1

    def stats(self):
        with self._lock:
            return dict(
                entries=len(self._entries),
                bytes=self._bytes + self._source_bytes + self._settle_bytes,
                image_bytes=self._bytes,
                source_bytes=self._source_bytes,
                settle_image_bytes=self._settle_bytes,
                source_entries=len(self._sources),
                source_hits=self._source_hits,
                source_misses=self._source_misses,
                full_verifications=self._verifications,
                budget_bytes=self._budget,
                hits=self._hits,
                misses=self._misses,
                conversions=self._conversions,
            )

    def _recently_verified_image(self, key):
        if self._settle_image is not None and self._settle_image[0] == key:
            return QImage(self._settle_image[1])
        return None

    def get_interactive_image(self, record, target, proof="off", *, reuse_for_settle=False):
        """Authenticate a held frame and optionally retain its final-size display image."""
        if proof not in {"off", "srgb", "adobe-rgb", "display-p3"}:
            raise ValueError("未知软打样色域")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("样片SHA256无效")
        path = Path(record["path"]).resolve(strict=True)
        packet = {**record, "path": str(path)}
        fit_record = record.get("transport") == FIT_TRANSPORT
        reuse_for_settle = reuse_for_settle and fit_record
        read_pixels = read_fit_record if fit_record else verified_pixels
        settle_key = (digest, target.icc, target.signature, proof)
        with self._lock:
            workers, epoch = self._workers, self._epoch
        for _ in range(3):
            identity = _file_identity(path)
            pixels, icc = read_pixels(packet)
            if _file_identity(path) == identity:
                image = None
                if reuse_for_settle:
                    with self._lock:
                        image = self._recently_verified_image(settle_key)
                if image is None:
                    image = _interactive_converted_image(pixels, icc, target, proof, workers)
                if _file_identity(path) == identity:
                    if reuse_for_settle:
                        with self._lock:
                            if epoch == self._epoch and image.sizeInBytes() <= self._budget // 8:
                                self._settle_image = (settle_key, QImage(image))
                                self._evict()
                    return image
        raise ValueError("样片在显示读取期间持续变化，请重试预览")

    def get_image(
        self, record, target, proof="off", full=True, max_edge=2048, sample_step=1,
        source_step=1, include_scopes=False, scope_orientation=None,
    ):
        """Convert an authenticated TIFF, optionally selecting its exact grid.

        source_step applies before quantization and ICC. full/max_edge retain
        their existing meaning for optional image thumbnailing after that grid
        selection; sample_step describes the incoming image's source scale.
        """
        if proof not in {"off", "srgb", "adobe-rgb", "display-p3"}:
            raise ValueError("未知软打样色域")
        if type(full) is not bool:
            raise ValueError("显示分辨率开关必须为布尔值")
        if type(max_edge) is not int or max_edge < 1:
            raise ValueError("显示缩略边长必须为正整数")
        if type(sample_step) is not int or sample_step < 1:
            raise ValueError("显示采样步长必须为正整数")
        if type(source_step) is not int or source_step < 1:
            raise ValueError("显示源采样步长必须为正整数")
        if type(include_scopes) is not bool:
            raise ValueError("分析图开关必须为布尔值")
        from .output_orientation import checked_document

        scope_orientation = checked_document(scope_orientation)
        scope_direction = tuple(sorted(scope_orientation.items())) if scope_orientation else None
        digest = record.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("样片SHA256无效")
        path = Path(record["path"]).resolve(strict=True)
        identity = _file_identity(path)
        target_digest = hashlib.sha256(target.icc).hexdigest()
        proof_digest = (
            None
            if proof == "off"
            else hashlib.sha256(named_reference_profile(proof)).hexdigest()
        )
        # Source ICC is authenticated by the file SHA; its explicit hash lives
        # in each entry.  A changed file identity cannot reuse that entry.
        key = (
            str(path),
            digest,
            identity,
            target_digest,
            target.signature,
            proof,
            proof_digest,
            full,
            None if full else max_edge,
            sample_step,
            source_step,
            scope_direction,
        )
        cached_image = None
        fit_record = record.get("transport") == FIT_TRANSPORT
        if fit_record:
            # A retained fit image is reusable across tokens, but every new
            # file still has to authenticate its RGB16 + ICC payload first.
            key = (FIT_TRANSPORT, digest, None, *key[3:])
        with self._lock:
            source_key = (str(path), digest, identity, source_step, scope_direction)
            source = self._sources.get(source_key)
            entry = self._entries.get(key)
            if (entry is not None and (not fit_record or source is not None)
                    and _file_identity(path) == identity):
                self._entries.move_to_end(key)
                self._touch("image", key)
                if not include_scopes or entry.scope_data is not None:
                    self._hits += 1
                    return (QImage(entry.image), entry.scope_data) if include_scopes else QImage(entry.image)
                cached_image = QImage(entry.image)
            self._misses += 1
            epoch = self._epoch
            workers = self._workers
            if (source is not None and _file_identity(path) == identity
                    and (not include_scopes or source.scope_data is not None or source_step == 1)):
                self._source_hits += 1
                self._touch("source", source_key)
            else:
                self._source_misses += 1
                source = None
        # Expensive image decode and LittleCMS work run outside the cache lock.
        for attempt in range(3):
            # A newly published temporary TIFF can acquire filesystem metadata
            # while being read. Retry its FULL hash/decode, never trust a changed
            # identity or merely ignore ctime. Content changes still fail SHA.
            if source is None:
                with self._lock:
                    self._verifications += 1
                read_pixels = read_fit_record if fit_record else verified_pixels
                pixels, source_icc = read_pixels({**record, "path": str(path)})
            else:
                pixels, source_icc = source.pixels, source.icc
            after_read = _file_identity(path)
            if after_read == identity:
                scopes = source.scope_data if source is not None else None
                if fit_record and cached_image is None:
                    with self._lock:
                        entry = self._entries.get(key)
                        if entry is not None:
                            cached_image = QImage(entry.image)
                            if scopes is None:
                                scopes = entry.scope_data
                        elif source_step == 1 and (full or max(pixels.shape[:2]) <= max_edge):
                            # The settled packet has been authenticated on its
                            # own filesystem identity. The same digest includes
                            # every pixel and the embedded source ICC profile.
                            cached_image = self._recently_verified_image(
                                (digest, target.icc, target.signature, proof)
                            )
                if include_scopes and scopes is None:
                    scopes = _scope_data(pixels, scope_orientation)
                grid = pixels if source is not None else pixels[::source_step, ::source_step]
                if cached_image is not None:
                    image = cached_image
                elif fit_record and (full or max(grid.shape[:2]) <= max_edge):
                    image = _interactive_converted_image(grid, source_icc, target, proof, workers)
                else:
                    image = _converted_image(grid, source_icc, target, proof, full, max_edge, workers)
                after_read = _file_identity(path)
                if after_read == identity:
                    break
            from .diagnostics import event

            event("display_source_identity_retry", before=identity, after=after_read, attempt=attempt)
            identity = after_read
            source, cached_image = None, None
        else:
            raise ValueError("样片在显示读取期间持续变化，请重试预览")
        if not fit_record:
            key = (*key[:2], identity, *key[3:])
        scope_bytes = scopes.nbytes if scopes is not None else 0
        entry = _Entry(image, hashlib.sha256(source_icc).hexdigest(),
                       int(image.sizeInBytes()) + scope_bytes, scopes)
        with self._lock:
            self._conversions += int(cached_image is None)
            source_key = (str(path), digest, identity, source_step, scope_direction)
            # Scope counts continue to come from the original authenticated
            # output, not from the reduced display grid. Share them across ICC
            # targets/proofs only when the source direction is identical.
            source_size = grid.nbytes + len(source_icc) + scope_bytes
            if source_size <= min(_MAX_SOURCE_ITEM_BYTES, self._budget // 2):
                if source is None:
                    owned = (grid if fit_record and source_step == 1
                             else np.frombuffer(grid.tobytes(), dtype=np.uint16).reshape(grid.shape))
                    source = _SourceEntry(owned, bytes(source_icc), scopes)
                elif scopes is not source.scope_data:
                    source = _SourceEntry(source.pixels, source.icc, scopes)
                self._remember_source(source_key, source, epoch)
            if epoch == self._epoch and entry.size_bytes <= self._budget:
                old = self._entries.pop(key, None)
                if old is not None:
                    self._bytes -= old.size_bytes
                self._entries[key] = entry
                self._bytes += entry.size_bytes
                self._touch("image", key)
            self._evict()
        return (QImage(image), scopes) if include_scopes else QImage(image)
