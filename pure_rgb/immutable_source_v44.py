"""Private, strongly immutable source lifetimes for repeated preview metering.

An ordinary ndarray, including an owning array marked read-only, is never
certified implicitly. A private copy backed by Python bytes cannot be made
writable through NumPy aliases. Its complete finite scan and SHA may therefore
be reused while its object, backing, pointer and layout remain unchanged.
"""

import hashlib
from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
import json
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
import weakref

import numpy as np


_CERTIFICATES = {}


def _layout(source):
    return (source.shape, source.dtype.str, source.strides,
            source.__array_interface__["data"][0], bool(source.flags.writeable),
            bool(source.flags.c_contiguous))


def _backing(source):
    value = source
    while isinstance(value, np.ndarray):
        value = value.base
    return value


class ImmutableSourceCertificate:
    @classmethod
    def create(cls, source):
        if (not isinstance(source, np.ndarray) or source.dtype != np.float32
                or source.ndim != 3 or source.shape[2] != 3 or not source.size
                or not source.flags.c_contiguous):
            raise ValueError("Preview source requires original C-order float32 RGB")
        backing = source.tobytes(order="C")
        frozen = np.frombuffer(backing, dtype=source.dtype).reshape(source.shape)
        flat = frozen.reshape(-1)
        chunk_size = 4 * 1024 * 1024
        if not all(
            np.isfinite(flat[start : start + chunk_size]).all()
            for start in range(0, flat.size, chunk_size)
        ):
            raise ValueError("Preview source contains nonfinite values")
        instance = cls.__new__(cls)
        instance.source, instance._backing = frozen, backing
        instance._layout = _layout(frozen)
        instance.sha256 = hashlib.sha256(backing).hexdigest()
        instance.closed = False
        instance.grids, instance.restored = OrderedDict(), OrderedDict()
        instance.automatic, instance.bundles = OrderedDict(), OrderedDict()
        instance.grid_hits = instance.grid_misses = instance.restore_hits = instance.restore_misses = 0
        instance.automatic_hits = instance.automatic_misses = 0
        instance.leaves = OrderedDict()
        instance.leaf_hits = instance.leaf_misses = 0
        instance._leaf_lock = threading.Lock()
        instance._leaf_pool = None
        instance.checkpoint = None
        _CERTIFICATES[id(frozen)] = weakref.ref(instance)
        return instance

    def validate(self, source=None, identity=None):
        source = self.source if source is None else source
        if (self.closed or source is not self.source or _layout(source) != self._layout
                or _backing(source) is not self._backing or type(self._backing) is not bytes
                or self._layout[-2:] != (False, True)):
            raise ValueError("Authenticated preview source layout/backing changed")
        if identity is not None and (
            list(source.shape) != identity["shape"] or source.dtype.str != identity["dtype"]
            or self.sha256 != identity["source_pixels_sha256"]
        ):
            raise ValueError("Authenticated preview source identity changed")
        return self.sha256

    def close(self):
        if not self.closed:
            _CERTIFICATES.pop(id(self.source), None)
            with self._leaf_lock:
                self.closed = True
                self.leaves.clear()
            self.grids.clear()
            self.restored.clear()
            self.automatic.clear()
            self.bundles.clear()
            if self._leaf_pool is not None:
                self._leaf_pool.shutdown(wait=True, cancel_futures=True)
                self._leaf_pool = None

    def prefilter_grid(self, roi):
        from negpy.features.exposure import normalization as norm
        from negpy.features.exposure.models import EXPOSURE_CONSTANTS

        self.validate()
        h, w = self.source.shape[:2]
        if (not isinstance(roi, (list, tuple)) or len(roi) != 4
                or any(type(value) is not int for value in roi)
                or not 0 <= roi[0] < roi[2] <= w or not 0 <= roi[1] < roi[3] <= h):
            raise ValueError("Auxiliary grid needs an exact full-source ROI")
        key = (*roi, int(EXPOSURE_CONSTANTS["analysis_grid"]))
        grid = self.grids.get(key)
        if grid is None:
            x0, y0, x1, y1 = roi
            # The SDK log transform is pixel-local. Crop its original input
            # first, retaining exactly the original ROI block origin/tail.
            grid = norm.prefilter_log_grid(np.ascontiguousarray(self.source[y0:y1, x0:x1]), None, 0.0)
            grid = np.frombuffer(grid.tobytes(), dtype=grid.dtype).reshape(grid.shape)
            self.grids[key] = grid
            self.grid_misses += 1
            while len(self.grids) > 4:
                self.grids.popitem(last=False)
        else:
            self.grid_hits += 1
        self.grids.move_to_end(key)
        self.validate()
        return grid

    @property
    def derived_bytes(self):
        return sum(grid.nbytes for grid in self.grids.values()) + sum(
            len(item[1]) if item[1] is not None else 0 for item in self.restored.values()) + sum(
            len(text.encode()) for text in self.automatic.values()) + sum(
            len(bundle.canonical_json.encode()) for bundle in self.bundles.values()) + len(self.leaves) * 256

    def leaf_statistics(self, box, support, factory, *, policy):
        self.validate()
        support_key = ("whole" if support is None else
                       hashlib.sha256(np.ascontiguousarray(support).tobytes()).hexdigest())
        key = (tuple(box), support_key, policy)
        with self._leaf_lock:
            value = self.leaves.get(key)
            if value is not None:
                self.leaf_hits += 1
                self.leaves.move_to_end(key)
                ready = Future()
                ready.set_result(value)
                return ready
            self.leaf_misses += 1

        checkpoint = self.checkpoint

        def compute():
            if checkpoint is not None:
                checkpoint()
            a, b, c = factory()
            result = (tuple(int(value) for value in a), tuple(int(value) for value in b),
                      tuple(float(value) for value in c))
            with self._leaf_lock:
                if not self.closed:
                    self.leaves[key] = result
                    while len(self.leaves) > 4096:
                        self.leaves.popitem(last=False)
            return result

        if self.source.shape[0] * self.source.shape[1] >= 1_000_000:
            if self._leaf_pool is None:
                workers = min(4, max(1, int(os.environ.get("RAYON_NUM_THREADS", "4"))))
                self._leaf_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="density-leaf")
            return self._leaf_pool.submit(compute)
        ready = Future()
        try:
            ready.set_result(compute())
        except BaseException as error:
            ready.set_exception(error)
        return ready

    def automatic_result(self, roi, document, factory):
        self.validate()
        key = json.dumps([list(roi), document], sort_keys=True, allow_nan=False)
        text = self.automatic.get(key)
        if text is None:
            text = json.dumps(factory(), sort_keys=True, allow_nan=False)
            self.automatic[key] = text
            self.automatic_misses += 1
            while len(self.automatic) > 4:
                self.automatic.popitem(last=False)
        else:
            self.automatic_hits += 1
        self.automatic.move_to_end(key)
        return json.loads(text)

    def density_bundle(self, roi, meter, identity, factory, *, policy=None):
        self.validate(identity=identity)
        key = json.dumps([list(roi), meter, identity, policy], sort_keys=True, allow_nan=False)
        value = self.bundles.get(key)
        if value is None:
            value = factory()
            self.bundles[key] = value
            while len(self.bundles) > 4:
                self.bundles.popitem(last=False)
        self.bundles.move_to_end(key)
        return value

    def restore_bundle(self, text, identity, factory, *, policy=None):
        self.validate(identity=identity)
        key = (text, json.dumps(identity, sort_keys=True, allow_nan=False), policy)
        found = self.restored.get(key)
        if found is None:
            value = factory()
            mask_bytes = value.selection_mask.mask.tobytes() if value.selection_mask is not None else None
            stored = deepcopy(value)
            # Keep only immutable raster bytes in the private proof cache.
            if stored.selection_mask is not None:
                stored = replace(stored, selection_mask=replace(stored.selection_mask,
                    mask=np.frombuffer(mask_bytes, dtype=np.bool_).reshape(stored.selection_mask.mask.shape)))
            self.restored[key] = found = (stored, mask_bytes)
            self.restore_misses += 1
            while len(self.restored) > 2:
                self.restored.popitem(last=False)
        else:
            self.restore_hits += 1
        self.restored.move_to_end(key)
        stored, mask_bytes = found
        selection = None
        if stored.selection_mask is not None:
            mask = stored.selection_mask
            selection = replace(mask, spec=deepcopy(mask.spec),
                mask=np.frombuffer(mask_bytes, dtype=np.bool_).reshape(mask.mask.shape))
        return replace(stored, window=deepcopy(stored.window), receipt=deepcopy(stored.receipt),
                       source_identity=deepcopy(stored.source_identity), config=deepcopy(stored.config),
                       selection_mask=selection)


def certificate_for(source):
    reference = _CERTIFICATES.get(id(source))
    certificate = reference() if reference else None
    if certificate is not None:
        certificate.validate(source)
    return certificate


def source_digest(source):
    certificate = certificate_for(source)
    return certificate.sha256 if certificate is not None else hashlib.sha256(memoryview(source).cast("B")).hexdigest()


def certificate_for_identity(identity):
    for reference in list(_CERTIFICATES.values()):
        certificate = reference()
        if certificate is not None and not certificate.closed and (
            certificate.sha256 == identity.get("source_pixels_sha256")
            and list(certificate.source.shape) == identity.get("shape")
            and certificate.source.dtype.str == identity.get("dtype")
        ):
            certificate.validate(identity=identity)
            return certificate
    return None
