"""Content-keyed disk cache for decoded RAW sources.

A hit rebinds the exact processor state a real decode leaves behind; cached
pixels are validated against frozen manifest hashes before use. Any doubt
decodes again.
"""

from copy import deepcopy
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import pickle
import tempfile

import numpy as np

from . import WORKSPACE_ROOT as ROOT
from .diagnostics import event, failure
from .raw_import import LEGACY_IMPORT, file_identity

_FORMAT = 1
_DEFAULT_BUDGET_BYTES = 16 * 1024**3


def _enabled():
    return os.environ.get("PURE_RGB_DECODED_CACHE", "1") != "0"


def _budget_bytes():
    try:
        return max(0, int(os.environ.get("PURE_RGB_DECODED_CACHE_BYTES", "")))
    except ValueError:
        return _DEFAULT_BUDGET_BYTES


def _directory():
    outputs = ROOT / "outputs"
    cache = outputs / "decoded-cache"
    if outputs.is_symlink() or cache.is_symlink():
        return None
    return cache


def _decoder_stamp():
    try:
        return version("rawpy")
    except PackageNotFoundError:
        return None


def cache_key(manifest, recipe, config_document, reference_shape):
    stamp = _decoder_stamp()
    if stamp is None:
        return None
    material = {
        "format": _FORMAT,
        "inputs": [record["sha256"] for record in manifest["inputs"]],
        "flatfield": bool(recipe["flatfield"]),
        "calibration": (manifest.get("calibration") or {}).get("sha256")
        if recipe["flatfield"]
        else None,
        "config": config_document,
        "raw_import": recipe.get("raw_import", LEGACY_IMPORT),
        **({"lens_correction": True} if recipe.get("lens_correction") else {}),
        "registration_check": recipe.get("registration_check"),
        "flatfield_strength": float(recipe.get("flatfield_strength", 1.0)),
        **({"flatfield_strength_curve": "exposure-stops-v1"}
           if recipe["flatfield"] and recipe.get("flatfield_strength", 1.0) > 1.0 else {}),
        "flatfield_reference_shape": list(reference_shape) if reference_shape else None,
        "rawpy": stamp,
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _entry_paths(directory, key):
    return directory / f"{key}.npy", directory / f"{key}.pkl"


def _drop(data_path, meta_path):
    for path in (data_path, meta_path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _rebind(processor, meta, source, file_path, params):
    key = meta["negpy_cache_key"]
    if (
        type(key) is not tuple
        or len(key) < 3
        or key[0] != file_path
        or type(key[1]) is not float
    ):
        raise ValueError("decoded cache decoder key differs")
    try:
        mtime = os.path.getmtime(file_path)
    except OSError:
        mtime = 0.0
    # The key's mtime is a freshness heuristic; content is already proven by
    # the manifest hashes, so the current mtime rebinds without a re-decode.
    processor._source_cache_key = (file_path, mtime, *key[2:])
    processor._source_cache_value = (source, None, meta["color_space"])
    processor._cam_xyz_by_path[file_path] = tuple(meta["camera_metadata"])
    if hasattr(processor, "pure_raw_import_token"):
        processor.pure_raw_import_receipt = deepcopy(meta["receipt"])
        processor.pure_raw_import_token = meta["import_token"]
        processor.pure_flatfield_sampled_source_mean_absolute_change = meta[
            "flatfield_sampled_change"
        ]
        paths = processor._paths(file_path, params)
        processor._pure_source_identity = (
            tuple((path, file_identity(path)) for path in paths),
            tuple(processor.pure_flatfield_reference_shape or ()),
        )


def restore_source(
    processor, manifest, recipe, config_document, reference_shape, file_path, params
):
    """Rebind a validated cached decode onto the processor, or return None."""
    if not _enabled():
        return None
    directory = _directory()
    if directory is None or not directory.is_dir():
        return None
    try:
        key = cache_key(manifest, recipe, config_document, reference_shape)
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return None
    if key is None:
        return None
    data_path, meta_path = _entry_paths(directory, key)
    if not data_path.is_file() or not meta_path.is_file():
        event("decoded_cache_miss", reason="absent")
        return None
    try:
        with meta_path.open("rb") as stream:
            meta = pickle.load(stream)
        if type(meta) is not dict or meta.get("format") != _FORMAT:
            raise ValueError("decoded cache metadata format differs")
        source = np.load(data_path, mmap_mode="r")
        shape = meta["shape"]
        if (
            type(shape) is not list
            or len(shape) != 3
            or shape[-1] != 3
            or any(type(value) is not int or value <= 0 for value in shape)
            or source.dtype != np.float32
            or list(source.shape) != shape
            or not source.flags.c_contiguous
            or source.flags.writeable
            or (
                manifest.get("source_shape") is not None
                and shape != list(manifest["source_shape"])
            )
        ):
            raise ValueError("decoded cache geometry differs")
        digest = hashlib.sha256(memoryview(source).cast("B")).hexdigest()
        if digest != meta["pixels_sha256"]:
            raise ValueError("decoded cache pixels differ")
        if recipe.get("raw_import", LEGACY_IMPORT) != LEGACY_IMPORT:
            expected = manifest.get("raw_import_receipt") or {}
            receipt = meta.get("receipt") or {}
            if (
                expected.get("source_pixels_sha256") != digest
                or expected.get("geometry_token") != receipt.get("geometry_token")
            ):
                raise ValueError("decoded cache import receipt differs")
        _rebind(processor, meta, source, file_path, params)
    except (OSError, EOFError, ValueError, KeyError, TypeError, pickle.UnpicklingError):
        _drop(data_path, meta_path)
        event("decoded_cache_miss", reason="invalid")
        return None
    for path in (data_path, meta_path):
        try:
            os.utime(path, None)
        except OSError:
            pass
    event("decoded_cache_hit", bytes=int(source.nbytes))
    return source


def store_source(
    processor,
    manifest,
    recipe,
    config_document,
    reference_shape,
    file_path,
    source,
    ir,
    color_space,
):
    """Persist a real decode for later sessions; caching never breaks a preview."""
    if not _enabled() or ir is not None:
        return
    directory = _directory()
    if directory is None:
        return
    try:
        key = cache_key(manifest, recipe, config_document, reference_shape)
        if key is None:
            return
        data_path, meta_path = _entry_paths(directory, key)
        if data_path.exists() or meta_path.exists():
            return
        standard = recipe.get("raw_import", LEGACY_IMPORT) != LEGACY_IMPORT
        meta = {
            "format": _FORMAT,
            "shape": list(source.shape),
            "pixels_sha256": hashlib.sha256(
                memoryview(source).cast("B")
            ).hexdigest(),
            "color_space": color_space,
            "negpy_cache_key": processor._source_cache_key,
            "camera_metadata": processor._cam_xyz_by_path.get(file_path),
        }
        if standard:
            meta.update(
                receipt=deepcopy(getattr(processor, "pure_raw_import_receipt", None)),
                import_token=getattr(processor, "pure_raw_import_token", None),
                flatfield_sampled_change=getattr(
                    processor,
                    "pure_flatfield_sampled_source_mean_absolute_change",
                    None,
                ),
            )
        if (
            source.dtype != np.float32
            or not isinstance(meta["color_space"], str)
            or not isinstance(meta["camera_metadata"], tuple)
            or (
                standard
                and not (
                    isinstance(meta["receipt"], dict)
                    and isinstance(meta["import_token"], str)
                )
            )
        ):
            return
        directory.mkdir(parents=True, exist_ok=True)
        temporary = []
        try:
            with tempfile.NamedTemporaryFile(
                "wb", dir=directory, delete=False
            ) as stream:
                temporary.append(Path(stream.name))
                np.save(stream, source)
            with tempfile.NamedTemporaryFile(
                "wb", dir=directory, delete=False
            ) as stream:
                temporary.append(Path(stream.name))
                pickle.dump(meta, stream)
            temporary[0].replace(data_path)
            temporary[1].replace(meta_path)
        finally:
            for path in temporary:
                path.unlink(missing_ok=True)
    except (OSError, TypeError, ValueError, KeyError, IndexError, pickle.PicklingError) as error:
        failure("decoded_cache_store_failed", error)
        return
    event("decoded_cache_store", bytes=int(source.nbytes))
    _evict_overflow(directory, _budget_bytes())


class RenderDecodeCache:
    """Render-mode cache context bound to a prior completed run's receipt.

    Render requests start a new run and hold no manifest of their own, so a
    cached decode may only restore against the import receipt/geometry of an
    earlier completed run with the same decode identity (the request anchor).
    ``records`` gates the triplet classification: user-confirmed roles that
    differ from automatic classification take the re-decode path instead.
    """

    def __init__(self, records, manifest, recipe, reference_shape):
        self.records = records
        self.manifest = manifest
        self.recipe = recipe
        self.reference_shape = reference_shape

    def restore(self, processor, config_document, file_path, params):
        return restore_source(
            processor,
            self.manifest,
            self.recipe,
            config_document,
            self.reference_shape,
            file_path,
            params,
        )

    def store(
        self, processor, config_document, file_path, source, ir, color_space
    ):
        store_source(
            processor,
            self.manifest,
            self.recipe,
            config_document,
            self.reference_shape,
            file_path,
            source,
            ir,
            color_space,
        )


def render_decode_context(request, recipe, anchor):
    """Build the render cache context, or None when the anchor cannot bind.

    Strict anchor validation keeps new photos (no completed run yet) exactly
    on the uncached path: a wrong or partial anchor disables caching instead
    of restoring an unproven decode.
    """
    from .raw_import import LEGACY_IMPORT

    if not isinstance(anchor, dict):
        return None
    shape = anchor.get("source_shape")
    reference = anchor.get("flatfield_reference_shape")
    receipt = anchor.get("raw_import_receipt")
    if receipt is not None and not isinstance(receipt, dict):
        return None
    if (
        type(shape) is not list
        or len(shape) != 3
        or any(type(value) is not int or value <= 0 for value in shape)
    ):
        return None
    if recipe["flatfield"]:
        if (
            type(reference) is not list
            or len(reference) != 3
            or any(type(value) is not int or value <= 0 for value in reference)
        ):
            return None
    else:
        reference = None
    policy = recipe.get("raw_import", LEGACY_IMPORT)
    if policy != LEGACY_IMPORT and (
        not isinstance(receipt, dict)
        or not isinstance(receipt.get("source_pixels_sha256"), str)
        or not isinstance(receipt.get("geometry_token"), str)
    ):
        return None
    manifest = {
        "inputs": request["inputs"],
        "calibration": request.get("calibration"),
        "raw_import_receipt": deepcopy(receipt)
        if isinstance(receipt, dict)
        else None,
        "source_shape": shape,
    }
    return RenderDecodeCache(
        request["inputs"], manifest, recipe, reference
    )


def _evict_overflow(directory, budget):
    entries = {}
    for path in directory.iterdir():
        stem = path.stem
        if (
            path.is_symlink()
            or not path.is_file()
            or path.suffix not in (".npy", ".pkl")
            or len(stem) != 64
            or any(character not in "0123456789abcdef" for character in stem)
        ):
            continue
        stat = path.lstat()
        entry = entries.setdefault(stem, {"mtime": stat.st_mtime_ns, "size": 0, "paths": []})
        entry["mtime"] = min(entry["mtime"], stat.st_mtime_ns)
        entry["size"] += stat.st_size
        entry["paths"].append(path)
    total = sum(entry["size"] for entry in entries.values())
    evicted = 0
    for entry in sorted(entries.values(), key=lambda item: item["mtime"]):
        if total <= budget:
            break
        for path in entry["paths"]:
            try:
                path.unlink()
            except OSError:
                continue
            evicted += 1
        total -= entry["size"]
    if evicted:
        event("decoded_cache_evict", count=evicted)
