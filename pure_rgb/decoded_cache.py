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
import shutil
import tempfile
import time
from types import SimpleNamespace
import weakref

import numpy as np

from . import WORKSPACE_ROOT as ROOT
from .diagnostics import event, failure
from .raw_import import LEGACY_IMPORT, STANDARD_MONO_IMPORT, file_identity

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


def _has_write_space(directory, required):
    """Skip optional writes only when the target volume is known to be too full."""
    path = directory
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        return shutil.disk_usage(path).free >= required
    except OSError:
        return True


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
        # Decoder metadata and import receipts retain source paths. Identical
        # files at different paths must have independent entries until that
        # state can be safely rebound across paths.
        "input_paths": [record.get("path") for record in manifest["inputs"]],
        "flatfield": bool(recipe["flatfield"]),
        "calibration": (manifest.get("calibration") or {}).get("sha256")
        if recipe["flatfield"]
        else None,
        # Private SDK state and export destinations have no effect on RAW
        # decoding. Workers can use different directories for the same source.
        "config": {key: value for key, value in config_document.items()
                   if key not in ("userDir", "export_path")},
        "raw_import": recipe.get("raw_import", LEGACY_IMPORT),
        **({"lens_correction": True} if recipe.get("lens_correction") else {}),
        **({"chromatic_aberration": True} if recipe.get("chromatic_aberration") else {}),
        **({"lens_profile": recipe["lens_profile"]} if recipe.get("lens_profile") else {}),
        "registration_check": recipe.get("registration_check"),
        "flatfield_strength": (
            float(recipe.get("flatfield_strength", 1.0)) if recipe["flatfield"] else 1.0
        ),
        **({"flatfield_strength_curve": "exposure-stops-v1"}
           if recipe["flatfield"] and recipe.get("flatfield_strength", 1.0) > 1.0 else {}),
        "flatfield_reference_shape": list(reference_shape) if reference_shape else None,
        "rawpy": stamp,
    }
    if recipe.get("raw_import") == STANDARD_MONO_IMPORT:
        from .raw_mono_import import MONO_NORMALIZATION_VERSION

        material["mono_normalization"] = MONO_NORMALIZATION_VERSION
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _mono_receipt_current(receipt):
    from .raw_mono_import import MONO_NORMALIZATION_VERSION

    if not isinstance(receipt, dict) or receipt.get("policy") != STANDARD_MONO_IMPORT:
        return False
    decode = receipt.get("decode")
    bands = receipt.get("bands")
    return (
        isinstance(decode, dict)
        and decode.get("normalization") == MONO_NORMALIZATION_VERSION
        and isinstance(bands, list)
        and len(bands) == 3
        and all(
            isinstance(band, dict)
            and isinstance(band.get("monochrome_sensor"), dict)
            and band["monochrome_sensor"].get("normalization") == MONO_NORMALIZATION_VERSION
            for band in bands
        )
    )


def _entry_paths(directory, key):
    return directory / f"{key}.npy", directory / f"{key}.pkl"


def _report_prewarm_entry(data_path, meta_path):
    """Report a validated entry to the optional, private prewarm subprocess."""
    destination = os.environ.get("PURE_RGB_PREWARM_RESULT")
    if not destination:
        return
    try:
        document = {
            "key": data_path.stem,
            "data_identity": file_identity(data_path),
            "meta_identity": file_identity(meta_path),
        }
        Path(destination).write_text(json.dumps(document))
    except (OSError, ValueError, TypeError):
        pass


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
    # The SDK includes a newly generated flat-field profile ID alongside the
    # gain digest, including in the embedded-lens token. Keep all decode
    # settings and the digest strict, and replace only the local handle.
    fields = list(key)
    # The pinned SDK load key puts embedded lens settings at 2 and gain at 10.
    if len(key) == 16 and isinstance(key[10], str) and key[10].startswith("|ff:"):
        from negpy.features.flatfield.logic import flatfield_token

        old = key[10]
        flatfield = getattr(params, "flatfield", None)
        token = flatfield_token(flatfield) if flatfield is not None else ""
        if not token or old.split(":", 2)[-1] != token.split(":", 2)[-1]:
            raise ValueError("decoded cache flat-field gain differs")
        fields[10] = token
        if isinstance(key[2], str) and key[2].endswith(old):
            fields[2] = key[2][:-len(old)] + token
    try:
        mtime = os.path.getmtime(file_path)
    except OSError:
        mtime = 0.0
    # The key's mtime is a freshness heuristic; content is already proven by
    # the manifest hashes, so the current mtime rebinds without a re-decode.
    fields[1] = mtime
    cache_value = (source, None, meta["color_space"])
    camera_metadata = tuple(meta["camera_metadata"])
    standard = hasattr(processor, "pure_raw_import_token")
    if standard:
        receipt = deepcopy(meta["receipt"])
        import_token = meta["import_token"]
        flatfield_change = meta["flatfield_sampled_change"]
        paths = processor._paths(file_path, params)
        source_identity = (processor.source_identity_for_paths(paths)
            if hasattr(processor, "source_identity_for_paths") else (
            tuple((path, file_identity(path)) for path in paths),
            tuple(processor.pure_flatfield_reference_shape or ()),
        ))
    processor._source_cache_key = tuple(fields)
    processor._source_cache_value = cache_value
    processor._cam_xyz_by_path[file_path] = camera_metadata
    if standard:
        processor.pure_raw_import_receipt = receipt
        processor.pure_raw_import_token = import_token
        processor.pure_flatfield_sampled_source_mean_absolute_change = flatfield_change
        processor._pure_source_identity = source_identity


def restore_source(
    processor, manifest, recipe, config_document, reference_shape, file_path, params,
    *, frozen=False,
):
    """Restore a validated decode, with its certificate if requested, or return None."""
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
    certificate = None
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
        if frozen:
            from .immutable_source_v44 import ImmutableSourceCertificate

            # Authenticate the exact immutable bytes that the preview will use.
            # A read-only mmap can still change if its backing file is edited.
            certificate = ImmutableSourceCertificate.create(source)
            source = certificate.source
            digest = certificate.sha256
        else:
            digest = hashlib.sha256(memoryview(source).cast("B")).hexdigest()
        if digest != meta["pixels_sha256"]:
            raise ValueError("decoded cache pixels differ")
        if recipe.get("raw_import", LEGACY_IMPORT) != LEGACY_IMPORT:
            expected = manifest.get("raw_import_receipt") or {}
            receipt = meta.get("receipt") or {}
            if (
                expected.get("source_pixels_sha256") != digest
                or expected.get("geometry_token") != receipt.get("geometry_token")
                or (recipe.get("raw_import") == STANDARD_MONO_IMPORT
                    and (not _mono_receipt_current(expected)
                         or not _mono_receipt_current(receipt)))
            ):
                raise ValueError("decoded cache import receipt differs")
        _rebind(processor, meta, source, file_path, params)
    except (OSError, EOFError, ValueError, KeyError, TypeError, pickle.UnpicklingError):
        if certificate is not None:
            certificate.close()
        _drop(data_path, meta_path)
        event("decoded_cache_miss", reason="invalid")
        return None
    for path in (data_path, meta_path):
        try:
            os.utime(path, None)
        except OSError:
            pass
    event("decoded_cache_hit", bytes=int(source.nbytes))
    _report_prewarm_entry(data_path, meta_path)
    return (source, certificate) if frozen else source


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
    # Avoid hashing and writing a frame that eviction would immediately remove.
    if _budget_bytes() < source.nbytes + 4096:
        return
    directory = _directory()
    if directory is None:
        return
    try:
        key = cache_key(manifest, recipe, config_document, reference_shape)
        if key is None:
            return
        data_path, meta_path = _entry_paths(directory, key)
        if data_path.exists() and meta_path.exists():
            return
        if not _has_write_space(directory, source.nbytes + 4096):
            event("decoded_cache_store_skipped", reason="insufficient_space")
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
            receipt = meta["receipt"]
            expected = manifest.get("raw_import_receipt") or {}
            if (
                not isinstance(receipt, dict)
                or not isinstance(meta["import_token"], str)
                or receipt.get("source_pixels_sha256") != meta["pixels_sha256"]
                or expected.get("source_pixels_sha256") != meta["pixels_sha256"]
                or expected.get("geometry_token") != receipt.get("geometry_token")
            ):
                return
        if (
            source.dtype != np.float32
            or (manifest.get("source_shape") is not None
                and list(source.shape) != list(manifest["source_shape"]))
            or not isinstance(meta["color_space"], str)
            or not isinstance(meta["camera_metadata"], tuple)
        ):
            return
        directory.mkdir(parents=True, exist_ok=True)
        temporary = []
        try:
            with tempfile.NamedTemporaryFile(
                "wb", dir=directory, prefix=".decoded-cache-tmp-", delete=False
            ) as stream:
                temporary.append(Path(stream.name))
                np.save(stream, source)
            with tempfile.NamedTemporaryFile(
                "wb", dir=directory, prefix=".decoded-cache-tmp-", delete=False
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
    try:
        _evict_overflow(directory, _budget_bytes())
    except OSError as error:
        failure("decoded_cache_evict_failed", error)
    _report_prewarm_entry(data_path, meta_path)


class RenderDecodeCache:
    """Render-mode cache context bound to a prior completed run's receipt.

    Render requests start a new run and hold no manifest of their own, so a
    cached decode may only restore against the import receipt/geometry of an
    earlier completed run with the same decode identity (the request anchor).
    ``records`` gates the triplet classification: user-confirmed roles that
    differ from automatic classification take the re-decode path instead.
    """

    def __init__(self, records, manifest, recipe, reference_shape, *, frozen=False):
        self.records = records
        self.manifest = manifest
        self.recipe = recipe
        self.reference_shape = reference_shape
        self.frozen = frozen
        self._certificate = None

    def restore(self, processor, config_document, file_path, params):
        if self._certificate is not None:
            self._certificate.close()
            self._certificate = None
        result = restore_source(
            processor,
            self.manifest,
            self.recipe,
            config_document,
            self.reference_shape,
            file_path,
            params,
            frozen=self.frozen,
        )
        if self.frozen and result is not None:
            source, self._certificate = result
            return source
        return result

    def take_source_certificate(self, source):
        certificate, self._certificate = self._certificate, None
        if certificate is not None:
            if certificate.source is source:
                certificate.validate(source)
                return certificate
            certificate.close()
        return None

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


def source_anchor(manifest):
    calibration = manifest.get("calibration") or {}
    provenance = calibration.get("provenance") or {}
    capture = provenance.get("capture") or {}
    return {
        "source_shape": manifest.get("source_shape"),
        "raw_import_receipt": manifest.get("raw_import_receipt"),
        "flatfield_reference_shape": capture.get("sensor_shape"),
    }


class ImportDecodeCache:
    """Capture decoder state, then cache it once the completed import is certified."""

    def __init__(self, request, recipe):
        self.records = request["inputs"]
        self.recipe = recipe
        self.pending = None

    def restore(self, processor, config_document, file_path, params):
        return None

    def store(self, processor, config_document, file_path, source, ir, color_space):
        if not _enabled() or ir is not None or _budget_bytes() < source.nbytes + 4096:
            return
        # Export can evict or replace the processor's working source. Keep the
        # decoder metadata without pinning another image or processor.
        state = SimpleNamespace(
            _source_cache_key=processor._source_cache_key,
            _cam_xyz_by_path={file_path: deepcopy(processor._cam_xyz_by_path.get(file_path))},
            pure_raw_import_receipt=deepcopy(getattr(processor, "pure_raw_import_receipt", None)),
            pure_raw_import_token=getattr(processor, "pure_raw_import_token", None),
            pure_flatfield_sampled_source_mean_absolute_change=getattr(
                processor, "pure_flatfield_sampled_source_mean_absolute_change", None
            ),
        )
        self.pending = weakref.ref(source), state, config_document, file_path, color_space

    def commit(self, manifest, source):
        pending, self.pending = self.pending, None
        if (
            pending is None
            or manifest.get("state") != "complete"
            or manifest.get("raw_hashes_unchanged") is not True
            or manifest.get("inputs") != self.records
            or manifest.get("automatic_roles") != self.records
        ):
            return
        reference, state, config_document, file_path, color_space = pending
        if reference() is not source:
            return
        context = render_decode_context(manifest, self.recipe, source_anchor(manifest))
        if context is not None:
            context.store(state, config_document, file_path, source, None, color_space)


def render_decode_context(request, recipe, anchor, *, frozen=False):
    """Build the render cache context, or None when the anchor cannot bind.

    Strict anchor validation keeps new photos (no completed run yet) exactly
    on the uncached path: a wrong or partial anchor disables caching instead
    of restoring an unproven decode.
    """
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
    if policy == STANDARD_MONO_IMPORT and not _mono_receipt_current(receipt):
        return None
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
        request["inputs"], manifest, recipe, reference, frozen=frozen
    )


def _evict_overflow(directory, budget):
    entries = {}
    now = time.time()
    for path in directory.iterdir():
        # A killed process cannot run its temporary-file cleanup. Remove only
        # our old files, leaving active writes and unrelated files untouched.
        transient = path.name.startswith((".decoded-cache-tmp-", ".prewarm-marker-"))
        marker = path.name.startswith("prewarm-") and path.suffix == ".json"
        if (transient or marker) and not path.is_symlink():
            try:
                max_age = (24 if transient else 30 * 24) * 60 * 60
                if path.is_file() and path.stat().st_mtime < now - max_age:
                    path.unlink()
            except OSError:
                pass
            continue
        stem = path.stem
        if (
            path.is_symlink()
            or not path.is_file()
            or path.suffix not in (".npy", ".pkl")
            or len(stem) != 64
            or any(character not in "0123456789abcdef" for character in stem)
        ):
            continue
        try:
            stat = path.lstat()
        except OSError:
            # Other render processes can evict the same entry concurrently.
            continue
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
