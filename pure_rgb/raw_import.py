"""Versioned RAW import on the unchanged NegPy decoder and rendering engine.

The explicit standard-crop policy selects each sensor band before registration.
Legacy recipes keep the original processor and their original coordinate space.
"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from negpy.services.rendering.image_processor import ImageProcessor

from .raw_validity import plan_standard_crop, StandardCropPlan

LEGACY_IMPORT = "legacy-v1"
STANDARD_IMPORT = "standard-crop-v1"
STANDARD_DNG_IMPORT = "standard-cfa-dng-v1"
STANDARD_MONO_IMPORT = "standard-mono-v1"
IMPORT_POLICIES = (LEGACY_IMPORT, STANDARD_IMPORT, STANDARD_DNG_IMPORT, STANDARD_MONO_IMPORT)
REGISTRATION_CHECK = "translation-consistency-v1"
REGISTRATION_SOURCE_CHECK = "translation-consistency-source-v2"
REGISTRATION_CHECKS = (REGISTRATION_CHECK, REGISTRATION_SOURCE_CHECK)


def _raw_decode_workers(source_shape):
    """Use independent LibRaw decoders only when CPU and memory allow them."""
    if source_shape[0] * source_shape[1] < 1_000_000:
        return 1
    try:
        from .workflow_exports import _available_memory

        available = _available_memory()
        threads = int(os.environ.get("RAYON_NUM_THREADS") or os.cpu_count() or 1)
    except (OSError, ValueError, TypeError):
        return 1
    if available is None:
        return 1
    # LibRaw maintains private decode buffers beyond its RGB16 result. Reserve
    # foreground memory and conservatively budget each simultaneous decode.
    pixels = source_shape[0] * source_shape[1]
    reserve = 4 * 1024**3
    per_decoder = max(1024**3, pixels * 48)
    return max(1, min(3, threads, (available - reserve) // per_decoder))


def checked_registration_check(policy, value):
    """Opt-in model checking is structural import state, never a live trim."""
    if value is None:
        return None
    if type(value) is not str or value not in REGISTRATION_CHECKS:
        raise ValueError("Unknown RAW registration check")
    if policy not in (STANDARD_IMPORT, STANDARD_DNG_IMPORT, STANDARD_MONO_IMPORT):
        raise ValueError("Registration model checking requires standard RAW import")
    return value


def validate_registration_requirement(receipt, requirement):
    """Validate an explicit new proof without upgrading historic v1 receipts."""
    registration = receipt.get("registration")
    if not isinstance(registration, dict):
        if requirement is not None:
            raise ValueError("Missing registration model evidence")
        return
    if requirement is None:
        if registration.get("version") in (2, 3) or "model_check" in registration:
            raise ValueError("Checked registration requires its explicit recipe policy")
        return
    from .raw_registration import validate_registration_model_metadata

    validate_registration_model_metadata(registration)
    proof = registration.get("model_check")
    if (
        type(registration.get("version")) is not int
        or registration["version"]
        != (3 if requirement == REGISTRATION_SOURCE_CHECK else 2)
        or not isinstance(proof, dict)
        or proof.get("policy") != requirement
        or proof.get("status") != "passed"
    ):
        raise ValueError("Frozen registration has no matching model-check proof")


def is_standard_import(policy):
    return policy in (STANDARD_IMPORT, STANDARD_DNG_IMPORT, STANDARD_MONO_IMPORT)


def probe_for_policy(policy):
    if policy == STANDARD_DNG_IMPORT:
        from .raw_dng_import import probe_cfa_dng_frame

        return probe_cfa_dng_frame
    if policy == STANDARD_IMPORT:
        return probe_standard_frame
    if policy == LEGACY_IMPORT:
        from negpy.features.rgbscan.logic import probe_frame

        return probe_frame
    raise ValueError("Unknown RAW import policy")


def file_identity(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def inspect_raw_header(path):
    """Read header geometry only; RawPy.open_file does not unpack the mosaic."""
    import rawpy
    from negpy.infrastructure.loaders.helpers import (
        read_orientation,
        read_exif_from_file,
    )

    before = file_identity(path)
    with rawpy.RawPy() as raw:
        raw.open_file(str(path))
        sizes = raw.sizes._asdict()
    plan = plan_standard_crop(
        sizes, (sizes["height"], sizes["width"], 3), user_flip=0, half_size=False
    )
    orientation = read_orientation(str(path))
    if type(orientation) is not int or not 1 <= orientation <= 8:
        raise ValueError("RAW EXIF orientation must be explicit 1..8")
    exif = read_exif_from_file(str(path)) or {}
    camera = [str(exif.get("0th", {}).get(key, "")) for key in (271, 272)]
    if file_identity(path) != before:
        raise ValueError("RAW changed while its header was read")
    return {"plan": plan, "orientation": orientation, "camera": camera}


def probe_standard_frame(path):
    """Use the existing classifier on the standard mosaic before bad tail rows."""
    import rawpy
    from negpy.features.rgbscan.logic import (
        FrameProbe,
        _channel_means,
        _scene_signature,
    )

    before = file_identity(path)
    with rawpy.imread(str(path)) as raw:
        sizes = raw.sizes._asdict()
        plan = plan_standard_crop(
            sizes, (sizes["height"], sizes["width"], 3), user_flip=0, half_size=False
        )
        x0, y0, x1, y1 = plan.visible_roi
        mosaic = raw.raw_image_visible[y0:y1, x0:x1].astype(np.float32)
        colors = raw.raw_colors_visible[y0:y1, x0:x1]
        black = float(np.mean(raw.black_level_per_channel))
        probe = FrameProbe(
            means=_channel_means(mosaic, colors, black),
            signature=_scene_signature(mosaic, black),
        )
    if file_identity(path) != before:
        raise ValueError("RAW changed during triplet classification")
    return probe


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class StandardRawProcessor(ImageProcessor):
    """CPU RGB-triplet importer; signed cubic output remains float32."""

    policy = STANDARD_IMPORT

    def __init__(self, *, flatfield_reference_shape=None, registration_check=None, lens_correction=False,
                 chromatic_aberration=False, lens_profile=None):
        super().__init__(use_gpu=False)
        self.pure_lens_correction = lens_correction
        self.pure_chromatic_aberration = chromatic_aberration
        self.pure_lens_profile = deepcopy(lens_profile)
        self._pure_registration_check = checked_registration_check(
            self.policy, registration_check
        )
        self.pure_flatfield_reference_shape = flatfield_reference_shape
        self.pure_raw_import_receipt = None
        self.pure_raw_import_token = self.policy
        self._pure_source_identity = None
        self.pure_flatfield_sampled_source_mean_absolute_change = None

    @property
    def pure_registration_check(self):
        return self._pure_registration_check

    @staticmethod
    def _standard_paths(file_path, params, *, suffix=".rw2"):
        from negpy.features.rgbscan.models import is_rgb_triplet
        from negpy.features.hdr.models import hdr_active
        from negpy.services.rendering.lens import metadata_lens_corrections
        from negpy.features.process.logic import (
            effective_linear_raw,
            effective_highlight_reconstruction,
        )

        if not is_rgb_triplet(params.rgbscan) or hdr_active(params.hdr):
            raise ValueError("Standard RAW import requires an RGB triplet without HDR")
        if params.stitch.stitch_enabled or metadata_lens_corrections(params):
            raise ValueError(
                "Standard RAW import does not support stitch/lens remapping"
            )
        if (
            not effective_linear_raw(params.process)
            or effective_highlight_reconstruction(params.process) != 0
        ):
            raise ValueError(
                "Standard RAW import requires linear RAW and no highlight reconstruction"
            )
        paths = tuple(
            str(Path(p).resolve(strict=True))
            for p in (file_path, params.rgbscan.green_path, params.rgbscan.blue_path)
        )
        if len(set(paths)) != 3 or any(Path(p).suffix.lower() != suffix for p in paths):
            label = "RW2" if suffix == ".rw2" else "CFA DNG"
            raise ValueError(
                f"Standard RAW import requires three distinct {label} files"
            )
        return paths

    @staticmethod
    def _paths(file_path, params):
        return StandardRawProcessor._standard_paths(file_path, params)

    def _header(self, path):
        return inspect_raw_header(path)

    def source_identity_for_paths(self, paths):
        identity = (tuple((p, file_identity(p)) for p in paths),
                    tuple(self.pure_flatfield_reference_shape or ()))
        if self.pure_chromatic_aberration or (self.pure_lens_profile and self.pure_lens_correction):
            identity += (_digest(dict(chromatic_aberration=self.pure_chromatic_aberration,
                                      lens_profile=self.pure_lens_profile,
                                      lens_correction=self.pure_lens_correction)),)
        return identity

    def _decode_band(self, path, band, plan, orientation, params):
        from negpy.kernel.image.logic import uint16_to_float32
        from negpy.features.process.path import positive_decode

        pixels, metadata = ImageProcessor._decode_sensor_rgb(
            self, path, True, fast=False, wb_override=[1, 1, 1, 1],
            demosaic=params.process.demosaic_export,
            positive_decode=positive_decode(params.process), highlight_mode=0,
        )
        if (pixels.dtype != np.uint16 or pixels.shape != plan.source_shape
                or metadata.get("orientation", 1) != orientation):
            raise ValueError("Decoded RAW does not match the frozen sensor geometry")
        x0, y0, x1, y1 = plan.visible_roi
        return uint16_to_float32(pixels[y0:y1, x0:x1, band]), metadata

    def _load_source_f32(self, file_path, params, fast_decode=False, unmix=True):
        if fast_decode:
            raise ValueError("Standard RAW import does not use half-size decoding")
        if self.pure_registration_check and not params.rgbscan.align:
            raise ValueError("Registration model checking requires alignment enabled")
        paths = self._paths(file_path, params)

        def identity_now():
            return self.source_identity_for_paths(paths)

        identity = identity_now()
        if identity != self._pure_source_identity:
            self._source_cache_key = self._source_cache_value = None
            self._prepare_slot = None
        try:
            result = super()._load_source_f32(file_path, params, False, unmix)
            if identity_now() != identity:
                raise ValueError("RAW changed during decoding or registration")
        except Exception:
            self._source_cache_key = self._source_cache_value = None
            self._prepare_slot = None
            self._pure_source_identity = None
            self.pure_raw_import_receipt = None
            self.pure_raw_import_token = self.policy
            raise
        self._pure_source_identity = identity
        return result

    def _decode_oriented_f32(
        self, file_path, params, fast_decode=False, wb_override=None, unmix=True
    ):
        from negpy.kernel.image.logic import apply_exif_orientation
        from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
        from .raw_registration import register_planes
        from .raw_flatfield import apply_standard_flatfield
        import rawpy

        if fast_decode or wb_override is not None:
            raise ValueError(
                "Standard triplets require full-resolution neutral decoding"
            )
        paths = self._paths(file_path, params)
        identities = [file_identity(p) for p in paths]
        headers = [self._header(p) for p in paths]
        first = headers[0]
        if any(header != first for header in headers[1:]):
            raise ValueError(
                "RGB RAW standard crops, sensor geometry, camera or EXIF differ"
            )
        plan, orientation = first["plan"], first["orientation"]
        planes, bands, primary_metadata = [], [], None
        lens_profiles = []
        def decode_band(item):
            band, (path, header) = item
            # The pinned decoder creates a fresh LibRaw handle per invocation;
            # it has no shared processor state. Keep only the selected band.
            plane, metadata = self._decode_band(path, band, plan, orientation, params)
            if file_identity(path) != identities[band]:
                raise ValueError("RAW changed during single-frame decoding")
            return plane, metadata

        workers = _raw_decode_workers(plan.source_shape)
        inputs = list(enumerate(zip(paths, headers)))
        # Mapping retains RGB order even when independent decoders finish in a
        # different order. Serial operation remains the low-memory fallback.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for band, (plane, metadata) in enumerate(executor.map(decode_band, inputs)):
                path, header = paths[band], headers[band]
                planes.append(plane)
                if self.pure_lens_correction or self.pure_chromatic_aberration:
                    lens_profiles.append(metadata.get("lens_correction"))
                if band == 0:
                    primary_metadata = metadata
                bands.append(
                    {
                        "role": "RGB"[band],
                        "path": path,
                        "geometry": asdict(plan),
                        "orientation": orientation,
                        "camera": header["camera"],
                    }
                )
        registration_options = (
            {"model_check": self.pure_registration_check}
            if self.pure_registration_check is not None
            else {}
        )
        merged, registration = register_planes(
            *planes, align=params.rgbscan.align, **registration_options
        )
        del planes
        left, top, right, bottom = registration["common_valid_roi_sensor_xyxy"]
        vx, vy, _, _ = plan.visible_roi
        rx, ry, _, _ = plan.raw_roi
        effective = replace(
            plan,
            raw_roi=(rx + left, ry + top, rx + right, ry + bottom),
            visible_roi=(vx + left, vy + top, vx + right, vy + bottom),
        )
        merged = apply_exif_orientation(merged, orientation)
        before_flatfield = merged[::64, ::64].copy() if params.flatfield.apply else None
        merged, flatfield = apply_standard_flatfield(
            merged,
            params.flatfield,
            effective,
            orientation=orientation,
            full_sensor_shape=self.pure_flatfield_reference_shape,
        )
        self.pure_flatfield_sampled_source_mean_absolute_change = (
            float(np.mean(np.abs(merged[::64, ::64] - before_flatfield)))
            if before_flatfield is not None else None
        )
        lens_receipt = None
        chromatic_receipt = None
        chromatic_reference = None
        if self.pure_lens_correction and not self.pure_lens_profile:
            from .raw_lens_correction import correct_registered_lens

            merged, lens_receipt = correct_registered_lens(
                merged, lens_profiles, plan, effective, orientation
            )
        baseline_lens_receipt = deepcopy(lens_receipt)
        if self.pure_chromatic_aberration or (self.pure_lens_profile and self.pure_lens_correction):
            from .raw_lens_correction import correct_additional_optics

            # This is the authenticated pre-CA source, in the same green
            # coordinate system as an existing dust selection. The mask may
            # be reused only when inputs, registration, flatfield and this
            # baseline all still match; it never authorizes arbitrary sources.
            baseline = merged
            baseline_context = {}
            merged, additional_lens, chromatic_receipt = correct_additional_optics(
                merged, lens_profiles, paths, plan, effective, orientation,
                chromatic_aberration=self.pure_chromatic_aberration,
                lens_profile=self.pure_lens_profile, distortion=self.pure_lens_correction,
                embedded_receipt=lens_receipt, registration=registration,
                baseline_context=baseline_context,
            )
            baseline = baseline_context.get("pixels", baseline)
            baseline_lens_receipt = baseline_context.get("lens_receipt", baseline_lens_receipt)
            baseline_sha = hashlib.sha256(memoryview(baseline).cast("B")).hexdigest()
            green_sha = hashlib.sha256(np.ascontiguousarray(baseline[..., 1])).hexdigest()
            if additional_lens:
                lens_receipt = additional_lens
            chromatic_reference = dict(source_pixels_sha256=baseline_sha,
                green_pixels_sha256=green_sha)
            chromatic_reference.update(
                green_unchanged=bool(np.array_equal(baseline[..., 1], merged[..., 1])),
                canvas_unchanged=baseline.shape == merged.shape)
        if tuple((p, file_identity(p)) for p in paths) != tuple(zip(paths, identities)):
            raise ValueError("RAW changed while the triplet was assembled")
        geometry = {
            "reference": "green full-visible RAW, then shared EXIF orientation",
            "effective_plan": asdict(effective),
            "orientation": orientation,
            "full_visible_oriented_roi_xyxy": list(
                effective.oriented_full_roi(orientation)
            ),
            "source_shape": list(merged.shape),
            **({"embedded_lens": lens_receipt} if lens_receipt else {}),
        }
        receipt = {
            "policy": self.policy,
            "version": 1,
            "bands": bands,
            "registration": registration,
            "effective_geometry": geometry,
            "geometry_token": _digest(
                {
                    "bands": bands,
                    "registration": registration,
                    "effective_geometry": geometry,
                }
            ),
            "flatfield": flatfield,
            "source_shape": list(merged.shape),
            "source_dtype": str(merged.dtype),
            "source_pixels_sha256": hashlib.sha256(
                memoryview(merged).cast("B")
            ).hexdigest(),
            "source_read_only": True,
            "decode": {
                "rawpy_version": rawpy.__version__,
                "LibRaw_version": list(rawpy.libraw_version),
                "linear_raw": True,
                "neutral_WB": True,
                "highlight_reconstruction": 0,
                "half_size": False,
                "user_flip": 0,
                "demosaic": str(params.process.demosaic_export),
            },
            "raw_files_modified": False,
            "LibRaw_decoder_patched": False,
        }
        if chromatic_receipt is not None and self.pure_chromatic_aberration:
            receipt["chromatic_aberration"] = chromatic_receipt
            baseline_geometry = {k: v for k, v in geometry.items() if k != "embedded_lens"}
            if baseline_lens_receipt:
                baseline_geometry["embedded_lens"] = baseline_lens_receipt
            chromatic_reference["effective_geometry"] = baseline_geometry
            chromatic_reference["geometry_token"] = _digest(dict(
                bands=bands, registration=registration, effective_geometry=baseline_geometry))
            receipt["chromatic_reference"] = chromatic_reference
        merged.setflags(write=False)
        self.pure_raw_import_receipt = receipt
        self.pure_raw_import_token = _digest(receipt)
        self._cam_xyz_by_path[file_path] = (primary_metadata.get("cam_xyz"), None)
        return (
            merged,
            None,
            str(primary_metadata.get("color_space") or WORKING_COLOR_SPACE),
        )


def create_processor(
    policy=LEGACY_IMPORT, *, flatfield_reference_shape=None, registration_check=None,
    lens_correction=False, chromatic_aberration=False, lens_profile=None,
):
    if policy not in IMPORT_POLICIES:
        raise ValueError("Unknown RAW import policy")
    registration_check = checked_registration_check(policy, registration_check)
    if type(chromatic_aberration) is not bool:
        raise ValueError("自动色差矫正必须为开关")
    if (chromatic_aberration or lens_profile) and policy not in (STANDARD_IMPORT, STANDARD_DNG_IMPORT):
        raise ValueError("自动色差矫正目前用于标准拜耳三色 RAW")
    options = (
        {"registration_check": registration_check}
        if registration_check is not None
        else {}
    )
    if lens_correction:
        if not is_standard_import(policy):
            raise ValueError("内嵌镜头矫正需要标准 RAW 导入")
        options["lens_correction"] = True
    if chromatic_aberration:
        options["chromatic_aberration"] = True
    if lens_profile:
        options["lens_profile"] = deepcopy(lens_profile)
    if policy == STANDARD_DNG_IMPORT:
        from .raw_dng_import import StandardCfaDngProcessor

        if flatfield_reference_shape is not None:
            raise ValueError("CFA DNG import does not support flatfield references")
        return StandardCfaDngProcessor(**options)
    if policy == STANDARD_MONO_IMPORT:
        from .raw_mono_import import StandardMonoProcessor

        return StandardMonoProcessor(flatfield_reference_shape=flatfield_reference_shape, **options)
    return (
        ImageProcessor(use_gpu=False)
        if policy == LEGACY_IMPORT
        else StandardRawProcessor(
            flatfield_reference_shape=flatfield_reference_shape, **options
        )
    )


def frozen_import_receipt(processor):
    return deepcopy(getattr(processor, "pure_raw_import_receipt", None))


def validate_frozen_import(processor, manifest, source):
    """Frozen analysis/fit may only be replayed on the same input geometry."""
    requirement = checked_registration_check(
        manifest["recipe"].get("raw_import", LEGACY_IMPORT),
        manifest["recipe"].get("registration_check"),
    )
    if manifest["recipe"].get("raw_import", LEGACY_IMPORT) == LEGACY_IMPORT:
        return
    if (bool(getattr(processor, "pure_chromatic_aberration", False))
            != bool(manifest["recipe"].get("chromatic_aberration", False))
            or getattr(processor, "pure_lens_profile", None) != manifest["recipe"].get("lens_profile")):
        raise ValueError("Frozen RAW optics differ from the recipe")
    expected = manifest.get("raw_import_receipt")
    actual = frozen_import_receipt(processor)
    if (
        not isinstance(expected, dict)
        or not isinstance(actual, dict)
        or expected.get("policy") != actual.get("policy")
        or actual.get("policy") != manifest["recipe"].get("raw_import", LEGACY_IMPORT)
        or expected.get("geometry_token") != actual.get("geometry_token")
        or expected.get("source_pixels_sha256") != actual.get("source_pixels_sha256")
        or list(source.shape) != manifest["source_shape"]
    ):
        raise ValueError("RAW import geometry changed; recompute analysis in a new job")
    validate_registration_requirement(expected, requirement)
    validate_registration_requirement(actual, requirement)
    if requirement is not None:
        if getattr(processor, "pure_registration_check", None) != requirement:
            raise ValueError("Processor registration check differs from frozen recipe")
        for receipt in (expected, actual):
            try:
                token = hashlib.sha256(
                    json.dumps(
                        {
                            key: receipt[key]
                            for key in ("bands", "registration", "effective_geometry")
                        },
                        sort_keys=True,
                        allow_nan=False,
                    ).encode()
                ).hexdigest()
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Invalid checked import geometry evidence") from error
            if receipt.get("geometry_token") != token:
                raise ValueError("Checked registration geometry token is inconsistent")
        if (
            not isinstance(source, np.ndarray)
            or source.dtype != np.float32
            or source.ndim != 3
            or source.shape[-1] != 3
            or not source.flags.c_contiguous
            or source.flags.writeable
            or not np.isfinite(source).all()
            or hashlib.sha256(memoryview(source).cast("B")).hexdigest()
            != actual.get("source_pixels_sha256")
        ):
            raise ValueError("Checked registration actual source identity differs")
        for receipt in (expected, actual):
            if (
                receipt.get("source_shape") != list(source.shape)
                or any(type(value) is not int for value in receipt["source_shape"])
                or receipt.get("source_dtype") != str(source.dtype)
                or receipt.get("source_read_only") is not True
                or not isinstance(receipt.get("effective_geometry"), dict)
                or receipt["effective_geometry"].get("source_shape")
                != list(source.shape)
            ):
                raise ValueError("Checked registration source shape/domain differs")
    if actual["policy"] == STANDARD_DNG_IMPORT:
        from .raw_dng_import import validate_dng_import_receipt_metadata

        validate_dng_import_receipt_metadata(expected)
        validate_dng_import_receipt_metadata(actual)


def migrate_full_visible_recipe(recipe, receipt):
    """Explicitly migrate old full-visible ROIs into a NEW standard-crop job.

    Registration/analysis/fit must be recomputed. A partially supported ROI is
    rejected; never silently intersect a requested historical output rectangle.
    """
    if recipe.get("raw_import", LEGACY_IMPORT) != LEGACY_IMPORT:
        raise ValueError("Only an explicitly legacy-coordinate recipe can be migrated")
    if receipt["policy"] != STANDARD_IMPORT:
        raise ValueError("A standard-crop import receipt is required")
    if recipe.get("crop", True) is False:
        raise ValueError(
            "Legacy full-visible output includes unsupported borders; select an explicit supported output ROI for the new job"
        )
    geometry = receipt["effective_geometry"]
    plan = StandardCropPlan(**geometry["effective_plan"])
    result = deepcopy(recipe)
    for key in ("roi", "output_roi"):
        if result.get(key) is not None:
            result[key] = list(
                plan.full_to_local_roi(result[key], orientation=geometry["orientation"])
            )
    result["raw_import"] = STANDARD_IMPORT
    return result
