"""Explicit M11 CFA-DNG import policy on the existing standard decoder.

Metadata gates reject unsupported/visible converted layouts. They are not an
authentication of camera origin. No existing RW2 policy/default is widened.
"""

from copy import deepcopy
from dataclasses import fields
import math
from numbers import Real
from pathlib import Path
import re

from .raw_import import (
    STANDARD_DNG_IMPORT,
    StandardRawProcessor,
    _digest,
    file_identity,
    probe_standard_frame,
)

HEADER_POLICY = "leica-m11-cfa16-rggb-camera-metadata-v1"
HEADER_FIELDS = {
    "schema",
    "version",
    "header_policy",
    "make",
    "model",
    "software",
    "dng_version",
    "raw_ifd_index",
    "raw_ifd_offset",
    "raw_shape",
    "samples_per_pixel",
    "bits_per_sample",
    "sample_format",
    "photometric",
    "cfa_repeat",
    "cfa_pattern",
    "cfa_plane_color",
    "default_crop_origin",
    "default_crop_size",
    "active_area",
    "orientation",
    "original_raw_filename",
    "source_authenticity_certified",
}


def validate_dng_cfa_header_metadata(header):
    """Validate and copy a JSON header, without opening a file or decoding RAW."""
    if not isinstance(header, dict) or set(header) != HEADER_FIELDS:
        raise ValueError("Malformed CFA DNG header metadata")
    h = deepcopy(header)
    for key in (
        "raw_ifd_index",
        "raw_ifd_offset",
        "samples_per_pixel",
        "bits_per_sample",
        "sample_format",
        "photometric",
        "orientation",
    ):
        if type(h[key]) is not int:
            raise ValueError("CFA DNG scalar fields require integer JSON values")
    for key in (
        "dng_version",
        "raw_shape",
        "cfa_repeat",
        "cfa_pattern",
        "cfa_plane_color",
        "default_crop_origin",
        "default_crop_size",
    ):
        if not isinstance(h[key], list) or any(type(v) is not int for v in h[key]):
            raise ValueError("CFA DNG vectors require integer JSON values")
    if (
        h["schema"] != "pure-rgb-cfa-dng-header"
        or type(h["version"]) is not int
        or h["version"] != 1
        or h["header_policy"] != HEADER_POLICY
        or h["make"] != "Leica Camera AG"
        or h["model"] != "LEICA M11"
        or not isinstance(h["software"], str)
        or re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", h["software"]) is None
        or h["source_authenticity_certified"] is not False
    ):
        raise ValueError(
            "Unsupported M11 firmware/source metadata; no camera-origin authentication"
        )
    if (
        h["photometric"] != 32803
        or h["samples_per_pixel"] != 1
        or h["bits_per_sample"] != 16
        or h["sample_format"] != 1
        or h["cfa_repeat"] != [2, 2]
        or h["cfa_pattern"] != [0, 1, 1, 2]
        or h["cfa_plane_color"] != [0, 1, 2]
        or h["active_area"] is not None
        or h["original_raw_filename"] is not None
        or type(h["raw_ifd_index"]) is not int
        or h["raw_ifd_index"] < 0
        or type(h["raw_ifd_offset"]) is not int
        or h["raw_ifd_offset"] < 8
        or type(h["orientation"]) is not int
        or not 1 <= h["orientation"] <= 8
    ):
        raise ValueError("Unsupported/mono/linear/converted CFA DNG layout")
    for key, count in (
        ("raw_shape", 2),
        ("default_crop_origin", 2),
        ("default_crop_size", 2),
        ("dng_version", 4),
    ):
        value = h[key]
        if (
            not isinstance(value, list)
            or len(value) != count
            or any(type(v) is not int for v in value)
        ):
            raise ValueError("Malformed CFA DNG geometry/version")
    height, width = h["raw_shape"]
    x, y = h["default_crop_origin"]
    cw, ch = h["default_crop_size"]
    if (
        min(height, width, cw, ch) < 2
        or min(x, y) < 0
        or x + cw > width
        or y + ch > height
        or h["dng_version"][:2] not in ([1, 3], [1, 4], [1, 5], [1, 6])
    ):
        raise ValueError("Unsupported CFA DNG crop/version")
    return h


def read_dng_cfa_header(path):
    """Read TIFF/DNG metadata only. Never call asarray, RAW unpack or postprocess."""
    import tifffile

    before = file_identity(path)
    with tifffile.TiffFile(path) as tif:
        candidates = []
        for index, page in enumerate(tif.pages):
            tag = page.tags.get(262)
            if tag is not None and int(tag.value) == 32803:
                candidates.append((index, page))
        if len(candidates) != 1:
            raise ValueError(
                "Require one primary CFA DNG IFD; linear/mono is unsupported"
            )
        index, page = candidates[0]

        def value(tag, default=None):
            found = page.tags.get(tag)
            return default if found is None else found.value

        def integer(found):
            if (
                isinstance(found, bool)
                or not isinstance(found, Real)
                or not math.isfinite(found)
                or int(found) != found
            ):
                raise ValueError("CFA DNG tag must be an exact integer")
            return int(found)

        def integers(tag, default=None):
            found = value(tag, default)
            return None if found is None else [integer(v) for v in found]

        header = {
            "schema": "pure-rgb-cfa-dng-header",
            "version": 1,
            "header_policy": HEADER_POLICY,
            "make": str(value(271, "")),
            "model": str(value(272, "")),
            "software": str(value(305, "")),
            "dng_version": integers(50706),
            "raw_ifd_index": index,
            "raw_ifd_offset": int(page.offset),
            "raw_shape": [int(page.imagelength), int(page.imagewidth)],
            "samples_per_pixel": integer(value(277, 1)),
            "bits_per_sample": integer(value(258, 0)),
            "sample_format": integer(value(339, 1)),
            "photometric": integer(value(262, 0)),
            "cfa_repeat": integers(33421),
            "cfa_pattern": integers(33422),
            "cfa_plane_color": integers(50710, (0, 1, 2)),
            "default_crop_origin": integers(50719),
            "default_crop_size": integers(50720),
            "active_area": integers(50829),
            "orientation": integer(value(274, 0)),
            "original_raw_filename": value(50827),
            "source_authenticity_certified": False,
        }
    if before != file_identity(path):
        raise ValueError("DNG changed during metadata inspection")
    return validate_dng_cfa_header_metadata(header)


def probe_cfa_dng_frame(path):
    read_dng_cfa_header(path)
    return probe_standard_frame(path)


def _validate_effective_geometry(receipt):
    from .raw_validity import StandardCropPlan

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def plan(document):
        names = {f.name for f in fields(StandardCropPlan)}
        require(
            isinstance(document, dict) and set(document) == names,
            "Malformed CFA DNG crop plan",
        )
        require(
            all(
                isinstance(v, (list, tuple)) and all(type(i) is int for i in v)
                for v in document.values()
            ),
            "CFA DNG crop coordinates must be integers",
        )
        return StandardCropPlan(**{k: tuple(v) for k, v in document.items()})

    geometry = receipt.get("effective_geometry")
    registration = receipt.get("registration")
    if isinstance(registration, dict) and (
        registration.get("version") in (2, 3) or "model_check" in registration
    ):
        from .raw_registration import validate_registration_model_metadata

        validate_registration_model_metadata(registration)
    require(
        isinstance(geometry, dict) and isinstance(registration, dict),
        "Missing CFA DNG effective geometry or registration",
    )
    try:
        plans = [plan(band["geometry"]) for band in receipt["bands"]]
        effective = plan(geometry["effective_plan"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Malformed CFA DNG crop plan") from error
    base = plans[0]
    require(all(p == base for p in plans[1:]), "CFA DNG band plans differ")
    orientation = geometry.get("orientation")
    require(
        type(orientation) is int
        and 1 <= orientation <= 8
        and all(
            type(b.get("orientation")) is int and b["orientation"] == orientation
            for b in receipt["bands"]
        ),
        "CFA DNG shared EXIF orientation mismatch",
    )
    x0, y0, x1, y1 = base.visible_roi
    width, height = x1 - x0, y1 - y0
    require(
        registration.get("reference_channel") == "green"
        and type(registration.get("align")) is bool
        and registration.get("input_shape_h_w") == [height, width],
        "CFA DNG registration input geometry mismatch",
    )
    common = registration.get("common_valid_roi_sensor_xyxy")
    require(
        isinstance(common, (list, tuple))
        and len(common) == 4
        and all(type(v) is int for v in common),
        "Malformed CFA DNG common support",
    )
    left, top, right, bottom = common
    require(
        0 <= left < right <= width and 0 <= top < bottom <= height,
        "Invalid CFA DNG common support",
    )
    require(
        registration["align"] or list(common) == [0, 0, width, height],
        "Disabled CFA DNG registration must retain full standard crop",
    )
    rx, ry, _, _ = base.raw_roi
    expected = StandardCropPlan(
        source_shape=base.source_shape,
        raw_shape=base.raw_shape,
        visible_origin_raw=base.visible_origin_raw,
        raw_roi=(rx + left, ry + top, rx + right, ry + bottom),
        visible_roi=(x0 + left, y0 + top, x0 + right, y0 + bottom),
    )
    require(effective == expected, "CFA DNG effective crop/support mismatch")
    full = list(expected.oriented_full_roi(orientation))
    shape = [full[3] - full[1], full[2] - full[0], 3]
    require(
        geometry.get("full_visible_oriented_roi_xyxy") == full
        and geometry.get("source_shape") == shape
        and receipt.get("source_shape") == shape,
        "CFA DNG effective EXIF output shape/ROI mismatch",
    )


def validate_dng_import_receipt_metadata(receipt):
    """Cross-check new policy/header/geometry metadata only; never read source."""
    if (
        not isinstance(receipt, dict)
        or receipt.get("policy") != STANDARD_DNG_IMPORT
        or type(receipt.get("version")) is not int
        or receipt.get("version") != 1
    ):
        raise ValueError("Require explicit CFA DNG import receipt")
    stamp = receipt.get("dng_cfa_validation")
    if (
        not isinstance(stamp, dict)
        or set(stamp)
        != {
            "schema",
            "version",
            "header_policy",
            "headers",
            "metadata_only",
            "source_authenticity_certified",
        }
        or stamp["schema"] != "pure-rgb-standard-cfa-dng-validation"
        or type(stamp["version"]) is not int
        or stamp["version"] != 1
        or stamp["header_policy"] != HEADER_POLICY
        or stamp["metadata_only"] is not True
        or stamp["source_authenticity_certified"] is not False
    ):
        raise ValueError("Malformed CFA DNG validation provenance")
    bands = receipt.get("bands", [])
    if (
        not isinstance(bands, list)
        or len(bands) != 3
        or any(not isinstance(b, dict) for b in bands)
        or [b.get("role") for b in bands] != list("RGB")
        or not isinstance(stamp["headers"], list)
        or len(stamp["headers"]) != 3
    ):
        raise ValueError("CFA DNG receipt requires three RGB bands")
    paths = [b.get("path") for b in bands]
    if (
        any(
            not isinstance(p, str)
            or not Path(p).is_absolute()
            or Path(p).suffix.lower() != ".dng"
            for p in paths
        )
        or len(set(paths)) != 3
    ):
        raise ValueError("CFA DNG receipt paths must be distinct absolute DNG files")
    signatures = []
    for band, header in zip(bands, stamp["headers"]):
        h = validate_dng_cfa_header_metadata(header)
        geometry = band.get("geometry")
        if not isinstance(geometry, dict) or not {
            "raw_shape",
            "raw_roi",
            "source_shape",
            "visible_roi",
            "visible_origin_raw",
        } <= set(geometry):
            raise ValueError("Missing CFA DNG band crop geometry")
        if any(
            not isinstance(geometry[key], (list, tuple))
            for key in (
                "raw_shape",
                "raw_roi",
                "source_shape",
                "visible_origin_raw",
                "visible_roi",
            )
        ):
            raise ValueError("Malformed CFA DNG band crop coordinates")
        x, y = h["default_crop_origin"]
        w, height = h["default_crop_size"]
        if (
            band.get("dng_cfa_header") != h
            or list(geometry["raw_shape"]) != h["raw_shape"]
            or list(geometry["raw_roi"]) != [x, y, x + w, y + height]
            or list(geometry["source_shape"]) != [*h["raw_shape"], 3]
            or list(geometry["visible_origin_raw"]) != [0, 0]
            or list(geometry["visible_roi"]) != list(geometry["raw_roi"])
            or band.get("orientation") != h["orientation"]
            or band.get("camera") != [h["make"], h["model"]]
        ):
            raise ValueError("CFA DNG header/band geometry disagrees")
        signatures.append({k: v for k, v in h.items() if k != "raw_ifd_offset"})
    if any(h != signatures[0] for h in signatures[1:]):
        raise ValueError("CFA DNG triplet header signatures differ")
    _validate_effective_geometry(receipt)
    if receipt.get("geometry_token") != _digest(
        {k: receipt[k] for k in ("bands", "registration", "effective_geometry")}
    ):
        raise ValueError("CFA DNG geometry token mismatch")
    decode = receipt.get("decode", {})
    if (
        not isinstance(decode, dict)
        or decode.get("linear_raw") is not True
        or decode.get("neutral_WB") is not True
        or decode.get("half_size") is not False
        or type(decode.get("user_flip")) is not int
        or decode["user_flip"] != 0
        or type(decode.get("highlight_reconstruction")) is not int
        or decode["highlight_reconstruction"] != 0
        or receipt.get("source_dtype") != "float32"
        or receipt.get("source_read_only") is not True
        or receipt.get("raw_files_modified") is not False
        or receipt.get("LibRaw_decoder_patched") is not False
        or receipt.get("source_shape")
        != receipt.get("effective_geometry", {}).get("source_shape")
    ):
        raise ValueError("Unsupported CFA DNG decode/source receipt domain")
    return deepcopy(receipt)


class StandardCfaDngProcessor(StandardRawProcessor):
    policy = STANDARD_DNG_IMPORT

    def __init__(self, *, registration_check=None, lens_correction=False):
        super().__init__(registration_check=registration_check, lens_correction=lens_correction)
        self._dng_headers = {}

    @staticmethod
    def _paths(file_path, params):
        if params.flatfield.apply:
            raise ValueError("CFA DNG import does not support flatfield")
        return StandardRawProcessor._standard_paths(file_path, params, suffix=".dng")

    def _header(self, path):
        metadata = read_dng_cfa_header(path)
        header = super()._header(path)
        plan = header["plan"]
        x, y = metadata["default_crop_origin"]
        w, h = metadata["default_crop_size"]
        if (
            list(plan.raw_shape) != metadata["raw_shape"]
            or list(plan.raw_roi) != [x, y, x + w, y + h]
            or header["orientation"] != metadata["orientation"]
        ):
            raise ValueError("LibRaw/DNG crop or EXIF header disagree")
        self._dng_headers[str(path)] = metadata
        return {**header, "camera": [metadata["make"], metadata["model"]]}

    def _decode_oriented_f32(
        self, file_path, params, fast_decode=False, wb_override=None, unmix=True
    ):
        result = super()._decode_oriented_f32(
            file_path, params, fast_decode, wb_override, unmix
        )
        receipt = deepcopy(self.pure_raw_import_receipt)
        headers = []
        for band in receipt["bands"]:
            header = deepcopy(self._dng_headers[band["path"]])
            band["dng_cfa_header"] = header
            headers.append(header)
        receipt["dng_cfa_validation"] = {
            "schema": "pure-rgb-standard-cfa-dng-validation",
            "version": 1,
            "header_policy": HEADER_POLICY,
            "headers": headers,
            "metadata_only": True,
            "source_authenticity_certified": False,
        }
        receipt["geometry_token"] = _digest(
            {k: receipt[k] for k in ("bands", "registration", "effective_geometry")}
        )
        validate_dng_import_receipt_metadata(receipt)
        self.pure_raw_import_receipt = receipt
        self.pure_raw_import_token = _digest(receipt)
        return result
