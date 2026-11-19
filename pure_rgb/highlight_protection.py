"""Source-bound explicit protection regions with an outward feather.

These are user/research annotations, not automatic person or skin recognition.
The mask remains one shared spatial weight for RGB and never edits the source.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import re

import cv2
import numpy as np

from .metering_masks import build_metering_mask, mask_from_spec


SCHEMA = "pure-rgb-highlight-protection"
VERSION = 1
RULE = "explicit region=1; outside distance/feather smoothstep to0; no inward erosion"


def _canonical(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pixels(mask):
    return hashlib.sha256(memoryview(mask).cast("B")).hexdigest()


def _source_key(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("Protection source_key must be the caller's explicit SHA256 identity")
    return value


@dataclass(frozen=True)
class HighlightProtection:
    mask: np.ndarray
    canonical_receipt: str

    @property
    def receipt(self):
        return json.loads(self.canonical_receipt)


def _feather(selection, feather_pixels, dtype):
    mask = selection.mask.astype(dtype)
    if feather_pixels > 0 and not selection.mask.all():
        distance = cv2.distanceTransform(
            np.ascontiguousarray(~selection.mask, dtype=np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        t = np.minimum(distance.astype(np.float64) / feather_pixels, 1.)
        mask = (1. - t * t * (3. - 2. * t)).astype(dtype)
        # Explicit interior protection must remain exactly1 despite any library
        # distance convention or rounding in the outward transition.
        mask[selection.mask] = 1.
    mask = np.ascontiguousarray(mask)
    mask.setflags(write=False)
    return mask


def build_highlight_protection(source_shape, source_key, *, include,
                               feather_pixels=64., dtype="float32"):
    """Build a full-source shared RGB mask from explicit vector annotations."""
    source_key = _source_key(source_key)
    if not isinstance(include, (list, tuple)) or not include:
        raise ValueError("Protection requires explicit nonempty regions; absent protection is None")
    if type(feather_pixels) not in (int, float):
        raise ValueError("Protection feather must be finite original-source pixels in [0,512]")
    try:
        feather_pixels = float(feather_pixels)
    except OverflowError as error:
        raise ValueError("Protection feather must be finite original-source pixels in [0,512]") from error
    if not math.isfinite(feather_pixels) or not 0 <= feather_pixels <= 512:
        raise ValueError("Protection feather must be finite original-source pixels in [0,512]")
    if dtype not in ("float32", "float64"):
        raise ValueError("Protection mask dtype must match float32/64 positive RGB")
    if not isinstance(source_shape, (tuple, list)) or len(source_shape) != 3 or source_shape[-1] != 3:
        raise ValueError("Protection requires the explicit full HWC RGB source shape")
    # Original vector rasterizer validates all shape integers and coordinates.
    selection = build_metering_mask(source_shape, (0, 0, source_shape[1], source_shape[0]), include=include)
    mask = _feather(selection, float(feather_pixels), dtype)
    receipt = {
        "schema": SCHEMA, "version": VERSION, "source_shape": list(selection.source_shape),
        "source_key": source_key, "coordinate_space": "full oriented merged source pixel centres",
        "vector_spec": selection.spec, "feather_pixels": float(feather_pixels),
        "feather_rule": RULE, "dtype": mask.dtype.name,
        "mask_pixels_sha256": _pixels(mask), "opencv_version": cv2.__version__,
        "explicit_annotation_not_automatic_recognition": True,
        "RGB_support_shared": True, "source_pixels_read_or_modified": False,
        "external_source_authenticity_certified": False,
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt).encode()).hexdigest()
    return HighlightProtection(mask, _canonical(receipt))


def restore_highlight_protection(document, *, source_shape, source_key):
    """Reconstruct exact source-coordinate weights; do not resize or guess origin."""
    if not isinstance(document, dict):
        raise ValueError("Protection receipt must be an object")
    try:
        encoded = _canonical(document)
        receipt = json.loads(encoded)
        checksum = receipt.pop("receipt_sha256")
        if (type(receipt["version"]) is not int or receipt["version"] != VERSION
                or receipt["schema"] != SCHEMA
                or checksum != hashlib.sha256(_canonical(receipt).encode()).hexdigest()
                or receipt["source_key"] != _source_key(source_key)
                or receipt["source_shape"] != list(source_shape)):
            raise ValueError("Protection source identity, version or checksum changed")
        vector = mask_from_spec(receipt["vector_spec"])
        if (list(vector.source_shape) != list(source_shape)
                or vector.spec["exclude"] or vector.spec["roi"] != [0, 0, source_shape[1], source_shape[0]]):
            raise ValueError("Protection must remain a full-source explicit inclusion")
        result = build_highlight_protection(
            source_shape, source_key, include=vector.spec["include"],
            feather_pixels=receipt["feather_pixels"], dtype=receipt["dtype"])
        if result.canonical_receipt != encoded:
            raise ValueError("Protection receipt, vector raster or feather implementation differs")
        return result
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("Malformed protection receipt") from error


def validate_highlight_protection(protection, *, source_shape, source_key):
    """Validate stored pixels and the independently re-rasterized vector binding."""
    if not isinstance(protection, HighlightProtection):
        raise ValueError("Expected HighlightProtection")
    rebuilt = restore_highlight_protection(protection.receipt, source_shape=source_shape, source_key=source_key)
    mask = protection.mask
    if (not isinstance(mask, np.ndarray) or mask.flags.writeable or not mask.flags.c_contiguous
            or mask.dtype != rebuilt.mask.dtype or mask.shape != rebuilt.mask.shape
            or _pixels(mask) != protection.receipt["mask_pixels_sha256"]
            or not np.array_equal(mask, rebuilt.mask)):
        raise ValueError("Protection mask changed or does not match its vector/source identity")
    return protection
