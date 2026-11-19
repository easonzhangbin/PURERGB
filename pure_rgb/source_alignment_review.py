"""Advisory native-resolution patch review after existing global alignment.

This is not a RAW registration policy or a replacement registration proof. It
reads finite distributed RGB patches at their current source pixel size, uses
the original gradient carrier/phase primitive and reports local remaining
displacements. No image, global shift, recipe or default is changed.
"""

from dataclasses import dataclass
import hashlib
import json
import re

import negpy_core
import numpy as np

from .raw_registration_model import (
    LOCAL_RESPONSE_MIN,
    MIN_SPAN_FRACTION,
    MIN_VALID_PATCHES,
    _local_carrier,
    source_residual_thresholds,
)


SCHEMA = "pure-rgb-native-source-alignment-review"
PATCH_SCHEMA = "pure-rgb-source-alignment-review-patches"
MAX_PATCH_SIDE = 512


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Source review metadata must be finite JSON") from error


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _pixels(image):
    return hashlib.sha256(memoryview(image).cast("B")).hexdigest()


def patch_rois(shape, roi, patch_side=MAX_PATCH_SIDE):
    _need(
        isinstance(shape, (tuple, list))
        and len(shape) == 3
        and all(type(v) is int and v > 0 for v in shape)
        and shape[2] == 3,
        "Source review requires a positive full HWC RGB source shape",
    )
    _need(
        type(patch_side) is int and 128 <= patch_side <= MAX_PATCH_SIDE,
        "Source review patch side must be128..512 native pixels",
    )
    _need(
        isinstance(roi, (tuple, list))
        and len(roi) == 4
        and all(type(v) is int for v in roi),
        "Source review ROI requires explicit integer source coordinates",
    )
    x0, y0, x1, y1 = roi
    _need(
        0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0],
        "Source review ROI outside source",
    )
    _need(
        x1 - x0 >= 2 * patch_side and y1 - y0 >= 2 * patch_side,
        "Source review requires distributed patch support",
    )
    xs = [x0, x0 + (x1 - x0 - patch_side) // 2, x1 - patch_side]
    ys = [y0, y0 + (y1 - y0 - patch_side) // 2, y1 - patch_side]
    return [[x, y, x + patch_side, y + patch_side] for y in ys for x in xs]


@dataclass(frozen=True)
class SourceAlignmentReviewPatches:
    pixels: tuple
    canonical_receipt: str

    @property
    def receipt(self):
        return json.loads(self.canonical_receipt)


def bind_source_alignment_patches(
    patches,
    *,
    source_shape,
    roi,
    source_key,
    geometry_token,
    provenance,
    patch_side=MAX_PATCH_SIDE,
):
    """Bind9 explicit native RGB arrays, including declared synthetic fixtures.

    source_key/geometry_token are caller provenance bindings, not authenticated
    source signatures. The actual reviewed arrays have independent pixel SHA.
    The campaign caller verifies real cache identities before building them.
    """
    boxes = patch_rois(source_shape, roi, patch_side)
    _need(
        all(
            isinstance(key, str) and re.fullmatch("[0-9a-f]{64}", key)
            for key in (source_key, geometry_token)
        ),
        "Explicit source and geometry SHA256 caller bindings required",
    )
    _need(
        isinstance(provenance, dict) and bool(provenance),
        "Explicit source or fixture provenance required",
    )
    _need(
        isinstance(patches, (tuple, list)) and len(patches) == 9,
        "Exactly9 distributed review patches required",
    )
    arrays, hashes = [], []
    for value in patches:
        _need(
            isinstance(value, np.ndarray)
            and value.dtype == np.float32
            and value.shape == (patch_side, patch_side, 3)
            and np.isfinite(value).all(),
            "Review patches require finite native float32 RGB",
        )
        image = np.array(value, copy=True, order="C")
        image.setflags(write=False)
        arrays.append(image)
        hashes.append(_pixels(image))
    receipt = {
        "schema": PATCH_SCHEMA,
        "version": 1,
        "source_shape": list(source_shape),
        "review_roi": list(roi),
        "patch_side": patch_side,
        "patch_rois_source_xyxy": boxes,
        "source_key": source_key,
        "geometry_token": geometry_token,
        "provenance": provenance,
        "coordinate_space": "current globally aligned oriented source pixels; no resize",
        "patches_dtype": "<f4",
        "patch_pixels_sha256": hashes,
        "full_source_authenticity_certified_by_this_module": False,
    }
    receipt["binding_sha256"] = _digest(receipt)
    return SourceAlignmentReviewPatches(tuple(arrays), _canonical(receipt))


def build_source_alignment_patches(
    source, *, roi, source_key, geometry_token, provenance, patch_side=MAX_PATCH_SIDE
):
    """Copy only9 bounded patches from an already globally registered source."""
    _need(
        isinstance(source, np.ndarray)
        and source.dtype == np.float32
        and source.ndim == 3
        and source.shape[-1] == 3
        and source.flags.c_contiguous
        and not source.flags.writeable,
        "Review source must be readonly C-order float32 HWC RGB",
    )
    boxes = patch_rois(list(source.shape), roi, patch_side)
    return bind_source_alignment_patches(
        [source[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes],
        source_shape=list(source.shape),
        roi=roi,
        source_key=source_key,
        geometry_token=geometry_token,
        provenance=provenance,
        patch_side=patch_side,
    )


def _validate(patches):
    _need(
        isinstance(patches, SourceAlignmentReviewPatches),
        "Bound source review patches required",
    )
    try:
        receipt = patches.receipt
    except (TypeError, ValueError) as error:
        raise ValueError("Source review patch metadata must be JSON") from error
    _need(isinstance(receipt, dict), "Source review patch metadata must be an object")
    checksum = receipt.pop("binding_sha256", None)
    _need(
        receipt.get("schema") == PATCH_SCHEMA
        and type(receipt.get("version")) is int
        and receipt["version"] == 1
        and checksum == _digest(receipt),
        "Source review patch metadata changed",
    )
    _need(
        set(receipt)
        == {
            "schema",
            "version",
            "source_shape",
            "review_roi",
            "patch_side",
            "patch_rois_source_xyxy",
            "source_key",
            "geometry_token",
            "provenance",
            "coordinate_space",
            "patches_dtype",
            "patch_pixels_sha256",
            "full_source_authenticity_certified_by_this_module",
        }
        and all(
            isinstance(receipt.get(key), str)
            and re.fullmatch("[0-9a-f]{64}", receipt[key])
            for key in ("source_key", "geometry_token")
        )
        and receipt["coordinate_space"]
        == "current globally aligned oriented source pixels; no resize"
        and receipt["patches_dtype"] == "<f4"
        and receipt["full_source_authenticity_certified_by_this_module"] is False
        and isinstance(receipt["provenance"], dict)
        and bool(receipt["provenance"]),
        "Source review domain or caller provenance differs",
    )
    boxes = patch_rois(
        receipt["source_shape"], receipt["review_roi"], receipt["patch_side"]
    )
    _need(
        receipt["patch_rois_source_xyxy"] == boxes
        and len(patches.pixels) == len(boxes),
        "Source review patch geometry differs",
    )
    for image, expected in zip(
        patches.pixels, receipt["patch_pixels_sha256"], strict=True
    ):
        _need(
            isinstance(image, np.ndarray)
            and image.shape == (receipt["patch_side"], receipt["patch_side"], 3)
            and image.dtype == np.float32
            and image.flags.c_contiguous
            and not image.flags.writeable
            and np.isfinite(image).all()
            and _pixels(image) == expected,
            "Source review patch pixels/layout/read-only identity changed",
        )
    receipt["binding_sha256"] = checksum
    return receipt


def source_alignment_review(patches):
    """Review native residuals relative to green; at most18 local phase calls."""
    bound = _validate(patches)
    rois, side = bound["patch_rois_source_xyxy"], bound["patch_side"]
    reference = [_local_carrier(image[..., 1]) for image in patches.pixels]
    channels, phase_calls = {}, 0
    budget = source_residual_thresholds()
    for channel, index in (("red", 0), ("blue", 2)):
        observations, valid, centers = [], [], []
        nonfinite = False
        for position, (image, roi, ref) in enumerate(
            zip(patches.pixels, rois, reference, strict=True)
        ):
            green, green_texture = ref
            moving, texture = _local_carrier(image[..., index])
            item = {
                "grid_rc": list(divmod(position, 3)),
                "source_roi_xyxy": roi,
                "reference_texture": green_texture,
                "moving_texture": texture,
            }
            if green is None or moving is None:
                item["status"] = "unusable-texture-or-ambiguous"
            else:
                phase_calls += 1
                assert phase_calls <= 18
                (dx, dy), response = negpy_core.phase_correlate(green, moving)
                dx, dy, response = float(dx), float(dy), float(response)
                if not np.isfinite([dx, dy, response]).all():
                    nonfinite = True
                    item.update(
                        status="nonfinite-phase-evidence",
                        residual_source_xy=None,
                        response=None,
                    )
                elif response < LOCAL_RESPONSE_MIN:
                    item.update(
                        status="unusable-local-response",
                        residual_source_xy=[dx, dy],
                        response=response,
                    )
                else:
                    norm = float(np.hypot(dx, dy))
                    valid.append(norm)
                    centers.append([(roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2])
                    item.update(
                        status="valid",
                        residual_source_xy=[dx, dy],
                        residual_norm_source_pixels=norm,
                        response=response,
                    )
            observations.append(item)
        span = (
            [
                float(
                    np.ptp(np.asarray(centers)[:, a])
                    / (bound["review_roi"][a + 2] - bound["review_roi"][a])
                )
                for a in (0, 1)
            ]
            if centers
            else [0.0, 0.0]
        )
        stats = (
            {
                "p90": float(np.quantile(valid, 0.9)),
                "median": float(np.median(valid)),
                "maximum": max(valid),
            }
            if valid
            else None
        )
        if nonfinite or len(valid) < MIN_VALID_PATCHES or min(span) < MIN_SPAN_FRACTION:
            status, reason = (
                "inconclusive",
                "insufficient-distributed-finite-local-evidence",
            )
        elif (
            stats["p90"] > budget["max_residual_p90_source_pixels"]
            or stats["maximum"] > budget["max_residual_max_source_pixels"]
        ):
            status, reason = (
                "outside-budget",
                "native-local-displacements-exceed-source-budget",
            )
        else:
            status, reason = (
                "within-budget",
                "supported-native-local-displacements-within-source-budget",
            )
        channels[channel] = {
            "status": status,
            "reason": reason,
            "valid_patch_count": len(valid),
            "coverage_span_fraction_xy": span,
            "residual_source_pixels": stats,
            "patches": observations,
        }
    _validate(patches)
    status = (
        "outside-budget"
        if any(v["status"] == "outside-budget" for v in channels.values())
        else "inconclusive"
        if any(v["status"] == "inconclusive" for v in channels.values())
        else "within-budget"
    )
    receipt = {
        "schema": SCHEMA,
        "version": 1,
        "status": status,
        "advisory_only": True,
        "input_binding_sha256": bound["binding_sha256"],
        "source": bound,
        "source_pixel_budgets": budget,
        "reference_channel": "green",
        "channels": channels,
        "local_response_min": LOCAL_RESPONSE_MIN,
        "minimum_valid_patches": MIN_VALID_PATCHES,
        "minimum_span_fraction_each_axis": MIN_SPAN_FRACTION,
        "native_patch_side": side,
        "phase_calls": phase_calls,
        "estimator_resize_performed": False,
        "coordinate_units": "current oriented source pixels; each carrier pixel corresponds to one source pixel",
        "global_registration_or_shift_changed": False,
        "new_RAW_registration_proof_generated": False,
        "absolute_alignment_accuracy_certified": False,
        "rotation_or_warp_corrected": False,
        "source_or_recipe_defaults_changed": False,
        "limitation": "Cross-band spectral texture, noise, interpolation and local phase estimation affect residuals; finite patch evidence is neither full-frame ground truth nor an automatic registration correction",
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt
