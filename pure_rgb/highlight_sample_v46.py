"""Exact source-coordinate protection and Native highlight point composition."""

from dataclasses import dataclass
import hashlib
import json
import math

import cv2
import numpy as np

from .highlight_protection import _source_key
from .metering_masks import MAX_SHAPES, _geometry, _raster, _shape


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _bounds(shape, full):
    if shape["type"] == "polygon":
        xx, yy = zip(*shape["points"])
        box = min(xx), min(yy), max(xx), max(yy)
    else:
        box = shape["bounds"]
    return (max(0, math.floor(box[0]) - 1), max(0, math.floor(box[1]) - 1),
            min(full[1], math.ceil(box[2]) + 1), min(full[0], math.ceil(box[3]) + 1))


@dataclass(frozen=True)
class SampledHighlightProtection:
    mask: np.ndarray
    canonical_receipt: str

    @property
    def receipt(self):
        return json.loads(self.canonical_receipt)


def sample_highlight_protection(source_shape, source_key, *, include,
                                feather_pixels=64.0, dtype="float64",
                                rectangle=None, sample_step=1,
                                automatic_feather_pixels=0.):
    """Sample the original outward EDT feather, never a resized vector mask.

    Only source pixels within feather distance of the requested rectangle can
    affect its weights. The original precise float32 distance is sampled before
    the original float64 smoothstep, keeping camera-size float64 arrays out of
    the preview. A full-frame fit still needs its full geometric binary plane.
    """
    source_key = _source_key(source_key)
    if not isinstance(source_shape, (list, tuple)) or len(source_shape) != 3:
        raise ValueError("Protection sampling requires complete HWC source geometry")
    full, _ = _geometry(source_shape, [0, 0, source_shape[1], source_shape[0]])
    _, roi = _geometry(full, [0, 0, full[1], full[0]] if rectangle is None else rectangle)
    try:
        feather = float(feather_pixels)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Protection feather must be finite original-source pixels") from error
    if (type(sample_step) is not int or sample_step < 1 or dtype not in ("float32", "float64")
            or type(feather_pixels) not in (int, float) or not math.isfinite(feather)
            or not 0 <= feather <= 512):
        raise ValueError("Protection sampling needs a positive integer step and source-pixel feather in [0,512]")
    feather_pixels = feather
    if type(automatic_feather_pixels) not in (int, float):
        raise ValueError("Automatic feather must be finite original-source pixels in [0,256]")
    try:
        automatic_feather_pixels = float(automatic_feather_pixels)
    except OverflowError as error:
        raise ValueError("Automatic feather must be finite original-source pixels in [0,256]") from error
    if not math.isfinite(automatic_feather_pixels) or not 0 <= automatic_feather_pixels <= 256:
        raise ValueError("Automatic feather must be finite original-source pixels in [0,256]")
    effective_feather = feather_pixels + automatic_feather_pixels
    if not isinstance(include, (list, tuple)) or not 1 <= len(include) <= MAX_SHAPES:
        raise ValueError("Protection requires explicit nonempty source regions")
    shapes = [_shape(shape, full) for shape in include]
    x0, y0, x1, y1 = roi
    if ((x1-x0+sample_step-1)//sample_step * ((y1-y0+sample_step-1)//sample_step) > 4_194_304):
        raise ValueError("Sampled protection exceeds the preview pixel limit")
    # The full builder rejects a vector set with no actual source pixel centre.
    # Prove the same fact using bounded rows, even if the requested tile is far
    # away from the protection and therefore has an empty local raster.
    valid = False
    for shape in shapes:
        a, b, c, d = _bounds(shape, full)
        if a >= c or b >= d:
            continue
        for first in range(b, d, 32):
            if _raster(shape, (a, first, c, min(d, first + 32))).any():
                valid = True
                break
        if valid:
            break
    if not valid:
        raise ValueError("Protection selection contains no source pixel centres")
    halo = math.ceil(effective_feather) + 1 if effective_feather else 0
    a, b, c, d = max(0, x0-halo), max(0, y0-halo), min(full[1], x1+halo), min(full[0], y1+halo)
    selected = np.zeros((d-b, c-a), dtype=bool)
    for shape in shapes:
        sub = (a, b, c, d)
        if shape["type"] != "polygon":
            left, top, right, bottom = _bounds(shape, full)
            sub = max(a, left), max(b, top), min(c, right), min(d, bottom)
        if sub[0] < sub[2] and sub[1] < sub[3]:
            selected[sub[1]-b:sub[3]-b, sub[0]-a:sub[2]-a] |= _raster(shape, sub)
    sy, sx = slice(y0-b, y1-b, sample_step), slice(x0-a, x1-a, sample_step)
    inside = selected[sy, sx]
    mask = inside.astype(dtype)
    if effective_feather > 0 and selected.any() and not selected.all():
        distance = cv2.distanceTransform(np.ascontiguousarray(~selected, dtype=np.uint8),
                                         cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        t = np.minimum(distance[sy, sx].astype(np.float64) / effective_feather, 1.0)
        mask = (1.0 - t*t*(3.0 - 2.0*t)).astype(dtype)
        mask[inside] = 1.0
    mask = np.frombuffer(np.ascontiguousarray(mask).tobytes(), dtype=np.dtype(dtype)).reshape(mask.shape)
    receipt = dict(schema="pure-rgb-sampled-highlight-protection-v46", version=1,
                   source_shape=list(full), source_key=source_key, include=shapes,
                   feather_pixels=float(feather_pixels), source_rectangle=list(roi),
                   sample_step=sample_step, sampled_shape=list(mask.shape), dtype=dtype,
                   halo_rectangle=[a, b, c, d], geometric_raster_pixels=int(selected.size),
                   opencv_version=cv2.__version__,
                   mask_pixels_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                   protection_domain="original full-source pixel centres and precise outward feather",
                   source_pixels_read_or_modified=False,
                   statistics_scope="requested source grid; not full-source protection counts")
    if automatic_feather_pixels:
        receipt["automatic_feather_pixels"] = float(automatic_feather_pixels)
    receipt["sha256"] = hashlib.sha256(_canonical(receipt).encode()).hexdigest()
    return SampledHighlightProtection(mask, _canonical(receipt))


def compose_native_sample(encoded, *, icc, reference_encoded=None, highlight=None,
                          point=None, protection_mask=None, effect_mask=None):
    """Apply the unchanged full-save float ordering before the one quantizer."""
    from negpy.kernel.image.logic import working_oetf_encode
    from .native_float import checked_native_float
    from .highlight_reference_guarded import apply_highlight_reference_guarded
    from .positive_float import apply_positive_float, decode_working_oetf_pixels, positive_domain

    point = checked_native_float(point)
    if not highlight and not point["values"]:
        return encoded, None
    linear_domain = positive_domain("linear", icc)
    linear = decode_working_oetf_pixels(encoded, icc_bytes=icc)
    output, high_receipt = linear, None
    if highlight:
        if (reference_encoded is None
                or bool(highlight["protection"]) != (protection_mask is not None)
                or bool(highlight.get("effect_area")) != (effect_mask is not None)):
            raise ValueError("Highlight sample requires its real reference and declared source protection")
        reference = decode_working_oetf_pixels(reference_encoded, icc_bytes=icc)
        output, high_receipt = apply_highlight_reference_guarded(
            linear, reference, highlight["controls"], domain=linear_domain,
            protection_mask=protection_mask, effect_mask=effect_mask,
            recover_clipped=bool(highlight.get("reference_density_offset")))
    if point["values"]:
        output = apply_positive_float(output, point, domain=linear_domain, icc_bytes=icc).image
    result = working_oetf_encode(np.clip(output, 0, 1))
    if not point["values"]:
        unchanged = np.all(output == linear, axis=2)
        result[unchanged] = encoded[unchanged]
    return np.ascontiguousarray(result, dtype=np.float32), dict(
        highlight=high_receipt, native_float_controls=point,
        statistics_scope="requested source grid only; reference meters use complete original support",
        working_OETF_encoded=True, integer_quantization_performed=False)
