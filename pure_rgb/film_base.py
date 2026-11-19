"""Explicit full-source film-rebate candidates; no physical calibration claim.

Visual classification is supplied by the researcher, never inferred from a
bright/flat patch. The unchanged NegPy adapter sampler is reused only for its
signal/MAD/saturation checks. A uniform holder or open path is not film base.
"""

import math

import numpy as np

from .engines import sample_film_base

VISUAL_CLASSES = (
    "clean-film-rebate-candidate",
    "printed-film-edge",
    "holder",
    "open-light-path",
    "flare",
    "image-area",
    "unknown",
)


def _patch(source, roi):
    x = np.asarray(source)
    if x.ndim != 3 or x.shape[2] != 3 or x.dtype.kind != "f" or not x.size:
        raise ValueError("Film-base study requires floating full-source HWC RGB")
    h, w = x.shape[:2]
    if (
        not isinstance(roi, (list, tuple))
        or len(roi) != 4
        or any(type(v) is not int for v in roi)
        or not 0 <= roi[0] < roi[2] <= w
        or not 0 <= roi[1] < roi[3] <= h
        or (roi[2] - roi[0]) * (roi[3] - roi[1]) < 64
    ):
        raise ValueError(
            "Candidate ROI requires >=64 full-source pixels, half-open xyxy"
        )
    patch = x[roi[1] : roi[3], roi[0] : roi[2]]
    if not np.isfinite(patch).all():
        raise ValueError("Candidate samples must be finite")
    return patch


def measure_candidate(source, roi, *, candidate_id, visual_class, visual_note):
    """Measure one manually classified patch without altering the full source."""
    patch = _patch(source, roi)
    if visual_class not in VISUAL_CLASSES:
        raise ValueError("An explicit supported visual classification is required")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("Candidate id required")
    if not isinstance(visual_note, str) or not visual_note:
        raise ValueError("Record the visual evidence, not only brightness")
    samples = patch.astype(np.float64)
    median = np.median(samples, axis=(0, 1))
    mad = np.median(np.abs(samples - median), axis=(0, 1)) / np.maximum(median, 1e-6)
    # This includes demosaicing, grain and fine texture; not independent sensor
    # noise or a calibrated SNR estimate. Neighbor differences reduce gradient.
    differences = np.diff(samples, axis=1 if samples.shape[1] > 1 else 0).reshape(-1, 3)
    noise = np.median(np.abs(differences - np.median(differences, axis=0)), axis=0)
    noise /= 0.6744897501960817 * math.sqrt(2)
    tiles = []
    for ys in np.array_split(np.arange(samples.shape[0]), 3):
        for xs in np.array_split(np.arange(samples.shape[1]), 3):
            if len(ys) and len(xs):
                tiles.append(np.median(samples[ys[:, None], xs], axis=(0, 1)))
    tile_values = np.asarray(tiles)
    record = {
        "id": candidate_id,
        "roi": list(roi),
        "coordinate_space": "EXIF-oriented full merged linear source, half-open xyxy; not preview coordinates",
        "visual_class": visual_class,
        "visual_note": visual_note,
        "median_rgb": median.tolist(),
        "relative_mad_rgb": mad.tolist(),
        "source_scale_near_0_98_fraction_rgb": (samples >= 0.98).mean((0, 1)).tolist(),
        "negative_sample_fraction_rgb": (samples < 0).mean((0, 1)).tolist(),
        "nonpositive_sample_fraction_rgb": (samples <= 0).mean((0, 1)).tolist(),
        "neighbor_difference_noise_proxy_rgb": noise.tolist(),
        "neighbor_difference_relative_noise_proxy_rgb": (
            noise / np.maximum(median, 1e-6)
        ).tolist(),
        "noise_scope": "Correlated demosaiced camera signal plus film grain/texture; not calibrated sensor noise or SNR",
        "tile_median_relative_range_rgb": (
            (tile_values.max(0) - tile_values.min(0)) / np.maximum(median, 1e-6)
        ).tolist(),
        "sample_count": int(samples.shape[0] * samples.shape[1]),
        "physical_film_base_certified": False,
        "scanner_density_measured": False,
    }
    try:
        local = sample_film_base(
            np.maximum(patch, 0), (0, 0, patch.shape[1], patch.shape[0])
        )
        record.update(
            legacy_sampler_passed=True,
            sampler_values=list(local.values),
            sampler_relative_mad=list(local.relative_mad),
            sampler_scope="Unchanged engines.sample_film_base on this patch, with ROI translated back to full source",
        )
    except ValueError as error:
        record.update(legacy_sampler_passed=False, sampler_rejection=str(error))
    record["eligible_provisional_candidate"] = (
        visual_class == "clean-film-rebate-candidate"
        and record["legacy_sampler_passed"]
        and bool(np.all(median > 1e-6))
    )
    return record


def candidate_consistency(candidates):
    """Spatial agreement is a diagnostic, not a certificate or a flat-field fix."""
    eligible = [c for c in candidates if c["eligible_provisional_candidate"]]
    if not eligible:
        return {
            "state": "unknown",
            "eligible_ids": [],
            "reason": "No visually and statistically eligible film rebate",
        }
    values = np.asarray([c["sampler_values"] for c in eligible], dtype=float)
    median = np.median(values, axis=0)
    return {
        "state": "provisional-spatial-comparison",
        "eligible_ids": [c["id"] for c in eligible],
        "patch_median_rgb_median": median.tolist(),
        "patch_median_minimum_rgb": values.min(0).tolist(),
        "patch_median_maximum_rgb": values.max(0).tolist(),
        "patch_median_relative_range_rgb": (
            (values.max(0) - values.min(0)) / median
        ).tolist(),
        "patch_median_relative_mad_rgb": (
            np.median(np.abs(values - median), axis=0) / median
        ).tolist(),
        "physical_uniformity_verified": False,
        "interpretation": "Spatial illumination, film variation and capture differences remain inseparable without calibration",
    }


def select_provisional_reference(candidates, candidate_id):
    """Keep an explicit selected patch and its global ROI, never choose max light."""
    selected = [c for c in candidates if c["id"] == candidate_id]
    if len(selected) != 1 or not selected[0]["eligible_provisional_candidate"]:
        raise ValueError("Selected patch must be one unique clean provisional rebate")
    c = selected[0]
    return {
        "values": c["sampler_values"],
        "roi": c["roi"],
        "relative_mad": c["sampler_relative_mad"],
        "candidate_id": candidate_id,
        "provenance": "manually inspected same-capture clean film-rebate candidate; camera linear signal only; provisional, physical calibration unverified",
    }
