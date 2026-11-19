"""Explicit small-motion feature proposal; never a production RAW fallback.

Matrices map moving source-pixel centres to reference source-pixel centres.
The model is a near-unit-scale similarity, not arbitrary affine/optical flow.
No image is warped and no successful estimate certifies all image locations.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import cv2
import numpy as np

PROVIDER = "sift-near-rigid-source-proposal-v2"
POLICY_VERSION = 2


@dataclass(frozen=True)
class RigidProbePolicy:
    maximum_width: int = 1600
    maximum_features: int = 1500
    ratio_threshold: float = 0.70
    minimum_relative_span: float = 0.01
    minimum_matches: int = 48
    minimum_validation_matches: int = 12
    minimum_distinctive_fraction: float = 0.20
    minimum_training_inliers: int = 24
    minimum_inlier_fraction: float = 0.70
    minimum_axis_span_fraction: float = 0.50
    minimum_occupied_sectors: int = 6
    ransac_source_threshold: float = 0.75
    maximum_rotation_degrees: float = 0.5
    maximum_scale_error: float = 0.002
    maximum_translation_source_pixels: float = 16.0
    maximum_validation_p95_source_pixels: float = 0.45
    good_validation_source_pixels: float = 0.5
    minimum_good_validation_fraction: float = 0.95
    all_match_outlier_source_pixels: float = 1.0
    maximum_all_bad_fraction: float = 0.02
    maximum_all_p95_source_pixels: float = 0.75


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _pixels(value):
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _policy(value):
    if not isinstance(value, RigidProbePolicy):
        raise ValueError("Explicit RigidProbePolicy required")
    integer_ranges = {
        "maximum_width": (128, 2048),
        "maximum_features": (100, 3000),
        "minimum_matches": (16, 1000),
        "minimum_validation_matches": (4, 500),
        "minimum_training_inliers": (8, 1000),
        "minimum_occupied_sectors": (4, 9),
    }
    for k, (lo, hi) in integer_ranges.items():
        x = getattr(value, k)
        if type(x) is not int or not lo <= x <= hi:
            raise ValueError("Invalid integer feature/support budget")
    for k, x in asdict(value).items():
        if k in integer_ranges:
            continue
        if type(x) not in (int, float):
            raise ValueError("Finite positive numeric policy required")
        try:
            valid = math.isfinite(float(x)) and x > 0
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError("Finite positive numeric policy required")
    for k in (
        "ratio_threshold",
        "minimum_distinctive_fraction",
        "minimum_inlier_fraction",
        "minimum_axis_span_fraction",
        "maximum_all_bad_fraction",
        "minimum_good_validation_fraction",
    ):
        if getattr(value, k) >= 1:
            raise ValueError("Feature/support fractions require(0,1)")
    if value.minimum_validation_matches >= value.minimum_matches:
        raise ValueError("Validation support must leave training matches")
    return value


def _plane(value):
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or value.ndim != 2
        or min(value.shape, default=0) < 64
        or not value.flags.c_contiguous
        or value.flags.writeable
        or not np.isfinite(value).all()
    ):
        raise ValueError(
            "Require finite readonly C-order float32 selected plane at least64x64"
        )


def _features(plane, policy):
    low, high = np.quantile(plane, [0.005, 0.995])
    relative = float((high - low) / max(abs(float(np.median(plane))), 1e-12))
    summary = {
        "relative_99percent_span": relative,
        "extraction_calls": 0,
        "source_shape": list(plane.shape),
    }
    if high <= low or relative < policy.minimum_relative_span:
        return None, None, {**summary, "reason": "insufficient-observable-contrast"}
    carrier = np.rint(
        np.clip((plane.astype(np.float64) - low) / (high - low), 0, 1) * 255
    ).astype(np.uint8)
    h, w = plane.shape
    ow, oh = (
        (policy.maximum_width, max(2, round(h * policy.maximum_width / w)))
        if w > policy.maximum_width
        else (w, h)
    )
    if (ow, oh) != (w, h):
        carrier = cv2.resize(carrier, (ow, oh), interpolation=cv2.INTER_AREA)
    detector = cv2.SIFT_create(
        nfeatures=policy.maximum_features,
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
    )
    keypoints, descriptors = detector.detectAndCompute(carrier, None)
    sx, sy = w / ow, h / oh
    points = np.asarray(
        [
            [(kp.pt[0] + 0.5) * sx - 0.5, (kp.pt[1] + 0.5) * sy - 0.5]
            for kp in keypoints
        ],
        dtype=np.float64,
    ).reshape(-1, 2)
    summary.update(
        extraction_calls=1,
        keypoint_count=len(points),
        estimator_shape=[oh, ow],
        source_scale_xy=[sx, sy],
        carrier="per-plane q.005..995linear→uint8; source pixels unchanged; estimator-only area resize",
        keypoint_coordinates="(carrier+.5)*actual_axis_scale-.5 source-pixel centres",
    )
    return points, descriptors, summary


def _coverage(points, shape):
    h, w = shape
    if not len(points):
        return {"axis_span_fraction_xy": [0.0, 0.0], "occupied_3x3_sectors": 0}
    sectors = np.clip(np.floor(points / np.array([w, h]) * 3).astype(int), 0, 2)
    return {
        "axis_span_fraction_xy": (np.ptp(points, axis=0) / np.array([w, h])).tolist(),
        "occupied_3x3_sectors": len(set(map(tuple, sectors.tolist()))),
    }


def _fit(moving, reference, shape, policy):
    """One RANSAC with a deterministic pre-fit validation partition."""
    n = len(moving)
    diagnostic = {
        "match_count": n,
        "ransac_calls": 0,
        "status": "inconclusive",
        "candidate_moving_to_reference": None,
    }
    if n < policy.minimum_matches:
        return {**diagnostic, "reason": "insufficient-distinctive-matches"}
    order = np.lexsort((moving[:, 0], moving[:, 1]))
    validation_idx, train_idx = order[::4], np.delete(order, np.arange(0, n, 4))
    if len(validation_idx) < policy.minimum_validation_matches:
        return {**diagnostic, "reason": "insufficient-pre-fit-validation-matches"}
    cv2.setRNGSeed(250116)
    matrix, mask = cv2.estimateAffinePartial2D(
        moving[train_idx],
        reference[train_idx],
        method=cv2.RANSAC,
        ransacReprojThreshold=policy.ransac_source_threshold,
        maxIters=3000,
        confidence=0.999,
        refineIters=10,
    )
    diagnostic["ransac_calls"] = 1
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return {**diagnostic, "reason": "RANSAC-no-finite-similarity"}
    inliers = mask.ravel().astype(bool)
    a, b = matrix[0, 0], matrix[1, 0]
    scale, angle = float(np.hypot(a, b)), float(np.degrees(np.arctan2(b, a)))
    predicted = moving @ matrix[:, :2].T + matrix[:, 2]
    residual = np.linalg.norm(predicted - reference, axis=1)
    held = residual[validation_idx]
    good_held = held <= policy.good_validation_source_pixels
    train_cov = _coverage(reference[train_idx][inliers], shape)
    held_cov = _coverage(reference[validation_idx][good_held], shape)
    failures = []
    if abs(angle) > policy.maximum_rotation_degrees:
        failures.append("rotation-outside-small-motion-policy")
    if abs(scale - 1) > policy.maximum_scale_error:
        failures.append("non-unit-scale-outside-policy")
    if np.max(np.abs(matrix[:, 2])) > policy.maximum_translation_source_pixels:
        failures.append("translation-outside-policy")
    if (
        int(inliers.sum()) < policy.minimum_training_inliers
        or inliers.mean() < policy.minimum_inlier_fraction
    ):
        failures.append("insufficient-training-inliers")
    if (
        min(train_cov["axis_span_fraction_xy"]) < policy.minimum_axis_span_fraction
        or train_cov["occupied_3x3_sectors"] < policy.minimum_occupied_sectors
    ):
        failures.append("insufficient-two-axis-spatial-training-support")
    if (
        min(held_cov["axis_span_fraction_xy"]) < policy.minimum_axis_span_fraction
        or held_cov["occupied_3x3_sectors"] < policy.minimum_occupied_sectors
    ):
        failures.append("insufficient-good-spatial-validation-support")
    if (
        np.quantile(held, 0.95) > policy.maximum_validation_p95_source_pixels
        or good_held.mean() < policy.minimum_good_validation_fraction
    ):
        failures.append("pre-fit-validation-residual-too-large")
    bad_fraction = float(np.mean(residual > policy.all_match_outlier_source_pixels))
    if (
        bad_fraction > policy.maximum_all_bad_fraction
        or np.quantile(residual, 0.95) > policy.maximum_all_p95_source_pixels
    ):
        failures.append("all-distinctive-match-nonrigid-or-mismatch-evidence")
    diagnostic.update(
        status="rejected" if failures else "proposed",
        reason=failures,
        candidate_moving_to_reference=matrix.tolist(),
        scale=scale,
        rotation_degrees=angle,
        translation_source_xy=matrix[:, 2].tolist(),
        training_count=len(train_idx),
        validation_count=len(validation_idx),
        validation_indices_before_fit=validation_idx.tolist(),
        training_inliers=int(inliers.sum()),
        training_inlier_fraction=float(inliers.mean()),
        training_coverage=train_cov,
        validation_coverage=held_cov,
        validation_coverage_uses_good_matches_only=True,
        good_validation_count=int(good_held.sum()),
        good_validation_fraction=float(good_held.mean()),
        good_validation_source_pixel_threshold=policy.good_validation_source_pixels,
        validation_residual_p95_source_pixels=float(np.quantile(held, 0.95)),
        validation_residual_max_source_pixels=float(held.max()),
        validation_max_is_diagnostic_only=True,
        all_match_residual_p95_source_pixels=float(np.quantile(residual, 0.95)),
        all_match_bad_fraction=bad_fraction,
        model="similarity constrained near-unit scale; no arbitrary affine or optical flow",
        heldout_match_gate_not_independent_image_holdout=True,
    )
    return diagnostic


def _pair(ref, moving, shape, policy):
    rp, rd, ri = ref
    mp, md, mi = moving
    base = {
        "status": "inconclusive",
        "ransac_calls": 0,
        "candidate_moving_to_reference": None,
    }
    if rd is None or md is None or min(len(rp), len(mp)) < 2:
        return {**base, "reason": "weak-or-insufficient-feature-evidence"}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(md, rd, k=2)
    reverse = matcher.knnMatch(rd, md, k=2)

    def qualified(rows):
        return {
            pair[0].queryIdx: pair[0]
            for pair in rows
            if len(pair) == 2
            and pair[0].distance < policy.ratio_threshold * pair[1].distance
        }

    f, r = qualified(forward), qualified(reverse)
    matches = [
        m for q, m in f.items() if m.trainIdx in r and r[m.trainIdx].trainIdx == q
    ]
    # Multiple SIFT orientations at one location must not count as independent support.
    matches.sort(key=lambda m: (m.distance, m.queryIdx, m.trainIdx))
    seen_m, seen_r, pairs = set(), set(), []
    for match in matches:
        qm, qr = (
            tuple(np.rint(mp[match.queryIdx] / 2).astype(int)),
            tuple(np.rint(rp[match.trainIdx] / 2).astype(int)),
        )
        if qm in seen_m or qr in seen_r:
            continue
        seen_m.add(qm)
        seen_r.add(qr)
        pairs.append((mp[match.queryIdx], rp[match.trainIdx]))
    fraction = len(pairs) / max(1, min(len(rp), len(mp)))
    if fraction < policy.minimum_distinctive_fraction:
        return {
            **base,
            "reason": "repetitive-or-low-distinctive-fraction",
            "mutual_unique_matches": len(pairs),
            "distinctive_fraction": fraction,
        }
    arrays = np.asarray(pairs, dtype=np.float64).reshape(-1, 2, 2)
    result = _fit(arrays[:, 0], arrays[:, 1], shape, policy)
    result.update(mutual_unique_matches=len(pairs), distinctive_fraction=fraction)
    return result


def propose_rigid_registration(red, green, blue, *, policy=RigidProbePolicy()):
    policy = _policy(policy)
    planes = [red, green, blue]
    for plane in planes:
        _plane(plane)
    if any(p.shape != green.shape for p in planes):
        raise ValueError("Matching source-plane geometry required")
    before = [_pixels(p) for p in planes]
    features = [_features(p, policy) for p in planes]
    channels = {
        "red": _pair(features[1], features[0], green.shape, policy),
        "blue": _pair(features[1], features[2], green.shape, policy),
    }
    for p, sha in zip(planes, before):
        _plane(p)
        if _pixels(p) != sha:
            raise ValueError("Source plane changed during feature proposal")
    status = (
        "proposed"
        if all(c["status"] == "proposed" for c in channels.values())
        else "rejected"
        if any(c["status"] == "rejected" for c in channels.values())
        else "inconclusive"
    )
    return {
        "schema": PROVIDER,
        "version": POLICY_VERSION,
        "status": status,
        "channels": channels,
        "policy": asdict(policy),
        "policy_sha256": _digest(asdict(policy)),
        "shape": list(green.shape),
        "source_plane_pixels_sha256": dict(zip(("red", "green", "blue"), before)),
        "feature_evidence": dict(
            zip(("red", "green", "blue"), [x[2] for x in features])
        ),
        "SIFT_extractions": sum(x[2]["extraction_calls"] for x in features),
        "RANSAC_calls": sum(c["ransac_calls"] for c in channels.values()),
        "coordinate_space": "moving→green source-pixel centres; no EXIF/RAW geometry change",
        "inlier_count_is_not_sufficient_acceptance": True,
        "no_translation_fallback": True,
        "source_values_changed": False,
        "image_warp_performed": False,
        "production_defaults_changed": False,
        "heuristic_not_world_geometry_certificate": True,
        "opencv_version": cv2.__version__,
    }
