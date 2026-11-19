"""Opt-in, bounded local evidence for the translation-only RAW motion model.

This check changes no global estimate, warp or pixel. Gradient magnitudes make
local evidence independent of affine intensity scale/offset (including sign).
Insufficient or repetitive evidence is inconclusive, never a geometry proof.
"""

from copy import deepcopy
import hashlib
import math
import re

import negpy_core
import numpy as np

POLICY = "translation-consistency-v1"
SOURCE_POLICY = "translation-consistency-source-v2"
MAX_RESIDUAL_P90_SOURCE_PIXELS = 0.25
MAX_RESIDUAL_MAX_SOURCE_PIXELS = 0.5
SOURCE_RESIDUAL_UNITS = "standard-crop-local source pixels before EXIF orientation"
SOURCE_RESIDUAL_FORMULA = (
    "hypot(residual_est_x*source_scale_x,residual_est_y*source_scale_y)"
)
LOCAL_RESPONSE_MIN = 0.15
LOCAL_TEXTURE_EIGEN_MIN = 1e-3
MIN_VALID_PATCHES = 5
MIN_SPAN_FRACTION = 0.5
MAX_RESIDUAL_P90_EST_PIXELS = 0.8
MAX_RESIDUAL_MEDIAN_EST_PIXELS = 0.5
MAX_RESIDUAL_MAX_EST_PIXELS = 1.5
PATCH_FRACTION = 0.4
MIN_PATCH_SIDE = 48


def _thresholds():
    return {
        "local_response_min": LOCAL_RESPONSE_MIN,
        "local_texture_eigen_min": LOCAL_TEXTURE_EIGEN_MIN,
        "min_valid_patches": MIN_VALID_PATCHES,
        "min_span_fraction_each_axis": MIN_SPAN_FRACTION,
        "max_residual_p90_est_pixels": MAX_RESIDUAL_P90_EST_PIXELS,
        "max_residual_median_est_pixels": MAX_RESIDUAL_MEDIAN_EST_PIXELS,
        "max_residual_max_est_pixels": MAX_RESIDUAL_MAX_EST_PIXELS,
        "min_patch_side": MIN_PATCH_SIDE,
        "patch_fraction": PATCH_FRACTION,
        "periodicity_correlation_min": 0.95,
        "periodicity_prominence_min": 0.1,
    }


class RegistrationModelCheckError(ValueError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__(
            f"{diagnostic.get('channel', 'registration')}: "
            f"translation model {diagnostic['status']}: {diagnostic['reason']}"
        )


def pixel_sha(plane):
    return hashlib.sha256(memoryview(np.ascontiguousarray(plane)).cast("B")).hexdigest()


def _gradient_magnitude(plane):
    a = plane.astype(np.float64)
    dx = (a[1:-1, 2:] - a[1:-1, :-2]) * 0.5
    dy = (a[2:, 1:-1] - a[:-2, 1:-1]) * 0.5
    return np.hypot(dx, dy)


def _periodicity(carrier):
    """Strong, locally prominent same-patch repetition; no global response use."""
    peaks = []
    for axis in (0, 1):
        values = []
        for lag in range(4, min(48, carrier.shape[axis] // 3) + 1):
            a = np.take(carrier, range(carrier.shape[axis] - lag), axis=axis)
            b = np.take(carrier, range(lag, carrier.shape[axis]), axis=axis)
            denominator = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
            values.append(float(np.sum(a * b)) / max(denominator, np.finfo(float).tiny))
        for i in range(2, len(values) - 2):
            prominence = values[i] - max(values[i - 2], values[i + 2])
            if values[i] > 0.95 and prominence > 0.1:
                peaks.append(
                    {
                        "axis": axis,
                        "lag": i + 4,
                        "correlation": values[i],
                        "prominence": prominence,
                    }
                )
    return peaks


def _local_carrier(patch):
    gradient = _gradient_magnitude(patch)
    mean = float(np.mean(gradient))
    centered = gradient - mean
    variance = float(np.mean(centered * centered))
    if variance <= np.finfo(np.float64).tiny:
        return None, {"status": "insufficient-texture", "variance": variance}
    gx, gy = np.diff(centered, axis=1)[:-1], np.diff(centered, axis=0)[:, :-1]
    eig = np.linalg.eigvalsh(
        [
            [float(np.mean(gx * gx)), float(np.mean(gx * gy))],
            [float(np.mean(gx * gy)), float(np.mean(gy * gy))],
        ]
    )
    ratio = max(0.0, float(eig[0])) / max(float(eig[1]), np.finfo(float).tiny)
    diagnostic = {
        "gradient_variance": variance,
        "gradient_structure_eigen_ratio": ratio,
    }
    if ratio < LOCAL_TEXTURE_EIGEN_MIN:
        return None, {**diagnostic, "status": "insufficient-two-axis-texture"}
    carrier = centered / math.sqrt(variance)
    repeated = _periodicity(carrier)
    if repeated:
        return None, {
            **diagnostic,
            "status": "ambiguous-repetition",
            "periodic_peaks": repeated,
        }
    return np.ascontiguousarray(carrier, dtype=np.float32), diagnostic


def check_translation_consistency(reference, moving, shift_est_xy, scale_xy, channel):
    """Return passed evidence or raise with complete failed/inconclusive evidence."""
    h, w = reference.shape
    dx, dy = map(float, shift_est_xy)
    ix, iy = round(dx), round(dy)
    left, top, right, bottom = max(0, -ix), max(0, -iy), min(w, w - ix), min(h, h - iy)
    aw, ah = right - left, bottom - top
    pw, ph = (
        min(aw, max(MIN_PATCH_SIDE, int(aw * PATCH_FRACTION))),
        min(ah, max(MIN_PATCH_SIDE, int(ah * PATCH_FRACTION))),
    )
    report = {
        "schema": "pure-rgb-translation-consistency",
        "version": 1,
        "policy": POLICY,
        "channel": channel,
        "status": "inconclusive",
        "reason": "insufficient-local-support",
        "estimator_shape_h_w": [h, w],
        "estimator_sha256": {
            "reference": pixel_sha(reference),
            "moving": pixel_sha(moving),
        },
        "global_shift_est_xy": [dx, dy],
        "source_scale_xy": list(map(float, scale_xy)),
        "carrier": "central-gradient-magnitude; independently mean/std normalized per patch",
        "thresholds_are_heuristics": True,
        "thresholds": _thresholds(),
        "patches": [],
        "valid_patch_count": 0,
        "coverage_span_fraction_xy": [0.0, 0.0],
        "residual_p90_est_pixels": None,
        "residual_median_est_pixels": None,
        "residual_max_est_pixels": None,
        "original_pixels_modified": False,
        "scope": "Finite local translation evidence, not an affine/nonlinear fit or a general geometry certificate",
    }
    if min(aw, ah) < MIN_PATCH_SIDE * 2:
        raise RegistrationModelCheckError(report)
    xs = [left, left + (aw - pw) // 2, right - pw]
    ys = [top, top + (ah - ph) // 2, bottom - ph]
    accepted = []
    for row, y in enumerate(ys):
        for column, x in enumerate(xs):
            patch = {
                "grid_rc": [row, column],
                "reference_roi_est_xyxy": [x, y, x + pw, y + ph],
                "moving_roi_est_xyxy": [x + ix, y + iy, x + ix + pw, y + iy + ph],
            }
            a, da = _local_carrier(reference[y : y + ph, x : x + pw])
            b, db = _local_carrier(moving[y + iy : y + iy + ph, x + ix : x + ix + pw])
            patch.update(reference_texture=da, moving_texture=db)
            if a is None or b is None:
                patch["status"] = "unusable-texture-or-ambiguous"
            else:
                (sx, sy), response = negpy_core.phase_correlate(a, b)
                sx, sy, response = float(sx), float(sy), float(response)
                if not all(math.isfinite(v) for v in (sx, sy, response)):
                    patch.update(
                        status="nonfinite-local-estimate",
                        local_shift_est_xy=None,
                        response=None,
                    )
                    report["patches"].append(patch)
                    report["reason"] = "nonfinite-local-estimate"
                    raise RegistrationModelCheckError(report)
                patch.update(local_shift_est_xy=[ix + sx, iy + sy], response=response)
                if response < LOCAL_RESPONSE_MIN:
                    patch["status"] = "unusable-local-response"
                else:
                    residual = [ix + sx - dx, iy + sy - dy]
                    patch.update(
                        status="valid",
                        residual_est_xy=residual,
                        residual_norm_est_pixels=math.hypot(*residual),
                    )
                    accepted.append(patch)
            report["patches"].append(patch)
    report["valid_patch_count"] = len(accepted)
    if len(accepted) < MIN_VALID_PATCHES:
        report["reason"] = "insufficient-nonrepetitive-local-constraints"
        raise RegistrationModelCheckError(report)
    centers = [
        (
            p["reference_roi_est_xyxy"][0] + pw / 2,
            p["reference_roi_est_xyxy"][1] + ph / 2,
        )
        for p in accepted
    ]
    span = [
        (max(v[axis] for v in centers) - min(v[axis] for v in centers)) / length
        for axis, length in ((0, aw), (1, ah))
    ]
    report["coverage_span_fraction_xy"] = span
    if min(span) < MIN_SPAN_FRACTION:
        report["reason"] = "local-constraints-do-not-cover-both-axes"
        raise RegistrationModelCheckError(report)
    residuals = [p["residual_norm_est_pixels"] for p in accepted]
    p90, median = float(np.quantile(residuals, 0.9)), float(np.median(residuals))
    maximum = max(residuals)
    report.update(
        residual_p90_est_pixels=p90,
        residual_median_est_pixels=median,
        residual_max_est_pixels=maximum,
    )
    if (
        p90 > MAX_RESIDUAL_P90_EST_PIXELS
        or median > MAX_RESIDUAL_MEDIAN_EST_PIXELS
        or maximum > MAX_RESIDUAL_MAX_EST_PIXELS
    ):
        report.update(
            status="rejected", reason="spatially-inconsistent-local-translations"
        )
        raise RegistrationModelCheckError(report)
    report.update(
        status="passed", reason="bounded-spatially-distributed-local-translations"
    )
    return report


def source_residual_thresholds():
    """Conservative engineering consistency budgets, never accuracy bounds.

    The90th-percentile local residual must be at most a quarter source pixel,
    reserving margin below the half-source-pixel worst-patch
    budget. These are fixed original-source units, not enlarged on resize.
    V24 stored evidence was used for development; held-out checks are separate.
    """
    return {
        "max_residual_p90_source_pixels": MAX_RESIDUAL_P90_SOURCE_PIXELS,
        "max_residual_max_source_pixels": MAX_RESIDUAL_MAX_SOURCE_PIXELS,
    }


def _source_evidence(evidence):
    """Add anisotropic source-coordinate diagnostics without another estimate."""
    result = deepcopy(evidence)
    scale = result["source_scale_xy"]
    residuals = []
    for patch in result["patches"]:
        if patch["status"] != "valid":
            continue
        residual = [a * b for a, b in zip(patch["residual_est_xy"], scale, strict=True)]
        norm = math.hypot(*residual)
        if not all(math.isfinite(value) for value in (*residual, norm)):
            raise ValueError("Source-coordinate registration residual is nonfinite")
        patch["residual_source_xy"] = residual
        patch["residual_norm_source_pixels"] = norm
        residuals.append(norm)
    result.update(
        version=2,
        policy=SOURCE_POLICY,
        estimator_gate_status=evidence["status"],
        estimator_gate_reason=evidence["reason"],
        source_thresholds=source_residual_thresholds(),
        source_residual_units=SOURCE_RESIDUAL_UNITS,
        source_residual_formula=SOURCE_RESIDUAL_FORMULA,
        source_budget_is_absolute_registration_accuracy=False,
        source_residual_p90_pixels=float(np.quantile(residuals, 0.9))
        if residuals
        else None,
        source_residual_median_pixels=float(np.median(residuals))
        if residuals
        else None,
        source_residual_max_pixels=max(residuals) if residuals else None,
        source_budget_status="not-evaluated-without-supported-estimator-proof",
        scope="Finite supported local translation consistency in original source pixels; not absolute alignment accuracy, rotation/nonlinear correction or a full-frame geometry certificate",
    )
    return result


def check_translation_consistency_source(
    reference, moving, shift_est_xy, scale_xy, channel
):
    """Opt-in source-unit gate using one unchanged V1 local evidence pass."""
    if (
        not isinstance(scale_xy, (list, tuple))
        or len(scale_xy) != 2
        or any(type(value) not in (int, float) for value in scale_xy)
    ):
        raise ValueError(
            "Source registration check requires finite actual resize scales>=1"
        )
    try:
        scales = [float(value) for value in scale_xy]
    except OverflowError as error:
        raise ValueError("Source registration resize scales overflow") from error
    if any(not math.isfinite(value) or value < 1 for value in scales):
        raise ValueError(
            "Source registration check requires finite actual resize scales>=1"
        )
    try:
        evidence = check_translation_consistency(
            reference, moving, shift_est_xy, scales, channel
        )
    except RegistrationModelCheckError as error:
        result = _source_evidence(error.diagnostic)
        if result["status"] == "rejected":
            result["source_budget_status"] = "upstream-estimator-consistency-rejected"
        raise RegistrationModelCheckError(result) from error
    result = _source_evidence(evidence)
    if (
        result["source_residual_p90_pixels"] > MAX_RESIDUAL_P90_SOURCE_PIXELS
        or result["source_residual_max_pixels"] > MAX_RESIDUAL_MAX_SOURCE_PIXELS
    ):
        result.update(
            status="rejected",
            reason="source-pixel-residual-budget-exceeded",
            source_budget_status="rejected",
        )
        raise RegistrationModelCheckError(result)
    result.update(
        status="passed",
        reason="bounded-source-pixel-local-translations",
        source_budget_status="passed",
    )
    return result


def _validate_registration_model_metadata(receipt):
    """Pure cross-field validation; v1 returns None without adding any proof.

    This validates a stored finite-evidence contract, not pixels or signatures.
    The containing import receipt/source identity must remain independently bound.
    """

    def need(condition):
        if not condition:
            raise ValueError(
                "Malformed or inconsistent RAW registration model evidence"
            )

    def number(value):
        need(type(value) in (int, float) and math.isfinite(value))
        return float(value)

    def vector(value, size, *, integer=False):
        need(isinstance(value, (list, tuple)) and len(value) == size)
        if integer:
            need(all(type(v) is int for v in value))
            return list(value)
        return [number(v) for v in value]

    def near(a, b):
        need(abs(number(a) - number(b)) <= 1e-9)

    def texture(value, *, require_usable=False):
        need(isinstance(value, dict))
        status = value.get("status")
        if require_usable:
            need(status is None)
        if status == "insufficient-texture":
            need(set(value) == {"status", "variance"})
            need(0 <= number(value["variance"]) <= np.finfo(np.float64).tiny)
            return False
        expected = {"gradient_variance", "gradient_structure_eigen_ratio"}
        if status is not None:
            need(status in ("insufficient-two-axis-texture", "ambiguous-repetition"))
            expected.add("status")
        if status == "ambiguous-repetition":
            expected.add("periodic_peaks")
        need(set(value) == expected)
        need(number(value["gradient_variance"]) > np.finfo(np.float64).tiny)
        eigen = number(value["gradient_structure_eigen_ratio"])
        need(0 <= eigen <= 1)
        if status == "insufficient-two-axis-texture":
            need(eigen < LOCAL_TEXTURE_EIGEN_MIN)
            return False
        need(eigen >= LOCAL_TEXTURE_EIGEN_MIN)
        peaks = value.get("periodic_peaks", [])
        need(
            isinstance(peaks, list)
            and (status != "ambiguous-repetition" or len(peaks) > 0)
        )
        for item in peaks:
            need(
                isinstance(item, dict)
                and type(item.get("axis")) is int
                and item["axis"] in (0, 1)
            )
            need(type(item.get("lag")) is int and item["lag"] >= 6)
            need(
                number(item.get("correlation")) > 0.95
                and number(item.get("prominence")) > 0.1
            )
        return status is None

    need(isinstance(receipt, dict) and type(receipt.get("version")) is int)
    if receipt["version"] == 1:
        need("model_check" not in receipt)
        return None
    need(
        receipt["version"] == 2
        and receipt.get("align") is True
        and receipt.get("status") == "registered"
    )
    model = receipt.get("model_check")
    need(
        isinstance(model, dict)
        and set(model)
        == {"schema", "version", "policy", "status", "source_plane_sha256", "channels"}
    )
    need(
        model["schema"] == "pure-rgb-registration-model-check"
        and type(model["version"]) is int
        and model["version"] == 1
        and model["policy"] == POLICY
        and model["status"] == "passed"
    )
    source_hashes = model["source_plane_sha256"]
    need(
        isinstance(source_hashes, dict)
        and set(source_hashes) == {"red", "green", "blue"}
    )
    need(
        all(
            isinstance(v, str) and re.fullmatch("[0-9a-f]{64}", v)
            for v in source_hashes.values()
        )
    )
    checks = model["channels"]
    need(isinstance(checks, dict) and set(checks) == {"red", "blue"})
    expected_shape = vector(receipt.get("phase_estimation_shape_h_w"), 2, integer=True)
    need(min(expected_shape) >= MIN_PATCH_SIDE * 2)
    expected_scales = vector(receipt.get("phase_estimation_to_source_scale_xy"), 2)
    need(min(expected_scales) >= 1)
    estimator_reference = None
    for channel in ("red", "blue"):
        check = checks[channel]
        need(isinstance(check, dict))
        need(
            check.get("schema") == "pure-rgb-translation-consistency"
            and type(check.get("version")) is int
            and check["version"] == 1
            and check.get("policy") == POLICY
            and check.get("channel") == channel
            and check.get("status") == "passed"
        )
        need(
            check.get("reason") == "bounded-spatially-distributed-local-translations"
            and check.get("original_pixels_modified") is False
            and check.get("thresholds_are_heuristics") is True
        )
        need(
            vector(check.get("estimator_shape_h_w"), 2, integer=True) == expected_shape
        )
        need(vector(check.get("source_scale_xy"), 2) == expected_scales)
        thresholds = check.get("thresholds")
        need(isinstance(thresholds, dict) and set(thresholds) == set(_thresholds()))
        need(all(number(thresholds[k]) == v for k, v in _thresholds().items()))
        for key in ("min_valid_patches", "min_patch_side"):
            need(type(thresholds[key]) is int)
        hashes = check.get("estimator_sha256")
        need(isinstance(hashes, dict) and set(hashes) == {"reference", "moving"})
        need(
            all(
                isinstance(v, str) and re.fullmatch("[0-9a-f]{64}", v)
                for v in hashes.values()
            )
        )
        if estimator_reference is not None:
            need(hashes["reference"] == estimator_reference)
        estimator_reference = hashes["reference"]
        shift = vector(check.get("global_shift_est_xy"), 2)
        channel_info = receipt.get("channels", {}).get(channel, {})
        actual_shift = vector(channel_info.get("shift_xy"), 2)
        for axis in (0, 1):
            near(shift[axis] * expected_scales[axis], actual_shift[axis])
        h, w = expected_shape
        ix, iy = map(round, shift)
        left, top, right, bottom = (
            max(0, -ix),
            max(0, -iy),
            min(w, w - ix),
            min(h, h - iy),
        )
        aw, ah = right - left, bottom - top
        pw, ph = (
            min(aw, max(MIN_PATCH_SIDE, int(aw * PATCH_FRACTION))),
            min(ah, max(MIN_PATCH_SIDE, int(ah * PATCH_FRACTION))),
        )
        xs, ys = (
            [left, left + (aw - pw) // 2, right - pw],
            [top, top + (ah - ph) // 2, bottom - ph],
        )
        patches = check.get("patches")
        need(isinstance(patches, list) and len(patches) == 9)
        accepted, centers = [], []
        for index, patch in enumerate(patches):
            need(isinstance(patch, dict))
            row, column = divmod(index, 3)
            need(vector(patch.get("grid_rc"), 2, integer=True) == [row, column])
            x, y = xs[column], ys[row]
            need(
                vector(patch.get("reference_roi_est_xyxy"), 4, integer=True)
                == [x, y, x + pw, y + ph]
            )
            need(
                vector(patch.get("moving_roi_est_xyxy"), 4, integer=True)
                == [x + ix, y + iy, x + ix + pw, y + iy + ph]
            )
            status = patch.get("status")
            need(
                status
                in ("valid", "unusable-texture-or-ambiguous", "unusable-local-response")
            )
            ref_ok = texture(
                patch.get("reference_texture"),
                require_usable=status != "unusable-texture-or-ambiguous",
            )
            moving_ok = texture(
                patch.get("moving_texture"),
                require_usable=status != "unusable-texture-or-ambiguous",
            )
            if status == "unusable-texture-or-ambiguous":
                need(not ref_ok or not moving_ok)
            elif status == "unusable-local-response":
                vector(patch.get("local_shift_est_xy"), 2)
                need(number(patch.get("response")) < LOCAL_RESPONSE_MIN)
            if status == "valid":
                need(number(patch.get("response")) >= LOCAL_RESPONSE_MIN)
                local = vector(patch.get("local_shift_est_xy"), 2)
                residual = vector(patch.get("residual_est_xy"), 2)
                for axis in (0, 1):
                    near(residual[axis], local[axis] - shift[axis])
                norm = number(patch.get("residual_norm_est_pixels"))
                near(norm, math.hypot(*residual))
                accepted.append(norm)
                centers.append((x + pw / 2, y + ph / 2))
        need(
            type(check.get("valid_patch_count")) is int
            and check["valid_patch_count"] == len(accepted)
            and len(accepted) >= MIN_VALID_PATCHES
        )
        span = [
            (max(v[axis] for v in centers) - min(v[axis] for v in centers)) / length
            for axis, length in ((0, aw), (1, ah))
        ]
        coverage = vector(check.get("coverage_span_fraction_xy"), 2)
        for axis in (0, 1):
            near(coverage[axis], span[axis])
            need(coverage[axis] >= MIN_SPAN_FRACTION)
        stats = {
            "residual_p90_est_pixels": float(np.quantile(accepted, 0.9)),
            "residual_median_est_pixels": float(np.median(accepted)),
            "residual_max_est_pixels": max(accepted),
        }
        for key, value in stats.items():
            near(check.get(key), value)
        need(
            stats["residual_p90_est_pixels"] <= MAX_RESIDUAL_P90_EST_PIXELS
            and stats["residual_median_est_pixels"] <= MAX_RESIDUAL_MEDIAN_EST_PIXELS
            and stats["residual_max_est_pixels"] <= MAX_RESIDUAL_MAX_EST_PIXELS
        )
    return None


def _validate_source_registration_model_metadata(receipt):
    """Validate new geometry/scales and source budgets, then unchanged V1 proof."""

    def need(condition):
        if not condition:
            raise ValueError("Malformed source-pixel RAW registration evidence")

    def number(value):
        need(type(value) in (int, float) and math.isfinite(value))
        return float(value)

    def near(actual, expected):
        need(abs(number(actual) - expected) <= 1e-9)

    from .raw_registration import EST_WIDTH

    need(
        isinstance(receipt, dict)
        and type(receipt.get("version")) is int
        and receipt["version"] == 3
    )
    source_shape = receipt.get("input_shape_h_w")
    need(
        isinstance(source_shape, (list, tuple))
        and len(source_shape) == 2
        and all(type(value) is int and value >= 2 for value in source_shape)
    )
    h, w = source_shape
    size = [max(2, round(h / (w / EST_WIDTH))), EST_WIDTH] if w > EST_WIDTH else [h, w]
    need(
        receipt.get("phase_estimation_shape_h_w") == size
        and type(receipt.get("phase_estimation_width_limit")) is int
        and receipt.get("phase_estimation_width_limit") == EST_WIDTH
    )
    scale = [w / size[1], h / size[0]]
    need(receipt.get("phase_estimation_to_source_scale_xy") == scale)
    model = receipt.get("model_check")
    need(
        isinstance(model, dict)
        and type(model.get("version")) is int
        and model["version"] == 2
        and model.get("policy") == SOURCE_POLICY
        and model.get("status") == "passed"
    )
    compatible = deepcopy(receipt)
    compatible["version"] = 2
    old_model = compatible["model_check"]
    old_model.update(version=1, policy=POLICY)
    checks = model.get("channels")
    need(isinstance(checks, dict) and set(checks) == {"red", "blue"})
    source_keys = {
        "source_thresholds",
        "source_residual_units",
        "source_residual_formula",
        "source_budget_is_absolute_registration_accuracy",
        "source_residual_p90_pixels",
        "source_residual_median_pixels",
        "source_residual_max_pixels",
        "source_budget_status",
        "estimator_gate_status",
        "estimator_gate_reason",
    }
    for channel in ("red", "blue"):
        proof = checks[channel]
        need(
            isinstance(proof, dict)
            and type(proof.get("version")) is int
            and proof["version"] == 2
            and proof.get("policy") == SOURCE_POLICY
            and proof.get("status") == "passed"
            and proof.get("reason") == "bounded-source-pixel-local-translations"
            and proof.get("source_budget_status") == "passed"
            and proof.get("estimator_gate_status") == "passed"
            and proof.get("estimator_gate_reason")
            == "bounded-spatially-distributed-local-translations"
        )
        need(
            proof.get("source_thresholds") == source_residual_thresholds()
            and all(
                type(value) is float for value in proof["source_thresholds"].values()
            )
            and proof.get("source_residual_units") == SOURCE_RESIDUAL_UNITS
            and proof.get("source_residual_formula") == SOURCE_RESIDUAL_FORMULA
            and proof.get("source_budget_is_absolute_registration_accuracy") is False
        )
        norms = []
        old_proof = old_model["channels"][channel]
        for item, old_item in zip(proof["patches"], old_proof["patches"], strict=True):
            if item["status"] != "valid":
                need(
                    "residual_source_xy" not in item
                    and "residual_norm_source_pixels" not in item
                )
                continue
            source_xy = item.get("residual_source_xy")
            need(isinstance(source_xy, (list, tuple)) and len(source_xy) == 2)
            expected = [
                number(value) * factor
                for value, factor in zip(item["residual_est_xy"], scale, strict=True)
            ]
            for actual, wanted in zip(source_xy, expected, strict=True):
                near(actual, wanted)
            norm = math.hypot(*expected)
            near(item.get("residual_norm_source_pixels"), norm)
            norms.append(norm)
            old_item.pop("residual_source_xy")
            old_item.pop("residual_norm_source_pixels")
        need(len(norms) >= MIN_VALID_PATCHES)
        p90, median, maximum = (
            float(np.quantile(norms, 0.9)),
            float(np.median(norms)),
            max(norms),
        )
        near(proof.get("source_residual_p90_pixels"), p90)
        near(proof.get("source_residual_median_pixels"), median)
        near(proof.get("source_residual_max_pixels"), maximum)
        need(
            p90 <= MAX_RESIDUAL_P90_SOURCE_PIXELS
            and maximum <= MAX_RESIDUAL_MAX_SOURCE_PIXELS
        )
        for key in source_keys:
            old_proof.pop(key)
        old_proof.update(
            version=1,
            policy=POLICY,
            reason="bounded-spatially-distributed-local-translations",
        )
    return _validate_registration_model_metadata(compatible)


def validate_registration_model_metadata(receipt):
    """Validate old/default and source-unit evidence without I/O or estimation."""
    try:
        if isinstance(receipt, dict) and receipt.get("version") == 3:
            return _validate_source_registration_model_metadata(receipt)
        return _validate_registration_model_metadata(receipt)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
        raise ValueError("Malformed RAW registration model evidence") from error
