"""Local edge-connected signal evidence on an explicit prepared source grid.

The output is an advisory union of actually supported small cells, never filled
bounding boxes or automatically accepted density exclusions. Broad structures
are retained as ambiguous rejections. A scene and a defect with identical RGB
cannot be distinguished by this module.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import cv2
import numpy as np

from .metering_regions import _rectangle
from .metering_snapshot import _check_source, _identity

PROVIDER = "localized-edge-signal-evidence-v1"


@dataclass(frozen=True)
class LocalizationPolicy:
    cell_pixels: int = 8
    normal_offsets_cells: tuple = (2, 4, 8)
    edge_depth_fraction: float = 0.25
    minimum_consistent_scales: int = 2
    minimum_common_shift: float = 0.18
    minimum_channel_span: float = 0.20
    minimum_component_cells: int = 3
    maximum_component_roi_fraction: float = 0.10
    maximum_tangential_fraction: float = 0.35
    guard_pixels: int = 4
    low_texture_spread: float = 0.015


@dataclass(frozen=True)
class LocalizedEdgeResult:
    evidence_mask: np.ndarray
    review_guard_mask: np.ndarray
    components: tuple
    rejected_components: tuple
    receipt: dict


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _pixels(value):
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _policy(value):
    if not isinstance(value, LocalizationPolicy):
        raise ValueError("Expected LocalizationPolicy")
    for key in (
        "cell_pixels",
        "minimum_consistent_scales",
        "minimum_component_cells",
        "guard_pixels",
    ):
        if type(getattr(value, key)) is not int:
            raise ValueError("Counts and source-grid margins require explicit integers")
    offsets = value.normal_offsets_cells
    if (
        type(offsets) is not tuple
        or not 2 <= len(offsets) <= 4
        or any(type(x) is not int or not 1 <= x <= 16 for x in offsets)
        or tuple(sorted(set(offsets))) != offsets
    ):
        raise ValueError(
            "Offsets require2..4 increasing distinct integer cell distances in[1,16]"
        )
    if (
        not 2 <= value.cell_pixels <= 32
        or not 2 <= value.minimum_consistent_scales <= len(offsets)
        or not 1 <= value.minimum_component_cells <= 1024
        or not 0 <= value.guard_pixels <= 32
    ):
        raise ValueError("Unsupported cell/support/guard limits")
    for key, upper in (
        ("edge_depth_fraction", 0.25),
        ("maximum_component_roi_fraction", 1.0),
        ("maximum_tangential_fraction", 1.0),
    ):
        x = getattr(value, key)
        if not _finite(x) or not 0 < x <= upper:
            raise ValueError("Invalid geometric evidence fraction")
    for key in ("minimum_common_shift", "minimum_channel_span", "low_texture_spread"):
        x = getattr(value, key)
        if not _finite(x) or x <= 0:
            raise ValueError("Signal thresholds must be positive finite numbers")
    return value


def _cells(source, roi, cell):
    x0, y0, x1, y1 = roi
    xs = list(range(x0, x1, cell)) + [x1]
    ys = list(range(y0, y1, cell)) + [y1]
    grid = np.empty((len(ys) - 1, len(xs) - 1, 3), np.float64)
    texture = np.empty(grid.shape[:2], np.float64)
    for y, (a, b) in enumerate(zip(ys[:-1], ys[1:])):
        for x, (c, d) in enumerate(zip(xs[:-1], xs[1:])):
            density = -np.log10(np.maximum(source[a:b, c:d].astype(np.float64), 1e-6))
            grid[y, x] = np.median(density, axis=(0, 1))
            low, high = np.quantile(density.mean(2), [0.1, 0.9])
            texture[y, x] = high - low
    return grid, texture, xs, ys


def _rle(mask):
    runs = []
    for y in np.flatnonzero(mask.any(axis=1)):
        row = mask[y]
        edges = np.diff(np.r_[False, row, False].astype(np.int8))
        for x0, x1 in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            runs.append([int(y), int(x0), int(x1)])
    return runs


def _raster(cells, xs, ys, shape):
    pixels = np.zeros(shape[:2], bool)
    for y, x in np.argwhere(cells):
        pixels[ys[y] : ys[y + 1], xs[x] : xs[x + 1]] = True
    return pixels


def localize_edge_evidence(
    source, picture_roi, current_identity, *, policy=LocalizationPolicy()
):
    """Return readonly source-grid masks and4-connected evidence components.

    ``current_identity`` must come from the caller's verified prepared grid. For
    a stride-sampled development cache its geometry token must name that grid;
    this API does not invent unsampled full-resolution pixel contours.
    """
    policy = _policy(policy)
    identity = _identity(current_identity)
    _check_source(source, identity)
    roi = _rectangle(picture_roi, source.shape, "Explicit picture ROI")
    x0, y0, x1, y1 = roi
    minimum = policy.cell_pixels * (max(policy.normal_offsets_cells) + 2)
    if min(x1 - x0, y1 - y0) < minimum:
        raise ValueError(
            "Picture ROI too small for the declared independent spatial scales"
        )
    grid, texture, xs, ys = _cells(source, roi, policy.cell_pixels)
    gh, gw = grid.shape[:2]
    area = np.diff(ys)[:, None] * np.diff(xs)[None, :]
    # Median3x3 context suppresses isolated cell noise; no source resampling.
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode="edge"), (3, 3), axis=(0, 1)
    )
    smoothed = np.median(windows, axis=(-2, -1))
    evidence = np.zeros(source.shape[:2], bool)
    accepted, rejected = [], []
    for side in ("top", "bottom", "left", "right"):
        iy, ix = np.mgrid[:gh, :gw]
        distance = (
            iy
            if side == "top"
            else gh - 1 - iy
            if side == "bottom"
            else ix
            if side == "left"
            else gw - 1 - ix
        )
        normal = gh if side in ("top", "bottom") else gw
        edge_band = distance < max(
            1, int(math.ceil(normal * policy.edge_depth_fraction))
        )
        deltas, valid_scales = [], []
        for offset in policy.normal_offsets_cells:
            ry = (
                iy + offset
                if side == "top"
                else iy - offset
                if side == "bottom"
                else iy
            )
            rx = (
                ix + offset
                if side == "left"
                else ix - offset
                if side == "right"
                else ix
            )
            valid = edge_band & (ry >= 0) & (ry < gh) & (rx >= 0) & (rx < gw)
            reference = smoothed[np.clip(ry, 0, gh - 1), np.clip(rx, 0, gw - 1)]
            deltas.append(smoothed - reference)
            valid_scales.append(valid)
        residuals, valid = np.asarray(deltas), np.asarray(valid_scales)
        kinds = [
            (
                "denser-edge",
                valid & np.all(residuals >= policy.minimum_common_shift, axis=3),
            ),
            (
                "less-dense-edge",
                valid & np.all(residuals <= -policy.minimum_common_shift, axis=3),
            ),
        ]
        span = np.ptp(residuals, axis=3) >= policy.minimum_channel_span
        for lo in range(3):
            for hi in range(3):
                if lo != hi:
                    kinds.append(
                        (
                            f"channel-drift-{lo}-to-{hi}",
                            valid
                            & span
                            & (np.argmin(residuals, axis=3) == lo)
                            & (np.argmax(residuals, axis=3) == hi),
                        )
                    )
        for kind, votes in kinds:
            counts = votes.sum(axis=0)
            selected = np.ascontiguousarray(
                counts >= policy.minimum_consistent_scales, dtype=np.uint8
            )
            number, labels, _, _ = cv2.connectedComponentsWithStats(
                selected, connectivity=4
            )
            for label in range(1, number):
                component = labels == label
                cell_count = int(component.sum())
                yy, xx = np.nonzero(component)
                reasons = []
                if not np.any(distance[component] == 0):
                    reasons.append("not-connected-to-explicit-picture-edge")
                if cell_count < policy.minimum_component_cells:
                    reasons.append("too-few-supported-cells")
                pixel_area = int(area[component].sum())
                fraction = pixel_area / ((x1 - x0) * (y1 - y0))
                tangent = (
                    (xs[int(xx.max()) + 1] - xs[int(xx.min())]) / (x1 - x0)
                    if side in ("top", "bottom")
                    else (ys[int(yy.max()) + 1] - ys[int(yy.min())]) / (y1 - y0)
                )
                if fraction > policy.maximum_component_roi_fraction:
                    reasons.append("broad-area-ambiguous")
                if tangent > policy.maximum_tangential_fraction:
                    reasons.append("long-edge-structure-ambiguous")
                bounds = [
                    xs[int(xx.min())],
                    ys[int(yy.min())],
                    xs[int(xx.max()) + 1],
                    ys[int(yy.max()) + 1],
                ]
                row = {
                    "side": side,
                    "kind": kind,
                    "cell_count": cell_count,
                    "supported_pixel_area": pixel_area,
                    "bbox": bounds,
                    "bbox_is_not_mask": True,
                    "tangential_fraction": float(tangent),
                    "minimum_supported_scales": int(counts[component].min()),
                    "maximum_supported_scales": int(counts[component].max()),
                    "median_cell_log_spread": float(np.median(texture[component])),
                    "low_texture_cell_fraction": float(
                        np.mean(texture[component] <= policy.low_texture_spread)
                    ),
                    "numeric_evidence_not_probability": True,
                    "classification": "indeterminate",
                    "physical_cause": None,
                    "automatic_exclusion_allowed": False,
                    "reasons": reasons,
                }
                if reasons:
                    rejected.append(row)
                else:
                    pixels = _raster(component, xs, ys, source.shape)
                    row.update(
                        mask_sha256=_pixels(pixels), pixel_runs_y_x0_x1=_rle(pixels)
                    )
                    evidence |= pixels
                    accepted.append(row)
    guard = evidence.copy()
    if policy.guard_pixels:
        k = 2 * policy.guard_pixels + 1
        guard = cv2.dilate(evidence.astype(np.uint8), np.ones((k, k), np.uint8)).astype(
            bool
        )
        inside = np.zeros_like(guard)
        inside[y0:y1, x0:x1] = True
        guard &= inside
    evidence, guard = np.ascontiguousarray(evidence), np.ascontiguousarray(guard)
    evidence.setflags(write=False)
    guard.setflags(write=False)
    _check_source(source, identity)
    receipt = {
        "schema": PROVIDER,
        "version": 1,
        "source_identity": deepcopy(identity),
        "picture_roi": list(roi),
        "policy": asdict(policy),
        "policy_sha256": _digest(asdict(policy)),
        "grid_shape": [gh, gw],
        "grid_x_edges": xs,
        "grid_y_edges": ys,
        "connectivity": 4,
        "source_pixel_mask_sha256": _pixels(evidence),
        "review_guard_mask_sha256": _pixels(guard),
        "evidence_pixel_area": int(evidence.sum()),
        "review_guard_pixel_area": int(guard.sum()),
        "guard_pixels_in_supplied_source_grid": policy.guard_pixels,
        "guard_is_not_measured_evidence": True,
        "accepted_component_count": len(accepted),
        "ambiguous_rejected_component_count": len(rejected),
        "status": "indeterminate-low-texture"
        if not accepted and np.median(texture) <= policy.low_texture_spread
        else "localized-signal-cause-indeterminate"
        if accepted
        else "no-localized-edge-evidence-not-a-health-certificate",
        "low_texture_cell_fraction": float(
            np.mean(texture <= policy.low_texture_spread)
        ),
        "log_policy": "float64 -log10(max(T,1e-6)); no upper clamp",
        "RGB_support_shared": True,
        "source_values_changed": False,
        "source_unchanged_checked_after_statistics": True,
        "automatic_exclusions": [],
        "automatic_weight_changes": [],
        "holdout_accuracy_certified": False,
        "film_boundary_inferred": False,
        "output_is_exact_supported_cell_union_not_filled_bbox": True,
        "source_geometry_external_authenticity_certified": False,
        "opencv_version": cv2.__version__,
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return LocalizedEdgeResult(
        evidence, guard, tuple(accepted), tuple(rejected), receipt
    )
