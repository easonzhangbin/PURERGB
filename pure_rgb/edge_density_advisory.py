"""Multiscale edge-to-inward signal drift suggestions, never film diagnoses.

An explicit picture rectangle separates the film-frame region from the source
canvas. Spatially corresponding edge/inward strips vote at several depths.
Normal scene objects can satisfy the same rules as leakage or development
faults. Nothing here changes pixels, density windows, masks or weights.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from .metering_regions import _rectangle

PROVIDER = "multiscale-edge-density-advisory-v1"


@dataclass(frozen=True)
class EdgeDensityPolicy:
    depth_fractions: tuple = (0.04, 0.08, 0.16)
    tangential_bins: int = 16
    minimum_consistent_scales: int = 2
    minimum_common_shift: float = 0.18
    minimum_channel_span: float = 0.20
    minimum_adjacent_bins: int = 2


@dataclass(frozen=True)
class EdgeDensityResult:
    candidates: tuple
    receipt: dict


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pixels(source):
    return hashlib.sha256(memoryview(source).cast("B")).hexdigest()


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _policy(value):
    if not isinstance(value, EdgeDensityPolicy):
        raise ValueError("Expected EdgeDensityPolicy")
    fractions = value.depth_fractions
    if (
        type(fractions) is not tuple
        or not 2 <= len(fractions) <= 4
        or any(not _finite_number(x) or not 0 < x <= 0.25 for x in fractions)
        or tuple(sorted(set(fractions))) != fractions
    ):
        raise ValueError(
            "Depth fractions require2..4 distinct increasing finite values in(0,.25]"
        )
    for name in (
        "tangential_bins",
        "minimum_consistent_scales",
        "minimum_adjacent_bins",
    ):
        if type(getattr(value, name)) is not int:
            raise ValueError("Policy counts require explicit integers")
    if (
        not 4 <= value.tangential_bins <= 32
        or not 2 <= value.minimum_consistent_scales <= len(fractions)
        or not 1 <= value.minimum_adjacent_bins <= value.tangential_bins
    ):
        raise ValueError("Invalid scale/bin support counts")
    for name in ("minimum_common_shift", "minimum_channel_span"):
        x = getattr(value, name)
        if not _finite_number(x) or x <= 0:
            raise ValueError("Signal thresholds require finite positive values")
    return value


def _runs(mask):
    start = None
    for i, selected in enumerate([*mask, False]):
        if selected and start is None:
            start = i
        if not selected and start is not None:
            yield start, i
            start = None


def _side_grid(density, side, fractions, bins):
    # Normal coordinate begins at the selected picture edge, never at the
    # source-canvas edge. Flips preserve matched tangential positions.
    grid = (
        density
        if side == "top"
        else density[::-1]
        if side == "bottom"
        else np.swapaxes(density, 0, 1)
        if side == "left"
        else np.swapaxes(density[:, ::-1], 0, 1)
    )
    normal, tangent = grid.shape[:2]
    depths = [max(1, int(normal * f)) for f in fractions]
    if len(set(depths)) != len(depths):
        raise ValueError("ROI too small to support distinct multiscale depths")
    edges = [tangent * i // bins for i in range(bins + 1)]
    residuals = np.empty((len(depths), bins, 3), dtype=np.float64)
    for si, depth in enumerate(depths):
        for bi, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            edge = grid[:depth, a:b]
            inner = grid[2 * depth : 3 * depth, a:b]
            residuals[si, bi] = np.median(edge, axis=(0, 1)) - np.median(
                inner, axis=(0, 1)
            )
    return depths, edges, residuals


def propose_edge_density_advisory(source, picture_roi, *, policy=EdgeDensityPolicy()):
    """Return evidence boxes in the supplied source coordinates; no auto masks.

    The rectangle must already describe the picture interior. This function
    does not infer film boundaries or distinguish a scene from a physical fault.
    """
    policy = _policy(policy)
    if (
        not isinstance(source, np.ndarray)
        or source.ndim != 3
        or source.shape[-1] != 3
        or source.dtype not in (np.float32, np.float64)
        or source.flags.writeable
        or not source.flags.c_contiguous
        or not np.isfinite(source).all()
    ):
        raise ValueError("Require finite readonly C-order float32/64 HWC RGB source")
    roi = _rectangle(picture_roi, source.shape, "Explicit picture ROI")
    x0, y0, x1, y1 = roi
    if min(x1 - x0, y1 - y0) < policy.tangential_bins * 4:
        raise ValueError("Picture ROI needs at least four pixels per tangential bin")
    before = _pixels(source)
    selected = source[y0:y1, x0:x1]
    density = -np.log10(np.maximum(selected.astype(np.float64), 1e-6))
    candidate_records, observations = [], []
    for side in ("top", "bottom", "left", "right"):
        depths, edges, residuals = _side_grid(
            density, side, policy.depth_fractions, policy.tangential_bins
        )
        positive = np.all(residuals >= policy.minimum_common_shift, axis=2)
        negative = np.all(residuals <= -policy.minimum_common_shift, axis=2)
        chromatic = np.ptp(residuals, axis=2) >= policy.minimum_channel_span
        kinds = [
            ("edge-denser-than-inward", positive),
            ("edge-less-dense-than-inward", negative),
        ]
        # The same largest/smallest channel pair must persist across scales.
        for lo in range(3):
            for hi in range(3):
                if lo != hi:
                    agreement = (
                        chromatic
                        & (np.argmin(residuals, axis=2) == lo)
                        & (np.argmax(residuals, axis=2) == hi)
                    )
                    kinds.append((f"edge-channel-drift-{lo}-to-{hi}", agreement))
        for kind, votes in kinds:
            supported = votes.sum(axis=0) >= policy.minimum_consistent_scales
            for first, last in _runs(supported):
                if last - first < policy.minimum_adjacent_bins:
                    continue
                scale_indices = np.nonzero(votes[:, first:last].any(axis=1))[0]
                depth = max(depths[i] for i in scale_indices)
                a, b = edges[first], edges[last]
                if side == "top":
                    box = [x0 + a, y0, x0 + b, y0 + depth]
                elif side == "bottom":
                    box = [x0 + a, y1 - depth, x0 + b, y1]
                elif side == "left":
                    box = [x0, y0 + a, x0 + depth, y0 + b]
                else:
                    box = [x1 - depth, y0 + a, x1, y0 + b]
                candidate_records.append(
                    {
                        "kind": kind,
                        "side": side,
                        "roi": box,
                        "tangential_bin_interval": [first, last],
                        "consistent_scale_count_by_bin": votes[:, first:last]
                        .sum(axis=0)
                        .tolist(),
                        "supporting_scale_indices_by_bin": [
                            np.nonzero(votes[:, bi])[0].tolist()
                            for bi in range(first, last)
                        ],
                        "box_is_evidence_envelope_not_exact_contour": True,
                        "physical_cause": None,
                        "classification": "indeterminate",
                        "automatic_exclusion_allowed": False,
                        "ambiguity": "Normal scene objects and physical faults can have identical RGB evidence",
                    }
                )
        observations.append(
            {
                "side": side,
                "depth_pixels": depths,
                "tangential_edges": edges,
                "edge_minus_inward_log_density_RGB_by_scale_and_bin": residuals.tolist(),
            }
        )
    if _pixels(source) != before or source.flags.writeable:
        raise ValueError("Source changed while collecting edge evidence")
    receipt = {
        "schema": PROVIDER,
        "version": 1,
        "policy": asdict(policy),
        "policy_sha256": hashlib.sha256(
            _canonical(asdict(policy)).encode()
        ).hexdigest(),
        "source_shape": list(source.shape),
        "source_dtype": source.dtype.str,
        "source_pixels_sha256_before": before,
        "source_pixels_sha256_after": before,
        "picture_roi": list(roi),
        "coordinate_space": "caller source half-open xyxy; no resize",
        "log_policy": "float64 -log10(max(T,1e-6)); no upper clamp",
        "selected_channel_samples_at_or_below_floor": (selected <= 1e-6)
        .sum((0, 1))
        .tolist(),
        "selected_channel_samples_above_one": (selected > 1).sum((0, 1)).tolist(),
        "strip_pairing": "at each normal depth d: edge[0:d] minus inward[2d:3d], identical tangential bin",
        "outside_picture_ROI_used": False,
        "film_boundary_inferred": False,
        "observations": observations,
        "candidate_count": len(candidate_records),
        "classification": "indeterminate",
        "automatic_exclusions": [],
        "automatic_weight_changes": [],
        "source_values_changed": False,
        "density_window_or_renderer_called": False,
        "scope": "Multiscale signal suggestions; not defect probabilities, semantic recognition or automatic metering support",
    }
    return EdgeDensityResult(tuple(candidate_records), receipt)
