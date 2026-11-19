"""Advisory spatial signal anomalies, never film diagnoses or density masks.

Scene objects and development faults can produce identical single-frame RGB.
These versioned heuristics expose measurements without removing any pixels.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from .metering_regions import _rectangle

PROVIDER = "edge-signal-anomaly-advisory-v1"


@dataclass(frozen=True)
class CandidatePolicy:
    rows: int = 24
    columns: int = 36
    edge_cells: int = 3
    minimum_component_cells: int = 2
    minimum_local_log_difference: float = 0.18
    maximum_flat_log_spread: float = 0.035
    minimum_channel_residual: float = 0.20
    local_radius: int = 3


@dataclass(frozen=True)
class CandidateResult:
    candidates: tuple
    receipt: dict


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pixel_sha(value):
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _components(mask):
    seen = np.zeros_like(mask)
    for row, column in zip(*np.nonzero(mask), strict=True):
        if seen[row, column]:
            continue
        stack, cells = [(int(row), int(column))], []
        seen[row, column] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for ny, nx in ((y - 1, x), (y, x - 1), (y, x + 1), (y + 1, x)):
                if (
                    0 <= ny < mask.shape[0]
                    and 0 <= nx < mask.shape[1]
                    and mask[ny, nx]
                    and not seen[ny, nx]
                ):
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        yield sorted(cells)


def _policy(policy):
    if not isinstance(policy, CandidatePolicy):
        raise ValueError("Expected CandidatePolicy")
    for name in (
        "rows",
        "columns",
        "edge_cells",
        "minimum_component_cells",
        "local_radius",
    ):
        value = getattr(policy, name)
        if type(value) is not int or value < 1:
            raise ValueError("Grid/support limits must be positive integers")
    if not 4 <= policy.rows <= 64 or not 4 <= policy.columns <= 64:
        raise ValueError("Grid must have 4..64 cells per axis")
    if (
        policy.edge_cells * 2 >= min(policy.rows, policy.columns)
        or policy.local_radius > 7
    ):
        raise ValueError("Edge/local neighborhood too broad")
    for name in (
        "minimum_local_log_difference",
        "maximum_flat_log_spread",
        "minimum_channel_residual",
    ):
        value = getattr(policy, name)
        if type(value) not in (float, int) or not np.isfinite(value) or value <= 0:
            raise ValueError("Thresholds must be finite and positive")
    return policy


def propose_defect_candidates(source, roi, *, policy=CandidatePolicy()):
    """Advisory boxes in original half-open source coordinates, not true contours."""
    policy = _policy(policy)
    if (
        not isinstance(source, np.ndarray)
        or source.ndim != 3
        or source.shape[-1] != 3
        or source.dtype not in (np.float32, np.float64)
        or not source.flags.c_contiguous
        or source.flags.writeable
        or not np.isfinite(source).all()
    ):
        raise ValueError("Require finite readonly C float32/64 RGB source")
    roi = _rectangle(roi, source.shape, "Candidate picture ROI")
    x0, y0, x1, y1 = roi
    if x1 - x0 < policy.columns * 4 or y1 - y0 < policy.rows * 4:
        raise ValueError("Cells need at least four source pixels per side")
    before = _pixel_sha(source)
    xs = [x0 + (x1 - x0) * i // policy.columns for i in range(policy.columns + 1)]
    ys = [y0 + (y1 - y0) * i // policy.rows for i in range(policy.rows + 1)]
    medians = np.empty((policy.rows, policy.columns, 3), dtype=np.float64)
    spread = np.empty((policy.rows, policy.columns), dtype=np.float64)
    area = np.empty((policy.rows, policy.columns), dtype=np.int64)
    floored = 0
    for y in range(policy.rows):
        for x in range(policy.columns):
            pixels = source[ys[y] : ys[y + 1], xs[x] : xs[x + 1]]
            floored += int(np.count_nonzero(pixels <= 1e-6))
            density = -np.log10(np.maximum(pixels.astype(np.float64), 1e-6))
            medians[y, x] = np.median(density, axis=(0, 1))
            low, high = np.quantile(density.mean(axis=2), [0.1, 0.9])
            spread[y, x] = high - low
            area[y, x] = pixels.shape[0] * pixels.shape[1]
    radius = policy.local_radius
    padded = np.pad(medians, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    neighbors = np.lib.stride_tricks.sliding_window_view(
        padded, (2 * radius + 1, 2 * radius + 1), axis=(0, 1)
    )
    residual = medians - np.median(neighbors, axis=(-2, -1))
    common = residual.mean(axis=2)
    chroma = residual.max(axis=2) - residual.min(axis=2)
    tone = medians.mean(axis=2)
    low_tone, high_tone = np.quantile(tone, [0.08, 0.92])
    flat = spread <= policy.maximum_flat_log_spread
    edge = np.zeros_like(flat)
    n = policy.edge_cells
    edge[:n] = edge[-n:] = True
    edge[:, :n] = edge[:, -n:] = True
    kinds = (
        (
            "smooth-low-density-edge-patch",
            flat
            & (residual.max(axis=2) <= -policy.minimum_local_log_difference)
            & (tone <= low_tone),
        ),
        (
            "smooth-high-density-edge-patch",
            flat
            & (residual.min(axis=2) >= policy.minimum_local_log_difference)
            & (tone >= high_tone),
        ),
        (
            "channel-shifted-edge-region",
            flat & (chroma >= policy.minimum_channel_residual),
        ),
    )
    candidates = []
    for kind, selected in kinds:
        for cells in _components(selected):
            if len(cells) < policy.minimum_component_cells or not any(
                edge[y, x] for y, x in cells
            ):
                continue
            yy, xx = np.array(cells).T
            mask = np.zeros_like(selected)
            mask[yy, xx] = True
            weight = int(area[mask].sum())
            candidates.append(
                {
                    "kind": kind,
                    "roi": [
                        xs[int(xx.min())],
                        ys[int(yy.min())],
                        xs[int(xx.max()) + 1],
                        ys[int(yy.max()) + 1],
                    ],
                    "grid_cells_yx": [[int(y), int(x)] for y, x in cells],
                    "cell_union_pixel_area": weight,
                    "fraction_of_requested_roi": weight / ((x1 - x0) * (y1 - y0)),
                    "median_log_density_residual_RGB": np.median(
                        residual[mask], axis=0
                    ).tolist(),
                    "median_common_log_density_residual": float(
                        np.median(common[mask])
                    ),
                    "median_channel_residual_range": float(np.median(chroma[mask])),
                    "median_within_cell_log_spread": float(np.median(spread[mask])),
                    "ranking_signal": float(np.max(np.abs(residual[mask]))),
                    "automatic_exclusion_allowed": False,
                    "physical_cause": None,
                    "ambiguity": "Normal bright/dark/coloured scene objects may have identical signals",
                }
            )
    candidates.sort(
        key=lambda item: (-item["ranking_signal"], item["kind"], item["roi"])
    )
    assert _pixel_sha(source) == before and not source.flags.writeable
    return CandidateResult(
        tuple(candidates),
        {
            "schema": PROVIDER,
            "version": 1,
            "source_shape": list(source.shape),
            "source_dtype": source.dtype.str,
            "source_pixels_sha256_before": before,
            "source_pixels_sha256_after": before,
            "requested_roi": list(roi),
            "policy": asdict(policy),
            "policy_sha256": hashlib.sha256(
                _canonical(asdict(policy)).encode()
            ).hexdigest(),
            "grid_x_edges": xs,
            "grid_y_edges": ys,
            "log_policy": "float64 -log10(max(T,1e-6)); no upper clip",
            "channel_samples_at_or_below_floor": floored,
            "candidate_count": len(candidates),
            "automatic_exclusions": [],
            "density_or_render_called": False,
            "source_values_changed": False,
            "output_crop_changed": False,
            "scope": "Signal suggestions only; no probability, physical diagnosis or metering weight change",
        },
    )
