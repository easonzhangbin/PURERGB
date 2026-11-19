"""Opt-in density metering around caller-confirmed unusable rectangles.

This selects a measurement rectangle, never an output crop or a repair mask.
Image colour alone does not identify a light leak or a processing defect. The
caller supplies confirmed exclusions in the same oriented source coordinates
as its original analysis ROI. No existing workflow default calls this module.
"""

from dataclasses import asdict, dataclass, replace
import math

import numpy as np

MAX_EXCLUSIONS = 64


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _rectangle(value, shape, label):
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 4
        or not all(_integer(v) for v in value)
    ):
        raise ValueError(f"{label} must contain four integer source coordinates")
    x0, y0, x1, y1 = (int(v) for v in value)
    h, w = shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f"{label} lies outside the source or is empty")
    return x0, y0, x1, y1


@dataclass(frozen=True)
class MeteringSelection:
    source_shape: tuple[int, ...]
    requested_roi: tuple[int, int, int, int]
    analysis_roi: tuple[int, int, int, int]
    excluded_rois: tuple[tuple[int, int, int, int], ...]
    padding: int
    min_area_fraction: float
    min_side: int
    retained_fraction: float
    available_fraction: float
    method: str = "largest-clear-metering-rectangle-v1"
    coordinate_space: str = "oriented merged source, half-open xyxy"
    automatic_defect_classification: bool = False
    output_crop_changed: bool = False
    requested_excluded_rois: tuple[tuple[int, int, int, int], ...] = ()


def select_metering_roi(
    shape,
    roi,
    excluded_rois=(),
    *,
    padding=0,
    min_area_fraction=0.25,
    min_side=16,
):
    """Select the largest rectangle disjoint from padded confirmed exclusions.

    Coordinate compression preserves exact pixel edges without allocating a
    full-resolution mask. A weighted histogram finds the maximum empty
    rectangle. Ties prefer the original ROI's centre, then top/left, so replay
    is deterministic. The minimum size/area are engineering gates, not proof
    that the remaining scene contains a representative range of densities.

    Exclusions must be valid source rectangles. They may lie outside the
    analysis ROI; their padded intersection with it is what is excluded.
    Insufficient usable area is an error, with no whole-image fallback.
    """
    if (
        not isinstance(shape, (tuple, list))
        or len(shape) not in (2, 3)
        or not all(_integer(v) and v > 0 for v in shape)
        or (len(shape) == 3 and shape[2] != 3)
    ):
        raise ValueError("Expected a positive source shape (H,W) or (H,W,3)")
    shape = tuple(int(v) for v in shape)
    roi = _rectangle(roi, shape, "Analysis ROI")
    if not _integer(padding) or padding < 0:
        raise ValueError("Exclusion padding must be a nonnegative integer")
    if not _integer(min_side) or min_side < 2:
        raise ValueError(
            "Minimum metering side must be an integer of at least two pixels"
        )
    if (
        isinstance(min_area_fraction, (bool, np.bool_))
        or not isinstance(min_area_fraction, (float, int, np.floating, np.integer))
        or not math.isfinite(min_area_fraction)
        or not 0 < min_area_fraction <= 1
    ):
        raise ValueError("Minimum metering area fraction must lie in (0,1]")
    if (
        not isinstance(excluded_rois, (list, tuple))
        or len(excluded_rois) > MAX_EXCLUSIONS
    ):
        raise ValueError(f"Provide at most {MAX_EXCLUSIONS} exclusion rectangles")
    padding, min_side = int(padding), int(min_side)
    exclusions = []
    requested_exclusions = []
    rx0, ry0, rx1, ry1 = roi
    for value in excluded_rois:
        x0, y0, x1, y1 = _rectangle(value, shape, "Excluded ROI")
        requested_exclusions.append((x0, y0, x1, y1))
        clipped = (
            max(rx0, x0 - padding),
            max(ry0, y0 - padding),
            min(rx1, x1 + padding),
            min(ry1, y1 + padding),
        )
        if clipped[0] < clipped[2] and clipped[1] < clipped[3]:
            exclusions.append(clipped)
    exclusions = tuple(sorted(set(exclusions)))
    xs = sorted({rx0, rx1, *(v for r in exclusions for v in (r[0], r[2]))})
    ys = sorted({ry0, ry1, *(v for r in exclusions for v in (r[1], r[3]))})
    xi, yi = {v: i for i, v in enumerate(xs)}, {v: i for i, v in enumerate(ys)}
    free = np.ones((len(ys) - 1, len(xs) - 1), dtype=bool)
    for x0, y0, x1, y1 in exclusions:
        free[yi[y0] : yi[y1], xi[x0] : xi[x1]] = False
    available_area = sum(
        (ys[row + 1] - ys[row]) * (xs[col + 1] - xs[col])
        for row, col in zip(*np.nonzero(free))
    )
    heights = [0] * free.shape[1]
    best, best_rank = None, None
    roi_width, roi_height = rx1 - rx0, ry1 - ry0
    for row in range(free.shape[0]):
        row_height = ys[row + 1] - ys[row]
        heights = [h + row_height if ok else 0 for h, ok in zip(heights, free[row])]
        stack = []
        for column, height in enumerate([*heights, 0]):
            start = column
            while stack and stack[-1][1] > height:
                left, previous_height = stack.pop()
                start = left
                width = xs[column] - xs[left]
                if previous_height < min_side or width < min_side:
                    continue
                candidate = (
                    xs[left],
                    ys[row + 1] - previous_height,
                    xs[column],
                    ys[row + 1],
                )
                cx2 = candidate[0] + candidate[2] - rx0 - rx1
                cy2 = candidate[1] + candidate[3] - ry0 - ry1
                distance = cx2**2 * roi_height**2 + cy2**2 * roi_width**2
                rank = (
                    width * previous_height,
                    -distance,
                    -candidate[1],
                    -candidate[0],
                    -candidate[3],
                    -candidate[2],
                )
                if best_rank is None or rank > best_rank:
                    best, best_rank = candidate, rank
            if height and (not stack or stack[-1][1] < height):
                stack.append((start, height))
    area = roi_width * roi_height
    if best is None or best_rank[0] / area < min_area_fraction:
        raise ValueError(
            "Insufficient clear metering rectangle; choose a new analysis ROI"
        )
    return MeteringSelection(
        shape,
        roi,
        best,
        exclusions,
        padding,
        float(min_area_fraction),
        min_side,
        best_rank[0] / area,
        int(available_area) / area,
        requested_excluded_rois=tuple(sorted(set(requested_exclusions))),
    )


def measure_avoiding_regions(
    image,
    config,
    roi,
    excluded_rois=(),
    *,
    density="negpy",
    padding=0,
    min_area_fraction=0.25,
    min_side=16,
):
    """Measure a frozen window on the selected clear rectangle, leaving source intact.

    The result can enter the existing fixed_process/render APIs. It uses the
    selected original density provider. The NegPy meter receives the selected
    contiguous pixel rectangle as a view with exact 0/1 normalized edges; a
    full-source integer -> fraction -> integer round trip can otherwise read
    a neighbouring excluded pixel. No masked-pixel packing, resampling or
    replacement of bad pixels occurs. Source-coordinate ROI is restored on the
    frozen window, for use on the original full image.
    """
    from .density_matrix import negpy_window, percentile_window

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype.kind != "f":
        raise ValueError("Expected floating-point HWC linear source bands")
    if not np.isfinite(image).all():
        raise ValueError("Metering source must contain only finite samples")
    if density not in ("negpy", "percentile"):
        raise ValueError("Unsupported density provider")
    selection = select_metering_roi(
        image.shape,
        roi,
        excluded_rois,
        padding=padding,
        min_area_fraction=min_area_fraction,
        min_side=min_side,
    )
    x0, y0, x1, y1 = selection.analysis_roi
    samples = image[y0:y1, x0:x1]
    if density == "negpy":
        window = negpy_window(samples, config, (0, 0, x1 - x0, y1 - y0))
    else:
        window = percentile_window(samples, (0, 0, x1 - x0, y1 - y0))
    window = replace(
        window, analysis=replace(window.analysis, roi=selection.analysis_roi)
    )
    if not np.isfinite([window.log_dense, window.log_thin]).all():
        raise ValueError("Selected metering region has nonfinite density endpoints")
    return replace(
        window,
        metadata={
            **window.metadata,
            "metering_selection": asdict(selection),
            "measurement_sampling": "Exact half-open source-pixel rectangle; NegPy normalized bounds are 0/1 within its view",
            "selected_nonpositive_sample_count": int(np.count_nonzero(samples <= 0)),
            "selected_below_log_floor_sample_count": int(
                np.count_nonzero(samples < 1e-6)
            ),
            "input_policy": "Reject source NaN/Inf. NegPy retains its existing signed-sample log floor; percentile requires nonnegative selected samples. Finite signed values outside the selected ROI are not measured. Source pixels are never clipped here.",
        },
    ), selection


def metering_process(process, window, selection):
    """Freeze the window and exact selected ROI for Native's auxiliary meters.

    Only this opt-in path adjusts normalized edges by the smallest necessary
    float step. JSON preserves those Python floats. The original fixed_process
    and existing saved-job behavior remain untouched.
    """
    from .density_matrix import fixed_process

    if tuple(window.analysis.roi) != selection.analysis_roi:
        raise ValueError("Frozen density window and metering selection disagree")
    h, w = selection.source_shape[:2]

    def exact_edge(pixel, length):
        edge = pixel / length
        for _ in range(4):
            recovered = int(edge * length)
            if recovered == pixel:
                return edge
            edge = float(
                np.nextafter(edge, math.inf if recovered < pixel else -math.inf)
            )
        raise ValueError("Metering edge cannot round-trip to its source pixel")

    edges = tuple(
        exact_edge(pixel, length)
        for pixel, length in zip(selection.analysis_roi, (w, h, w, h))
    )
    return replace(
        fixed_process(process, window, selection.source_shape), analysis_rect=edges
    )
