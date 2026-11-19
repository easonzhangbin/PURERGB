"""A separately versioned regional model for gradual corner/edge contamination.

Unlike the strip model this compares local RGB density with a two-dimensional
median background, seeds coherent extrema, and grows only the nearby shoulder.
Corner or thin-band topology is required. Natural scenes may still match;
the result records that ambiguity and remains an explicit user-selected mode.
"""

from copy import deepcopy

import cv2
import numpy as np
from .immutable_source_v44 import source_digest

from .metering_regions import _rectangle


POLICY = dict(
    rows=48,
    columns=72,
    background_cells=25,
    background_hole_cells=9,
    seed_common_density=0.45,
    grow_common_density=0.15,
    seed_channel_span=0.45,
    grow_channel_span=0.20,
    maximum_seed_spread=0.18,
    shoulder_cells=4,
    near_edge_fraction=0.18,
    maximum_corner_extent=0.32,
    minimum_band_span=0.55,
    maximum_band_depth=0.12,
)


def detect_adaptive_edge(source, roi, document):
    from .automatic_metering_v31 import _boxes, _components

    if (
        not isinstance(source, np.ndarray)
        or source.ndim != 3
        or source.shape[-1] != 3
        or source.dtype not in (np.float32, np.float64)
        or source.flags.writeable
        or not source.flags.c_contiguous
    ):
        raise ValueError(
            "Adaptive edge model requires readonly contiguous floating RGB"
        )
    x0, y0, x1, y1 = _rectangle(roi, source.shape, "Adaptive edge picture ROI")
    before = source_digest(source)
    rows, cols = (
        min(POLICY["rows"], (y1 - y0) // 2),
        min(POLICY["columns"], (x1 - x0) // 2),
    )
    if min(rows, cols) < 16:
        raise ValueError("Adaptive edge model requires at least32x32 source pixels")
    xs = [x0 + (x1 - x0) * i // cols for i in range(cols + 1)]
    ys = [y0 + (y1 - y0) * i // rows for i in range(rows + 1)]
    med = np.empty((rows, cols, 3), np.float64)
    spread = np.empty((rows, cols), np.float64)
    protected = np.zeros((rows, cols), bool)
    for box in document["protected_rois"]:
        _rectangle(box, source.shape, "Protected ROI")
    from .parallel_meter_cells_v44 import parallel_rows

    def measure_row(y):
        row_med = np.empty((cols, 3), np.float64)
        row_spread = np.empty(cols, np.float64)
        row_protected = np.zeros(cols, bool)
        for x in range(cols):
            pixels = source[ys[y] : ys[y + 1], xs[x] : xs[x + 1]]
            if not np.isfinite(pixels).all():
                raise ValueError("Nonfinite source before adaptive density measurement")
            density = -np.log10(np.maximum(pixels.astype(np.float64), 1e-6))
            row_med[x] = np.median(density, axis=(0, 1))
            row_spread[x] = np.ptp(np.quantile(density.mean(axis=2), [0.1, 0.9]))
            row_protected[x] = any(
                xs[x] < c and xs[x + 1] > a and ys[y] < d and ys[y + 1] > b
                for a, b, c, d in document["protected_rois"]
            )
        return row_med, row_spread, row_protected

    measured = parallel_rows(source, rows, measure_row)
    if measured is None:
        for y in range(rows):
            for x in range(cols):
                pixels = source[ys[y] : ys[y + 1], xs[x] : xs[x + 1]]
                if not np.isfinite(pixels).all():
                    raise ValueError("Nonfinite source before adaptive density measurement")
                density = -np.log10(np.maximum(pixels.astype(np.float64), 1e-6))
                med[y, x] = np.median(density, axis=(0, 1))
                spread[y, x] = np.ptp(np.quantile(density.mean(axis=2), [0.1, 0.9]))
                protected[y, x] = any(
                    xs[x] < c and xs[x + 1] > a and ys[y] < d and ys[y + 1] > b
                    for a, b, c, d in document["protected_rois"]
                )
    else:
        for y, (row_med, row_spread, row_protected) in enumerate(measured):
            med[y], spread[y], protected[y] = row_med, row_spread, row_protected
    size = POLICY["background_cells"]
    hole = POLICY["background_hole_cells"]
    start = (size - hole) // 2
    footprint = np.ones((size, size), bool)
    footprint[start : start + hole, start : start + hole] = False
    radius = size // 2
    padded = np.pad(
        med, ((radius, radius), (radius, radius), (0, 0)), constant_values=np.nan
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (size, size), axis=(0, 1)
    )
    # Grid size is capped48x72: no full-camera-size annular temporary is created.
    ring = windows[..., footprint]
    residual = np.nanmedian(ring, axis=-1, overwrite_input=True) - med
    common = np.median(residual, axis=2)
    span = np.ptp(residual, axis=2)
    flat = spread <= POLICY["maximum_seed_spread"]
    selected = np.zeros((rows, cols), bool)
    observations = []

    def admit(seed, growth, kind):
        nonlocal selected
        for core in _components(seed & ~protected):
            radius = POLICY["shoulder_cells"]
            neighbourhood = cv2.dilate(
                core.astype(np.uint8),
                np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
            ).astype(bool)
            permitted = (growth & neighbourhood & ~protected) | core
            component = core.copy()
            cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
            for _ in range(rows * cols):
                expanded = (
                    cv2.dilate(component.astype(np.uint8), cross).astype(bool)
                    & permitted
                )
                if np.array_equal(expanded, component):
                    break
                component = expanded
            else:
                raise ValueError(
                    "Adaptive propagation did not converge within finite grid bound"
                )
            yy, xx = np.nonzero(component)
            width = (xx.max() - xx.min() + 1) / cols
            height = (yy.max() - yy.min() + 1) / rows
            margin = POLICY["near_edge_fraction"]
            top = yy.min() / rows <= margin
            bottom = (rows - 1 - yy.max()) / rows <= margin
            left = xx.min() / cols <= margin
            right = (cols - 1 - xx.max()) / cols <= margin
            corner = (
                (top or bottom)
                and (left or right)
                and max(width, height) <= POLICY["maximum_corner_extent"]
            )
            band = kind.startswith("channel") and (
                (
                    (top or bottom)
                    and width >= POLICY["minimum_band_span"]
                    and height <= POLICY["maximum_band_depth"]
                )
                or (
                    (left or right)
                    and height >= POLICY["minimum_band_span"]
                    and width <= POLICY["maximum_band_depth"]
                )
            )
            eligible = bool(corner or band)
            observations.append(
                dict(
                    kind=kind,
                    core_cells=int(core.sum()),
                    grown_cells=int(component.sum()),
                    width_fraction=float(width),
                    height_fraction=float(height),
                    compact_corner=bool(corner),
                    thin_band=bool(band),
                    topology_admitted=eligible,
                    boxes=_boxes(component, xs, ys),
                    classification="possible-contamination-or-scene",
                    physical_cause=None,
                )
            )
            if eligible:
                selected |= component

    for sign in (1, -1):
        score = common * sign
        admit(
            (score >= POLICY["seed_common_density"]) & flat,
            score >= POLICY["grow_common_density"],
            f"common-density-{sign}",
        )
    for lo in range(3):
        for hi in range(3):
            if lo == hi:
                continue
            pair = (residual.argmin(axis=2) == lo) & (residual.argmax(axis=2) == hi)
            admit(
                pair & (span >= POLICY["seed_channel_span"]) & flat,
                pair & (span >= POLICY["grow_channel_span"]),
                f"channel-{lo}-to-{hi}",
            )
    areas = np.outer(np.diff(ys), np.diff(xs))
    fraction = float(areas[selected].sum() / areas.sum())
    boxes = _boxes(selected, xs, ys)
    fallback = (
        "area-budget-exceeded"
        if fraction > document["max_fraction"]
        else "contour-box-budget-exceeded"
        if len(boxes) > 64
        else None
    )
    if source_digest(source) != before:
        raise ValueError("Adaptive source changed")
    return dict(
        schema="adaptive-regional-edge-density-v31",
        version=1,
        source_pixels_sha256=before,
        source_shape=list(source.shape),
        roi=list(roi),
        document=deepcopy(document),
        fixed_policy=deepcopy(POLICY),
        grid_x=xs,
        grid_y=ys,
        candidates=observations,
        candidate_exclusions=boxes,
        candidate_area_fraction=fraction,
        automatic_exclusions=[] if fallback else boxes,
        automatic_excluded_fraction=0.0 if fallback else fraction,
        fallback=fallback,
        protected_cells=int(protected.sum()),
        shared_RGB_support=True,
        physical_defect_classification_proven=False,
        source_values_changed=False,
        output_crop_changed=False,
        default_route_changed=False,
        ambiguity="Compact natural corner objects and long thin coloured scene bands can have the same evidence",
    )
