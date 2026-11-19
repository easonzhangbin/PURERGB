"""Bounded automatic measurement avoidance with explicit ambiguity records.

Conservative mode only excludes localized edge-connected clipped plateaus.
Edge-model mode also permits strongly supported edge/corner contamination
patterns. Those patterns are hypotheses, not proofs of development or leakage.
Every exclusion shares RGB support and leaves the source and output untouched.
"""

from copy import deepcopy
import json

import cv2
import numpy as np

from .metering_regions import _rectangle
from .immutable_source_v44 import source_digest


PROVIDER = "automatic-edge-metering-avoidance-v31"
POLICY = dict(
    rows=24,
    columns=36,
    edge_depth_fraction=0.16,
    inward_offsets=[2, 4],
    minimum_common_shift=0.5,
    minimum_chroma_shift=0.35,
    maximum_cell_spread=0.065,
    clipped_level=65534 / 65535,
    minimum_clipped_fraction=0.75,
    minimum_component_cells=2,
    maximum_boxes=64,
)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def checked_automatic_avoidance(document):
    _require(type(document) is dict, "Automatic avoidance needs a versioned object")
    if type(document.get("version")) is int and document["version"] == 2:
        from .defect_reliability_v32 import checked_context_avoidance

        return checked_context_avoidance(document)
    _require(
        type(document.get("version")) is int and document["version"] == 1,
        "Unsupported automatic avoidance version",
    )
    _require(type(document.get("enabled")) is bool, "Automatic enabled must be boolean")
    if not document["enabled"]:
        _require(
            set(document) == {"version", "enabled"},
            "Disabled automatic avoidance cannot hide fields",
        )
        return None
    _require(
        not set(document)
        - {"version", "enabled", "mode", "max_fraction", "protected_rois"},
        "Unknown automatic avoidance field",
    )
    mode = document.get("mode", "conservative")
    _require(
        mode in ("advisory", "conservative", "edge-model", "adaptive-edge"),
        "Unknown automatic avoidance mode",
    )
    fraction = document.get("max_fraction", 0.12)
    _require(
        type(fraction) in (int, float)
        and np.isfinite(fraction)
        and 0 < fraction <= 0.2,
        "Automatic removal fraction must lie in (0,.2]",
    )
    boxes = document.get("protected_rois", [])
    _require(
        type(boxes) is list and len(boxes) <= 64, "At most64 explicitly protected ROIs"
    )
    normalized = []
    for box in boxes:
        _require(
            type(box) is list
            and len(box) == 4
            and all(type(x) is int for x in box)
            and 0 <= box[0] < box[2]
            and 0 <= box[1] < box[3],
            "Invalid protected source ROI",
        )
        normalized.append(list(box))
    return dict(
        version=1,
        enabled=True,
        mode=mode,
        max_fraction=float(fraction),
        protected_rois=normalized,
    )


def _components(selected):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        selected.astype(np.uint8), connectivity=4
    )
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= POLICY["minimum_component_cells"]:
            yield labels == label


def _boxes(mask, xs, ys):
    boxes = []
    # Horizontal runs, vertically coalesced only when their exact x interval matches.
    active = {}
    for y, row in enumerate(mask):
        starts = np.flatnonzero(np.diff(np.r_[False, row, False].astype(np.int8)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[False, row, False].astype(np.int8)) == -1)
        current = {}
        for x0, x1 in zip(starts, ends):
            key = (int(x0), int(x1))
            box = active.pop(key, [xs[x0], ys[y], xs[x1], ys[y + 1]])
            box[3] = ys[y + 1]
            current[key] = box
        boxes.extend(active.values())
        active = current
    boxes.extend(active.values())
    return [list(map(int, box)) for box in boxes]


def detect_automatic_avoidance(source, roi, document):
    document = checked_automatic_avoidance(document)
    _require(document is not None, "Active automatic avoidance required")
    if document["version"] == 2:
        from .defect_reliability_v32 import detect_context_avoidance

        return detect_context_avoidance(source, roi, document)
    if document["mode"] == "adaptive-edge":
        from .adaptive_edge_v31 import detect_adaptive_edge

        return detect_adaptive_edge(source, roi, document)
    _require(
        isinstance(source, np.ndarray)
        and source.dtype in (np.float32, np.float64)
        and source.ndim == 3
        and source.shape[-1] == 3
        and source.flags.c_contiguous
        and not source.flags.writeable,
        "Automatic avoidance needs readonly contiguous float RGB",
    )
    roi = _rectangle(roi, source.shape, "Automatic measurement ROI")
    x0, y0, x1, y1 = roi
    before = source_digest(source)
    rows, cols = (
        min(POLICY["rows"], (y1 - y0) // 4),
        min(POLICY["columns"], (x1 - x0) // 4),
    )
    _require(
        min(rows, cols) >= 4, "Automatic measurement ROI needs at least16x16 pixels"
    )
    xs = [x0 + (x1 - x0) * i // cols for i in range(cols + 1)]
    ys = [y0 + (y1 - y0) * i // rows for i in range(rows + 1)]
    med = np.empty((rows, cols, 3), np.float64)
    spread = np.empty((rows, cols), np.float64)
    clipped = np.empty((rows, cols), np.float64)
    protected = np.zeros((rows, cols), bool)
    areas = np.outer(np.diff(ys), np.diff(xs))
    for box in document["protected_rois"]:
        _rectangle(box, source.shape, "Protected measurement ROI")
    from .parallel_meter_cells_v44 import parallel_rows

    def measure_row(y):
        row_med = np.empty((cols, 3), np.float64)
        row_spread = np.empty(cols, np.float64)
        row_clipped = np.empty(cols, np.float64)
        row_protected = np.zeros(cols, bool)
        for x in range(cols):
            values = source[ys[y] : ys[y + 1], xs[x] : xs[x + 1]]
            _require(
                np.isfinite(values).all(),
                "Nonfinite source must be corrected before density measurement",
            )
            log = -np.log10(np.maximum(values.astype(np.float64), 1e-6))
            row_med[x] = np.median(log, axis=(0, 1))
            row_spread[x] = float(np.ptp(np.quantile(log.mean(axis=2), [0.1, 0.9])))
            row_clipped[x] = float(
                np.mean(np.all(values >= POLICY["clipped_level"], axis=2))
            )
            row_protected[x] = any(
                xs[x] < c and xs[x + 1] > a and ys[y] < d and ys[y + 1] > b
                for a, b, c, d in document["protected_rois"]
            )
        return row_med, row_spread, row_clipped, row_protected

    measured = parallel_rows(source, rows, measure_row)
    if measured is None:
        for y in range(rows):
            for x in range(cols):
                values = source[ys[y] : ys[y + 1], xs[x] : xs[x + 1]]
                _require(
                    np.isfinite(values).all(),
                    "Nonfinite source must be corrected before density measurement",
                )
                log = -np.log10(np.maximum(values.astype(np.float64), 1e-6))
                med[y, x] = np.median(log, axis=(0, 1))
                spread[y, x] = float(np.ptp(np.quantile(log.mean(axis=2), [0.1, 0.9])))
                clipped[y, x] = float(
                    np.mean(np.all(values >= POLICY["clipped_level"], axis=2))
                )
                protected[y, x] = any(
                    xs[x] < c and xs[x + 1] > a and ys[y] < d and ys[y + 1] > b
                    for a, b, c, d in document["protected_rois"]
                )
    else:
        for y, (row_med, row_spread, row_clipped, row_protected) in enumerate(measured):
            med[y], spread[y], clipped[y], protected[y] = row_med, row_spread, row_clipped, row_protected
    yy, xx = np.mgrid[:rows, :cols]
    edge = (yy == 0) | (xx == 0) | (yy == rows - 1) | (xx == cols - 1)
    hard = clipped >= POLICY["minimum_clipped_fraction"]
    soft = np.zeros((rows, cols), bool)
    observations = []
    for side in ("top", "bottom", "left", "right"):
        distance = (
            yy
            if side == "top"
            else rows - 1 - yy
            if side == "bottom"
            else xx
            if side == "left"
            else cols - 1 - xx
        )
        normal = rows if side in ("top", "bottom") else cols
        band = distance < max(1, int(np.ceil(normal * POLICY["edge_depth_fraction"])))
        common_votes = np.zeros((rows, cols), int)
        chroma_votes = np.zeros((rows, cols), int)
        for offset in POLICY["inward_offsets"]:
            ry = (
                yy + offset
                if side == "top"
                else yy - offset
                if side == "bottom"
                else yy
            )
            rx = (
                xx + offset
                if side == "left"
                else xx - offset
                if side == "right"
                else xx
            )
            valid = band & (ry >= 0) & (ry < rows) & (rx >= 0) & (rx < cols)
            residual = med - med[np.clip(ry, 0, rows - 1), np.clip(rx, 0, cols - 1)]
            # Both higher and lower transmission can be unreliable; no preferred colour.
            common_votes += valid & (
                (residual.min(axis=2) >= POLICY["minimum_common_shift"])
                | (residual.max(axis=2) <= -POLICY["minimum_common_shift"])
            )
            chroma_votes += valid & (
                np.ptp(residual, axis=2) >= POLICY["minimum_chroma_shift"]
            )
        selected = (
            band
            & (spread <= POLICY["maximum_cell_spread"])
            & ((common_votes == 2) | (chroma_votes == 2))
        )
        for component in _components(selected):
            if not (component & edge).any():
                continue
            sy, sx = np.nonzero(component)
            tangent_span = (
                (sx.max() - sx.min() + 1) / cols
                if side in ("top", "bottom")
                else (sy.max() - sy.min() + 1) / rows
            )
            corner = bool(
                component[0, 0]
                or component[0, -1]
                or component[-1, 0]
                or component[-1, -1]
            )
            # Short isolated coloured scene objects are reported, never automatically excluded.
            eligible = tangent_span >= 0.5 or (corner and tangent_span >= 0.08)
            observations.append(
                dict(
                    side=side,
                    kind="edge-signal-pattern",
                    cells=int(component.sum()),
                    tangent_fraction=float(tangent_span),
                    corner=corner,
                    edge_model_gate=bool(eligible),
                    physical_cause=None,
                    classification="possible-contamination-or-scene",
                    boxes=_boxes(component, xs, ys),
                )
            )
            if eligible:
                soft |= component
    selected_hard = np.zeros_like(hard)
    for component in _components(hard):
        if (component & edge).any():
            selected_hard |= component
            observations.append(
                dict(
                    kind="edge-connected-clipped-plateau",
                    boxes=_boxes(component, xs, ys),
                    cells=int(component.sum()),
                    physical_cause=None,
                    classification="clipped-measurement-support",
                )
            )
    selected = (
        selected_hard if document["mode"] == "conservative" else selected_hard | soft
    )
    selected &= ~protected
    fraction = float(areas[selected].sum() / areas.sum())
    boxes = _boxes(selected, xs, ys)
    fallback = None
    if document["mode"] == "advisory":
        fallback = "advisory-only"
    elif fraction > document["max_fraction"]:
        fallback = "area-budget-exceeded"
    elif len(boxes) > POLICY["maximum_boxes"]:
        fallback = "contour-box-budget-exceeded"
    automatic = [] if fallback else boxes
    _require(
        source_digest(source) == before
        and not source.flags.writeable,
        "Source changed during automatic measurement detection",
    )
    return dict(
        schema=PROVIDER,
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
        automatic_exclusions=automatic,
        automatic_excluded_fraction=fraction if automatic else 0.0,
        fallback=fallback,
        protected_cells=int(protected.sum()),
        shared_RGB_support=True,
        physical_defect_classification_proven=False,
        ambiguity="Edge-model selection can also match natural scene regions; conservative mode uses clipped plateaus only",
        source_values_changed=False,
        output_crop_changed=False,
        default_route_changed=False,
    )
