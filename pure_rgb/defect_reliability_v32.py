"""Validate anomaly topology in source context, independent of a metering crop.

The V31 proposals and thresholds are unchanged. This additional gate cannot
invent pixels: it rejects proposals which only look like corners because of
the arbitrary analysis selection. The default context is the prepared source
canvas, not a claim to have identified a physical film boundary.
"""

from copy import deepcopy

import numpy as np
from .immutable_source_v44 import source_digest

from .metering_regions import _rectangle


MODE = "context-checked"
CONTEXT_POLICY = {
    "algorithm_revision": 2,
    "maximum_empty_grid_gap": 1,
    "opposite_common_density_only": True,
    "direct_anchor_required": True,
    "recursive_linking": False,
    "new_or_expanded_pixels": False,
}


def checked_context_avoidance(document):
    from .automatic_metering_v31 import checked_automatic_avoidance

    if (
        type(document) is not dict
        or type(document.get("version")) is not int
        or document["version"] != 2
    ):
        raise ValueError("Context avoidance requires version2")
    if type(document.get("enabled")) is not bool:
        raise ValueError("Context enabled requires a boolean")
    if not document["enabled"]:
        if set(document) != {"version", "enabled"}:
            raise ValueError("Disabled context controls cannot hide fields")
        return None
    if set(document) - {
        "version",
        "enabled",
        "mode",
        "max_fraction",
        "protected_rois",
        "context_roi",
        "action",
    }:
        raise ValueError("Unknown context avoidance field")
    if document.get("mode", MODE) != MODE:
        raise ValueError("Version2 requires context-checked mode")
    if "action" in document and (
        type(document["action"]) is not str
        or document["action"] not in ("suggest", "exclude")
    ):
        raise ValueError("Context action requires suggest or exclude")
    legacy = {
        k: deepcopy(v)
        for k, v in document.items()
        if k not in {"context_roi", "action"}
    }
    legacy.update(version=1, mode="adaptive-edge")
    normalized = checked_automatic_avoidance(legacy)
    context = document.get("context_roi")
    if context is not None and (
        type(context) is not list
        or len(context) != 4
        or any(type(v) is not int for v in context)
        or not 0 <= context[0] < context[2]
        or not 0 <= context[1] < context[3]
    ):
        raise ValueError("Context ROI requires four integer source coordinates")
    normalized.update(version=2, mode=MODE, context_roi=deepcopy(context))
    if "action" in document:
        normalized["action"] = document["action"]
    return normalized


def _candidate_topology(candidate, context):
    from .adaptive_edge_v31 import POLICY

    boxes = candidate["boxes"]
    if not boxes:
        return dict(eligible=False, reason="empty-proposal")
    a = min(b[0] for b in boxes)
    b = min(b[1] for b in boxes)
    c = max(b[2] for b in boxes)
    d = max(b[3] for b in boxes)
    x0, y0, x1, y1 = context
    width, height = x1 - x0, y1 - y0
    inside = x0 <= a < c <= x1 and y0 <= b < d <= y1
    left = (a - x0) / width <= POLICY["near_edge_fraction"]
    right = (x1 - c) / width <= POLICY["near_edge_fraction"]
    top = (b - y0) / height <= POLICY["near_edge_fraction"]
    bottom = (y1 - d) / height <= POLICY["near_edge_fraction"]
    wf, hf = (c - a) / width, (d - b) / height
    corner = (
        inside
        and (top or bottom)
        and (left or right)
        and max(wf, hf) <= POLICY["maximum_corner_extent"]
    )
    band = (
        inside
        and candidate["kind"].startswith("channel")
        and (
            (
                (top or bottom)
                and wf >= POLICY["minimum_band_span"]
                and hf <= POLICY["maximum_band_depth"]
            )
            or (
                (left or right)
                and hf >= POLICY["minimum_band_span"]
                and wf <= POLICY["maximum_band_depth"]
            )
        )
    )
    eligible = bool(candidate["topology_admitted"] and (corner or band))
    return dict(
        eligible=eligible,
        source_box=[a, b, c, d],
        context_roi=list(context),
        width_fraction=wf,
        height_fraction=hf,
        corner=bool(corner),
        band=bool(band),
        reason="context-supported"
        if eligible
        else "meter-corner-not-supported-by-source-context",
    )


def _context_checks(candidates, context, xs, ys):
    """Retain adjacent signed shoulders only beside a directly valid anchor.

    The annular residual can change sign between a compact core and its shoulder,
    leaving one grid cell in the deadband. This links existing proposals across
    that single gap; it neither fills the gap nor recursively grows support.
    """
    checks = [_candidate_topology(c, context) for c in candidates]
    direct = [check["eligible"] for check in checks]
    ix, iy = {v: i for i, v in enumerate(xs)}, {v: i for i, v in enumerate(ys)}

    def adjacent(first, second):
        for a, b, c, d in first["boxes"]:
            for e, f, g, h in second["boxes"]:
                dx = max(ix[a] - ix[g], ix[e] - ix[c], 0)
                dy = max(iy[b] - iy[h], iy[f] - iy[d], 0)
                if max(dx, dy) <= CONTEXT_POLICY["maximum_empty_grid_gap"]:
                    return True
        return False

    opposite = {
        "common-density-1": "common-density--1",
        "common-density--1": "common-density-1",
    }
    for index, candidate in enumerate(candidates):
        if direct[index] or not candidate["topology_admitted"]:
            continue
        other_kind = opposite.get(candidate["kind"])
        if other_kind is None:
            continue
        for anchor_index, anchor in enumerate(candidates):
            if not direct[anchor_index] or anchor["kind"] != other_kind:
                continue
            if not adjacent(candidate, anchor):
                continue
            union = _candidate_topology(
                dict(
                    kind=candidate["kind"],
                    boxes=candidate["boxes"] + anchor["boxes"],
                    topology_admitted=True,
                ),
                context,
            )
            if union["eligible"]:
                checks[index] = {
                    **checks[index],
                    "eligible": True,
                    "direct_context_eligible": False,
                    "reason": "adjacent-opposite-shoulder-of-direct-context-anchor",
                    "anchor_candidate_index": anchor_index,
                    "combined_context_check": union,
                }
                break
    return checks


def _filter_context_proposals(original, context, document):
    """Pure metadata gate, also usable to audit already-frozen proposals."""
    from .automatic_metering_v31 import _boxes

    xs, ys = original["grid_x"], original["grid_y"]
    ix = {v: i for i, v in enumerate(xs)}
    iy = {v: i for i, v in enumerate(ys)}
    selected = np.zeros((len(ys) - 1, len(xs) - 1), bool)
    proposed = np.zeros_like(selected)
    observations = []
    checks = _context_checks(original["candidates"], context, xs, ys)
    for candidate, check in zip(original["candidates"], checks, strict=True):
        observations.append({**deepcopy(candidate), "source_context_check": check})
        if not candidate["topology_admitted"]:
            continue
        for a, b, c, d in candidate["boxes"]:
            proposed[iy[b] : iy[d], ix[a] : ix[c]] = True
            if check["eligible"]:
                selected[iy[b] : iy[d], ix[a] : ix[c]] = True
    if np.any(selected & ~proposed):
        raise ValueError("Context gate invented support outside V31 proposals")
    area = np.outer(np.diff(ys), np.diff(xs))
    fraction = float(area[selected].sum() / area.sum())
    boxes = _boxes(selected, xs, ys)
    fallback = (
        "area-budget-exceeded"
        if fraction > document["max_fraction"]
        else "contour-box-budget-exceeded"
        if len(boxes) > 64
        else None
    )
    action = document.get("action", "exclude")
    applied = action == "exclude" and fallback is None
    return {
        **original,
        "schema": "source-context-checked-density-v32",
        "version": 2,
        "context_gate_policy": deepcopy(CONTEXT_POLICY),
        "document": document,
        "source_context_roi": list(context),
        "context_kind": "caller-context"
        if document["context_roi"]
        else "prepared-source-canvas-assumption",
        "candidates": observations,
        "upstream_candidate_exclusions": deepcopy(original["candidate_exclusions"]),
        "upstream_candidate_area_fraction": original["candidate_area_fraction"],
        "candidate_exclusions": boxes,
        "candidate_area_fraction": fraction,
        "automatic_exclusions": boxes if applied else [],
        "automatic_excluded_fraction": fraction if applied else 0.0,
        "fallback": fallback,
        "application_action": action,
        "application_status": "suggestions-only"
        if action == "suggest"
        else "budget-fallback"
        if fallback
        else "automatic-exclusions-applied"
        if boxes
        else "no-automatic-exclusions",
        "physical_health_assessed": False,
        "physical_film_boundary_inferred": False,
        "physical_defect_classification_proven": False,
        "scope": "Subset of V31 proposals validated against a distinct source context; natural edge objects can still match",
    }


def detect_context_avoidance(source, roi, document):
    from .adaptive_edge_v31 import detect_adaptive_edge

    document = checked_context_avoidance(document)
    if document is None:
        raise ValueError("Active context avoidance required")
    if (
        not isinstance(source, np.ndarray)
        or source.ndim != 3
        or source.shape[-1] != 3
        or source.dtype not in (np.float32, np.float64)
        or not source.flags.c_contiguous
        or source.flags.writeable
    ):
        raise ValueError("Context avoidance requires readonly contiguous floating RGB")
    roi = _rectangle(roi, source.shape, "Metering ROI")
    context = _rectangle(
        document["context_roi"] or [0, 0, source.shape[1], source.shape[0]],
        source.shape,
        "Source context ROI",
    )
    if not (
        context[0] <= roi[0] < roi[2] <= context[2]
        and context[1] <= roi[1] < roi[3] <= context[3]
    ):
        raise ValueError("Source context must contain the metering ROI")
    digest = source_digest(source)
    legacy = {
        k: deepcopy(v)
        for k, v in document.items()
        if k not in {"context_roi", "action"}
    }
    legacy.update(version=1, mode="adaptive-edge")
    original = detect_adaptive_edge(source, roi, legacy)
    result = _filter_context_proposals(original, context, document)
    if source_digest(source) != digest:
        raise ValueError("Source changed during context check")
    return result
