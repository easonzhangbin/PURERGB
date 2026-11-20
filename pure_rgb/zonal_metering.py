"""Independent experimental density window from spatially shared leaf medians.

This opt-in provider does not identify defects, crop/repair the image, replace
an existing density provider, or integrate Native's auxiliary exposure meters.
Confirmed source-coordinate exclusions alone remove support. Scene brightness
and colour never change a retained leaf's original pixel-area weight.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from .density_matrix import DensityWindow
from .engines import Analysis
from .metering_regions import MAX_EXCLUSIONS, _integer, _rectangle


PROVIDER = "zonal-log-block-median-v1"
LOG_FLOOR = 1e-6
MAX_LEAVES = 65536


@dataclass(frozen=True)
class ZonalMeteringResult:
    window: DensityWindow
    receipt: dict


def _pair(value, name):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(_integer(item) and item > 0 for item in value)
    ):
        raise ValueError(f"{name} must be positive integer (rows, columns)")
    return tuple(int(item) for item in value)


def _edges(low, high, count):
    return tuple(low + (high - low) * i // count for i in range(count + 1))


def _intersects(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _union_area(rectangles):
    """Exact rectangle union via x strips and merged y intervals, no pixel mask."""
    xs = sorted({coordinate for box in rectangles for coordinate in (box[0], box[2])})
    area = 0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted(
            (y0, y1) for x0, y0, x1, y1 in rectangles if x0 <= left and right <= x1
        )
        covered, previous_high = 0, None
        for low, high in intervals:
            if previous_high is None or low > previous_high:
                covered += high - low
            elif high > previous_high:
                covered += high - previous_high
            previous_high = high if previous_high is None else max(previous_high, high)
        area += (right - left) * covered
    return area


def _masked_excluded_area(mask, roi, rectangles):
    """Count selected pixels in the exclusion union without another ROI raster."""
    xs = sorted({v for box in rectangles for v in (box[0], box[2])})
    area = 0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted(
            (y0, y1) for x0, y0, x1, y1 in rectangles if x0 <= left and right <= x1
        )
        merged = []
        for low, high in intervals:
            if not merged or low > merged[-1][1]:
                merged.append([low, high])
            else:
                merged[-1][1] = max(merged[-1][1], high)
        for low, high in merged:
            area += int(
                np.count_nonzero(
                    mask[low - roi[1] : high - roi[1], left - roi[0] : right - roi[0]]
                )
            )
    return area


def _weighted_inverse_cdf(values, weights, numerator, denominator):
    """No interpolation: lowest value reaching ceil(p * original pixel area).

    p is an exact rational. For p=0 the minimum value is returned. Sorting is
    channel-wise, but every RGB channel uses the identical leaves and weights.
    """
    total = sum(int(weight) for weight in weights)
    threshold = max(1, (total * numerator + denominator - 1) // denominator)
    result = []
    weights = np.asarray(weights, dtype=np.int64)
    for channel in range(3):
        order = np.argsort(values[:, channel], kind="stable")
        cumulative = np.cumsum(weights[order], dtype=np.int64)
        index = int(np.searchsorted(cumulative, threshold, side="left"))
        result.append(float(values[order[index], channel]))
    return result


def _summary(values, weights):
    if not weights:
        return {
            "median_log_rgb": None,
            "low_log_rgb": None,
            "high_log_rgb": None,
            "observed_leaf_log_range_rgb": None,
            "channel_differences_advisory": {
                "R_minus_G": None,
                "B_minus_G": None,
                "automatic_weight_adjustment": False,
            },
        }
    values = np.asarray(values, dtype=np.float64)
    median = _weighted_inverse_cdf(values, weights, 1, 2)
    return {
        "median_log_rgb": median,
        "low_log_rgb": _weighted_inverse_cdf(values, weights, 1, 200),
        "high_log_rgb": _weighted_inverse_cdf(values, weights, 199, 200),
        "observed_leaf_log_range_rgb": [
            values.min(axis=0).tolist(),
            values.max(axis=0).tolist(),
        ],
        "channel_differences_advisory": {
            "R_minus_G": median[0] - median[1],
            "B_minus_G": median[2] - median[1],
            "automatic_weight_adjustment": False,
            "scope": "Signal differences may reflect scene, film and illumination; not a defect classification",
        },
    }


def _measure_leaf(source, box, leaf_mask, selected_area, area):
    from .film_density_domain_v45 import film_domain_for

    domain = film_domain_for(source)
    samples = np.asarray(source[box[1]:box[3], box[0]:box[2]], dtype=np.float64)
    original = (np.asarray(domain.original[box[1]:box[3], box[0]:box[2]], dtype=np.float64)
                if domain is not None else samples)
    axes = (0, 1)
    if leaf_mask is not None and selected_area != area:
        samples = samples[leaf_mask]
        original = original[leaf_mask]
        axes = 0
    nonpositive = np.array([
        np.count_nonzero(original[..., channel] <= 0) for channel in range(3)
    ], dtype=np.int64)
    below_floor = np.array([
        np.count_nonzero(original[..., channel] < LOG_FLOOR) for channel in range(3)
    ], dtype=np.int64)
    logs = samples if domain is not None else np.log10(np.maximum(samples, LOG_FLOOR))
    median = np.median(logs, axis=axes).tolist()
    return nonpositive, below_floor, median


def analyze_zonal_density(
    image,
    roi,
    excluded_rois=(),
    *,
    zones=(8, 12),
    blocks_per_zone=(4, 4),
    padding=0,
    min_area_fraction=0.25,
    selection_mask=None,
):
    """Pool all retained log-median leaves by original pixel area.

    Zone and leaf edges use floor(i*length/count). Every ROI pixel belongs to
    exactly one leaf before exclusions; an intersecting confirmed rectangle
    rejects that whole leaf conservatively. No channel-specific support,
    brightness/colour rejection, per-zone endpoint averaging, or sample packing
    into a fake image is used. An optional vector-defined selection contributes
    only its true original pixels within each leaf, weighted by their count.
    Rectangular confirmed exclusions still reject any intersecting whole leaf.
    The resulting single window is density-only.
    """
    source = np.asarray(image)
    if (
        source.ndim != 3
        or source.shape[2] != 3
        or not source.size
        or source.dtype.kind != "f"
    ):
        raise ValueError("Expected nonempty floating-point HWC linear source bands")
    from .immutable_source_v44 import certificate_for

    if certificate_for(source) is None and not np.isfinite(source).all():
        raise ValueError("Zonal source must contain only finite samples")
    shape = tuple(int(value) for value in source.shape)
    requested_roi = _rectangle(roi, shape, "Zonal analysis ROI")
    mask = None
    mask_receipt = None
    if selection_mask is not None:
        from .metering_masks import validate_metering_mask

        mask = validate_metering_mask(selection_mask, shape, requested_roi)
        mask_receipt = {
            "mask_sha256": selection_mask.mask_sha256,
            "spec": deepcopy(selection_mask.spec),
            "selected_pixel_area": int(np.count_nonzero(mask)),
            "RGB_support_shared": True,
        }
    zone_rows, zone_columns = zones = _pair(zones, "Zones")
    block_rows, block_columns = blocks_per_zone = _pair(
        blocks_per_zone, "Blocks per zone"
    )
    leaf_count = zone_rows * zone_columns * block_rows * block_columns
    if leaf_count > MAX_LEAVES:
        raise ValueError(f"Zonal study is limited to {MAX_LEAVES} leaves")
    if not _integer(padding) or padding < 0:
        raise ValueError("Exclusion padding must be a nonnegative integer")
    if (
        isinstance(min_area_fraction, (bool, np.bool_))
        or not isinstance(min_area_fraction, (float, int, np.floating, np.integer))
        or not math.isfinite(min_area_fraction)
        or not 0 < min_area_fraction <= 1
    ):
        raise ValueError("Minimum retained area fraction must lie in (0,1]")
    if (
        not isinstance(excluded_rois, (list, tuple))
        or len(excluded_rois) > MAX_EXCLUSIONS
    ):
        raise ValueError(
            f"Provide at most {MAX_EXCLUSIONS} confirmed exclusion rectangles"
        )
    padding = int(padding)
    rx0, ry0, rx1, ry1 = requested_roi
    if ry1 - ry0 < zone_rows * block_rows or rx1 - rx0 < zone_columns * block_columns:
        raise ValueError("ROI resolution cannot support nonempty zones and leaf blocks")
    requested_exclusions = tuple(
        sorted(set(_rectangle(box, shape, "Excluded ROI") for box in excluded_rois))
    )
    effective = []
    for x0, y0, x1, y1 in requested_exclusions:
        box = (
            max(rx0, x0 - padding),
            max(ry0, y0 - padding),
            min(rx1, x1 + padding),
            min(ry1, y1 + padding),
        )
        if box[0] < box[2] and box[1] < box[3]:
            effective.append(box)
    effective = tuple(sorted(set(effective)))
    total_area = (rx1 - rx0) * (ry1 - ry0)
    available_area = (
        total_area - _union_area(effective)
        if mask is None
        else int(np.count_nonzero(mask))
        - _masked_excluded_area(mask, requested_roi, effective)
    )
    leaf_records, zone_records, all_logs, all_weights = [], [], [], []
    nonpositive = np.zeros(3, dtype=np.int64)
    below_floor = np.zeros(3, dtype=np.int64)
    ys, xs = _edges(ry0, ry1, zone_rows), _edges(rx0, rx1, zone_columns)
    certificate = certificate_for(source)
    pending_leaves = {}
    if certificate is not None:
        # Leaves are independent original-pixel statistics. Compute them in
        # parallel, then consume in the unchanged row/zone order below.
        for zr in range(zone_rows):
            for zc in range(zone_columns):
                by = _edges(ys[zr], ys[zr + 1], block_rows)
                bx = _edges(xs[zc], xs[zc + 1], block_columns)
                for br in range(block_rows):
                    for bc in range(block_columns):
                        box = (bx[bc], by[br], bx[bc + 1], by[br + 1])
                        if any(_intersects(box, bad) for bad in effective):
                            continue
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        leaf = None if mask is None else mask[box[1]-ry0:box[3]-ry0, box[0]-rx0:box[2]-rx0]
                        selected = area if leaf is None else int(np.count_nonzero(leaf))
                        if not selected:
                            continue
                        support = leaf if selected != area else None
                        pending_leaves[box] = certificate.leaf_statistics(
                            box, support,
                            lambda box=box, leaf=leaf, selected=selected, area=area:
                                _measure_leaf(source, box, leaf, selected, area),
                            policy=("original-f64-log-median-v1", float(LOG_FLOOR), np.__version__),
                        )
    for zr in range(zone_rows):
        for zc in range(zone_columns):
            zone_box = (xs[zc], ys[zr], xs[zc + 1], ys[zr + 1])
            zone_logs, zone_weights = [], []
            by, bx = (
                _edges(ys[zr], ys[zr + 1], block_rows),
                _edges(xs[zc], xs[zc + 1], block_columns),
            )
            for br in range(block_rows):
                for bc in range(block_columns):
                    box = (bx[bc], by[br], bx[bc + 1], by[br + 1])
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    excluded = any(_intersects(box, bad) for bad in effective)
                    leaf_mask = (
                        None
                        if mask is None
                        else mask[
                            box[1] - ry0 : box[3] - ry0, box[0] - rx0 : box[2] - rx0
                        ]
                    )
                    selected_area = (
                        area if leaf_mask is None else int(np.count_nonzero(leaf_mask))
                    )
                    supported = 0 if excluded else selected_area
                    record = {
                        "id": f"Z{zr:02d}_{zc:02d}_L{br:02d}_{bc:02d}",
                        "zone_index": [zr, zc],
                        "block_index": [br, bc],
                        "source_xyxy": list(box),
                        "pixel_area": area,
                        "valid": bool(supported),
                        "excluded_intersection": excluded,
                        "median_log_rgb": None,
                    }
                    if mask is not None:
                        record.update(
                            mask_pixel_area=selected_area,
                            weight_pixel_area=supported,
                            supported_pixel_area=supported,
                        )
                    if supported:
                        statistics = (pending_leaves[box].result() if certificate is not None
                                      else _measure_leaf(source, box, leaf_mask, selected_area, area))
                        nonpositive += statistics[0]
                        below_floor += statistics[1]
                        median = list(statistics[2])
                        record["median_log_rgb"] = median
                        zone_logs.append(median)
                        zone_weights.append(supported)
                        all_logs.append(median)
                        all_weights.append(supported)
                    leaf_records.append(record)
            zone_area = (zone_box[2] - zone_box[0]) * (zone_box[3] - zone_box[1])
            zone_supported = sum(zone_weights)
            zone_records.append(
                {
                    "zone_index": [zr, zc],
                    "source_xyxy": list(zone_box),
                    "total_pixel_area": zone_area,
                    "supported_pixel_area": zone_supported,
                    "total_leaf_count": block_rows * block_columns,
                    "valid_leaf_count": len(zone_weights),
                    "retained_fraction": zone_supported / zone_area,
                    **_summary(zone_logs, zone_weights),
                }
            )
    supported_area = sum(all_weights)
    if not supported_area or supported_area / total_area < min_area_fraction:
        raise ValueError("Insufficient retained zonal metering coverage")
    global_summary = _summary(all_logs, all_weights)
    dense, thin = (
        np.asarray(global_summary["low_log_rgb"]),
        np.asarray(global_summary["high_log_rgb"]),
    )
    if np.any(thin <= dense):
        raise ValueError("Insufficient per-channel zonal density dynamic range")
    with np.errstate(over="ignore", invalid="ignore"):
        dark, base = np.power(10.0, dense), np.power(10.0, thin)
    try:
        analysis = Analysis(
            tuple(float(v) for v in dark),
            tuple(float(v) for v in base),
            requested_roi,
            (0.5, 99.5),
        )
    except ValueError as error:
        raise ValueError("Insufficient or nonfinite zonal signal endpoints") from error
    support_document = {
        "source_shape": list(shape),
        "requested_roi": list(requested_roi),
        "leaves": [
            [r["source_xyxy"], r["pixel_area"], r["valid"]] for r in leaf_records
        ],
    }
    if mask_receipt is not None:
        # The mask digest binds the vector request as well as every selected
        # pixel, so equal per-leaf counts cannot alias different spatial sets.
        support_document["selection_mask_sha256"] = mask_receipt["mask_sha256"]
        support_document["selected_leaf_weights"] = [
            r["weight_pixel_area"] for r in leaf_records
        ]
        validate_metering_mask(selection_mask, shape, requested_roi)
    support_digest = hashlib.sha256(
        json.dumps(support_document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt = {
        "schema": "pure-rgb-zonal-log-block-density",
        "version": 1,
        "provider": PROVIDER,
        "source_shape": list(shape),
        "requested_roi": list(requested_roi),
        "requested_excluded_rois": [list(box) for box in requested_exclusions],
        "effective_excluded_rois": [list(box) for box in effective],
        "padding": padding,
        "zones": list(zones),
        "blocks_per_zone": list(blocks_per_zone),
        "total_pixel_area": total_area,
        "available_pixel_area": available_area,
        "supported_pixel_area": supported_area,
        "available_fraction": available_area / total_area,
        "retained_fraction": supported_area / total_area,
        "min_area_fraction": float(min_area_fraction),
        "valid_leaf_count": len(all_weights),
        "total_leaf_count": leaf_count,
        "support_signature_sha256": support_digest,
        "selection_mask": mask_receipt,
        "leaf_area_weight_field": "pixel_area" if mask is None else "weight_pixel_area",
        "selection_rule": "Whole original leaf"
        if mask is None
        else "Only true original pixel centres within each leaf; partial leaf allowed; no feathering",
        "confirmed_rectangle_exclusion_rule": "Any intersection rejects the whole leaf, including with a selection mask",
        "leaf_records": leaf_records,
        "zone_records": zone_records,
        "global_log_distribution": global_summary,
        "log_floor": LOG_FLOOR,
        "retained_nonpositive_sample_count_rgb": nonpositive.tolist(),
        "retained_below_log_floor_sample_count_rgb": below_floor.tolist(),
        "leaf_statistic": "float64 median of log10(max(original signal,1e-6)); log precedes median, including even-sized leaves",
        "pooling": "All retained leaf logRGB medians; each channel uses the same spatial support and count of original pixels actually used in that leaf",
        "quantile_definition": "Weighted inverse CDF without interpolation: first cumulative integer pixel area >=ceil(p*total); p=1/200,1/2,199/200 computed with integer rational thresholds",
        "shared_RGB_support_and_area_weights": True,
        "per_channel_quantiles_need_not_come_from_same_leaf": True,
        "original_NegPy_same_pixel_bound_rule": False,
        "zone_endpoint_averaging": False,
        "automatic_color_or_brightness_rejection": False,
        "automatic_defect_classification": False,
        "density_only": True,
        "native_auxiliary_meter_masks_integrated": False,
        "source_values_changed": False,
        "output_crop_changed": False,
        "original_provider_or_three_engine_defaults_replaced": False,
        "scope": "Independent experimental density window only; no factory matrix metering, end-to-end rendering optimization or physical colour calibration claim",
    }
    from .film_density_domain_v45 import LEAF_STATISTIC, film_domain_for

    if film_domain_for(source) is not None:
        receipt["leaf_statistic"] = LEAF_STATISTIC
    window = DensityWindow(
        PROVIDER,
        analysis,
        tuple(float(v) for v in dense),
        tuple(float(v) for v in thin),
        {"zonal_metering": deepcopy(receipt)},
    )
    return ZonalMeteringResult(window, receipt)
