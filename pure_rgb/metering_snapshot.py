"""Source-bound persistence for the opt-in experimental zonal density meter.

A checksum detects a changed document; it is not an authenticity signature.
Callers must obtain current_identity from the current verified source context.
Restoration validates stored geometry/support and reconstructs a vector mask,
but never reads RAW, reads source pixels, or measures density again. Native's
auxiliary meters and all existing providers/defaults remain unconnected.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

import numpy as np

from .density_matrix import DensityWindow
from .engines import Analysis
from .metering_masks import mask_from_spec, validate_metering_mask
from .zonal_metering import (
    LOG_FLOOR,
    MAX_LEAVES,
    PROVIDER,
    _edges,
    _intersects,
    _masked_excluded_area,
    _summary,
    _union_area,
    analyze_zonal_density,
)


SCHEMA = "pure-rgb-source-bound-zonal-metering"
VERSION = 1
SCOPE = "density-window-only-v1"
COORDINATE_SPACE = "oriented-merged-source-local-half-open-xyxy"
SOURCE_DOMAIN = "linear-selected-sensor-bands"
MAX_BUNDLE_BYTES = 32 * 1024 * 1024
IDENTITY_FIELDS = {
    "policy",
    "shape",
    "dtype",
    "source_pixels_sha256",
    "geometry_token",
    "coordinate_space",
    "domain",
    "channel_order",
    "source_read_only",
}
ALGORITHM = {
    "provider": PROVIDER,
    "version": 1,
    "log_floor": LOG_FLOOR,
    "quantile_definition_version": "integer-area-inverse-cdf-1-200-v1",
    "input_log_policy": "log10(max(T,1e-6)); no high clipping",
    "original_NegPy_equivalent": False,
}
RECEIPT_TEXT = {
    "leaf_statistic": "float64 median of log10(max(original signal,1e-6)); log precedes median, including even-sized leaves",
    "pooling": "All retained leaf logRGB medians; each channel uses the same spatial support and count of original pixels actually used in that leaf",
    "quantile_definition": "Weighted inverse CDF without interpolation: first cumulative integer pixel area >=ceil(p*total); p=1/200,1/2,199/200 computed with integer rational thresholds",
    "confirmed_rectangle_exclusion_rule": "Any intersection rejects the whole leaf, including with a selection mask",
    "scope": "Independent experimental density window only; no factory matrix metering, end-to-end rendering optimization or physical colour calibration claim",
}


def _domain_policy(identity):
    from .film_density_domain_v45 import DOMAIN, LEAF_STATISTIC, LOG_POLICY

    algorithm, text = deepcopy(ALGORITHM), deepcopy(RECEIPT_TEXT)
    if identity["domain"] == DOMAIN:
        algorithm["input_log_policy"] = LOG_POLICY
        text["leaf_statistic"] = LEAF_STATISTIC
    return algorithm, text
CONFIG_FIELDS = {
    "roi",
    "excluded_rois",
    "zones",
    "blocks_per_zone",
    "padding",
    "min_area_fraction",
}
RECEIPT_FIELDS = {
    "schema",
    "version",
    "provider",
    "source_shape",
    "requested_roi",
    "requested_excluded_rois",
    "effective_excluded_rois",
    "padding",
    "zones",
    "blocks_per_zone",
    "total_pixel_area",
    "available_pixel_area",
    "supported_pixel_area",
    "available_fraction",
    "retained_fraction",
    "min_area_fraction",
    "valid_leaf_count",
    "total_leaf_count",
    "support_signature_sha256",
    "selection_mask",
    "leaf_area_weight_field",
    "selection_rule",
    "confirmed_rectangle_exclusion_rule",
    "leaf_records",
    "zone_records",
    "global_log_distribution",
    "log_floor",
    "retained_nonpositive_sample_count_rgb",
    "retained_below_log_floor_sample_count_rgb",
    "leaf_statistic",
    "pooling",
    "quantile_definition",
    "shared_RGB_support_and_area_weights",
    "per_channel_quantiles_need_not_come_from_same_leaf",
    "original_NegPy_same_pixel_bound_rule",
    "zone_endpoint_averaging",
    "automatic_color_or_brightness_rejection",
    "automatic_defect_classification",
    "density_only",
    "native_auxiliary_meter_masks_integrated",
    "source_values_changed",
    "output_crop_changed",
    "original_provider_or_three_engine_defaults_replaced",
    "scope",
}


@dataclass(frozen=True)
class FrozenMeteringBundle:
    """Immutable canonical JSON; no mutable receipt or array is retained."""

    canonical_json: str
    sha256: str

    def to_json(self):
        return self.canonical_json


@dataclass(frozen=True)
class RestoredMeteringBundle:
    window: DensityWindow
    receipt: dict
    selection_mask: object | None
    source_identity: dict
    config: dict
    sha256: str


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _keys(value, fields, name):
    _require(
        isinstance(value, dict) and set(value) == fields, f"Malformed {name} fields"
    )


def _json_values(value, depth=0):
    _require(depth <= 32, "Metering JSON is nested too deeply")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        _require(math.isfinite(value), "Metering JSON contains a nonfinite number")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _json_values(item, depth + 1)
        return
    if isinstance(value, dict) and all(type(key) is str for key in value):
        for item in value.values():
            _json_values(item, depth + 1)
        return
    raise ValueError("Metering JSON contains a non-JSON value")


def _sha(value, name):
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
        f"Invalid {name}",
    )
    return value


def _shape(value):
    _require(
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(type(v) is int and v > 0 for v in value)
        and value[2] == 3,
        "Expected positive integer HWC RGB source shape",
    )
    return list(value)


def _roi(value, shape, name):
    _require(
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(type(v) is int for v in value),
        f"Invalid {name}",
    )
    x0, y0, x1, y1 = value
    _require(0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0], f"Invalid {name}")
    return list(value)


def _pair(value, name):
    _require(
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(type(v) is int and v > 0 for v in value),
        f"Invalid {name}",
    )
    return list(value)


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _triple(value, name):
    _require(
        isinstance(value, list) and len(value) == 3 and all(_number(v) for v in value),
        f"Invalid {name}",
    )
    return tuple(float(v) for v in value)


def _identity(value):
    _keys(value, IDENTITY_FIELDS, "source identity")
    _json_values(value)
    result = deepcopy(value)
    result["shape"] = _shape(value["shape"])
    policy = value["policy"]
    _require(
        isinstance(policy, str)
        and len(policy) <= 128
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*-v[1-9][0-9]*", policy),
        "Source policy must be an explicit versioned label",
    )
    _require(isinstance(value["dtype"], str), "Invalid source dtype")
    try:
        dtype = np.dtype(value["dtype"])
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid source dtype") from error
    _require(
        dtype.kind == "f" and dtype.itemsize in (4, 8),
        "Only float32/float64 source bytes are supported",
    )
    result["dtype"] = dtype.str
    _sha(value["source_pixels_sha256"], "source pixel SHA")
    _sha(value["geometry_token"], "geometry token")
    _require(
        value["coordinate_space"] == COORDINATE_SPACE,
        "Unsupported source coordinate space",
    )
    from .film_density_domain_v45 import DOMAIN

    _require(value["domain"] in (SOURCE_DOMAIN, DOMAIN), "Unsupported source signal domain")
    _require(value["domain"] != DOMAIN or dtype == np.dtype(np.float32),
             "SDK film log density requires float32 source bytes")
    _require(value["channel_order"] == "RGB", "Unsupported source channel order")
    _require(
        value["source_read_only"] is True,
        "Source identity must declare read-only pixels",
    )
    return result


def identity_from_import_receipt(receipt):
    """Adapt an explicit standard import receipt, never a NPY/file checksum.

    The receipt alone is not proof that current pixels are these pixels:
    freeze_zonal_metering additionally measures the supplied array's byte SHA.
    Legacy caches and generated fixtures must supply their own explicit policy.
    The CFA-DNG branch validates stored metadata without opening the DNG or
    authenticating that the file came directly from its claimed camera.
    """
    _require(isinstance(receipt, dict), "Expected standard RAW import receipt")
    _json_values(receipt)
    _require(
        receipt.get("policy") in ("standard-crop-v1", "standard-cfa-dng-v1")
        and type(receipt.get("version")) is int
        and receipt["version"] == 1,
        "Expected an explicit standard RAW import receipt",
    )
    if receipt["policy"] == "standard-cfa-dng-v1":
        from .raw_dng_import import validate_dng_import_receipt_metadata

        validate_dng_import_receipt_metadata(receipt)
    registration = receipt.get("registration")
    if isinstance(registration, dict) and (
        registration.get("version") in (2, 3) or "model_check" in registration
    ):
        from .raw_registration import validate_registration_model_metadata

        validate_registration_model_metadata(registration)
    _require(receipt.get("source_read_only") is True, "Import receipt is not read-only")
    shape = _shape(receipt.get("source_shape"))
    geometry = receipt.get("effective_geometry")
    _require(
        isinstance(geometry, dict) and geometry.get("source_shape") == shape,
        "Import geometry/source shape mismatch",
    )
    fields = ("bands", "registration", "effective_geometry")
    _require(
        all(field in receipt for field in fields), "Missing import geometry evidence"
    )
    # Match the existing raw_import token convention (default JSON separators).
    token = hashlib.sha256(
        json.dumps(
            {key: receipt[key] for key in fields}, sort_keys=True, allow_nan=False
        ).encode()
    ).hexdigest()
    _require(receipt.get("geometry_token") == token, "Import geometry token mismatch")
    decode = receipt.get("decode")
    _require(
        isinstance(decode, dict)
        and decode.get("linear_raw") is True
        and decode.get("neutral_WB") is True
        and decode.get("half_size") is False,
        "Import receipt does not describe full-size linear neutral-WB sensor bands",
    )
    _require(
        receipt.get("raw_files_modified") is False
        and receipt.get("LibRaw_decoder_patched") is False,
        "Unsupported modified RAW import receipt",
    )
    return _identity(
        {
            "policy": receipt["policy"],
            "shape": shape,
            "dtype": receipt.get("source_dtype"),
            "source_pixels_sha256": receipt.get("source_pixels_sha256"),
            "geometry_token": token,
            "coordinate_space": COORDINATE_SPACE,
            "domain": SOURCE_DOMAIN,
            "channel_order": "RGB",
            "source_read_only": True,
        }
    )


def _source_sha(source):
    from .immutable_source_v44 import source_digest

    return source_digest(source)


def _check_source(source, identity):
    _require(isinstance(source, np.ndarray), "Expected original source ndarray")
    _require(
        source.flags.c_contiguous and not source.flags.writeable,
        "Source must be C-contiguous and read-only",
    )
    _require(
        list(source.shape) == identity["shape"]
        and source.dtype.str == identity["dtype"],
        "Source shape/dtype identity mismatch",
    )
    from .film_density_domain_v45 import DOMAIN, film_domain_for

    domain = film_domain_for(source)
    _require((identity["domain"] == DOMAIN) == (domain is not None),
             "Source bytes do not belong to the declared density domain")
    if domain is not None:
        domain.validate()
    from .immutable_source_v44 import certificate_for

    certificate = certificate_for(source)
    if certificate is not None:
        certificate.validate(source, identity)
        return
    _require(np.isfinite(source).all(), "Source contains nonfinite values")
    _require(
        _source_sha(source) == identity["source_pixels_sha256"],
        "Source pixel SHA mismatch",
    )


def _config(value, shape):
    _keys(value, CONFIG_FIELDS, "metering configuration")
    roi = _roi(value["roi"], shape, "metering ROI")
    exclusions = value["excluded_rois"]
    _require(
        isinstance(exclusions, list) and len(exclusions) <= 64, "Invalid exclusion list"
    )
    boxes = [_roi(box, shape, "excluded ROI") for box in exclusions]
    _require(
        boxes == [list(box) for box in sorted(set(map(tuple, boxes)))],
        "Exclusion list is not canonical",
    )
    zones = _pair(value["zones"], "zones")
    blocks = _pair(value["blocks_per_zone"], "blocks per zone")
    count = math.prod(zones) * math.prod(blocks)
    _require(count <= MAX_LEAVES, "Too many metering leaves")
    _require(
        roi[3] - roi[1] >= zones[0] * blocks[0]
        and roi[2] - roi[0] >= zones[1] * blocks[1],
        "Empty metering grid cells",
    )
    _require(
        type(value["padding"]) is int and value["padding"] >= 0,
        "Invalid exclusion padding",
    )
    _require(
        _number(value["min_area_fraction"]) and 0 < value["min_area_fraction"] <= 1,
        "Invalid minimum coverage",
    )
    return roi, boxes, zones, blocks


def _restore_mask(document, shape, roi):
    if document is None:
        return None
    _keys(document, {"spec", "mask_sha256"}, "selection mask")
    _sha(document["mask_sha256"], "selection mask SHA")
    mask = mask_from_spec(document["spec"])
    validate_metering_mask(mask, tuple(shape), tuple(roi))
    _require(
        mask.mask_sha256 == document["mask_sha256"], "Selection mask digest mismatch"
    )
    return mask


def _support_receipt(receipt, config, shape, selection, identity):
    """Validate recorded support/statistics without reading any source pixels."""
    _keys(receipt, RECEIPT_FIELDS, "zonal receipt")
    roi, boxes, zones, blocks = _config(config, shape)
    _require(
        receipt["schema"] == "pure-rgb-zonal-log-block-density"
        and type(receipt["version"]) is int
        and receipt["version"] == 1
        and receipt["provider"] == PROVIDER,
        "Unsupported zonal receipt version",
    )
    links = {
        "source_shape": shape,
        "requested_roi": roi,
        "requested_excluded_rois": boxes,
        "zones": zones,
        "blocks_per_zone": blocks,
        "padding": config["padding"],
        "min_area_fraction": config["min_area_fraction"],
        "log_floor": LOG_FLOOR,
    }
    _require(
        all(
            _canonical(receipt[key]) == _canonical(value)
            for key, value in links.items()
        ),
        "Configuration/receipt mismatch",
    )
    true_flags = (
        "shared_RGB_support_and_area_weights",
        "per_channel_quantiles_need_not_come_from_same_leaf",
        "density_only",
    )
    false_flags = (
        "original_NegPy_same_pixel_bound_rule",
        "zone_endpoint_averaging",
        "automatic_color_or_brightness_rejection",
        "automatic_defect_classification",
        "native_auxiliary_meter_masks_integrated",
        "source_values_changed",
        "output_crop_changed",
        "original_provider_or_three_engine_defaults_replaced",
    )
    _require(
        all(receipt[key] is True for key in true_flags)
        and all(receipt[key] is False for key in false_flags),
        "Unsupported metering scope/flags",
    )
    _require(
        all(receipt[key] == value for key, value in _domain_policy(identity)[1].items()),
        "Recorded algorithm/scope description mismatch",
    )
    rx0, ry0, rx1, ry1 = roi
    effective = []
    for x0, y0, x1, y1 in boxes:
        pad = config["padding"]
        box = (
            max(rx0, x0 - pad),
            max(ry0, y0 - pad),
            min(rx1, x1 + pad),
            min(ry1, y1 + pad),
        )
        if box[0] < box[2] and box[1] < box[3]:
            effective.append(box)
    effective = sorted(set(effective))
    _require(
        receipt["effective_excluded_rois"] == [list(box) for box in effective],
        "Effective exclusions mismatch",
    )
    mask = None if selection is None else selection.mask
    mask_receipt = (
        None
        if selection is None
        else {
            "mask_sha256": selection.mask_sha256,
            "spec": selection.spec,
            "selected_pixel_area": int(np.count_nonzero(mask)),
            "RGB_support_shared": True,
        }
    )
    _require(
        _canonical(receipt["selection_mask"]) == _canonical(mask_receipt),
        "Mask/receipt mismatch",
    )
    weight_key = "pixel_area" if mask is None else "weight_pixel_area"
    _require(
        receipt["leaf_area_weight_field"] == weight_key, "Unsupported leaf weight field"
    )
    _require(
        receipt["selection_rule"]
        == (
            "Whole original leaf"
            if mask is None
            else "Only true original pixel centres within each leaf; partial leaf allowed; no feathering"
        ),
        "Recorded selection rule mismatch",
    )
    total = (rx1 - rx0) * (ry1 - ry0)
    available = (
        total - _union_area(effective)
        if mask is None
        else int(np.count_nonzero(mask)) - _masked_excluded_area(mask, roi, effective)
    )
    leaves, zone_records = receipt["leaf_records"], receipt["zone_records"]
    _require(
        isinstance(leaves, list)
        and len(leaves) == math.prod(zones) * math.prod(blocks),
        "Invalid leaf count",
    )
    _require(
        isinstance(zone_records, list) and len(zone_records) == math.prod(zones),
        "Invalid zone count",
    )
    source_leaves, all_logs, all_weights = [], [], []
    ys, xs = _edges(ry0, ry1, zones[0]), _edges(rx0, rx1, zones[1])
    leaf_index, zone_index = 0, 0
    for zr in range(zones[0]):
        for zc in range(zones[1]):
            zone_box = [xs[zc], ys[zr], xs[zc + 1], ys[zr + 1]]
            by, bx = (
                _edges(ys[zr], ys[zr + 1], blocks[0]),
                _edges(xs[zc], xs[zc + 1], blocks[1]),
            )
            logs, weights = [], []
            for br in range(blocks[0]):
                for bc in range(blocks[1]):
                    leaf = leaves[leaf_index]
                    fields = {
                        "id",
                        "zone_index",
                        "block_index",
                        "source_xyxy",
                        "pixel_area",
                        "valid",
                        "excluded_intersection",
                        "median_log_rgb",
                    }
                    if mask is not None:
                        fields |= {
                            "mask_pixel_area",
                            "weight_pixel_area",
                            "supported_pixel_area",
                        }
                    _keys(leaf, fields, "leaf record")
                    box = [bx[bc], by[br], bx[bc + 1], by[br + 1]]
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    excluded = any(_intersects(box, bad) for bad in effective)
                    selected = (
                        area
                        if mask is None
                        else int(
                            np.count_nonzero(
                                mask[
                                    box[1] - ry0 : box[3] - ry0,
                                    box[0] - rx0 : box[2] - rx0,
                                ]
                            )
                        )
                    )
                    weight = 0 if excluded else selected
                    expected = {
                        "id": f"Z{zr:02d}_{zc:02d}_L{br:02d}_{bc:02d}",
                        "zone_index": [zr, zc],
                        "block_index": [br, bc],
                        "source_xyxy": box,
                        "pixel_area": area,
                        "valid": bool(weight),
                        "excluded_intersection": excluded,
                    }
                    if mask is not None:
                        expected.update(
                            mask_pixel_area=selected,
                            weight_pixel_area=weight,
                            supported_pixel_area=weight,
                        )
                    _require(
                        all(
                            _canonical(leaf[key]) == _canonical(value)
                            for key, value in expected.items()
                        ),
                        "Leaf support/geometry mismatch",
                    )
                    if weight:
                        median = _triple(leaf["median_log_rgb"], "leaf log median")
                        _require(
                            all(v >= math.log10(LOG_FLOOR) for v in median),
                            "Leaf median below log floor",
                        )
                        logs.append(list(median))
                        weights.append(weight)
                        all_logs.append(list(median))
                        all_weights.append(weight)
                    else:
                        _require(
                            leaf["median_log_rgb"] is None,
                            "Unsupported leaf retains statistics",
                        )
                    source_leaves.append([box, area, bool(weight)])
                    leaf_index += 1
            zone_area = (zone_box[2] - zone_box[0]) * (zone_box[3] - zone_box[1])
            expected_zone = {
                "zone_index": [zr, zc],
                "source_xyxy": zone_box,
                "total_pixel_area": zone_area,
                "supported_pixel_area": sum(weights),
                "total_leaf_count": math.prod(blocks),
                "valid_leaf_count": len(weights),
                "retained_fraction": sum(weights) / zone_area,
                **_summary(logs, weights),
            }
            _require(
                _canonical(zone_records[zone_index]) == _canonical(expected_zone),
                "Recorded zone statistics/support mismatch",
            )
            zone_index += 1
    supported = sum(all_weights)
    _require(
        supported > 0 and supported / total >= config["min_area_fraction"],
        "Insufficient stored metering coverage",
    )
    counts = {
        "total_pixel_area": total,
        "available_pixel_area": available,
        "supported_pixel_area": supported,
        "available_fraction": available / total,
        "retained_fraction": supported / total,
        "valid_leaf_count": len(all_weights),
        "total_leaf_count": len(leaves),
    }
    _require(
        all(
            receipt[key] == value and type(receipt[key]) is type(value)
            for key, value in counts.items()
        ),
        "Recorded metering area/count mismatch",
    )
    # Only stored leaf summaries are checked, never original pixels or a meter.
    _require(
        _canonical(receipt["global_log_distribution"])
        == _canonical(_summary(all_logs, all_weights)),
        "Recorded global distribution mismatch",
    )
    support = {"source_shape": shape, "requested_roi": roi, "leaves": source_leaves}
    if selection is not None:
        support.update(
            selection_mask_sha256=selection.mask_sha256,
            selected_leaf_weights=[leaf["weight_pixel_area"] for leaf in leaves],
        )
    _require(
        receipt["support_signature_sha256"] == _digest(support),
        "Recorded support signature mismatch",
    )
    for key in (
        "retained_nonpositive_sample_count_rgb",
        "retained_below_log_floor_sample_count_rgb",
    ):
        _require(
            isinstance(receipt[key], list)
            and len(receipt[key]) == 3
            and all(type(v) is int and 0 <= v <= supported for v in receipt[key]),
            "Invalid retained signal counts",
        )
    _require(
        all(
            a <= b
            for a, b in zip(
                receipt["retained_nonpositive_sample_count_rgb"],
                receipt["retained_below_log_floor_sample_count_rgb"],
            )
        ),
        "Inconsistent log-floor counts",
    )


def _window(document, receipt, roi):
    _keys(document, {"provider", "analysis", "log_dense", "log_thin"}, "density window")
    _keys(
        document["analysis"], {"dark", "base", "roi", "percentiles"}, "frozen analysis"
    )
    analysis = document["analysis"]
    _require(
        document["provider"] == PROVIDER
        and _canonical(analysis["roi"]) == _canonical(roi)
        and analysis["percentiles"] == [0.5, 99.5],
        "Frozen analysis/provider mismatch",
    )
    dense, thin = (
        _triple(document["log_dense"], "dense log endpoint"),
        _triple(document["log_thin"], "thin log endpoint"),
    )
    _require(
        all(a < b for a, b in zip(dense, thin)), "Invalid frozen log dynamic range"
    )
    global_summary = receipt["global_log_distribution"]
    _require(
        list(dense) == global_summary["low_log_rgb"]
        and list(thin) == global_summary["high_log_rgb"],
        "Window/distribution mismatch",
    )
    dark, base = (
        _triple(analysis["dark"], "dark endpoint"),
        _triple(analysis["base"], "base endpoint"),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        _require(
            np.array_equal(dark, np.power(10.0, dense))
            and np.array_equal(base, np.power(10.0, thin)),
            "Signal/log endpoint mismatch",
        )
    return DensityWindow(
        PROVIDER,
        Analysis(dark, base, tuple(roi), (0.5, 99.5)),
        dense,
        thin,
        {"zonal_metering": deepcopy(receipt)},
    )


def _load_document(text):
    _require(
        isinstance(text, str) and len(text.encode("utf-8")) <= MAX_BUNDLE_BYTES,
        "Expected bounded metering JSON string",
    )

    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate metering JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    try:
        document = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Malformed metering JSON") from error
    _json_values(document)
    return document


def restore_zonal_metering(bundle_json, current_identity):
    from .immutable_source_v44 import certificate_for_identity

    certificate = certificate_for_identity(current_identity) if isinstance(current_identity, dict) else None
    if certificate is not None and type(bundle_json) is str:
        return certificate.restore_bundle(bundle_json, current_identity,
            lambda: _restore_zonal_metering_uncached(bundle_json, current_identity),
            policy=_canonical([SCHEMA, VERSION, SCOPE, ALGORITHM, RECEIPT_TEXT]))
    return _restore_zonal_metering_uncached(bundle_json, current_identity)


def _restore_zonal_metering_uncached(bundle_json, current_identity):
    """Restore against an identity from the current verified source context.

    Rebuilds vector geometry and checks recorded leaf summaries internally;
    does not read/measure a source, decode RAW, or invoke analyze_zonal_density.
    Each call returns independent dictionaries/mask arrays.
    """
    document = _load_document(bundle_json)
    _keys(
        document,
        {
            "schema",
            "version",
            "source_identity",
            "algorithm",
            "config",
            "window",
            "receipt",
            "selection_mask",
            "payload_sha256",
            "scope",
            "native_auxiliary_meter_masks_integrated",
        },
        "metering bundle",
    )
    _require(
        document["schema"] == SCHEMA
        and type(document["version"]) is int
        and document["version"] == VERSION,
        "Unsupported metering bundle version",
    )
    _require(
        document["scope"] == SCOPE
        and document["native_auxiliary_meter_masks_integrated"] is False,
        "Unsupported metering bundle scope",
    )
    checksum = _sha(document["payload_sha256"], "bundle checksum")
    payload = {key: value for key, value in document.items() if key != "payload_sha256"}
    _require(_digest(payload) == checksum, "Metering bundle checksum mismatch")
    identity = _identity(document["source_identity"])
    _require(
        _canonical(document["algorithm"]) == _canonical(_domain_policy(identity)[0]),
        "Unsupported metering algorithm configuration",
    )
    _require(
        document["source_identity"] == identity,
        "Stored source identity is not canonical",
    )
    _require(
        identity == _identity(current_identity),
        "Metering bundle belongs to a different source identity",
    )
    config, receipt = document["config"], document["receipt"]
    roi, _, _, _ = _config(config, identity["shape"])
    selection = _restore_mask(document["selection_mask"], identity["shape"], roi)
    _support_receipt(receipt, config, identity["shape"], selection, identity)
    window = _window(document["window"], receipt, roi)
    return RestoredMeteringBundle(
        window,
        deepcopy(receipt),
        selection,
        deepcopy(identity),
        deepcopy(config),
        checksum,
    )


def freeze_zonal_metering(
    source,
    identity,
    roi,
    excluded_rois=(),
    *,
    zones=(8, 12),
    blocks_per_zone=(4, 4),
    padding=0,
    min_area_fraction=0.25,
    selection_mask=None,
):
    """Measure once and bind the result to unchanged read-only source bytes."""
    identity = _identity(identity)
    _check_source(source, identity)
    try:
        result = analyze_zonal_density(
            source,
            roi,
            excluded_rois,
            zones=zones,
            blocks_per_zone=blocks_per_zone,
            padding=padding,
            min_area_fraction=min_area_fraction,
            selection_mask=selection_mask,
        )
    finally:
        _check_source(source, identity)
    receipt = deepcopy(result.receipt)
    window = result.window
    config = {
        "roi": receipt["requested_roi"],
        "excluded_rois": receipt["requested_excluded_rois"],
        "zones": receipt["zones"],
        "blocks_per_zone": receipt["blocks_per_zone"],
        "padding": receipt["padding"],
        "min_area_fraction": receipt["min_area_fraction"],
    }
    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "scope": SCOPE,
        "native_auxiliary_meter_masks_integrated": False,
        "source_identity": identity,
        "algorithm": _domain_policy(identity)[0],
        "config": config,
        "window": {
            "provider": window.provider,
            "analysis": asdict(window.analysis),
            "log_dense": list(window.log_dense),
            "log_thin": list(window.log_thin),
        },
        "receipt": receipt,
        "selection_mask": None
        if selection_mask is None
        else {
            "spec": deepcopy(selection_mask.spec),
            "mask_sha256": selection_mask.mask_sha256,
        },
    }
    checksum = _digest(payload)
    text = _canonical({**payload, "payload_sha256": checksum})
    # Use the same structural validation at creation and restore, without a
    # second measurement; canonical JSON also severs mutable object aliases.
    restore_zonal_metering(text, identity)
    return FrozenMeteringBundle(text, checksum)
