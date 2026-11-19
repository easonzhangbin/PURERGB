"""Opt-in sensitivity audit of an already frozen zonal density window.

Each probe removes one supported original zone from the recorded population,
then repeats the provider's exact area-weighted endpoint rule. No source pixels
are read, regridded, repaired, or excluded from the actual measurement. Stable
endpoints under these probes do not certify representative content or a clean
negative. A normal scene and a fault can have identical recorded observations.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from .metering_snapshot import FrozenMeteringBundle, restore_zonal_metering
from .zonal_metering import _weighted_inverse_cdf


PROVIDER = "frozen-zonal-endpoint-sensitivity-v1"
MAX_SUPPORTED_ZONES = 256
MAX_RECORDED_LEAVES = 8192


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class ReliabilityPolicy:
    # 0.05 log10 transmission is about 0.166 EV. This is a predeclared
    # engineering warning threshold, not a fitted film-defect probability.
    maximum_endpoint_delta_log10: float = 0.05
    minimum_supported_zones: int = 4


@dataclass(frozen=True)
class ReliabilityReport:
    canonical_json: str
    sha256: str

    def to_json(self):
        return self.canonical_json

    @property
    def receipt(self):
        """Every caller receives an independent, JSON-native copy."""
        return json.loads(self.canonical_json)


def _policy(value):
    if not isinstance(value, ReliabilityPolicy):
        raise ValueError("Expected an explicit ReliabilityPolicy")
    delta = value.maximum_endpoint_delta_log10
    if type(delta) not in (float, int) or not math.isfinite(delta) or delta <= 0:
        raise ValueError("Endpoint warning threshold must be finite and positive")
    zones = value.minimum_supported_zones
    if type(zones) is not int or not 2 <= zones <= MAX_SUPPORTED_ZONES:
        raise ValueError("Minimum supported zones must be an integer in2..256")
    return value


def _endpoints(values, weights):
    values = np.asarray(values, dtype=np.float64)
    return (
        _weighted_inverse_cdf(values, weights, 1, 200),
        _weighted_inverse_cdf(values, weights, 199, 200),
    )


def assess_metering_reliability(
    bundle_json, current_identity, *, policy=ReliabilityPolicy()
):
    """Report endpoint sensitivity; never return a replacement density window.

    ``current_identity`` must come from the caller's verified source context.
    Restore checks the original source/geometry/domain and full leaf/support
    receipt, but this metadata-only audit does not independently rehash pixels.
    Mask restoration can allocate its declared ROI bitmap; leaf calculations
    themselves are bounded to8192 leaves and256 supported-zone probes.
    """
    policy = _policy(policy)
    text = (
        bundle_json.to_json()
        if isinstance(bundle_json, FrozenMeteringBundle)
        else bundle_json
    )
    restored = restore_zonal_metering(text, current_identity)
    receipt = restored.receipt
    leaves = receipt["leaf_records"]
    if len(leaves) > MAX_RECORDED_LEAVES:
        raise ValueError("Reliability audit exceeds8192 recorded leaves")
    kept = [leaf for leaf in leaves if leaf["valid"]]
    weight_field = receipt["leaf_area_weight_field"]
    weights = [leaf[weight_field] for leaf in kept]
    values = [leaf["median_log_rgb"] for leaf in kept]
    zone_keys = [tuple(leaf["zone_index"]) for leaf in kept]
    zones = sorted(set(zone_keys))
    if len(zones) > MAX_SUPPORTED_ZONES:
        raise ValueError("Reliability audit exceeds256 supported zones")
    dense, thin = _endpoints(values, weights)
    if dense != list(restored.window.log_dense) or thin != list(
        restored.window.log_thin
    ):
        raise ValueError("Recorded leaves do not replay the frozen endpoints")
    span = np.asarray(thin) - dense
    original_area = receipt["total_pixel_area"]
    retained_area = sum(weights)
    minimum_fraction = restored.config["min_area_fraction"]
    probes = []
    for zone in zones:
        keep_indices = [index for index, item in enumerate(zone_keys) if item != zone]
        probe_weights = [weights[index] for index in keep_indices]
        remaining = sum(probe_weights)
        record = {
            "removed_zone_yx": list(zone),
            "removed_supported_pixel_area": retained_area - remaining,
            "removed_fraction_of_baseline_retained": (retained_area - remaining)
            / retained_area,
            "remaining_supported_pixel_area": remaining,
            "remaining_fraction_of_original_roi": remaining / original_area,
            "remaining_leaf_count": len(keep_indices),
            "state": "insufficient-support",
            "reason": None,
            "log_dense": None,
            "log_thin": None,
            "delta_log_dense_RGB": None,
            "delta_log_thin_RGB": None,
            "maximum_absolute_endpoint_delta_log10": None,
            "maximum_delta_fraction_of_baseline_span": None,
            "exceeds_warning_threshold": None,
        }
        if not remaining or remaining / original_area < minimum_fraction:
            record["reason"] = "remaining-area-below-original-provider-minimum"
        else:
            low, high = _endpoints(
                [values[index] for index in keep_indices], probe_weights
            )
            record["log_dense"], record["log_thin"] = low, high
            if any(h <= d for d, h in zip(low, high, strict=True)):
                record["reason"] = "remaining-channel-has-no-density-dynamic-range"
            else:
                low_delta, high_delta = np.asarray(low) - dense, np.asarray(high) - thin
                change = np.maximum(np.abs(low_delta), np.abs(high_delta))
                maximum = float(change.max())
                record.update(
                    state="measured",
                    delta_log_dense_RGB=low_delta.tolist(),
                    delta_log_thin_RGB=high_delta.tolist(),
                    maximum_absolute_endpoint_delta_log10=maximum,
                    maximum_delta_fraction_of_baseline_span=float(
                        np.max(change / span)
                    ),
                    exceeds_warning_threshold=maximum
                    > policy.maximum_endpoint_delta_log10,
                )
        probes.append(record)
    measured = [probe for probe in probes if probe["state"] == "measured"]
    failed = [probe for probe in probes if probe["state"] != "measured"]
    too_few = len(zones) < policy.minimum_supported_zones
    if too_few or failed:
        status = "insufficient-support"
    elif any(probe["exceeds_warning_threshold"] for probe in measured):
        status = "sensitive-to-zone-removal"
    else:
        status = "stable-under-listed-probes"
    influential = sorted(
        measured,
        key=lambda probe: (
            -probe["maximum_absolute_endpoint_delta_log10"],
            probe["removed_zone_yx"],
        ),
    )
    document = {
        "schema": "pure-rgb-frozen-zonal-reliability",
        "version": 1,
        "provider": PROVIDER,
        "status": status,
        "source_identity": restored.source_identity,
        "density_bundle_payload_sha256": restored.sha256,
        "original_support_signature_sha256": receipt["support_signature_sha256"],
        "selection_mask_sha256": None
        if receipt["selection_mask"] is None
        else receipt["selection_mask"]["mask_sha256"],
        "analysis_roi": list(restored.window.analysis.roi),
        "policy": asdict(policy),
        "policy_sha256": _digest(asdict(policy)),
        "threshold_operator": "Strictly greater than maximum_endpoint_delta_log10",
        "threshold_equivalent_absolute_EV": policy.maximum_endpoint_delta_log10
        / math.log10(2),
        "baseline": {
            "log_dense": dense,
            "log_thin": thin,
            "supported_pixel_area": retained_area,
            "total_roi_pixel_area": original_area,
            "supported_zone_count": len(zones),
            "supported_leaf_count": len(kept),
            "maximum_single_zone_fraction_of_retained_area": max(
                probe["removed_fraction_of_baseline_retained"] for probe in probes
            ),
            "retained_nonpositive_sample_count_rgb": receipt[
                "retained_nonpositive_sample_count_rgb"
            ],
            "retained_below_log_floor_sample_count_rgb": receipt[
                "retained_below_log_floor_sample_count_rgb"
            ],
        },
        "minimum_original_provider_retained_fraction": minimum_fraction,
        "too_few_supported_zones": too_few,
        "measured_probe_count": len(measured),
        "insufficient_probe_count": len(failed),
        "largest_measured_endpoint_delta_log10": max(
            (probe["maximum_absolute_endpoint_delta_log10"] for probe in measured),
            default=None,
        ),
        "most_influential_measured_zone_yx": None
        if not influential
        else influential[0]["removed_zone_yx"],
        "probes": probes,
        "probe_rule": "Remove exactly one supported original zone; keep the existing leaves and original-area weights; do not regrid or remeasure pixels",
        "endpoint_rule": "Original shared-RGB leaf population; channel-wise weighted inverse CDF at1/200 and199/200, integer ceil thresholds, no interpolation",
        "RGB_support_shared": True,
        "source_pixels_read": False,
        "source_pixels_modified": False,
        "meter_or_renderer_called": False,
        "cross_scale_comparison_performed": False,
        "automatic_exclusions": [],
        "automatic_weight_changes": False,
        "replacement_window": None,
        "defect_probability": None,
        "physical_cause": None,
        "scope": "Metadata-only endpoint sensitivity, not a film diagnosis or representative-density/colour calibration certificate",
        "limitations": [
            "A normal sky, wall, dark subject or bright sign may dominate endpoints exactly like a film fault; sensitivity does not authorize exclusion.",
            "Stable endpoints do not prove a clean or representative selection, nor detect faults distributed across zones.",
            "The warning threshold is an explicit engineering policy, not a learned probability or threshold tuned to sample05.",
            "Finite signed and above-one source values remain governed by the original low-floor provider; no normalized value is inferred to be sensor saturation.",
            "Nonfinite sources cannot be rescued by masking in the current full-source finite rendering contract.",
            "Restoring recorded identities does not revalidate source pixels on disk; the caller owns the verified current source context.",
        ],
    }
    canonical = _canonical(document)
    return ReliabilityReport(canonical, hashlib.sha256(canonical.encode()).hexdigest())
