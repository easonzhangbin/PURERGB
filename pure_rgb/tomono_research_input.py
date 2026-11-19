"""Isolated toMONO LinearRaw experiment; never a production CFA importer.

Only explicitly bound build16, zero-EV, strict same-colour legacy or D10
experiments are accepted. The result has its own signal domain and cannot masquerade as a
standard-cfa-dng-v1 receipt or a source-bound Native metering snapshot.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import negpy_core
import numpy as np
import tifffile

from .raw_validity import StandardCropPlan, _orient


POLICY = "tomono-linear-dng-triplet-study-v1"
DOMAIN = "linear-tomono-selected-band-reconstruction-v1"
SOFTWARE = "toMONO 1.2.0 (build 16)"
PATTERNS = ((0, 1, 1, 2), (2, 1, 1, 0), (1, 0, 2, 1), (1, 2, 0, 1))


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def pixels_sha(value):
    require(value.flags.c_contiguous, "C-order pixels required for byte digest")
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, name + " requires an integer")
    return value


def _sha(value):
    require(
        isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value),
        "SHA256 required",
    )
    return value


def _rect(value, width, height):
    require(
        isinstance(value, (list, tuple)) and len(value) == 4, "xyxy rectangle required"
    )
    x0, y0, x1, y1 = [_integer(v, "rectangle edge") for v in value]
    require(
        0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, "Rectangle outside its canvas"
    )
    return x0, y0, x1, y1


def _number(value, name):
    require(
        type(value) in (float, int) and math.isfinite(value), name + " must be finite"
    )
    return float(value)


def _rational(value, count, name):
    values = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    require(len(values) == count * 2, name + " rational field count differs")
    output = []
    for n, d in zip(values[::2], values[1::2], strict=True):
        require(
            type(n) is int and type(d) is int and d > 0 and n % d == 0,
            name + " must be integral",
        )
        output.append(n // d)
    return output


@dataclass(frozen=True)
class MonoPlane:
    signal: np.ndarray
    receipt: dict


@dataclass(frozen=True)
class ResearchTriplet:
    source: np.ndarray
    receipt: dict


def validate_baseline_report(report, *, role, original, expected_detail_mode="legacy"):
    """Validate conversion semantics independently of its output file digest."""
    require(role in "RGB" and len(role) == 1, "Explicit physical RGB role required")
    require(
        expected_detail_mode in ("legacy", "directional"),
        "Explicit supported detail mode required",
    )
    require(
        report.get("state") in ("saved", "saved_with_warning")
        and report.get("roundtrip_exact") is True,
        "toMONO conversion was not completely verified",
    )
    require(
        report.get("dng_saved") is True and report.get("sidecar_saved") is True,
        "DNG and sidecar must both be saved",
    )
    require(
        str(Path(report.get("input", "")).resolve())
        == str(Path(original["path"]).resolve()),
        "Conversion input differs from classified original",
    )
    _sha(original["sha256"])
    recipe = report.get("recipe", {})
    require(
        type(recipe.get("version")) is int and recipe["version"] == 2,
        "Explicit v2 recipe required",
    )
    require(
        recipe.get("response_mode")
        == "narrow_" + {"R": "red", "G": "green", "B": "blue"}[role],
        "Narrow response does not match physical illumination",
    )
    require(
        recipe.get("detail_mode") == expected_detail_mode
        and recipe.get("narrow_interpolation") == "cubic",
        "Conversion detail mode differs from this explicitly requested experiment",
    )
    require(
        _number(recipe.get("response_exposure"), "EV") == 0, "Baseline requires zero EV"
    )
    require(
        recipe.get("green_mode") == "off"
        and recipe.get("defect_profile") is None
        and recipe.get("lens_correction") is None
        and recipe.get("match_preview") is False,
        "Baseline requires disabled optional processing",
    )
    parameters = report.get("resolved_parameters", {})
    response = parameters.get("response", {})
    require(
        parameters.get("processing_path")
        == "pure_narrowband_"
        + ("cubic" if expected_detail_mode == "legacy" else "directional")
        and parameters.get("selected_channel") == role
        and parameters.get("requested_detail_mode") == expected_detail_mode
        and parameters.get("effective_detail_mode") == expected_detail_mode
        and parameters.get("detail_fallback_reason") is None,
        "Conversion used an unbound reconstruction or fallback",
    )
    if expected_detail_mode == "directional":
        detail = parameters.get("detail_reconstruction", {})
        require(
            detail.get("mode") == "directional"
            and detail.get("code") == "D10"
            and detail.get("same_colour_only") is True
            and detail.get("automatic_noise_estimation") is False
            and _number(detail.get("noise_sigma"), "D10 noise sigma") == 0
            and detail.get("RB_directional_fallback") is (role in "RB"),
            "D10 must declare same-colour G direction/RB bounded cubic and no noise estimation",
        )
    require(
        response.get("digital_channel_isolation") is True
        and response.get("target_channel") == role
        and response.get("mode") == recipe["response_mode"]
        and response.get("weights") == [int(c == role) for c in "RGB"]
        and _number(response.get("exposure_ev"), "resolved EV") == 0,
        "Resolved channel response differs from recipe",
    )
    require(
        _number(parameters.get("green_compensation"), "green compensation") == 0
        and parameters.get("green_replacements") == 0,
        "Green correction entered strict baseline",
    )
    pattern = np.asarray(parameters.get("pattern"))
    require(
        pattern.shape == (2, 2)
        and all(type(v) is int for row in parameters["pattern"] for v in row)
        and tuple(pattern.ravel()) in PATTERNS,
        "Verified RGB Bayer phase metadata required",
    )
    blacks, whites = [], []
    for key, destination in (
        ("source_black_levels_2x2", blacks),
        ("source_white_levels_2x2", whites),
    ):
        rows = parameters.get(key)
        require(
            isinstance(rows, list)
            and len(rows) == 2
            and all(isinstance(row, list) and len(row) == 2 for row in rows),
            "Recorded phase black/white levels required",
        )
        destination.extend([[_number(v, key) for v in row] for row in rows])
    blacks, whites = np.asarray(blacks), np.asarray(whites)
    require(
        np.all(blacks >= 0) and np.all(whites <= 65535) and np.all(whites > blacks),
        "Invalid source phase levels",
    )
    gy, gx = np.argwhere(pattern == 1)[0]
    black = _integer(parameters.get("output_black"), "output black")
    white = _integer(parameters.get("output_white"), "output white", 1)
    require(
        black < white <= 65535 and black == blacks[gy, gx] and white == whites[gy, gx],
        "Output DN container is not the declared common-G scale",
    )
    return parameters, black, white


def _linear_page(tif):
    candidates = []

    def visit(pages):
        for page in pages:
            if (
                int(page.photometric) == 34892
                and page.samplesperpixel == 1
                and not int(page.subfiletype) & 5
            ):
                candidates.append(page)
            if page.pages is not None:
                visit(page.pages)

    visit(tif.pages)
    require(len(candidates) == 1, "Exactly one single-channel LinearRaw IFD required")
    return candidates[0]


def read_tomono_plane(
    binding,
    *,
    role,
    original,
    band_geometry,
    orientation,
    selected_site_oracles=(),
    expected_detail_mode="legacy",
):
    """Read one bound LinearRaw plane, normalized/cropped but never demosaiced.

    binding explicitly carries original_file_sha256, output_file_sha256 and
    report_file_sha256. Producer raw_sha256 is the output DN byte digest only.
    No original RAW is opened by this function.
    """
    require(binding["role"] == role == original["role"], "Physical RGB role mismatch")
    require(
        binding["original_file_sha256"] == _sha(original["sha256"]),
        "Original RAW identity differs",
    )
    output, report_path = Path(binding["output"]), Path(binding["report"])
    require(
        file_sha(output) == _sha(binding["output_file_sha256"])
        and file_sha(report_path) == _sha(binding["report_file_sha256"]),
        "Converted file identity changed",
    )
    report = json.loads(report_path.read_text())
    parameters, black, white = validate_baseline_report(
        report, role=role, original=original, expected_detail_mode=expected_detail_mode
    )
    require(
        Path(report["output"]).resolve() == output.resolve(),
        "DNG path differs from conversion report",
    )
    plan = StandardCropPlan(**band_geometry)
    rh, rw = plan.raw_shape
    active_t, active_l, active_b, active_r = [
        _integer(v, "source ActiveArea")
        for v in parameters.get("source_active_area", [])
    ]
    _rect((active_l, active_t, active_r, active_b), rw, rh)
    _integer(orientation, "orientation", 1)
    require(orientation <= 8, "EXIF orientation must be1..8")
    with tifffile.TiffFile(output) as tif:
        page, root = _linear_page(tif), tif.pages[0]

        def tag(name, default=None):
            item = page.tags.get(name) or root.tags.get(name)
            return item.value if item is not None else default

        require(tag("Software") == SOFTWARE, "Unbound toMONO software version")
        require(
            tag("Orientation") == orientation,
            "Converted EXIF orientation differs from original",
        )
        require(
            page.dtype == np.dtype("uint16")
            and page.ndim == 2
            and page.bitspersample == 16,
            "Unsigned16 single-plane DN raster required",
        )
        for name in (
            "CFAPattern",
            "CFARepeatPatternDim",
            "LinearizationTable",
            "BlackLevelDeltaH",
            "BlackLevelDeltaV",
            "OpcodeList1",
            "OpcodeList2",
            "OpcodeList3",
        ):
            require(tag(name) is None, "Unsupported LinearRaw transformation: " + name)
        require(
            tag("DefaultScale", (1, 1, 1, 1)) == (1, 1, 1, 1),
            "Non-unit pixel scale unsupported",
        )
        require(
            tuple(page.shape) == (active_b - active_t, active_r - active_l),
            "DNG raster does not match original active sensor view",
        )
        require(
            list(tag("ActiveArea", ())) == [0, 0, *page.shape],
            "Derived ActiveArea must cover its full raster",
        )
        require(
            tuple(tag("BlackLevelRepeatDim", ())) == (1, 1)
            and _rational(tag("BlackLevel"), 1, "BlackLevel") == [black]
            and tag("WhiteLevel") == white,
            "DNG black/white metadata differs from conversion",
        )
        crop_origin = _rational(tag("DefaultCropOrigin"), 2, "DefaultCropOrigin")
        crop_size = _rational(tag("DefaultCropSize"), 2, "DefaultCropSize")
        x0, y0, x1, y1 = plan.raw_roi
        require(
            crop_origin == [x0 - active_l, y0 - active_t]
            and crop_size == [x1 - x0, y1 - y0],
            "DNG DefaultCrop and original standard sensor crop disagree",
        )
        _rect(
            (
                crop_origin[0],
                crop_origin[1],
                crop_origin[0] + crop_size[0],
                crop_origin[1] + crop_size[1],
            ),
            page.shape[1],
            page.shape[0],
        )
        pixels = page.asarray()
    require(
        pixels_sha(pixels) == _sha(report.get("raw_sha256")),
        "Output DN pixels differ from producer digest",
    )
    require(report.get("shape") == list(pixels.shape), "Report raster shape differs")
    selected_checks = []
    target = "RGB".index(role)
    for oracle in selected_site_oracles:
        x, y = [_integer(v, "oracle raw coordinate") for v in oracle["source_raw_xy"]]
        require(
            x0 <= x < x1 and y0 <= y < y1, "Selected-site oracle outside standard crop"
        )
        phase_x, phase_y = (x - active_l) % 2, (y - active_t) % 2
        require(
            parameters["pattern"][phase_y][phase_x] == target,
            "Oracle does not sample the selected CFA colour",
        )
        source_dn = _integer(oracle["source_DN"], "oracle source DN")
        source_black = _number(oracle["source_black"], "oracle source black")
        source_white = _number(oracle["source_white"], "oracle source white")
        require(
            source_dn <= 65535
            and source_black == parameters["source_black_levels_2x2"][phase_y][phase_x]
            and source_white == parameters["source_white_levels_2x2"][phase_y][phase_x],
            "Oracle source phase levels differ from conversion",
        )
        signal_before_container = (source_dn - source_black) / (
            source_white - source_black
        )
        mapped = float(np.rint(black + signal_before_container * (white - black)))
        expected = int(np.clip(mapped, 0, 65535))
        actual = int(pixels[y - active_t, x - active_l])
        require(
            actual == expected, "Selected-site independent DN rounding oracle differs"
        )
        selected_checks.append(
            {
                "source_raw_xy": [x, y],
                "active_phase_xy": [phase_x, phase_y],
                "source_DN": source_dn,
                "source_black": source_black,
                "source_white": source_white,
                "expected_DN": expected,
                "actual_DN": actual,
                "container_clipped": not 0 <= mapped <= 65535,
                "normalized_roundtrip_error": (actual - black) / (white - black)
                - signal_before_container,
            }
        )
    ox, oy = crop_origin
    width, height = crop_size
    crop = pixels[oy : oy + height, ox : ox + width]
    signal = np.empty(crop.shape, np.float32)
    for top in range(0, height, 256):
        signal[top : top + 256] = (crop[top : top + 256].astype(np.float64) - black) / (
            white - black
        )
    require(np.isfinite(signal).all(), "Nonfinite normalized mono signal")
    signal.setflags(write=False)
    receipt = {
        "policy": POLICY,
        "domain": DOMAIN,
        "role": role,
        "original": deepcopy(original),
        "binding": deepcopy(binding),
        "software": SOFTWARE,
        "detail_mode": expected_detail_mode,
        "effective_channel_reconstruction": (
            "same-colour bounded cubic"
            if expected_detail_mode == "legacy" or role in "RB"
            else "same-colour green directional"
        ),
        "recipe": report["recipe"],
        "source_active_area_tlbr": [active_t, active_l, active_b, active_r],
        "standard_crop_raw_xyxy": list(plan.raw_roi),
        "orientation_not_applied": orientation,
        "cfa_pattern_active_origin": parameters["pattern"],
        "source_black_levels_2x2": parameters["source_black_levels_2x2"],
        "source_white_levels_2x2": parameters["source_white_levels_2x2"],
        "output_black": black,
        "output_white": white,
        "normalization": "float64(DN-commonGblack)/(commonGwhite-commonGblack), cast once tofloat32; EV0; no clipping",
        "content_gain_or_mean_fit": False,
        "display_metadata_used": False,
        "normalized_signal_sha256": pixels_sha(signal),
        "shape": list(signal.shape),
        "source_dtype": "<f4",
        "source_read_only": True,
        "common_crop_counts": {
            "DN_zero": int(np.count_nonzero(crop == 0)),
            "DN_container_max": int(np.count_nonzero(crop == 65535)),
            "below_black": int(np.count_nonzero(crop < black)),
            "at_or_above_white": int(np.count_nonzero(crop >= white)),
        },
        "producer_full_active_clip_counts": {
            key: parameters["response"].get(key)
            for key in (
                "container_clipped_low",
                "container_clipped_high",
                "at_or_above_white",
            )
        },
        "quantization_half_step_normalized": 0.5 / (white - black),
        "selected_site_oracles": selected_checks,
        "selected_site_oracle_scope": "Bounded original sensor samples captured by the conversion's single unpack; no new RAW read; not a full-pixel truth image",
        "scope": "Independent derived LinearRaw input experiment; no CFA importer, snapshot or factory rendering support",
    }
    require(
        file_sha(output) == binding["output_file_sha256"]
        and file_sha(report_path) == binding["report_file_sha256"],
        "Conversion changed during read",
    )
    return MonoPlane(signal, receipt)


def _original_support(shape, shift):
    h, w = shape
    dx, dy = shift
    return (
        max(0, math.ceil(2 - dx)),
        max(0, math.ceil(2 - dy)),
        min(w, math.floor(w - 3 - dx) + 1),
        min(h, math.floor(h - 3 - dy) + 1),
    )


def assemble_frozen_triplet(planes, a_receipt):
    """Use A's already frozen warps/support, without estimating a B transform."""
    require(
        len(planes) == 3 and [p.receipt["role"] for p in planes] == list("RGB"),
        "Exactly ordered RGB planes required",
    )
    require(
        len({p.receipt["detail_mode"] for p in planes}) == 1,
        "Mixed legacy/D10 frames are not one controlled triplet",
    )
    require(
        a_receipt.get("policy") in ("standard-crop-v1", "standard-cfa-dng-v1")
        and a_receipt.get("source_read_only") is True,
        "Verified standard A receipt required",
    )
    require(
        a_receipt.get("flatfield", {}).get("identity") is True
        and a_receipt["flatfield"].get("applied") is False,
        "A flatfield must be an explicit identity",
    )
    decode = a_receipt.get("decode", {})
    require(
        decode.get("linear_raw") is True
        and decode.get("neutral_WB") is True
        and decode.get("half_size") is False
        and decode.get("user_flip") == 0
        and decode.get("highlight_reconstruction") == 0,
        "A must use neutral full-resolution sensor decoding",
    )
    require(
        a_receipt["geometry_token"]
        == hashlib.sha256(
            json.dumps(
                {
                    key: a_receipt[key]
                    for key in ("bands", "registration", "effective_geometry")
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "A geometry token differs",
    )
    registration, geometry = a_receipt["registration"], a_receipt["effective_geometry"]
    shape = tuple(registration["input_shape_h_w"])
    require(
        all(
            p.signal.shape == shape
            and p.signal.dtype == np.float32
            and p.signal.flags.c_contiguous
            and not p.signal.flags.writeable
            and np.isfinite(p.signal).all()
            and p.receipt.get("policy") == POLICY
            and p.receipt.get("domain") == DOMAIN
            and pixels_sha(p.signal) == p.receipt.get("normalized_signal_sha256")
            for p in planes
        ),
        "Finite read-only common standard sensor planes required",
    )
    for plane, band in zip(planes, a_receipt["bands"], strict=True):
        require(
            plane.receipt["original"]["path"] == band["path"]
            and plane.receipt["role"] == band["role"]
            and plane.receipt["standard_crop_raw_xyxy"] == band["geometry"]["raw_roi"]
            and plane.receipt["orientation_not_applied"] == band["orientation"],
            "B frame identity or sensor geometry differs from A",
        )
    require(
        type(registration.get("align")) is bool
        and registration.get("reference_channel") == "green"
        and registration.get("exif_orientation_applied") is False,
        "Unrecognized A registration contract",
    )
    channels = registration["channels"]
    require(
        channels["green"]["shift_xy"] == [0.0, 0.0],
        "Green reference must not be warped",
    )
    h, w = shape
    supports = [(0, 0, w, h)]
    shifts = []
    for name in ("red", "blue"):
        shift = channels[name]["shift_xy"]
        require(isinstance(shift, list) and len(shift) == 2, "Frozen xy shift required")
        shift = [_number(v, "frozen shift") for v in shift]
        shifts.append(shift)
        supports.append(
            _original_support(shape, shift) if registration["align"] else (0, 0, w, h)
        )
        require(
            registration["align"] or shift == [0.0, 0.0],
            "Disabled registration has nonzero shift",
        )
    common = (
        max(v[0] for v in supports),
        max(v[1] for v in supports),
        min(v[2] for v in supports),
        min(v[3] for v in supports),
    )
    require(
        list(common) == registration["common_valid_roi_sensor_xyxy"],
        "A common support is not the exact frozen-warp intersection",
    )
    x0, y0, x1, y1 = _rect(common, w, h)
    if registration["align"]:
        require(
            registration.get("interpolation") == "cubic"
            and registration.get("warp_border") == "replicate"
            and registration.get("warp_quantization_pixels") == 1 / 32,
            "Only A's pinned cubic warp is supported",
        )
    standard = StandardCropPlan(**a_receipt["bands"][1]["geometry"])
    effective = StandardCropPlan(**geometry["effective_plan"])
    sx, sy, _, _ = standard.raw_roi
    require(
        effective.raw_roi == (sx + x0, sy + y0, sx + x1, sy + y1),
        "Effective A crop differs from frozen support",
    )
    merged = np.empty((y1 - y0, x1 - x0, 3), np.float32)
    merged[..., 1] = planes[1].signal[y0:y1, x0:x1]
    for band, shift in zip((0, 2), shifts, strict=True):
        if registration["align"]:
            dx, dy = shift
            warped = negpy_core.warp_affine(
                planes[band].signal,
                np.array([[1, 0, -dx], [0, 1, -dy]], np.float64),
                (w, h),
                interpolation="cubic",
                border="replicate",
            )
            merged[..., band] = warped[y0:y1, x0:x1]
        else:
            merged[..., band] = planes[band].signal[y0:y1, x0:x1]
    orientation = _integer(geometry["orientation"], "orientation", 1)
    require(
        orientation <= 8
        and all(p.receipt["orientation_not_applied"] == orientation for p in planes),
        "Shared original EXIF orientation differs",
    )
    source = np.ascontiguousarray(_orient(merged, orientation))
    require(
        list(source.shape) == a_receipt["source_shape"] == geometry["source_shape"]
        and np.isfinite(source).all(),
        "B source shape/values differ from declared A common canvas",
    )
    source.setflags(write=False)
    receipt = {
        "policy": POLICY,
        "domain": DOMAIN,
        "version": 1,
        "source_shape": list(source.shape),
        "source_dtype": "<f4",
        "source_read_only": True,
        "source_pixels_sha256": pixels_sha(source),
        "channel_order": "RGB",
        "coordinate_space": "EXIF-oriented A-common-support local pixels; half-open xyxy",
        "a_geometry_token": a_receipt["geometry_token"],
        "a_source_pixels_sha256": a_receipt["source_pixels_sha256"],
        "frozen_registration": deepcopy(registration),
        "geometry": deepcopy(geometry),
        "planes": [deepcopy(p.receipt) for p in planes],
        "B_registration_reestimated": False,
        "gain_fit": False,
        "resized": False,
        "EXIF_application_count": 1,
        "production_CFA_import_or_snapshot_supported": False,
    }
    receipt["identity_sha256"] = digest(receipt)
    return ResearchTriplet(source, receipt)
