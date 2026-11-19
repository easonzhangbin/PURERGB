"""Explicit positive float point controls, separate from bounded legacy post.

Signed RGB is a positive image's linear colour representation, never signed
normalized negative density. ICC adaptation is matrix relative colourimetry;
it is not scanner characterization, perceptual CMS or gamut mapping. The odd
extension outside nominal code range is a declared research policy, not an ICC
specification or evidence that already clipped/quantized values are recoverable.
"""

from dataclasses import dataclass
import hashlib
import json
import struct

import numpy as np

from .professional_controls import Control, OUTPUT_SPECS, checked_controls
from .tone_study import GAMMA, ICC_SHA256


DOMAIN_SCHEMA = "pure-rgb-positive-float-domain"
ENCODINGS = ("linear", "working-oetf", "icc-encoded")
DOMAIN_FIELDS = {"schema", "version", "signal_role", "encoding", "icc_sha256"}
PIVOT = 0.18
# Same working-space luminance as the existing output_codes saturation control.
LUMA = np.array([0.297344975, 0.627363566, 0.075291459], dtype=np.float64)
FLOAT_SPECS = OUTPUT_SPECS + (
    Control("positive_gamma", "正像 Gamma（幂指数倒数）", "显式浮点后期", 1.0, 0.25, 4),
    Control("shadow_ev", "阴影曝光（EV）", "显式浮点后期", 0.0, -2, 2),
    Control("highlight_ev", "高光曝光（EV）", "显式浮点后期", 0.0, -2, 2),
)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _pixel_sha(image):
    return _sha(memoryview(image).cast("B"))


def working_icc_bytes():
    """Actual pinned profile; never substitute another profile by its name."""
    from negpy.infrastructure.display.color_mgmt import icc_bytes_for_space
    from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE

    data = icc_bytes_for_space(WORKING_COLOR_SPACE)
    _require(
        isinstance(data, bytes) and _sha(data) == ICC_SHA256,
        "Pinned working ICC unavailable or changed",
    )
    return data


def positive_domain(encoding, icc_bytes):
    _require(
        encoding in ENCODINGS and isinstance(icc_bytes, bytes),
        "Explicit encoding and ICC bytes required",
    )
    return {
        "schema": DOMAIN_SCHEMA,
        "version": 1,
        "signal_role": "positive-rgb",
        "encoding": encoding,
        "icc_sha256": _sha(icc_bytes),
    }


def checked_float_recipe(document):
    """JSON controls only; bounded post recipes have a separate entry point."""
    return checked_controls(document, FLOAT_SPECS)


def _input(image):
    _require(
        isinstance(image, np.ndarray)
        and image.dtype in (np.dtype("float32"), np.dtype("float64"))
        and image.ndim == 3
        and image.shape[-1] == 3
        and image.size > 0
        and image.flags.c_contiguous
        and np.isfinite(image).all(),
        "Positive float requires finite C-order float32/float64 HWC RGB; no unsigned or density input",
    )
    return image


def _range(image):
    return {
        "dtype": image.dtype.str,
        "minimum_rgb": image.min((0, 1)).tolist(),
        "maximum_rgb": image.max((0, 1)).tolist(),
        "below_zero_sample_count_rgb": (image < 0).sum((0, 1)).tolist(),
        "above_one_sample_count_rgb": (image > 1).sum((0, 1)).tolist(),
    }


def _domain(document, icc_bytes):
    _require(
        isinstance(document, dict)
        and set(document) == DOMAIN_FIELDS
        and document["schema"] == DOMAIN_SCHEMA
        and type(document["version"]) is int
        and document["version"] == 1
        and document["signal_role"] == "positive-rgb"
        and document["encoding"] in ENCODINGS
        and isinstance(icc_bytes, bytes)
        and document["icc_sha256"] == _sha(icc_bytes),
        "Wrong positive signal domain, encoding, version or ICC identity",
    )
    return document["encoding"]


def _profile(data):
    """Strict RGB matrix profile container, without evaluating unused TRCs."""
    _require(
        isinstance(data, bytes) and 132 <= len(data) <= 16 * 1024 * 1024,
        "ICC byte length invalid",
    )
    (size,) = struct.unpack_from(">I", data)
    _require(
        size == len(data)
        and data[8] in (2, 4)
        and data[36:40] == b"acsp"
        and data[12:16] in (b"mntr", b"spac")
        and data[16:20] == b"RGB "
        and data[20:24] == b"XYZ ",
        "Only display/space RGB matrix ICC profiles with XYZ PCS are supported",
    )
    (count,) = struct.unpack_from(">I", data, 128)
    _require(
        1 <= count <= 1024 and 132 + 12 * count <= len(data), "ICC tag table invalid"
    )
    tags = {}
    for index in range(count):
        signature, offset, length = struct.unpack_from(">4sII", data, 132 + 12 * index)
        _require(
            signature not in tags
            and length >= 8
            and offset >= 132 + 12 * count
            and offset % 4 == 0
            and offset + length <= len(data),
            "ICC tag extent or duplicate invalid",
        )
        tags[signature] = data[offset : offset + length]
    _require(
        not any(
            signature.startswith((b"A2B", b"B2A", b"D2B", b"B2D")) for signature in tags
        ),
        "LUT/device transforms are outside this matrix float adapter",
    )
    colorants = []
    curves = []
    for color in (b"r", b"g", b"b"):
        xyz = tags.get(color + b"XYZ", b"")
        curve = tags.get(color + b"TRC", b"")
        _require(
            len(xyz) == 20 and xyz[:4] == b"XYZ " and len(curve) >= 12,
            "All three XYZ colorants and TRC tags required",
        )
        if curve[:4] == b"curv":
            (samples,) = struct.unpack_from(">I", curve, 8)
            _require(12 + samples * 2 <= len(curve), "ICC curve payload truncated")
            if samples == 1:
                _require(
                    0.1 <= struct.unpack_from(">H", curve, 12)[0] / 256 <= 10,
                    "ICC gamma invalid",
                )
        elif curve[:4] == b"para":
            (function,) = struct.unpack_from(">H", curve, 8)
            parameters = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}.get(function)
            _require(
                parameters is not None and 12 + parameters * 4 <= len(curve),
                "ICC parametric TRC payload invalid",
            )
        else:
            raise ValueError("Unknown ICC TRC container; no implicit transfer fallback")
        colorants.append(
            np.asarray(struct.unpack_from(">3i", xyz, 8), dtype=np.float64) / 65536
        )
        # Record the exact tag type. Linear/working-oetf do not sample it.
        curves.append(curve)
    matrix = np.column_stack(colorants)
    _require(
        np.isfinite(matrix).all()
        and abs(float(np.linalg.det(matrix))) > 1e-10
        and np.linalg.cond(matrix) < 1e6,
        "ICC colorant matrix singular or unstable",
    )
    return matrix, tuple(curves)


def _power_trc(curve):
    _require(
        curve[:4] == b"curv",
        "ICC-encoded float accepts curv0/1 only; parametric/LUT curves unsupported",
    )
    (count,) = struct.unpack_from(">I", curve, 8)
    _require(
        count in (0, 1) and len(curve) >= 12 + 2 * count,
        "Tabulated ICC curves cannot be silently extrapolated",
    )
    gamma = 1.0 if count == 0 else struct.unpack_from(">H", curve, 12)[0] / 256
    _require(0.1 <= gamma <= 10, "ICC power gamma invalid")
    return gamma


def _signed_power(values, power):
    return np.copysign(np.power(np.abs(values), power), values)


def _stage(name, before, after, **extra):
    return {
        "stage": name,
        "input_range": _range(before),
        "output_range": _range(after),
        "range_policy": "unbounded-float; no signal clamp, quantization or encode",
        "signal_hard_clipped_samples": 0,
        **extra,
    }


@dataclass(frozen=True)
class PositiveFloatResult:
    image: np.ndarray
    domain: dict
    receipt: dict


def adapt_positive_float(image, *, domain, icc_bytes):
    """Explicit source primaries/TRC -> linear working RGB, retaining headroom.

    ICC XYZ colorants already inhabit D50 PCS. No second Bradford adaptation
    is applied, including when a chad tag is present. There is no BPC, gamut
    mapping or perceptual intent. A native working OETF has its own encoding
    declaration because the pinned ICC's para3 TRC is not that power function.
    """
    image = _input(image)
    encoding = _domain(domain, icc_bytes)
    source_matrix, curves = _profile(icc_bytes)
    target_bytes = working_icc_bytes()
    target_matrix, _ = _profile(target_bytes)
    stages = []
    output = image
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            if encoding == "working-oetf":
                _require(
                    _sha(icc_bytes) == ICC_SHA256,
                    "working-oetf requires the actual pinned working ICC",
                )
                output = _signed_power(image.astype(np.float64), GAMMA)
                stages.append(
                    _stage(
                        "pinned-working-OETF-decode",
                        image,
                        output,
                        power=GAMMA,
                        extension_policy="explicit odd-sign power outside nominal[0,1]; not an ICC standard extrapolation",
                    )
                )
            elif encoding == "icc-encoded":
                powers = [_power_trc(curve) for curve in curves]
                if any(value != 1 for value in powers):
                    output = _signed_power(image.astype(np.float64), np.asarray(powers))
                    stages.append(
                        _stage(
                            "ICC-curv-power-decode",
                            image,
                            output,
                            powers=powers,
                            extension_policy="explicit odd-sign power outside nominal[0,1]; not ICC-defined colour accuracy",
                        )
                    )
            if not np.array_equal(source_matrix, target_matrix):
                matrix = np.linalg.solve(target_matrix, source_matrix)
                before = output
                output = np.einsum("...c,dc->...d", output.astype(np.float64), matrix)
                stages.append(
                    _stage(
                        "ICC-PCS-matrix-to-working",
                        before,
                        output,
                        matrix=matrix.tolist(),
                        PCS="D50 colorant matrices, no repeated chad/Bradford",
                        gamut_mapping=False,
                    )
                )
        except FloatingPointError as error:
            raise ValueError(
                "Float colour adaptation overflow; no clipping fallback"
            ) from error
    _require(np.isfinite(output).all(), "Nonfinite float colour adaptation")
    if output is not image:
        output = np.ascontiguousarray(output)
        output.setflags(write=False)
    return PositiveFloatResult(
        output,
        positive_domain("linear", target_bytes),
        {
            "version": 1,
            "identity": output is image,
            "input_domain": dict(domain),
            "output_domain": positive_domain("linear", target_bytes),
            "stages": stages,
            "input_TRC_formats": [
                curve[:4].decode("ascii", "replace") for curve in curves
            ],
            "ICC_role_for_linear_input": "primaries reference only; no inverse TRC applied",
            "ICC_role_for_working_oetf": "primaries reference plus explicitly declared pinned563/256 transfer; not profile TRC",
            "ICC_relative_matrix_only": True,
            "unsigned_or_nominal_range_encoding_performed": False,
        },
    )


def apply_positive_float(image, recipe, *, domain, icc_bytes):
    """Point operations in linear working RGB; no legacy bounded stage is called."""
    checked = checked_float_recipe(recipe)
    source = _input(image)
    source_sha = _pixel_sha(source)
    adapted = adapt_positive_float(source, domain=domain, icc_bytes=icc_bytes)
    output = adapted.image
    values = checked["values"]
    stages = []

    def apply(name, function, **description):
        nonlocal output
        before = output
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                output = function(output.astype(np.float64, copy=False))
        except FloatingPointError as error:
            raise ValueError(name + " overflow; no clipping fallback") from error
        _require(np.isfinite(output).all(), name + " produced nonfinite output")
        stages.append(_stage(name, before, output, **description))

    ev = np.asarray(
        [
            values.get("exposure_ev", 0) + values.get(color + "_ev", 0)
            for color in ("red", "green", "blue")
        ],
        dtype=np.float64,
    )
    if np.any(ev != 0):
        apply("linear-exposure", lambda x: x * np.exp2(ev), EV_RGB=ev.tolist())
    contrast = values.get("contrast", 1)
    if contrast != 1:
        apply(
            "signed-power-contrast",
            lambda x: PIVOT * _signed_power(x / PIVOT, contrast),
            pivot=PIVOT,
            exponent=contrast,
            formula=".18*sign(x)*abs(x/.18)**contrast",
        )
    gamma = values.get("positive_gamma", 1)
    if gamma != 1:
        apply(
            "signed-positive-gamma",
            lambda x: _signed_power(x, 1 / gamma),
            gamma=gamma,
            formula="sign(x)*abs(x)**(1/gamma); gamma1 identity; independent of Native paper midtone_gamma",
        )
    shadow, highlight = values.get("shadow_ev", 0), values.get("highlight_ev", 0)
    if shadow != 0 or highlight != 0:

        def zones(x):
            y = np.maximum(np.einsum("...c,c->...", x, LUMA), 0)
            high = y / (y + PIVOT)
            return x * np.exp2(shadow * (1 - high) + highlight * high)[..., None]

        apply(
            "shared-luma-shadow-highlight-EV",
            zones,
            shadow_ev=shadow,
            highlight_ev=highlight,
            formula="Yplus=max(working_luma,0); h=Yplus/(Yplus+.18); RGB*=2**(shadow*(1-h)+highlight*h)",
            weight_bounds=[0, 1],
            RGB_signal_clamped=False,
            limitation="New explicit point mathematics, not original Transfer shadow/highlight_density",
        )
    saturation = values.get("saturation", 1)
    if saturation != 1:

        def saturate(x):
            luma = np.einsum("...c,c->...", x, LUMA)[..., None]
            return luma + saturation * (x - luma)

        apply(
            "working-linear-saturation",
            saturate,
            saturation=saturation,
            luminance_weights=LUMA.tolist(),
            note="Same positive working-RGB formula as output_codes; colour mixing can change individual channel signs",
        )
    if output is not source:
        output = np.ascontiguousarray(output)
        output.setflags(write=False)
    _require(
        _pixel_sha(source) == source_sha, "Caller source changed during float post"
    )
    return PositiveFloatResult(
        output,
        adapted.domain,
        {
            "schema": "pure-rgb-positive-float-receipt",
            "version": 1,
            "recipe": checked,
            "stage_order": [stage["stage"] for stage in stages],
            "identity": output is source,
            "input_domain": dict(domain),
            "output_domain": adapted.domain,
            "input_pixels_sha256": source_sha,
            "output_pixels_sha256": _pixel_sha(output),
            "input_range": _range(source),
            "adaptation": adapted.receipt,
            "stages": stages,
            "bounded_stages": [],
            "output_range": _range(output),
            "source_unchanged": True,
            "integer_encode_or_clip_performed": False,
            "range_observations_are_clip_counts": False,
            "already_clipped_samples_recovered": False,
            "new_point_semantics_not_Native_paper_or_scanner_equivalence": True,
        },
    )


def apply_bounded_positive(image, post_document, *, domain, icc_bytes, sample_step=1):
    """Separate opt-in entry to unchanged legacy Transfer/PhotoLab contracts.

    Requires unit-range linear working input. This function provides no signed
    headroom promise. It never silently clamps an out-of-range input to make an
    original bounded operator run.
    """
    from .positive_post import post_linear

    source = _input(image)
    _require(
        _domain(domain, icc_bytes) == "linear" and _sha(icc_bytes) == ICC_SHA256,
        "Bounded post requires explicit linear pinned working primaries",
    )
    _profile(icc_bytes)
    _require(
        np.all(source >= 0) and np.all(source <= 1),
        "Original bounded post requires[0,1]; retained headroom is not silently discarded",
    )
    _require(
        type(sample_step) is int and sample_step >= 1,
        "Explicit positive integer sample_step required",
    )
    before = _pixel_sha(source)
    output, old_receipt = post_linear(source, post_document, sample_step=sample_step)
    _require(_pixel_sha(source) == before, "Caller source changed in bounded post")
    return PositiveFloatResult(
        output,
        dict(domain),
        {
            "schema": "pure-rgb-explicit-bounded-positive-receipt",
            "version": 1,
            "input_domain": dict(domain),
            "input_range": _range(source),
            "output_range": _range(output),
            "legacy_post": old_receipt,
            "contract": "Original operators, explicitly unit-range entry; no whole-route headroom guarantee",
            "hard_clipped_fraction_rgb": None,
            "pre_final_clamp_values_observed": False,
            "source_unchanged": True,
            "integer_encode_performed": False,
        },
    )


def float_recipe_json(document):
    """Canonical replay document, not a serializer for caller pixel arrays."""
    return json.dumps(
        checked_float_recipe(document),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
