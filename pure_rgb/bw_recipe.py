"""Explicit single-RAW Native B&W recipes; missing source_mode remains RGB.

This document selects the unchanged NegPy B&W renderer. C41 experimental
renderers and RGB-triplet-only input operations are deliberately unavailable.
"""

from copy import deepcopy
import json
import math
from pathlib import Path

from .native_headroom import HEADROOM_CONTROL, HEADROOM_STRENGTH
from .professional_controls import NATIVE_SPECS, PRINT_EXPOSURE_EV, checked_controls
from .output_orientation import checked_document as checked_output_orientation


BW_SOURCE_MODE = "bw-single-v1"
BW_SUFFIXES = (".rw2", ".dng")
_KEYS = {
    "source_mode",
    "raw",
    "raw_import",
    "roi",
    "output_roi",
    "output_orientation",
    "density",
    "measurement_method",
    "paper",
    "crop",
    "exposure_ev",
    "flatfield",
    "native_controls",
    "highlight_controls",
    "bw_metering",
    "native_spatial_controls",
    "native_float_controls",
    "post_controls",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def is_bw_recipe(recipe):
    return isinstance(recipe, dict) and recipe.get("source_mode") == BW_SOURCE_MODE


def bw_papers():
    from negpy.features.exposure.papers import profiles_for_mode
    from negpy.features.process.models import ProcessMode

    return tuple(
        (key, profile.label) for key, profile in profiles_for_mode(ProcessMode.BW)
    )


def _rectangle(value, label):
    _require(
        type(value) is list and len(value) == 4 and all(type(v) is int for v in value),
        label + " requires four integer source coordinates",
    )
    x0, y0, x1, y1 = value
    _require(0 <= x0 < x1 and 0 <= y0 < y1, label + " requires a nonempty source ROI")
    return list(value)


def checked_bw_recipe(recipe, *, verify_resources=True):
    """Validate the complete Native-only contract without opening a RAW file."""
    _require(type(verify_resources) is bool, "Resource policy must be boolean")
    _require(
        type(recipe) is dict and not set(recipe) - _KEYS, "Unknown B&W recipe field"
    )
    _require(
        recipe.get("source_mode") == BW_SOURCE_MODE, "Explicit B&W source mode required"
    )
    try:
        encoded = json.dumps(recipe, allow_nan=False)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("B&W recipe must contain finite JSON values") from error
    _require(len(encoded.encode()) <= 131072, "B&W recipe exceeds JSON size bound")
    result = deepcopy(recipe)
    raw = result.get("raw")
    _require(
        type(raw) is list
        and len(raw) == 1
        and type(raw[0]) is str
        and bool(raw[0].strip())
        and Path(raw[0]).suffix.lower() in BW_SUFFIXES,
        "B&W requires one original RW2 or DNG; no RGB triplet or converted surrogate",
    )
    result["raw"] = [str(Path(raw[0]).expanduser().resolve())]
    _require(
        result.get("raw_import", "legacy-v1") == "legacy-v1",
        "Single B&W uses the original NegPy loader, not a triplet import policy",
    )
    result["raw_import"] = "legacy-v1"
    _require(result.get("flatfield") is None, "Single B&W flatfield is not joined")
    result["flatfield"] = None
    roi = result.get("roi")
    result["roi"] = None if roi is None else _rectangle(roi, "B&W analysis ROI")
    _require(
        result.get("density", "negpy") == "negpy", "B&W uses NegPy density analysis"
    )
    result["density"] = "negpy"
    paper = result.get("paper", "neutral")
    _require(
        type(paper) is str and paper in dict(bw_papers()), "Unknown B&W print material"
    )
    result["paper"] = paper
    ev = result.get("exposure_ev", 0.0)
    _require(
        type(ev) in (int, float) and math.isfinite(ev) and -4 <= ev <= 4,
        "B&W exposure must be finite within[-4,4]EV",
    )
    result["exposure_ev"] = ev
    _require(type(result.get("crop", True)) is bool, "B&W crop must be boolean")
    result.setdefault("crop", True)
    if "output_roi" in result:
        if result["output_roi"] is None:
            result["crop"] = False
            result.pop("output_roi")
        else:
            output = _rectangle(result["output_roi"], "B&W output ROI")
            result["crop"] = True
            if output == result["roi"]:
                result.pop("output_roi")
            else:
                result["output_roi"] = output
    if "output_orientation" in result:
        value = checked_output_orientation(result["output_orientation"])
        if value is None:
            result.pop("output_orientation")
        else:
            result["output_orientation"] = value
    if "native_controls" in result:
        controls = checked_controls(result["native_controls"], NATIVE_SPECS)
        _require(not {HEADROOM_CONTROL, HEADROOM_STRENGTH}.intersection(controls["values"]),
                 "负片动态范围映射目前适用于彩色 NegPy 原生转换")
        _require(PRINT_EXPOSURE_EV not in controls["values"],
                 "RAW 印相曝光目前适用于彩色 NegPy 原生转换")
        paper_override = controls["values"].get(
            "exposure.paper_profile", "__reference__"
        )
        _require(
            paper_override in ("__reference__", "neutral"),
            "Select B&W print material through the main paper field; C41 overrides are not allowed",
        )
        if controls["values"]:
            result["native_controls"] = controls
        else:
            result.pop("native_controls")
    if "post_controls" in result:
        from .positive_post import checked_post

        document = result["post_controls"]
        _require(type(document) is dict and set(document) == {"native"},
                 "Unknown B&W positive geometry mode")
        native = document["native"]
        _require(type(native) is dict and type(native.get("values")) is dict,
                 "B&W positive geometry requires a versioned values document")
        _require(all(key.startswith("geometry.")
                     for key in native["values"]),
                 "B&W positive post permits geometry only; no C41 colour controls")
        controls = checked_post(document)
        if controls:
            result["post_controls"] = controls
        else:
            result.pop("post_controls")
    if "highlight_controls" in result:
        from .highlight_workflow import checked_highlight_workflow

        document = result["highlight_controls"]
        _require(type(document) is dict, "B&W highlight controls must be an object")
        checked = checked_highlight_workflow(document) if document else None
        if checked is not None and checked["controls"]["strength"] > 0:
            result["highlight_controls"] = checked
        else:
            result.pop("highlight_controls")
    if "bw_metering" in result:
        from .bw_metering import checked_bw_metering

        checked = checked_bw_metering(result["bw_metering"])
        if checked is None:
            result.pop("bw_metering")
        else:
            result["bw_metering"] = checked
    if "native_spatial_controls" in result:
        from .spatial_workflow_v31 import check_spatial_recipe

        check_spatial_recipe(result)
    if "native_float_controls" in result:
        from .workspace import checked_native_float_controls

        _require(
            bool(result.get("native_spatial_controls")),
            "Unknown B&W recipe field: float controls require the explicit spatial workflow",
        )
        checked = checked_native_float_controls(result["native_float_controls"])
        if checked is None:
            result.pop("native_float_controls")
        else:
            _require(
                bool(result.get("native_spatial_controls")),
                "BW float controls require the explicit spatial workflow",
            )
            result["native_float_controls"] = checked
    from .measurement_method import checked_method

    checked_method(result)
    from .native_metering_controls import process_controls_active
    _require(not process_controls_active(result.get("native_controls", {}).get("values", {})),
             "当前黑白测量未开放独立白点／黑点及范围截断，请在原生黑白参数范围内调整。")
    return result
