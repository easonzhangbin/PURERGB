"""Engine-agnostic positive-image finishing stage.

Unlike ``highlight_controls`` (a Native SDK source re-render bound to a
pre-edit reference), this stage operates on the final uint16 working-RGB
grid produced by any engine.  It is deliberately simple and deterministic:
per-pixel luminance curves with no channel rebalance, so white balance and
engine color intent are preserved.  Disabled (all zeros) is a no-op and must
remain byte-identical to the engine output.
"""

import json
from copy import deepcopy

import numpy as np

SCHEMA = "pure-rgb-common-positive-controls"
VERSION = 1
MAX_BYTES = 4096


def checked_common_positive_controls(document):
    if document is None:
        return None
    if not isinstance(document, dict):
        raise ValueError("跨引擎公共正光参数必须为对象")
    if document.get("version") != VERSION:
        raise ValueError("跨引擎公共正光参数版本不支持")
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(raw.encode()) > MAX_BYTES:
        raise ValueError("跨引擎公共正光参数超出大小限制")
    result = deepcopy(document)
    enabled = bool(result.get("enabled", True))
    if not enabled:
        return None
    for key in ("highlight", "shadow", "tone"):
        value = result.get(key, 0.0)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{key} 必须为数值")
        result[key] = float(value)
    if not (0.0 <= result["highlight"] <= 100.0):
        raise ValueError("highlight 必须在 0..100")
    if not (0.0 <= result["shadow"] <= 100.0):
        raise ValueError("shadow 必须在 0..100")
    if not (-100.0 <= result["tone"] <= 100.0):
        raise ValueError("tone 必须在 -100..100")
    return result


def _curve_lut(highlight, shadow, tone):
    """Return a uint16 -> uint16 LUT combining highlight/shadow/tone.

    The curve is identity at 0 and 65535 to protect black point and clipping.
    A soft S-curve lifts mid-shadows (shadow) and rolls off near highlights
    (highlight); tone bends the midpoint without changing endpoints.
    """
    x = np.arange(65536, dtype=np.float64) / 65535.0
    h = highlight / 100.0
    s = shadow / 100.0
    t = tone / 100.0
    # shadow lift: stronger below midpoint, identity at 0 and 1.
    shadow_curve = 1.0 - (1.0 - x) ** (1.0 + 1.5 * s) if s > 0 else x
    # highlight roll-off: compress near 1, identity elsewhere.
    if h > 0:
        k = 1.0 + 3.0 * h
        highlight_curve = 1.0 - (1.0 - x) ** k
    else:
        highlight_curve = x
    # tone: symmetric S around 0.5.
    tone_curve = x + t * 0.15 * np.sin(np.pi * x)
    combined = np.clip(tone_curve, 0.0, 1.0)
    combined = combined * (1.0 - 0.5 * s) + shadow_curve * (0.5 * s)
    combined = combined * (1.0 - 0.5 * h) + highlight_curve * (0.5 * h)
    combined = np.clip(combined, 0.0, 1.0)
    # restore exact endpoints.
    combined[0] = 0.0
    combined[-1] = 1.0
    return np.rint(combined * 65535.0).astype(np.uint16)


def apply_common_positive(px, controls):
    """Apply the controls in place-free manner; returns a new uint16 array."""
    if not controls:
        return px
    lut = _curve_lut(controls["highlight"], controls["shadow"], controls["tone"])
    out = lut[px]
    return out
