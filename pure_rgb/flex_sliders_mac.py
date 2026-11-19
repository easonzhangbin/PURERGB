"""Mac FlexColor 4.8.13 Contrast/Brightness curve on explicit RGB14 codes.

This is the original i386 CContrastCurve polynomial, independently replayed in
an isolated emulator. It differs from the older Windows linear implementation.
Control values are raw signed-byte integers, not EV or percentages. The field
storage range does not establish the original application's UI slider limits.
"""

from functools import lru_cache

import numpy as np


def _control(value, name):
    if type(value) is not int or not -128 <= value <= 127:
        raise ValueError(
            f"Mac Flex {name} requires an integer signed-byte field (-128..127)"
        )
    return value


@lru_cache(maxsize=256)
def _build(contrast, brightness, enabled):
    codes = np.arange(16384, dtype=np.float64)
    if not enabled:
        return np.frombuffer(codes.astype("<u2").tobytes(), dtype="<u2")

    # Keep the original SSE2 binary64 operation order, including the exact
    # powers-of-two constants. No pow/libm or x87 service is used by this stage.
    brightness_normalized = float(brightness) / 50.0
    contrast_normalized = float(contrast) / 50.0
    brightness_amplitude = brightness_normalized * 8192.0
    contrast_doubled = contrast_normalized + contrast_normalized
    lower, upper = codes[:8192], codes[8192:]

    lower_contrast = (lower - 8192.0) * contrast_doubled
    lower_contrast = lower_contrast * lower
    lower_contrast = lower_contrast * 0.00006103515625
    lower_brightness = lower * 0.0001220703125 if brightness > 0 else np.ones(8192)
    lower_brightness = lower_brightness * brightness_amplitude
    lower_result = (lower_brightness + lower_contrast) + lower

    upper_contrast = (upper - 8192.0) * contrast_doubled
    upper_contrast = upper_contrast * (16384.0 - upper)
    upper_contrast = upper_contrast * 0.00006103515625
    upper_brightness = (
        np.ones(8192) if brightness > 0 else (16383.0 - upper) * 0.0001220703125
    )
    upper_brightness = upper_brightness * brightness_amplitude
    upper_result = (upper_brightness + upper_contrast) + upper

    values = np.concatenate((lower_result, upper_result))
    # Original cvttsd2si truncates binary64 toward zero, then the evaluator
    # clamps to 0..16383. There is no Windows-style signed-int16 narrowing.
    output = np.clip(np.trunc(values), 0, 16383).astype("<u2")
    return np.frombuffer(output.tobytes(), dtype="<u2")


def contrast_brightness_lut_mac(contrast=0, brightness=0, *, enabled=True):
    """Return immutable uint16[16384] matching the Mac original scalar stage.

    ``enabled`` is the CContrastCurve active state, controlled by ApplySliders
    in the original graph. Disabled returns identity even for nonzero fields.
    Out-of-range values are rejected instead of silently wrapping their low
    byte as the original raw setter can; caller policy must remain explicit.
    No scanner calibration, ICC conversion, OETF, or full factory render is
    performed. The complete stored-field range is supported, including its
    original clipping and potentially non-monotonic extreme curves.
    """
    contrast = _control(contrast, "Contrast")
    brightness = _control(brightness, "Brightness")
    if type(enabled) is not bool:
        raise ValueError("Mac Flex enabled requires an explicit boolean")
    return _build(contrast, brightness, enabled)
