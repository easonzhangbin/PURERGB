"""NegPy's signed/logarithmic metering slider units, stored as native values."""

import math

RANGE_FIELDS = {"process.luma_range_clip", "process.color_range_clip"}
PROCESS_DEFAULTS = {
    "process.white_point_offset": 0.,
    "process.black_point_offset": 0.,
    "process.luma_range_clip": 0.,
    "process.color_range_clip": 1.,
}


def process_controls_active(values):
    return any(values.get(key, default) != default for key, default in PROCESS_DEFAULTS.items())


def slider_to_native(key, position):
    if key == "process.luma_range_clip":
        # Keep the upstream config's exact neutral zero when resetting a group.
        if position == 0:
            return 0.
        return 10 ** (.05 * position - 5) if position > 0 else -10 ** (-6 - .06 * position)
    if key == "process.color_range_clip":
        return 10 ** (abs(position) / 100 * (math.log10(5) if position >= 0 else -6))
    return position


def native_to_slider(key, value):
    if key == "process.luma_range_clip":
        if value >= 0:
            return 20 * (math.log10(max(value, 1e-5)) + 5)
        return -100 * (math.log10(-value) + 6) / 6
    if key == "process.color_range_clip":
        return 100 * math.log10(value) / math.log10(5) if value >= 1 else 100 * math.log10(value) / 6
    return value
