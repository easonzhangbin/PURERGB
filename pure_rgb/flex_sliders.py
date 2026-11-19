"""Original FlexColor Contrast/Brightness scalar tables for explicit code domains.

The signed-byte fields and table arithmetic are checked independently against
the pinned 4.8.9.1 x86 functions. This is one stage, not a scanner input model,
seven-Gradations routing, histogram, negative gamma or factory CMS recreation.
"""

from functools import lru_cache

import numpy as np


def _parameters(contrast, brightness, bits):
    if type(bits) is not int or bits not in (8, 14):
        raise ValueError("Flex 亮度／反差表只接受 8-bit 或 14-bit")
    result = []
    for value in (contrast, brightness):
        if (
            type(value) not in (int, float)
            or not np.isfinite(value)
            or value != int(value)
            or not -128 <= value <= 127
        ):
            raise ValueError("原始 Flex 亮度／反差字段需 -128..127 整数")
        result.append(int(value))
    return (*result, bits)


@lru_cache(maxsize=64)
def _build(contrast, brightness, bits):
    # The x87 result is spilled into two float32 members before evaluation.
    slope = np.float32((contrast + 50) / 50 if contrast < 0 else 1 + contrast / 30)
    offset = np.float32(8192 - (8192 - 160 * brightness) * float(slope))
    codes = np.arange(256 if bits == 8 else 16384, dtype=np.int64)
    if bits == 8:
        codes *= 64
    values = np.trunc(codes * float(slope) + float(offset)).astype(np.int64)
    # Original word evaluator narrows to signed int16 BEFORE clamping. Preserve
    # its wrap behavior even for extreme combinations that reverse the curve.
    values = (values + 32768) % 65536 - 32768
    values = np.clip(values, 0, 16383)
    output = (values >> 6).astype(np.uint8) if bits == 8 else values.astype("<u2")
    return np.frombuffer(output.tobytes(), dtype=output.dtype)


def slider_lut(contrast=0, brightness=0, bits=14):
    return _build(*_parameters(contrast, brightness, bits))


def apply_sliders(codes, contrast=0, brightness=0, bits=14):
    """Apply to integer codes only; RAW/normalized floats need an explicit adapter."""
    values = np.asarray(codes)
    table = slider_lut(contrast, brightness, bits)
    if values.dtype.kind not in "ui" or (
        values.size and (values.min() < 0 or values.max() >= len(table))
    ):
        raise ValueError("输入必须是明确的 8/14-bit 整数编码")
    return table[values]
