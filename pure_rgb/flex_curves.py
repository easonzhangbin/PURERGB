"""FlexColor CPointCurve scalar tables, independently checked against x86 code.

This kernel is distinct from the historical linear Gradations approximation.
It does not establish the routing of seven Gradations, the scanner input
transform, histogram, ColorCorr, or the complete FlexColor rendering pipeline.
"""

from decimal import Decimal, localcontext
from functools import lru_cache
import hashlib
import json
import math

import numpy as np


def _points(points):
    if not isinstance(points, (list, tuple)) or not 2 <= len(points) <= 10:
        raise ValueError("CPointCurve 需 2–10 个控制点")
    result = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) not in (2, 3):
            raise ValueError("曲线控制点需 X、Y，可含 DY")
        x, y = point[:2]
        if any(
            type(v) not in (int, float)
            or not math.isfinite(v)
            or v != int(v)
            or not 0 <= v <= 255
            for v in (x, y)
        ):
            raise ValueError("CPointCurve 坐标需 0..255 整数")
        if len(point) == 3 and (
            type(point[2]) not in (int, float) or not math.isfinite(point[2])
        ):
            raise ValueError("DY 需有限数值")
        result.append((int(x), int(y)))
    if (
        result[0][0] != 0
        or result[-1][0] != 255
        or any(a[0] >= b[0] for a, b in zip(result, result[1:]))
    ):
        raise ValueError("当前完整 LUT 契约要求 X 递增且覆盖 0..255")
    # DY is serialized by the application but is not read by these two builders.
    return tuple(result)


def _divide(numerator, denominator):
    """x86 signed integer division truncates toward zero."""
    return (
        numerator // denominator if numerator >= 0 else -((-numerator) // denominator)
    )


def _segment(output, x0, y0, x1, y1):
    for x in range(x0, x1 + 1):
        output[x] = _divide((x - x0) * (y1 - y0), x1 - x0) + y0


@lru_cache(maxsize=32)
def _build(points, bits):
    maximum, scale = (255, 1) if bits == 8 else (16383, 64)
    output = np.zeros(maximum + 1, dtype=np.uint8 if bits == 8 else np.uint16)
    if len(points) == 2 or (bits == 14 and all(x == y for x, y in points)):
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            _segment(
                output,
                x0 * scale,
                y0 * scale,
                maximum if bits == 14 and x1 == 255 else x1 * scale,
                maximum if bits == 14 and y1 == 255 else y1 * scale,
            )
    else:
        count = len(points)
        knots = [0] * count + [1] * (count + 1)
        previous_x = previous_y = 0
        parameter, step = 0.0, 1.0 / maximum
        # ARM macOS long double is only binary64. Decimal prevents loss of the
        # x87 accumulator precision; explicit binary64 spills below mirror the
        # original recursion. Independent DLL oracle fixtures test this path.
        with localcontext() as context:
            context.prec = 30
            for index in range(maximum):
                t = Decimal.from_float(parameter)
                parameter += step  # Original loop stores the parameter as f64.
                cache = {}

                def basis(i, order):
                    if (i, order) in cache:
                        return cache[i, order]
                    if order == 1:
                        value = Decimal(knots[i] <= t < knots[i + 1])
                    else:
                        left = knots[i + order - 1] - knots[i]
                        right = knots[i + order] - knots[i + 1]
                        if left and right:
                            tail = Decimal.from_float(
                                float(
                                    (knots[i + order] - t)
                                    / right
                                    * basis(i + 1, order - 1)
                                )
                            )
                            value = (t - knots[i]) / left * basis(i, order - 1) + tail
                        elif left:
                            value = (t - knots[i]) / left * basis(i, order - 1)
                        elif right:
                            value = (
                                (knots[i + order] - t) / right * basis(i + 1, order - 1)
                            )
                        else:
                            value = Decimal(0)
                    cache[i, order] = value
                    return value

                x = y = 0
                for i, (px, py) in enumerate(points):
                    weight = basis(i, count)
                    # The original truncates after every control-point addition.
                    x = int(x + px * scale * weight)
                    y = int(y + py * scale * weight)
                x, y = min(max(x, 0), maximum), min(max(y, 0), maximum)
                if index:
                    if x > previous_x:
                        _segment(output, previous_x, previous_y, x, y)
                    # Repeated x still updates y before the next line segment.
                    previous_x, previous_y = x, y
        if previous_x < maximum:
            _segment(output, previous_x, previous_y, maximum, maximum)
    # A bytes-backed array cannot be made writable again by a cache consumer.
    return np.frombuffer(output.tobytes(), dtype=output.dtype)


def point_curve_lut(points, bits=14):
    if type(bits) is not int or bits not in (8, 14):
        raise ValueError("原始 CPointCurve 表只接受 8-bit 或 14-bit")
    return _build(_points(points), bits)


def apply_point_curve(codes, points, bits=14):
    """Apply only to explicit integer codes in the selected table domain."""
    values = np.asarray(codes)
    table = point_curve_lut(points, bits)
    if values.dtype.kind not in "ui" or (
        values.size and (values.min() < 0 or values.max() >= len(table))
    ):
        raise ValueError(
            "输入必须是对应 8/14-bit 域内的整数编码；不可隐式解释 RAW 或 RGB16"
        )
    return table[values]


def main():
    import argparse
    from pathlib import Path
    from .workspace import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("points", type=Path, help="JSON array of [X,Y] or [X,Y,DY]")
    parser.add_argument("output", type=Path, help="New .npy LUT artifact")
    parser.add_argument("--bits", type=int, choices=(8, 14), default=14)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出已存在；请使用新的文件名")
    points = json.loads(args.points.read_text())
    table = point_curve_lut(points, args.bits)
    with args.output.open("xb") as stream:
        np.save(stream, table, allow_pickle=False)
    write_json(
        args.output.with_suffix(".json"),
        {
            "kernel": "FlexColor CPointCurve reconstruction v1",
            "bits": args.bits,
            "points": points,
            "table_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
            "scope": "Scalar curve only; complete factory pipeline not implied",
        },
    )


if __name__ == "__main__":
    main()
