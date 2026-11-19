"""Isolated FlexColor CAggregateCurve 14-bit arithmetic.

Checked against the pinned original x86 evaluator. This composes explicit LUTs;
it does not claim that seven factory Gradations can be composed without the
negative, gamma, contrast and histogram stages between them. The original graph
uses Gradations[0] as a sequential stage, adds [1+c], and subtracts [4+c], with
both channel corrections evaluated at the same saved reference (c=0,1,2).
"""

import numpy as np


MAXIMUM = 16383


def _table(table):
    values = np.asarray(table)
    if values.shape != (MAXIMUM + 1,) or values.dtype.kind not in "ui" or (
        values.min() < 0 or values.max() > 65535
    ):
        raise ValueError("曲线必须是含 16384 项的无符号 16-bit 整数查找表")
    return values.astype(np.int64, copy=False)


def aggregate_lut(tables, operations):
    """Build an exact 14-bit aggregate table from explicit child tables.

    Operation 0 evaluates at the running value and saves a new reference.
    Operation 1 adds child(reference), wrapping at 16 bits without clipping.
    Operation 2 subtracts child(reference), floors at zero, narrows to a word,
    then clamps to 16383. Operations 1/2 do not update the reference.
    """
    if not isinstance(tables, (list, tuple)) or not isinstance(
        operations, (list, tuple)
    ) or len(tables) != len(operations) or len(tables) > 30:
        raise ValueError("聚合曲线需匹配的查找表与操作列表，最多 30 项")
    if any(type(op) is not int or op not in (0, 1, 2) for op in operations):
        raise ValueError("聚合操作仅接受 0=串行、1=相加、2=相减")
    children = [_table(table) for table in tables]
    value = np.arange(MAXIMUM + 1, dtype=np.int64)
    reference = value.copy()
    for child, operation in zip(children, operations):
        indices = value if operation == 0 else reference
        if indices.min() < 0 or indices.max() > MAXIMUM:
            raise ValueError("聚合阶段输入越出已验证的 14-bit 查找域")
        sampled = child[indices]
        if operation == 0:
            value = sampled.copy()
            reference = value.copy()
        elif operation == 1:
            value = (value + sampled) & 65535
        else:
            difference = value - sampled
            value = np.where(
                difference < 0, 0, np.minimum(difference & 65535, MAXIMUM)
            )
    return np.frombuffer(value.astype("<u2").tobytes(), dtype="<u2")
