"""Recovered LS5000 4.0.3 per-channel inversion LUT numerical kernel.

This is a research entry point for *explicit* seven parameters and two 32-point
control arrays. It does not infer parameters from a camera image, assign scanner
exposure ticks, or claim a match to an independently captured acquisition.

The portable arithmetic follows original x86 functions 0x10010C30 (builder)
and 0x1000F470 (piecewise linear curve), independently replayed in Unicorn from
SHA-256 45afd6fb61a9517ff95d1896dc4257779c319310c2e8bbe75f3b4f3dada920af.
Argument order is A,B,C,E,a,f,g; the captured analyzer block instead stores E,C.
No proprietary executable is needed to use this module.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BuilderParameters:
    A: float
    B: float
    C: float
    E: float
    a: float
    f: float
    g: float

    @classmethod
    def from_analyzer_scalars(cls, values):
        """Convert original in-memory A,B,E,C,a,f,g to public call order."""
        if len(values) != 7:
            raise ValueError("Nikon analyzer scalars must have seven entries")
        A, B, E, C, a, f, g = values
        return cls(A, B, C, E, a, f, g)

    def arguments(self):
        return self.A, self.B, self.C, self.E, self.a, self.f, self.g


def _controls(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != (32,) or y.shape != (32,):
        raise ValueError("Nikon curve needs two 32-point arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nikon control points must be finite")
    if np.any(np.diff(x) < 0):
        raise ValueError("Nikon curve X must be nondecreasing")
    return x, y


def _curve(x, y, t):
    # The original chooses the first X strictly greater than the query, starting
    # at index 1. Below X[0] it extrapolates; above X[31] it returns Y[31].
    right = np.clip(np.searchsorted(x, t, side="right"), 1, 32)
    index = np.minimum(right, 31)
    left = index - 1
    denominator = x[index] - x[left]
    delta = np.divide(
        (y[index] - y[left]) * (t - x[left]),
        denominator,
        out=np.zeros_like(t, dtype=np.float64),
        where=denominator != 0,
    )
    result = delta + y[left]
    return np.where(right == 32, y[31], result)


def curve(x, y, t):
    """Original 32-point linear evaluation, including lower extrapolation.

    Repeated X values use the last matching interval, matching the source's
    strict-greater search. A zero-width first interval returns its left Y.
    """
    x, y = _controls(x, y)
    t = np.asarray(t, dtype=np.float64)
    if not np.isfinite(t).all():
        raise ValueError("Nikon curve query must be finite")
    result = _curve(x, y, t)
    return float(result) if result.ndim == 0 else result


def build_lut(A, B, C, E, a, f, g, x, y, *, original_prefix_count=None):
    """Return the original pre-fixed 65,536-entry uint16 lookup table.

    Inputs are supplied research arguments, not camera acquisition evidence.
    The prefix saturates to 65535. The original stores the low 16 bits of the
    truncated result afterwards; this behavior is preserved for diagnosis,
    rather than silently adding a modern shoulder/clipping policy. At exact
    integer prefix boundaries, x87 CRT pow and platform libm can differ by one
    entry. ``original_prefix_count`` may replay the count observed from the
    original numerical microfunction, with explicit provenance at the caller.
    """
    values = tuple(float(v) for v in (A, B, C, E, a, f, g))
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nikon builder parameters must be finite")
    A, B, C, E, a, f, g = values
    if C <= 0:
        raise ValueError("Nikon builder C must be positive")
    if f < g:
        raise ValueError("Nikon builder f must be >= g to keep prefix in bounds")
    x, y = _controls(x, y)
    density_range = min(max((A - ((1.0 - a) * B + E * a)) / C, 0.01), 4.0)
    prefix = math.trunc(65535.0 * math.pow(10.0, -(f - g))) + 1
    if original_prefix_count is not None:
        if (
            type(original_prefix_count) is not int
            or not 1 <= original_prefix_count <= 65536
            or abs(original_prefix_count - prefix) > 1
        ):
            raise ValueError(
                "Original prefix count must be an adjacent 1..65536 integer"
            )
        prefix = original_prefix_count
    table = np.full(65536, 65535, dtype=np.uint16)
    if prefix >= 65536:
        return table
    code = np.arange(prefix, 65536, dtype=np.float64)
    query = np.log10(65535.0 / code) + g
    mapped = _curve(x, y, query)
    # Original computes (A - mapped) * (-1 / density_range), in this order.
    exponent = (A - mapped) * (-1.0 / density_range)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.power(10.0, exponent) * 65535.0
    if not np.isfinite(result).all() or np.any(result >= 2.0**63):
        raise ValueError(
            "Nikon LUT results exceed the original signed conversion domain"
        )
    table[prefix:] = np.remainder(np.trunc(result), 65536).astype(np.uint16)
    return table
