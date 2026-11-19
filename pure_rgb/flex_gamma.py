"""Explicit Mac 4.8.13 inner CGammaCurve arithmetic for research compilers.

This is distinct from outer CGammaNegCurve and its preference-driven formula.
Original numeric loops are checked with an explicitly substituted math.pow
service; original Darwin libm last-bit equivalence remains unverified.
"""

from functools import lru_cache
import math

import numpy as np

from .flex_routing import MAXIMUM


SUPPORTED_GAMMA_RANGE = (0.8, 8.0)


def gamma_curve_parameters(gamma, *, enabled=True):
    """Report the float32 profile field and subsequent original double mapping.

    The currently supported positive domain is bounded by original-loop probes.
    Values near/below 0.75 make the effective parameter nonpositive and are not
    assigned invented behavior. The original catalog's 114 Gamma fields are 2.
    """
    if type(enabled) is not bool:
        raise ValueError("CGammaCurve enabled 必须为布尔值")
    if (
        isinstance(gamma, (bool, np.bool_))
        or not isinstance(gamma, (int, float, np.integer, np.floating))
        or not math.isfinite(float(gamma))
    ):
        raise ValueError("Gamma 字段需有限数值")
    if not SUPPORTED_GAMMA_RANGE[0] <= float(gamma) <= SUPPORTED_GAMMA_RANGE[1]:
        raise ValueError("当前原函数验证合同仅支持 0.8..8.0 的 Gamma 字段")
    field = float(np.float32(gamma))
    effective = field - 1.0
    if effective < 1.0:
        effective = (1.0 - effective) * (-0.8) + 1.0
    return {
        "gamma_field_float32": field,
        "effective_parameter": effective,
        "exponent": 1.0 / effective,
        "enabled": enabled,
        "arithmetic": "mac-4.8.13-i386-CGammaCurve",
        "pow_service": "math.pow substitute; original Darwin libm not verified",
    }


@lru_cache(maxsize=128)
def _cached_gamma(field, enabled):
    parameters = gamma_curve_parameters(field, enabled=enabled)
    if not enabled:
        values = np.arange(MAXIMUM + 1, dtype="<u2")
    else:
        exponent = parameters["exponent"]
        # Scalar libm calls preserve the oracle's external service contract;
        # numpy.power may choose a different vector math implementation.
        values = np.fromiter(
            (
                int(math.pow(code / float(MAXIMUM), exponent) * MAXIMUM)
                for code in range(MAXIMUM + 1)
            ),
            dtype="<u2",
            count=MAXIMUM + 1,
        )
    return np.frombuffer(values.tobytes(), dtype="<u2")


def gamma_curve_lut(gamma, *, enabled=True):
    """Return immutable uint16[16384] for an explicit inner Gamma field.

    Gamma=2 produces identity; it is not a 1/2 exponent. ApplyHistogram controls
    this stage's original active flag. No XML, preferences, CMS or RAW domain is
    inferred by this function.
    """
    parameters = gamma_curve_parameters(gamma, enabled=enabled)
    return _cached_gamma(parameters["gamma_field_float32"], enabled)
