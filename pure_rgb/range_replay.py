"""Research range controls before bounded engine stages, without output recovery.

The coordinate is positive normalized density (thin=0, dense=1), not radiance,
EV, a measured film base, or a calibrated scanner response. Keep its signed
floating source and frozen measurement separately; every request starts there.
No production renderer imports this opt-in API.
"""

from dataclasses import asdict, dataclass
import math
from numbers import Real

import numpy as np

from .stage_trace import clip_observation, rgb_array


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True)
class DensityRange:
    """Explicit common-channel boundaries on the original density coordinate.

    A lower shadow_boundary includes more low-density values; a higher
    highlight_boundary includes more high-density values. Contracting either
    end deliberately increases truncation at the subsequent bounded adapter.
    This affine diagnostic changes midtones too; it is not the final selective
    shoulder/toe UI or a claim of noise-free scene recovery.
    """

    shadow_boundary: float = 0.0
    highlight_boundary: float = 1.0

    def __post_init__(self):
        low = _finite(self.shadow_boundary, "shadow_boundary")
        high = _finite(self.highlight_boundary, "highlight_boundary")
        if not math.isfinite(high - low) or high - low < 1e-6:
            raise ValueError("Density range width must be at least 1e-6")
        object.__setattr__(self, "shadow_boundary", low)
        object.__setattr__(self, "highlight_boundary", high)

    @property
    def identity(self):
        return self.shadow_boundary == 0 and self.highlight_boundary == 1


def map_density_range(signed_density, window=DensityRange()):
    """Map signed floats once; leave all values unbounded and source untouched."""
    x = rgb_array(signed_density)
    if x.dtype.kind != "f" or not isinstance(window, DensityRange):
        raise ValueError("Explicit DensityRange and signed floating RGB required")
    if window.identity:
        return x
    with np.errstate(over="ignore", invalid="ignore"):
        result = (x.astype(np.float64) - window.shadow_boundary) / (
            window.highlight_boundary - window.shadow_boundary
        )
    if not np.isfinite(result).all():
        raise ValueError("Density range mapping overflowed; retain the original floats")
    return result


def quantize_density_range(signed_density, window=DensityRange(), *, bits=14):
    """The only bounded preparation step; returns codes plus a loss receipt.

    Do not feed these codes back into map_density_range. Replay from the signed
    source when changing a boundary. No gamma, inversion, ICC or fit occurs.
    """
    if type(bits) is not int or bits not in (14, 16):
        raise ValueError("Range replay encoding supports explicit 14 or 16 bits")
    mapped = map_density_range(signed_density, window)
    # One arithmetic policy even when the identity map returns float32 input.
    bounded = np.clip(mapped.astype(np.float64), 0, 1)
    maximum = (1 << bits) - 1
    codes = np.rint(bounded * maximum).astype(np.uint16)
    return codes, {
        "schema": "pure-rgb-signed-density-range-replay",
        "version": 1,
        "domain": "signed positive normalized density; not absolute optical density or EV",
        "window": asdict(window),
        "formula": "(source_density-shadow_boundary)/(highlight_boundary-shadow_boundary)",
        "identity_window": window.identity,
        "same_mapping_for_RGB": True,
        "measurement_and_fit_recomputed": False,
        "upstream_float_source_required_for_next_request": True,
        "physical_recoverability_certified": False,
        "clip": clip_observation(mapped, bounded),
        "quantization": f"rint(clip(mapped,0,1)*{maximum}); no OETF",
        "endpoint_fraction_rgb": {
            "zero": (codes == 0).mean((0, 1)).tolist(),
            "maximum": (codes == maximum).mean((0, 1)).tolist(),
        },
    }


def lookup_loss_report(codes, tables, *, output_maximum):
    """Exact occupied code collisions at a bounded scalar LUT, per channel.

    A plateau may be rounding, curve shape or explicit clamping. Endpoint
    occupancy alone cannot distinguish these causes; this reports code loss,
    not sensor clipping or a percentage of physically recoverable scene detail.
    """
    x = rgb_array(codes)
    if type(output_maximum) is not int or not 1 <= output_maximum <= 65535:
        raise ValueError("Declare the output code ceiling explicitly")
    if x.dtype != np.uint16 or len(tables) != 3:
        raise ValueError("Lookup report requires RGB16 codes and three tables")
    result = []
    for channel, table in enumerate(tables):
        table = np.asarray(table)
        if table.ndim != 1 or table.dtype != np.uint16 or table.size < 2:
            raise ValueError("Lookup tables must be one-dimensional uint16 arrays")
        if table.max() > output_maximum:
            raise ValueError("Lookup output exceeds declared output ceiling")
        values, counts = np.unique(x[..., channel], return_counts=True)
        if values[-1] >= table.size:
            raise ValueError("Input code outside lookup table domain")
        mapped = table[values]
        low, high = mapped == 0, mapped == output_maximum
        result.append(
            {
                "occupied_input_codes": int(values.size),
                "occupied_output_codes": int(np.unique(mapped).size),
                "merged_occupied_input_code_count": int(
                    values.size - np.unique(mapped).size
                ),
                "occupied_input_codes_mapped_to_zero": int(low.sum()),
                "occupied_input_codes_mapped_to_maximum": int(high.sum()),
                "sample_fraction_mapped_to_zero": float(
                    counts[low].sum() / counts.sum()
                ),
                "sample_fraction_mapped_to_maximum": float(
                    counts[high].sum() / counts.sum()
                ),
            }
        )
    return {
        "declared_output_maximum": output_maximum,
        "per_channel": result,
        "interpretation": "Occupied scalar code collisions; not certified clipping, spatial detail, or physical dynamic range",
        "source_must_be_retained_before_this_boundary": True,
    }
