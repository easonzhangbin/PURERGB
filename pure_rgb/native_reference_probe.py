"""Native-only diagnostic: move endpoints after measuring original references."""

from dataclasses import replace

import numpy as np

from .density_matrix import fixed_process


def post_reference_process(process, baseline, target, shape):
    """Use existing Native point trims, leaving the base analysis unchanged.

    This is not a shared renderer bridge and does not enable point trims in it.
    Native's pre-trim color/anchor measurements keep the original base window.
    """
    if baseline.analysis.roi != target.analysis.roi:
        raise ValueError("Reference isolation requires one frozen picture region")
    base = fixed_process(process, baseline, shape)
    wp = np.subtract(target.log_dense, baseline.log_dense)
    bp = np.subtract(target.log_thin, baseline.log_thin)
    return replace(
        base,
        white_point_trim_red=float(wp[0]),
        white_point_trim_green=float(wp[1]),
        white_point_trim_blue=float(wp[2]),
        black_point_trim_red=float(bp[0]),
        black_point_trim_green=float(bp[1]),
        black_point_trim_blue=float(bp[2]),
    )
