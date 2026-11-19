"""Display-referred paired color observations, not scanner colorimetry.

Inputs must already be explicitly converted to sRGB16 with their actual ICC.
CIE Lab D65 is used only to describe differences between paired hypotheses;
there is no claim that a photographed object has a known neutral color.
"""

import numpy as np


def srgb16_to_lab_d65(pixels):
    values = np.asarray(pixels)
    if (
        values.dtype != np.uint16
        or values.ndim != 3
        or values.shape[-1] != 3
        or not values.size
    ):
        raise ValueError("Require nonempty explicitly ICC-converted sRGB16 HWC")
    rgb = values.astype(np.float64) / 65535
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = np.einsum(
        "...c,dc->...d",
        linear,
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
    )
    relative = xyz / np.asarray([0.95047, 1, 1.08883])
    threshold = 6 / 29
    f = np.where(
        relative > threshold**3,
        np.cbrt(relative),
        relative / (3 * threshold**2) + 4 / 29,
    )
    return np.stack(
        [
            116 * f[..., 1] - 16,
            500 * (f[..., 0] - f[..., 1]),
            200 * (f[..., 1] - f[..., 2]),
        ],
        axis=-1,
    )


def paired_color_observation(before_srgb16, after_srgb16, *, mask=None):
    before = srgb16_to_lab_d65(before_srgb16)
    after = srgb16_to_lab_d65(after_srgb16)
    if before.shape != after.shape:
        raise ValueError("Paired observations require identical geometry")
    if mask is None:
        mask = np.ones(before.shape[:2], dtype=bool)
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != bool
        or mask.shape != before.shape[:2]
        or not mask.any()
    ):
        raise ValueError("Observation mask must be nonempty bool HW")
    delta = (after - before)[mask]
    return {
        "domain": "paired ICC-converted sRGB16 -> CIE Lab D65 display observations",
        "selected_pixels": int(mask.sum()),
        "selected_pixel_fraction": float(mask.mean()),
        "mean_delta_L_a_b": delta.mean(0).tolist(),
        "median_delta_L_a_b": np.median(delta, axis=0).tolist(),
        "ab_difference_p50_p90": np.quantile(
            np.linalg.norm(delta[:, 1:], axis=1), [0.5, 0.9]
        ).tolist(),
        "not_color_accuracy_or_factory_equivalence": True,
        "brightness_and_chroma_reported_separately": True,
    }
