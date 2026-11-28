"""Run the original Lab stages on only the input pixels needed by fit grids."""

from dataclasses import dataclass

import numpy as np
from negpy.features.lab.logic import chroma_denoise_sigma, gaussian_kernel_1d
from negpy.features.lab.models import SharpenMethod

from .diagnostics import span


_MIN_SOURCE_PIXELS = 1024 * 1024
_TILE_PIXELS = 4 * 1024 * 1024
_FILTER_ALIGNMENT = 16


def _lab_halo(config, scale_factor):
    if config.glow_amount > 0 or config.halation_strength > 0:
        return None
    if config.sharpen > 0 and (config.sharpen_auto_masking or config.sharpen_method != SharpenMethod.USM):
        return None
    sigma = chroma_denoise_sigma(config.chroma_denoise, scale_factor)
    chroma = len(gaussian_kernel_1d(sigma)) // 2 + 1 if sigma else 0
    sharpen = max(len(gaussian_kernel_1d(config.sharpen_radius)) // 2, 2) if config.sharpen > 0 else 0
    # Chroma's low-pass gradient reads one extra pixel. Fixed USM reads
    # blur or gradient/morphology support from the preceding chroma result.
    return chroma + sharpen


@dataclass(frozen=True)
class LabGridSchedule:
    grids: tuple
    source_pixels: int
    scheduled_pixels: int
    left: int
    right: int
    bands: tuple

    def render(self, image, processor, context, *, check=None, fused=False):
        outputs = tuple(np.empty(((y1 - y0 + step - 1) // step, (x1 - x0 + step - 1) // step, 3),
                                 dtype=np.float32)
                        for (x0, y0, x1, y1), step in self.grids)
        name = "native_preview_fused_rgb_tiles" if fused else "native_preview_sampled_lab"
        with span(name, source_pixels=self.source_pixels, scheduled_pixels=self.scheduled_pixels,
                  grids=len(self.grids), bands=len(self.bands)):
            for start, end, low, high in self.bands:
                if check is not None:
                    check()
                tile = processor(image[low:high, self.left:self.right], context)
                for output, ((x0, y0, x1, y1), step) in zip(outputs, self.grids):
                    first = max(0, (start - y0 + step - 1) // step)
                    stop = min(len(output), (min(end, y1) - y0 + step - 1) // step)
                    if first < stop:
                        output[first:stop] = tile[y0 + first * step - low:y0 + stop * step - low:step,
                                                  x0 - self.left:x1 - self.left:step]
        return outputs


def schedule_lab_grids(shape, config, context, grids):
    """Plan only profitable tiles with the original scale, phase and support."""
    height, width = shape[:2]
    if context.cache_stages or height * width < _MIN_SOURCE_PIXELS:
        return None
    halo = _lab_halo(config, context.scale_factor)
    if not halo:
        return None

    starts_x, starts_y, ends_x, ends_y = zip(*(bounds for bounds, _ in grids))
    left = max(0, (min(starts_x) - halo) // _FILTER_ALIGNMENT * _FILTER_ALIGNMENT)
    right = min(width, (max(ends_x) + halo + _FILTER_ALIGNMENT - 1)
                // _FILTER_ALIGNMENT * _FILTER_ALIGNMENT)
    top, bottom = min(starts_y), max(ends_y)
    tile_width = right - left
    rows = max(1, _TILE_PIXELS // tile_width)
    bands = [(start, min(start + rows, bottom)) for start in range(top, bottom, rows)]
    # Keep the horizontal SIMD/tail position from the full source. Since the
    # candidate ends on a vector boundary or the real edge, no new scalar
    # tail can reach a requested pixel through the halo.
    read_bounds = [(max(0, start - halo), min(height, end + halo)) for start, end in bands]
    scheduled_pixels = tile_width * sum(high - low for low, high in read_bounds)
    if scheduled_pixels > height * width * 0.85:
        return None
    return LabGridSchedule(tuple(grids), height * width, scheduled_pixels, left, right,
                           tuple((*band, *read) for band, read in zip(bands, read_bounds)))


def sample_lab_grids(image, config, processor, context, grids, *, check=None):
    """Sample the original Lab stage after any full-image CLAHE."""
    schedule = schedule_lab_grids(image.shape, config, context, grids)
    return None if schedule is None else schedule.render(image, processor, context, check=check)
