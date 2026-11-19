"""Original-width Native USM bands for exact saved-image detail samples.

Keeping both the full original row width and the SDK's original row-band
boundaries preserves its SIMD/FMA tails and float64 box-gradient recurrence.
The caller samples only after this original dense band has been processed.
"""

import numpy as np
import negpy_core

from negpy.domain.interfaces import PipelineContext
from negpy.features.lab.logic import _usm_sharpening_block, apply_saturation, gaussian_kernel_1d
from negpy.kernel.system.config import render_scale_factor

from .film_density_domain_v45 import film_domain_for
from .metering_snapshot import _check_source
from .native_spatial_v30 import _SpatialEngine


class _USMBandEngine(_SpatialEngine):
    def _run_stage(self, image, config, cache_field, processor_fn, context, pipeline_changed):
        if cache_field == "lab":
            # Same PhotoLab order as the SDK, but its outer _map_row_blocks
            # has already selected this exact original-width expanded band.
            # Calling apply_usm_sharpening here would partition it a second time.
            def original_band_lab(img, ctx):
                if config.saturation != 1.0 or config.skin_protection > 0:
                    img = apply_saturation(img, config.saturation, config.skin_protection)
                img = _usm_sharpening_block(img, config.sharpen,
                                            config.sharpen_radius, config.sharpen_masking)
                return negpy_core.clip01(np.ascontiguousarray(img, dtype=np.float32))

            processor_fn = original_band_lab
        return super()._run_stage(image, config, cache_field, processor_fn, context, pipeline_changed)


def render_original_usm_band(runner, source, render_config, band_start):
    from .density_preview_v44 import native_usm_supported
    from .live_worker import preview_checkpoint

    if not native_usm_supported(render_config):
        raise ValueError("Exact Native band sampling supports only fixed-mask USM")
    if (render_config.process != runner._effective.process
            or render_config.geometry != runner._effective.geometry
            or render_config.local != runner._effective.local):
        raise ValueError("Native USM cannot change frozen source, geometry or local masks")
    runner._validate()
    _check_source(source, runner._identity)
    height, width = source.shape[:2]
    rows = max(1, 1048576 // width)
    if type(band_start) is not int or not 0 <= band_start < height or band_start % rows:
        raise ValueError("Native USM detail requires an original SDK band boundary")
    end = min(height, band_start + rows)
    halo = max(len(gaussian_kernel_1d(render_config.lab.sharpen_radius)) // 2, 2)
    low, high = max(0, band_start - halo), min(height, end + halo)
    checkpoint = preview_checkpoint()
    checkpoint()
    dense = np.array(source[low:high], dtype=np.float32, order="C", copy=True)
    context = PipelineContext(original_size=source.shape[:2], scale_factor=render_scale_factor(source.shape),
        process_mode=render_config.process.process_mode, wants_uv_grid=False,
        readback_metrics=False, cache_stages=False)
    engine = _USMBandEngine(runner._metrics, runner._sha256, log_domain=film_domain_for(source))
    image = engine.process(dense, render_config, f"{runner._sha256}:usm:{band_start}", context)
    if image.shape != dense.shape or not np.isfinite(image).all():
        raise ValueError("Native USM band produced invalid or changed-geometry output")
    _check_source(source, runner._identity)
    runner._validate()
    checkpoint()
    return np.ascontiguousarray(image, dtype=np.float32), dict(
        source_rectangle=[0, low, width, high], original_core_rows=[band_start, end],
        original_row_band_size=rows, source_halo=halo,
        original_width_and_SIMD_tail_preserved=True, original_box_gradient_reset_preserved=True,
        processed_pixels=int(dense.shape[0] * width), approximate=False)
