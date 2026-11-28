"""Carry unclamped negative highlight detail into the positive working space."""

from contextlib import contextmanager
from dataclasses import replace
from math import log10

import numpy as np


HEADROOM_CONTROL = "exposure.negative_headroom"
HEADROOM_STRENGTH = "exposure.negative_headroom_strength"
PRINT_EXPOSURE_POLICY = "native-print-luminance-v1"
_BLOCK_PIXELS = 2_097_152
_KNEE_FRACTION = 0.35
_SHOULDER_FRACTION = 0.94
_DENSE_RESERVE = 0.18


def headroom_selected(recipe):
    values = recipe.get("native_controls", {}).get("values", {})
    return HEADROOM_STRENGTH in values or values.get(HEADROOM_CONTROL) is True


def native_headroom_strength(recipe):
    values = recipe.get("native_controls", {}).get("values", {})
    return values.get(HEADROOM_STRENGTH, float(values.get(HEADROOM_CONTROL) is True))


def native_headroom_supported(recipe):
    values = recipe.get("native_controls", {}).get("values", {})
    return (recipe.get("measurement_method") == "negpy" and not recipe.get("source_mode")
            and not recipe.get("native_spatial_controls") and not recipe.get("density_metering")
            and not any(values.get(key) for key in ("exposure.diffusion", "exposure.contrast_mask_strength")))


def native_headroom_active(recipe):
    return headroom_selected(recipe) and native_headroom_supported(recipe)


def native_local_highlight(recipe):
    values = recipe.get("native_controls", {}).get("values", {})
    if recipe.get("measurement_method") == "negpy" and not recipe.get("source_mode"):
        if HEADROOM_STRENGTH in values or native_headroom_active(recipe):
            return None
    return recipe.get("highlight_controls")


class NativeHeadroom:
    """Use pre-paper densities for luminance and the original print for chroma."""

    def __init__(self, strength=1.0):
        self.strength = strength
        self.shoulder = None
        self.receipt = None

    def _calibrate(self, context, processor_fn):
        from negpy.features.exposure.logic import channel_density_ranges, normalized_neutral_axis
        from .positive_float import LUMA

        bounds = context.metrics["final_bounds"]
        ranges = np.asarray(channel_density_ranges(bounds), dtype=np.float32)
        references = context.metrics.get("neutral_axis_refs")
        neutral = normalized_neutral_axis(bounds, references) if references is not None else None
        anchor = np.asarray(neutral[0] if neutral is not None else (0.5, 0.5, 0.5), dtype=np.float32)
        radius = max(float(ranges.max()), 0.25) * 2 + 1.5
        levels = np.linspace(-radius, radius, 4097, dtype=np.float32)
        probe = np.ascontiguousarray((anchor - levels[:, None] / ranges)[None])
        response = processor_fn(probe, replace(context, metrics=context.metrics.copy()))[0]
        luma = np.asarray(LUMA, dtype=np.float32)
        peak = np.maximum.accumulate(response @ luma)
        white = float(peak[-1])
        knee_y, shoulder_y = white * _KNEE_FRACTION, white * _SHOULDER_FRACTION
        knee = float(np.interp(knee_y, peak, levels))
        limit = float(np.interp(white * .98, peak, levels)) + _DENSE_RESERVE
        width = limit - knee
        if not np.isfinite((white, knee, width)).all() or white <= 0 or width <= 0:
            raise ValueError("当前相纸设置无法建立负片高光映射")
        weights = ranges * luma
        self.shoulder = (luma, weights, float(anchor @ weights), knee, knee_y, shoulder_y,
                         (shoulder_y - knee_y) / width, white)
        self.receipt = dict(
            policy="native-negative-density-luminance-v1",
            strength=self.strength,
            luminance_source="signed, calibrated density before the NegPy paper curve",
            chroma_source="original NegPy print RGB ratios",
            output_domain="working-linear SDR before Lab, OETF and integer quantization",
            knee_density=knee,
            shoulder_span_density=width,
            extra_dense_range=_DENSE_RESERVE,
            paper_peak_linear=white,
            paper_knee_linear=knee_y,
            display_shoulder_linear=shoulder_y,
        )

    def _map(self, image, context, processor_fn):
        luma, weights, anchor, knee, knee_y, shoulder_y, slope, white = self.shoulder
        # Density still has the variations that the stock's print response
        # will compress. The terminal rational shoulder has unit derivative
        # at onset, then converges to the measured paper white.
        density = anchor - np.einsum("...c,c->...", image, weights)
        guide = knee_y + slope * np.maximum(density - knee, 0)
        roll = np.maximum(guide - shoulder_y, 0)
        limit = white - shoulder_y
        target = np.minimum(guide, shoulder_y) + limit * (roll / (limit + roll))
        native = processor_fn(image, context)
        native_y = np.einsum("...c,c->...", native, luma)
        delta = np.maximum(native_y - target, 0)
        # A smooth minimum is increasing in both luminances and retains the
        # existing output below the target. A common RGB gain keeps its hue.
        reduction = delta * (delta / (white * .015 + delta))
        if self.strength != 1:
            reduction *= self.strength
        rolled_y = native_y - reduction
        gain = np.ones_like(native_y)
        np.divide(rolled_y, native_y, out=gain, where=native_y > 0)
        return np.ascontiguousarray(native * gain[..., None])

    def render(self, image, context, processor_fn):
        if self.strength == 0:
            self.receipt = dict(policy="native-negative-density-luminance-v1", strength=0.0, identity=True)
            return processor_fn(image, context)
        if self.shoulder is None:
            self._calibrate(context, processor_fn)
        rows = max(1, _BLOCK_PIXELS // image.shape[1])
        if rows >= len(image):
            return self._map(image, context, processor_fn)
        from .live_worker import preview_checkpoint

        checkpoint = preview_checkpoint()
        result = np.empty_like(image)
        for start in range(0, len(image), rows):
            checkpoint()
            block = image[start:start + rows]
            result[start:start + rows] = self._map(block, context, processor_fn)
        return result


def print_exposure_receipt(ev):
    return {"policy": PRINT_EXPOSURE_POLICY, "ev": ev}


def _print_luminance_at_baseline_color(original, exposed):
    from .positive_float import LUMA

    luma = LUMA.astype(np.float32)
    target = np.einsum("...c,c->...", exposed, luma)
    baseline = np.einsum("...c,c->...", original, luma)
    peak = np.max(original, axis=-1)
    gain = np.zeros_like(baseline)
    np.divide(np.minimum(target, baseline / np.maximum(peak, 1e-10)), baseline,
              out=gain, where=baseline > 0)
    result = original * gain[..., None]
    np.copyto(result, target[..., None], where=(baseline <= 0)[..., None])
    return result


class NativePrintExposure:
    """Take exposed paper luminance and retain the unshifted linear RGB ratios."""

    def __init__(self, config, ev, *, strength=0.0):
        from negpy.features.exposure.logic import density_to_cmy

        self.ev = ev
        self.headroom = NativeHeadroom(strength)
        offset = density_to_cmy(-ev * log10(2.0))
        filters = config.exposure
        self.shifted = replace(config, exposure=replace(
            filters, wb_cyan=filters.wb_cyan + offset, wb_magenta=filters.wb_magenta + offset,
            wb_yellow=filters.wb_yellow + offset,
        ))

    @property
    def receipt(self):
        return self.headroom.receipt

    def _shifted_print(self, image, context):
        from negpy.features.process.fade import render_white_space, with_fade_white
        from negpy.features.process.hue import apply_hue_trim
        from negpy.services.rendering.engine import exposure_processor

        config = self.shifted
        measured = replace(config, process=with_fade_white(config.process, context.metrics.get("fade_white")))
        white = render_white_space(config.process, context.process_mode, context.cam_xyz, context.camera_wb)
        return apply_hue_trim(exposure_processor(measured, white).process(image, context), config.process.hue_trim)

    def _block(self, image, context, processor_fn):
        original = processor_fn(image, context)
        exposed = self.headroom.render(image, replace(context, metrics=context.metrics.copy()), self._shifted_print)
        return _print_luminance_at_baseline_color(original, exposed)

    def render(self, image, context, processor_fn):
        if not self.ev:
            return self.headroom.render(image, context, processor_fn)
        # SDK spatial inputs are defined on the original exposure raster.
        rows = max(1, _BLOCK_PIXELS // image.shape[1])
        if (rows >= len(image) or self.shifted.local.masks
                or context.metrics.get("contrast_mask_plane") is not None
                or context.metrics.get("diffusion_plane") is not None):
            return self._block(image, context, processor_fn)
        from .live_worker import preview_checkpoint

        check = preview_checkpoint()
        result = np.empty_like(image)
        for start in range(0, len(image), rows):
            check()
            result[start:start + rows] = self._block(image[start:start + rows], context, processor_fn)
        return result


@contextmanager
def native_headroom_engine(processor, *, strength=1.0, print_ev=0.0):
    """Install the same CPU print-stage adapter used by sampled previews."""
    from negpy.services.rendering.engine import DarkroomEngine

    class HeadroomEngine(DarkroomEngine):
        mapping = None

        @property
        def receipt(self):
            return self.mapping.receipt if self.mapping is not None else None

        def process(self, img, settings, source_hash, context=None):
            self.mapping = (NativePrintExposure(settings, print_ev, strength=strength)
                            if print_ev else NativeHeadroom(strength))
            return super().process(img, settings, source_hash, context)

        def _run_stage(self, img, config, cache_field, processor_fn, context, pipeline_changed):
            if cache_field == "exposure":
                print_pixels = processor_fn

                def render_print(pixels, ctx):
                    return self.mapping.render(pixels, ctx, print_pixels)

                processor_fn = render_print
            return super()._run_stage(img, config, cache_field, processor_fn, context, pipeline_changed)

    previous = processor.engine_cpu
    engine = HeadroomEngine()
    processor.engine_cpu = engine
    try:
        yield engine
    finally:
        processor.engine_cpu = previous
