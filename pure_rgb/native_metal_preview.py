"""Qualified Metal rendering for Native fit previews with exact CPU metering."""

from collections import OrderedDict
import os
import sys

import numpy as np

from .diagnostics import event, span
from .native_print_metrics import native_print_values


_SUPPORTED_OVERRIDES = {
    "exposure.density", "exposure.grade", "exposure.shadow_density", "exposure.highlight_density",
    "exposure.paper_profile", "exposure.wb_cyan", "exposure.wb_magenta", "exposure.wb_yellow",
    "exposure.auto_exposure", "exposure.auto_normalize_contrast",
    "exposure.shadow_grade", "exposure.highlight_grade", "exposure.preflash",
    "exposure.cast_removal_strength",
}


def metal_tone_allowed(config, rectangle, step, controls):
    backend = os.environ.get("PURE_RGB_COMPUTE_BACKEND", "auto")
    if (sys.platform != "darwin" or backend not in ("auto", "gpu") or step <= 1
            or set(controls["values"]) - _SUPPORTED_OVERRIDES):
        return False
    x0, y0, x1, y1 = rectangle
    if ((x1 - x0 + step - 1) // step) * ((y1 - y0 + step - 1) // step) < 256 * 1024:
        return False
    exposure = config.exposure
    # Leave extreme filtration to the existing CPU screen renderer. CPU and Metal
    # filter/denoise thresholds can diverge on more pixels with aggressive CMY.
    return max(abs(exposure.wb_cyan), abs(exposure.wb_magenta), abs(exposure.wb_yellow)) <= .35


def _agrees_with_cpu(cpu, gpu):
    if gpu.shape != cpu.shape or not np.isfinite(gpu).all():
        return False
    delta = np.abs(gpu - cpu)
    # Compare encoded working RGB. Bound both broad tonal shifts and localized
    # exceptions caused by the two screen grids' Lab threshold rounding.
    return (
        float(delta.mean()) <= 1 / 65535
        and np.count_nonzero(delta > 16 / 65535) <= delta.size // 1000
        and float(delta.max()) <= 8 / 255
    )


class NativeMetalTone:
    def __init__(self):
        self.engine = None
        self.device_failed = False
        self.meter_identity = None
        self.source_key = None
        self.source_grid = None
        self.base_key = None
        self.qualified = OrderedDict()
        self.rejected = set()

    def close(self):
        engine, self.engine = self.engine, None
        self.source_grid = self.source_key = self.base_key = self.meter_identity = None
        self.qualified.clear()
        self.rejected.clear()
        if engine is not None:
            try:
                engine.destroy_all()
            except Exception as error:
                event("native_preview_metal_cleanup_failed", reason=str(error)[:500])

    @property
    def retained_cpu_bytes(self):
        return self.source_grid.nbytes if self.source_grid is not None else 0

    def _failed(self, error):
        self.device_failed = True
        event("native_preview_metal_fallback", reason=str(error)[:500])
        self.close()

    @staticmethod
    def _signature(config):
        exposure = config.exposure
        # A previously qualified neutral print must not activate an untested
        # split-grade branch, preflash, or a different neutral-axis treatment.
        split_signs = tuple(int(value > 0) - int(value < 0)
                           for value in (exposure.shadow_grade, exposure.highlight_grade))
        cast = exposure.cast_removal_strength
        cast_mode = "off" if cast == 0 else "default" if cast == .5 else "custom"
        return (config.lab, exposure.paper_profile, split_signs, exposure.preflash > 0, cast_mode)

    def _bind(self, cache, rectangle, step, base_key, meter_identity):
        from .density_preview_v44 import _key

        x0, y0, x1, y1 = rectangle
        sample = cache.session.source[y0:y1:step, x0:x1:step]
        if self.source_grid is not None and self.source_grid.shape != sample.shape:
            self.close()
        self.source_key = _key(["native-tone-metal", base_key])
        self.source_grid = np.ascontiguousarray(sample)
        self.source_grid.setflags(write=False)
        self.base_key = base_key
        self.meter_identity = meter_identity
        self.qualified.clear()
        self.rejected.clear()

    def _render(self, cache, config, meters):
        overrides = dict(
            bounds_override=meters.get("pre_trim_bounds"),
            metered_anchor_override=meters.get("metered_anchor"),
            textural_range_override=meters.get("textural_range"),
            neutral_axis_override=meters.get("neutral_axis_refs"),
            shadow_point_override=meters.get("shadow_point"),
            highlight_point_override=meters.get("highlight_point"),
        )
        if any(value is None for value in overrides.values()):
            return None
        try:
            from negpy.infrastructure.gpu.device import GPUDevice
            from negpy.kernel.system.config import render_scale_factor
            from negpy.services.rendering.gpu_engine import GPUEngine

            if self.engine is None:
                device = GPUDevice.get()
                if not device.is_available or "Metal" not in str(device.backend_name):
                    raise RuntimeError("A pinned NegPy Metal device is unavailable")
                self.engine = GPUEngine()
            grid = self.source_grid
            with span("native_preview_metal", pixels=int(grid.shape[0] * grid.shape[1])):
                self.engine.evict_stale_textures()
                texture, report = self.engine.process_to_texture(
                    grid, config, scale_factor=render_scale_factor(cache.session.source.shape),
                    readback_metrics=False, source_hash=self.source_key, analysis_source_hash=self.source_key,
                    render_size_ref=max(grid.shape[:2]), **overrides,
                )
                if (texture.height, texture.width) != grid.shape[:2]:
                    raise ValueError("Metal preview texture does not match the certified screen grid")
                handle = self.engine._submit_readback(texture)
                if handle is None:
                    raise RuntimeError("Metal preview readback is unavailable")
                # Pin the same unpacking primitive the SDK uses for tiled output.
                # Its native row loop strips padded alpha directly into RGB.
                pixels = np.empty(grid.shape, dtype=np.float32)
                self.engine._resolve_readback(handle, dest=pixels)
            expected, measured = meters["final_bounds"], report["final_bounds"]
            if (pixels.shape != grid.shape or pixels.dtype != np.float32
                    or not np.isfinite(pixels).all()
                    or not np.allclose(measured.floors, expected.floors, rtol=0, atol=1e-10)
                    or not np.allclose(measured.ceils, expected.ceils, rtol=0, atol=1e-10)
                    or any(report.get(field) != meters[field] for field in (
                        "metered_anchor", "textural_range", "neutral_axis_refs", "shadow_point", "highlight_point"
                    ))):
                raise ValueError("Metal returned different full-source bounds or a malformed preview")
            pixels.setflags(write=False)
            return pixels
        except Exception as error:
            self._failed(error)
            return None

    def qualify(self, cache, config, rectangle, step, meter_identity, cpu):
        key = cpu["base_sample_key"]
        entry = cache.original_base_samples.get(key)
        if self.device_failed or entry is None:
            return
        if key != self.base_key:
            try:
                self._bind(cache, rectangle, step, key, meter_identity)
            except MemoryError as error:
                self._failed(error)
                return
        signature = self._signature(config)
        candidate = (signature, config.exposure)
        if signature in self.qualified or candidate in self.rejected or len(self.rejected) >= 32:
            return
        _, metrics, _ = entry
        if metrics.get("active_roi") is not None:
            return
        pixels = self._render(cache, config, metrics)
        if pixels is None:
            return
        try:
            match = _agrees_with_cpu(cpu["encoded"], pixels)
        except MemoryError as error:
            self._failed(error)
            return
        if match:
            self.qualified[signature] = cpu["target"]
            while len(self.qualified) > 16:
                self.qualified.popitem(last=False)
        else:
            # One failed exposure must stay on CPU without disabling its paper
            # for every later edit. Keep retry cost bounded to distinct recipes.
            self.rejected.add(candidate)
        event("native_preview_metal_qualification", matched=match, paper=config.exposure.paper_profile,
              sample_step=step)

    def render(self, cache, config, export, meter_identity):
        if self.device_failed or meter_identity != self.meter_identity:
            return None
        signature = self._signature(config)
        entry = cache.original_base_samples.get(self.base_key)
        if (signature not in self.qualified or (signature, config.exposure) in self.rejected
                or entry is None):
            return None
        target = self.qualified[signature]
        self.qualified.move_to_end(signature)
        cache.original_base_samples.move_to_end(self.base_key)
        _, meters, _ = entry
        pixels = self._render(cache, config, meters)
        if pixels is None:
            return None
        bounds = meters["final_bounds"]
        return dict(encoded=pixels, target=target, bounds=dict(log_dense=list(bounds.floors),
                    log_thin=list(bounds.ceils)), native_print=native_print_values(config, meters),
                    export=export, base_cache_hit=True, base_sample_key=self.base_key,
                    tone_backend="metal")
