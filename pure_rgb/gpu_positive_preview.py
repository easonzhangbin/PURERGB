"""Explicit Metal preview of the shared positive-domain output controls.

The verified CPU save path is deliberately outside this module.  This renderer
accepts already inverted RGB16 working pixels; it never meters a negative,
opens RAW, calls scanner CMS, or changes a file/profile.  Float32 Metal math can
differ from the float64 CPU reference by a small number of RGB16 codes.
"""

import threading
import time
import os
from functools import lru_cache

import numpy as np

from .professional_controls import OUTPUT_SPECS, checked_controls
from .tone_study import GAMMA
from .positive_lut import exact_cpu_lut_codes as exact_cpu_lut_codes, exact_output_lut


_SHADER = r"""
struct Parameters {
    gain_contrast: vec4<f32>,
    saturation_gamma: vec4<f32>,
    counts: vec4<u32>,
};
@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> destination: array<u32>;
@group(0) @binding(2) var<uniform> settings: Parameters;
@group(0) @binding(3) var<storage, read_write> clipping: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> exact_lut: array<u32>;

fn rawcode(i: u32) -> u32 {
    let packed = source[i / 2u];
    return (packed >> ((i & 1u) * 16u)) & 65535u;
}
fn code(i: u32) -> f32 {
    return f32(rawcode(i)) / 65535.0;
}
fn pixel(i: u32) -> vec3<u32> {
    if (i >= settings.counts.x) { return vec3<u32>(0u); }
    let base = i * 3u;
    if (settings.counts.y == 1u) {
        let r = exact_lut[rawcode(base) * 3u];
        let g = exact_lut[rawcode(base + 1u) * 3u + 1u];
        let b = exact_lut[rawcode(base + 2u) * 3u + 2u];
        let above = (r >> 16u) + (g >> 16u) + (b >> 16u);
        if (above > 0u) { atomicAdd(&clipping[1], above); }
        return vec3<u32>(r, g, b) & vec3<u32>(65535u);
    }
    let encoded = vec3<f32>(code(base), code(base + 1u), code(base + 2u));
    var linear = pow(encoded, vec3<f32>(settings.saturation_gamma.y));
    linear *= settings.gain_contrast.xyz;
    if (settings.gain_contrast.w != 1.0) {
        linear = 0.18 * pow(linear / 0.18, vec3<f32>(settings.gain_contrast.w));
    }
    if (settings.saturation_gamma.x != 1.0) {
        let luma = dot(linear, vec3<f32>(0.297344975, 0.627363566, 0.075291459));
        linear = vec3<f32>(luma) + settings.saturation_gamma.x * (linear - vec3<f32>(luma));
    }
    let below = u32(linear.x < 0.0) + u32(linear.y < 0.0) + u32(linear.z < 0.0);
    let above = u32(linear.x > 1.0) + u32(linear.y > 1.0) + u32(linear.z > 1.0);
    if (below > 0u) { atomicAdd(&clipping[0], below); }
    if (above > 0u) { atomicAdd(&clipping[1], above); }
    let bounded = clamp(linear, vec3<f32>(0.0), vec3<f32>(1.0));
    let result = pow(bounded, vec3<f32>(settings.saturation_gamma.z)) * 65535.0;
    return vec3<u32>(round(result));
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {
    let first = index.x * 2u;
    if (first >= settings.counts.x) { return; }
    let a = pixel(first);
    let b = pixel(first + 1u);
    let packed = index.x * 3u;
    destination[packed] = a.x | (a.y << 16u);
    destination[packed + 1u] = a.z | (b.x << 16u);
    destination[packed + 2u] = b.y | (b.z << 16u);
}
"""


def available_adapter():
    """Inspect a real high-performance adapter; no CPU/software fallback claim."""
    import wgpu

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        return None, dict(available=False, reason="未发现可用 GPU")
    info = dict(adapter.info)
    metal = info.get("backend_type") == "Metal"
    return adapter, dict(
        available=metal,
        device=info.get("device"),
        backend=info.get("backend_type"),
        adapter_type=info.get("adapter_type"),
        scope="Positive RGB controls preview only; CPU verified saves unchanged",
        reason="" if metal else "当前预览加速仅验证 Metal",
    )


class MetalPositivePreview:
    """Bounded storage-buffer compute, including host upload and GPU readback."""

    def __init__(self, *, chunk_pixels=4 * 1024 * 1024):
        import wgpu

        if type(chunk_pixels) is not int or not 1 <= chunk_pixels <= 8 * 1024 * 1024:
            raise ValueError("GPU预览块大小无效")
        started = time.perf_counter()
        adapter, self.report = available_adapter()
        if not self.report["available"]:
            raise RuntimeError(self.report["reason"])
        self.device = adapter.request_device_sync()
        self.chunk_pixels = chunk_pixels
        self._lock = threading.Lock()
        self._buffers = None
        self._capacity = 0
        self._closed = False
        self._lut_key = None
        self._wgpu = wgpu
        shader = self.device.create_shader_module(code=_SHADER)
        self.pipeline = self.device.create_compute_pipeline(
            layout="auto", compute={"module": shader, "entry_point": "main"}
        )
        self.report["initialization_seconds"] = time.perf_counter() - started

    def _prepare(self, count):
        W = self._wgpu
        pairs = (count + 1) // 2
        capacity = pairs * 12
        if capacity <= self._capacity:
            return
        if self._buffers:
            for buffer in self._buffers:
                buffer.destroy()
        self._buffers = [
            self.device.create_buffer(
                size=capacity, usage=W.BufferUsage.STORAGE | W.BufferUsage.COPY_DST
            ),
            self.device.create_buffer(
                size=capacity, usage=W.BufferUsage.STORAGE | W.BufferUsage.COPY_SRC
            ),
            self.device.create_buffer(
                size=48, usage=W.BufferUsage.UNIFORM | W.BufferUsage.COPY_DST
            ),
            self.device.create_buffer(
                size=8,
                usage=W.BufferUsage.STORAGE
                | W.BufferUsage.COPY_SRC
                | W.BufferUsage.COPY_DST,
            ),
            self.device.create_buffer(
                size=65536 * 3 * 4, usage=W.BufferUsage.STORAGE | W.BufferUsage.COPY_DST
            ),
        ]
        self._capacity = capacity
        self.group = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                dict(binding=i, resource=dict(buffer=buffer))
                for i, buffer in enumerate(self._buffers)
            ],
        )

    def output_codes(self, pixels, document, common_ev=0.0):
        """Apply only the existing shared RGB controls; returns codes and receipt."""
        overrides = checked_controls(document, OUTPUT_SPECS)["values"]
        if (
            type(common_ev) not in (float, int)
            or not np.isfinite(common_ev)
            or not -12 <= common_ev <= 12
        ):
            raise ValueError("GPU预览曝光无效")
        if (
            not isinstance(pixels, np.ndarray)
            or pixels.dtype != np.uint16
            or pixels.ndim != 3
            or pixels.shape[2] != 3
            or not pixels.size
        ):
            raise ValueError("GPU预览仅接受非空 RGB16 工作像素")
        if not overrides and common_ev == 0:
            return pixels, dict(identity=True, backend="identity-no-dispatch")
        values = {
            spec.key: overrides.get(spec.key, spec.default) for spec in OUTPUT_SPECS
        }
        gains = [
            2 ** (common_ev + values["exposure_ev"] + values[f"{c}_ev"])
            for c in ("red", "green", "blue")
        ]
        started = time.perf_counter()
        flat = np.ascontiguousarray(pixels).reshape(-1, 3)
        result = np.empty_like(flat)
        below = above = 0
        dispatches = 0
        with self._lock:
            if self._closed:
                raise RuntimeError("GPU预览设备已释放")
            for start in range(0, len(flat), self.chunk_pixels):
                block = flat[start : start + self.chunk_pixels]
                count = len(block)
                self._prepare(count)
                source, output, uniform, metrics, lut_buffer = self._buffers
                queue = self.device.queue
                exact = values["saturation"] == 1
                if exact:
                    key = (*gains, values["contrast"])
                    if key != self._lut_key:
                        self._lut = exact_output_lut(document, common_ev)
                        self._lut_key = key
                    queue.write_buffer(lut_buffer, 0, self._lut)
                # Pairs of pixels occupy three u32s; pad the odd last pixel.
                if count % 2:
                    padded = np.zeros((count + 1, 3), dtype=np.uint16)
                    padded[:count] = block
                    upload = padded
                else:
                    upload = block
                queue.write_buffer(source, 0, upload)
                params = np.zeros(12, dtype=np.float32)
                params[:4] = [*gains, values["contrast"]]
                params[4:7] = [values["saturation"], GAMMA, 1 / GAMMA]
                params.view(np.uint32)[8] = count
                params.view(np.uint32)[9] = int(exact)
                queue.write_buffer(uniform, 0, params)
                queue.write_buffer(metrics, 0, np.zeros(2, dtype=np.uint32))
                encoder = self.device.create_command_encoder()
                compute = encoder.begin_compute_pass()
                compute.set_pipeline(self.pipeline)
                compute.set_bind_group(0, self.group)
                compute.dispatch_workgroups(((count + 1) // 2 + 255) // 256)
                compute.end()
                queue.submit([encoder.finish()])
                raw = queue.read_buffer(output, 0, ((count + 1) // 2) * 12)
                result[start : start + count] = np.frombuffer(raw, dtype=np.uint16)[
                    : count * 3
                ].reshape(count, 3)
                clipping = np.frombuffer(queue.read_buffer(metrics), dtype=np.uint32)
                below += int(clipping[0])
                above += int(clipping[1])
                dispatches += 1
        return result.reshape(pixels.shape), dict(
            identity=False,
            backend="Metal",
            device=self.report["device"],
            scope="Positive RGB preview; original CPU saves unchanged",
            arithmetic="exact-uint16-LUT"
            if values["saturation"] == 1
            else "float32-coupled-RGB",
            elapsed_seconds=time.perf_counter() - started,
            dispatches=dispatches,
            below_zero_fraction=below / pixels.size,
            above_one_fraction=above / pixels.size,
            maximum_expected_code_error_not_certified=values["saturation"] != 1,
            profile_modified=False,
            source_modified=False,
        )

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._buffers:
                for buffer in self._buffers:
                    buffer.destroy()
            self._buffers = None
            self.device.destroy()


@lru_cache(maxsize=1)
def _gpu_capabilities():
    try:
        _, report = available_adapter()
        return report
    except Exception as error:
        return dict(
            available=False,
            backend=None,
            device=None,
            reason=str(error),
            scope="Positive controls preview only",
        )


def gpu_info():
    """Actual adapter status, cached independently of device/shader startup."""
    return dict(_gpu_capabilities())


_DEFAULT_LOCK = threading.Lock()
_DEFAULT_GPU = None
_INITIALIZING = None
_INITIALIZATION_ERROR = None
AUTO_MIN_PIXELS = 500_000


def warm_gpu():
    """Initialize once in the background; Auto never blocks its first edit."""
    global _INITIALIZING
    with _DEFAULT_LOCK:
        if (
            _DEFAULT_GPU is not None
            or _INITIALIZING is not None
            or _INITIALIZATION_ERROR
        ):
            return _INITIALIZING

        def initialize():
            global _DEFAULT_GPU, _INITIALIZATION_ERROR
            try:
                renderer = MetalPositivePreview()
                renderer.output_codes(
                    np.zeros((1, 2, 3), np.uint16),
                    dict(version=1, values=dict(exposure_ev=0.01)),
                )
                with _DEFAULT_LOCK:
                    _DEFAULT_GPU = renderer
            except Exception as error:
                with _DEFAULT_LOCK:
                    _INITIALIZATION_ERROR = str(error)

        _INITIALIZING = threading.Thread(
            target=initialize, name="Metal-preview-init", daemon=True
        )
        _INITIALIZING.start()
        return _INITIALIZING


def preview_output_codes(pixels, document=None, common_ev=0.0, *, backend=None):
    """Runtime dispatch for PREVIEWS; never called by the save/export pipeline.

    Auto permits only exact separable math, uses CPU for small frames, and
    warms GPU asynchronously. Explicit GPU also permits approximate coupled
    saturation. Device failure falls back to the original checked CPU route.
    """
    from .professional_controls import output_codes

    document = document or dict(version=1, values={})
    values = checked_controls(document, OUTPUT_SPECS)["values"]
    selected = backend or os.environ.get("PURE_RGB_COMPUTE_BACKEND", "auto")
    if selected not in {"auto", "cpu", "gpu"}:
        raise ValueError("未知计算后端")
    if not values and common_ev == 0:
        return pixels, dict(
            identity=True, backend="identity-no-dispatch", scope="Positive RGB preview"
        )
    exact = values.get("saturation", 1) == 1
    count = pixels.shape[0] * pixels.shape[1]
    permitted = selected == "gpu" or (
        selected == "auto" and exact and count >= AUTO_MIN_PIXELS
    )
    renderer = None
    reason = (
        "CPU selected"
        if selected == "cpu"
        else "Small preview uses CPU LUT"
        if count < AUTO_MIN_PIXELS
        else "Coupled saturation keeps exact CPU in Auto"
    )
    if permitted:
        initialization = warm_gpu()
        if selected == "gpu" and initialization is not None:
            initialization.join(timeout=10)
        with _DEFAULT_LOCK:
            renderer = _DEFAULT_GPU
            reason = (
                _INITIALIZATION_ERROR
                or "GPU warming asynchronously; current edit uses CPU"
            )
        if renderer is not None:
            try:
                return renderer.output_codes(pixels, document, common_ev)
            except Exception as error:
                reason = "GPU dispatch failed; CPU fallback: " + str(error)
    codes, receipt = output_codes(pixels, document, common_ev)
    return codes, {
        **receipt,
        "backend": "CPU",
        "fallback_reason": reason,
        "scope": "Positive RGB preview",
        "arithmetic": "exact-LUT" if exact else "reference-float64",
    }
