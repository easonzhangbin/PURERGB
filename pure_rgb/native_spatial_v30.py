"""Opt-in full-source spatial rendering around an existing frozen meter packet.

C41/BW base packets remain immutable. Only explicitly bounded post-exposure Lab
settings and original, unkeyed local exposure masks may differ from the meter
configuration. Defects are excluded from metering only: these original spatial
kernels still see the entire source and provide no defect-repair guarantee.
"""

from copy import deepcopy
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import negpy_core

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.lab.models import LabConfig, SharpenMethod
from negpy.features.local.models import (
    LocalAdjustmentsConfig,
    LocalMask,
    MaskKey,
    MaskShape,
)
from negpy.features.process.models import ProcessMode
from negpy.kernel.system.config import render_scale_factor

from .frozen_native_cpu import (
    FrozenNativeCPU,
    FrozenNativeCPUResult,
    _FrozenEngine,
    _config_document,
    _supported_config,
)
from .metering_aux_snapshot import _restore_metrics
from .metering_snapshot import _check_source, _identity, _json_values
from .native_masked_meters import native_masked_metrics_payload


SCHEMA = "pure-rgb-native-spatial-v30"
SCOPE = "frozen-meter-identity-full-source-spatial-cpu-v1"
MAX_JSON_BYTES = 4 * 1024 * 1024
LAB_RANGES = {
    "clahe_strength": (0, 1),
    "chroma_denoise": (0, 10),
    "sharpen": (0, 2),
    "sharpen_radius": (0.1, 10),
    "sharpen_masking": (0, 1),
    "glow_amount": (0, 1),
    "halation_strength": (0, 1),
}
LAB_FIELDS = set(LAB_RANGES) | {"sharpen_method", "sharpen_auto_masking"}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _runtime():
    # The frozen test app runs its own pure_rgb snapshot from the bundle while
    # the pinned negpy sources stay in the workspace; hash each from where it
    # actually loads. In the repository both roots coincide.
    from . import VENDOR_ROOT

    local = Path(__file__).resolve().parents[1]
    own = [
        "pure_rgb/native_spatial_v30.py",
        "pure_rgb/frozen_native_cpu.py",
        "pure_rgb/bw_metering.py",
        "pure_rgb/native_masked_meters.py",
        "pure_rgb/metering_aux_snapshot.py",
        "pure_rgb/film_density_domain_v45.py",
    ]
    vendor = [
        "vendor/negpy/negpy/services/rendering/engine.py",
        "vendor/negpy/negpy/features/exposure/processor.py",
        "vendor/negpy/negpy/features/exposure/normalization.py",
        "vendor/negpy/negpy/features/local/logic.py",
        "vendor/negpy/negpy/features/lab/logic.py",
        "vendor/negpy/negpy/features/lab/processor.py",
    ]
    files = {
        name: hashlib.sha256((local / name).read_bytes()).hexdigest()
        for name in own
    }
    files.update(
        {
            name: hashlib.sha256((VENDOR_ROOT / name.removeprefix("vendor/")).read_bytes()).hexdigest()
            for name in vendor
        }
    )
    return dict(
        files=files,
        kernel=hashlib.sha256(Path(negpy_core.__file__).read_bytes()).hexdigest(),
        numpy=np.__version__,
        python=list(sys.version_info[:2]),
        byteorder=sys.byteorder,
    )


def _meter_configuration(config):
    if config.process.process_mode == ProcessMode.C41:
        _supported_config(config)
    elif config.process.process_mode == ProcessMode.BW:
        from .bw_metering import _configuration

        _configuration(config)
    else:
        raise ValueError(
            "Spatial replay requires a supported C41 or BW meter configuration"
        )


def _local(config):
    _require(type(config) is LocalAdjustmentsConfig, "Unknown local configuration")
    _require(
        type(config.masks) is tuple and len(config.masks) <= 64,
        "At most64 explicit local masks",
    )
    defaults = LocalMask()
    for mask in config.masks:
        _require(type(mask) is LocalMask, "Unknown local mask")
        _require(mask.shape in tuple(MaskShape), "Unknown local mask shape")
        _require(
            mask.key == MaskKey.OFF and mask.grade == mask.flash == 0,
            "Spatial replay allows unkeyed exposure-only local masks",
        )
        _require(
            mask.key_zone == defaults.key_zone
            and mask.key_softness == defaults.key_softness,
            "Inactive local tone-key settings must remain default",
        )
        _require(
            type(mask.enabled) is bool and type(mask.invert) is bool,
            "Local switches must be boolean",
        )
        _require(
            type(mask.stops) in (int, float) and -5 <= mask.stops <= 5,
            "Local exposure must lie in [-5,5] stops",
        )
        _require(
            type(mask.feather) in (int, float) and 0 <= mask.feather <= 0.25,
            "Local feather must lie in [0,.25]",
        )
        vertices = mask.vertices
        count = len(vertices) if type(vertices) is tuple else 0
        expected = 2 if mask.shape == MaskShape.GRADIENT else 3
        _require(
            count == expected if mask.shape != MaskShape.POLYGON else 3 <= count <= 128,
            "Invalid local mask vertex count",
        )
        _require(
            all(
                type(point) is tuple
                and len(point) == 2
                and all(
                    type(v) in (int, float) and np.isfinite(v) and 0 <= v <= 1
                    for v in point
                )
                for point in vertices
            ),
            "Local vertices require finite normalized coordinates",
        )
        if mask.shape == MaskShape.GRADIENT:
            _require(
                sum((a - b) ** 2 for a, b in zip(*vertices)) > 1e-12,
                "Degenerate local gradient",
            )
        elif mask.shape == MaskShape.OVAL:
            c, p, q = vertices
            _require(
                abs((p[0] - c[0]) * (q[1] - c[1]) - (p[1] - c[1]) * (q[0] - c[0]))
                > 1e-8,
                "Degenerate local oval",
            )
        else:
            area = sum(
                a[0] * b[1] - a[1] * b[0]
                for a, b in zip(vertices, vertices[1:] + vertices[:1])
            )
            _require(abs(area) > 1e-8, "Degenerate local polygon")


def _render_configuration(meter, render):
    _meter_configuration(meter)
    _config_document(render)
    _require(type(render.lab) is LabConfig, "Unknown Lab configuration")
    _require(
        _config_document(replace(render, lab=meter.lab, local=meter.local))
        == _config_document(meter),
        "Spatial render may change only Lab spatial settings and local exposure masks",
    )
    for field in fields(LabConfig):
        if field.name not in LAB_FIELDS:
            _require(
                getattr(render.lab, field.name) == getattr(meter.lab, field.name),
                "Spatial render cannot change nonspatial Lab settings",
            )
    for name, (lo, hi) in LAB_RANGES.items():
        value = getattr(render.lab, name)
        _require(
            type(value) in (int, float) and np.isfinite(value) and lo <= value <= hi,
            f"Spatial Lab {name} outside supported range [{lo},{hi}]",
        )
    _require(
        render.lab.sharpen_auto_masking is False,
        "Spatial replay excludes auto-mask sharpening",
    )
    _require(
        render.lab.sharpen_method in tuple(SharpenMethod), "Unknown sharpening method"
    )
    _local(render.local)


class _SpatialEngine(_FrozenEngine):
    def _normalize(self, image, context):
        normalized = super()._normalize(image, context)
        # This is an existing masked auxiliary value, not a full-source
        # diagnostic fallback. Preserve every field of the immutable packet.
        context.metrics["scan_clip_fractions"] = deepcopy(
            self._frozen_metrics["scan_clip_fractions"]
        )
        return normalized


class NativeSpatialCPU:
    """One immutable source/meter/render binding, built from existing base meters.

    This is deliberately opt-in; neither the original frozen providers nor the
    default runtime dispatch are changed. Rendering and restore never meter.
    BW accepts its base record only; registered highlight-reference configs are
    rejected. This render API is not an SDK Session engine replacement.
    """

    @classmethod
    def from_c41(cls, frozen, meter_config, render_config):
        _require(
            type(frozen) is FrozenNativeCPU,
            "Expected an existing FrozenNativeCPU packet",
        )
        _render_configuration(meter_config, render_config)
        payload = native_masked_metrics_payload(frozen._metrics)
        _require(
            _config_document(meter_config) == frozen._input_config_document
            and _digest(frozen._binding) == frozen.binding_sha256
            and frozen._identity == frozen._binding["source_identity"]
            and frozen._bundle_sha256 == frozen._binding["density_bundle_sha256"]
            and _digest(payload) == frozen._binding["metrics_sha256"]
            and _digest(_config_document(frozen.effective_config))
            == frozen._binding["effective_config_sha256"],
            "C41 frozen meter packet or input configuration changed",
        )
        instance = cls.__new__(cls)
        instance._initialize(
            meter_config,
            render_config,
            frozen.effective_config,
            frozen._identity,
            payload,
            "C41",
            frozen.binding_sha256,
            frozen._bundle_sha256,
        )
        return instance

    @classmethod
    def from_bw(cls, state, meter_config, render_config):
        from .bw_metering import BWMaskedState

        _require(
            type(state) is BWMaskedState, "Expected an existing BWMaskedState packet"
        )
        _render_configuration(meter_config, render_config)
        state._validate()
        _require(
            _config_document(meter_config) == state._input_config,
            "BW meter configuration changed",
        )
        record = state._records[_digest(_config_document(state.effective_config))]
        _require(
            record["role"] == "base", "Spatial BW replay requires the base meter packet"
        )
        instance = cls.__new__(cls)
        instance._initialize(
            meter_config,
            render_config,
            state.effective_config,
            state.identity,
            native_masked_metrics_payload(record["metrics"]),
            "BW",
            json.loads(state.to_json())["sha256"],
            state.bundle.sha256,
        )
        return instance

    def _initialize(
        self,
        meter,
        render,
        effective_meter,
        identity,
        metrics,
        parent_type,
        parent_sha,
        bundle_sha,
    ):
        _render_configuration(meter, render)
        _meter_configuration(effective_meter)
        _require(
            parent_type in ("C41", "BW")
            and (parent_type == "BW") == (meter.process.process_mode == ProcessMode.BW),
            "Spatial parent mode mismatch",
        )
        _require(
            all(
                type(value) is str
                and len(value) == 64
                and all(c in "0123456789abcdef" for c in value)
                for value in (parent_sha, bundle_sha)
            ),
            "Invalid spatial parent binding",
        )
        _require(
            _config_document(replace(effective_meter, process=meter.process))
            == _config_document(meter),
            "Effective meter changed outside its frozen process",
        )
        process = replace(
            effective_meter.process,
            analysis_rect=meter.process.analysis_rect,
            analysis_buffer=meter.process.analysis_buffer,
            local_floors=meter.process.local_floors,
            local_ceils=meter.process.local_ceils,
        )
        _require(
            process == meter.process and effective_meter.process.analysis_buffer == 0,
            "Effective meter changed outside frozen bounds/ROI",
        )
        self._meter = deepcopy(meter)
        self._render = deepcopy(render)
        self._effective_meter = deepcopy(effective_meter)
        self._effective = replace(
            deepcopy(effective_meter),
            lab=deepcopy(render.lab),
            local=deepcopy(render.local),
        )
        self._identity = _identity(identity)
        self._metrics = _restore_metrics(deepcopy(metrics))
        _require(
            tuple(effective_meter.process.local_floors)
            == self._metrics["log_bounds_base"].floors
            and tuple(effective_meter.process.local_ceils)
            == self._metrics["log_bounds_base"].ceils,
            "Effective process differs from frozen base bounds",
        )
        _require(
            parent_type != "BW" or self._metrics["neutral_axis_refs"] is None,
            "BW packet cannot contain a colour neutral axis",
        )
        self._payload = dict(
            schema=SCHEMA,
            version=1,
            scope=SCOPE,
            source_identity=deepcopy(self._identity),
            meter_config=_config_document(meter),
            render_config=_config_document(render),
            effective_meter_config=_config_document(effective_meter),
            metrics=deepcopy(metrics),
            parent_type=parent_type,
            parent_binding_sha256=parent_sha,
            density_bundle_sha256=bundle_sha,
            runtime=_runtime(),
        )
        self._sha256 = _digest(self._payload)
        self._engine = _SpatialEngine(self._metrics, self._sha256)

    @property
    def effective_config(self):
        return deepcopy(self._effective)

    @property
    def binding_sha256(self):
        return self._sha256

    @property
    def base_normalization_runs(self):
        return self._engine.base_normalization_runs

    def _validate(self):
        _require(
            _digest(self._payload) == self._sha256
            and _runtime() == self._payload["runtime"]
            and _config_document(self._meter) == self._payload["meter_config"]
            and _config_document(self._render) == self._payload["render_config"]
            and _config_document(self._effective_meter)
            == self._payload["effective_meter_config"]
            and native_masked_metrics_payload(self._metrics) == self._payload["metrics"]
            and self._identity == self._payload["source_identity"]
            and _config_document(self._effective)
            == _config_document(
                replace(
                    self._effective_meter,
                    lab=self._render.lab,
                    local=self._render.local,
                )
            ),
            "Spatial source/config/metrics/runtime binding changed",
        )

    def to_json(self):
        self._validate()
        return _canonical({**self._payload, "sha256": self._sha256})

    @classmethod
    def restore(cls, text, source, meter_config, render_config):
        _render_configuration(meter_config, render_config)
        _require(
            type(text) is str and len(text.encode()) <= MAX_JSON_BYTES,
            "Invalid spatial snapshot size",
        )

        def unique(items):
            result = {}
            for key, value in items:
                _require(key not in result, "Duplicate spatial snapshot key")
                result[key] = value
            return result

        try:
            parsed = json.loads(
                text,
                object_pairs_hook=unique,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
            _json_values(parsed)
        except (TypeError, ValueError, RecursionError, OverflowError) as error:
            raise ValueError("Invalid spatial snapshot JSON") from error
        keys = {
            "schema",
            "version",
            "scope",
            "source_identity",
            "meter_config",
            "render_config",
            "effective_meter_config",
            "metrics",
            "parent_type",
            "parent_binding_sha256",
            "density_bundle_sha256",
            "runtime",
            "sha256",
        }
        _require(
            type(parsed) is dict
            and set(parsed) == keys
            and parsed["schema"] == SCHEMA
            and type(parsed["version"]) is int
            and parsed["version"] == 1
            and parsed["scope"] == SCOPE,
            "Unsupported spatial snapshot schema/version/scope",
        )
        _require(
            parsed["sha256"]
            == _digest(
                {key: value for key, value in parsed.items() if key != "sha256"}
            ),
            "Spatial snapshot checksum changed",
        )
        _require(
            parsed["meter_config"] == _config_document(meter_config)
            and parsed["render_config"] == _config_document(render_config)
            and parsed["runtime"] == _runtime(),
            "Spatial snapshot configuration/runtime changed",
        )
        _check_source(source, parsed["source_identity"])
        effective = WorkspaceConfig.from_flat_dict(parsed["effective_meter_config"])
        instance = cls.__new__(cls)
        instance._initialize(
            meter_config,
            render_config,
            effective,
            parsed["source_identity"],
            parsed["metrics"],
            parsed["parent_type"],
            parsed["parent_binding_sha256"],
            parsed["density_bundle_sha256"],
        )
        _require(
            instance._sha256 == parsed["sha256"],
            "Spatial snapshot effective packet changed on restore",
        )
        return instance

    def render(
        self,
        source,
        meter_config,
        render_config,
        *,
        prefer_gpu=False,
        cache_stages=True,
    ):
        _require(prefer_gpu is False, "Spatial replay supports CPU only")
        _require(type(cache_stages) is bool, "cache_stages must be boolean")
        _render_configuration(meter_config, render_config)
        self._validate()
        _require(
            _config_document(meter_config) == self._payload["meter_config"]
            and _config_document(render_config) == self._payload["render_config"],
            "Spatial instance belongs to different meter/render configurations",
        )
        _check_source(source, self._identity)
        context = PipelineContext(
            original_size=source.shape[:2],
            scale_factor=render_scale_factor(source.shape),
            process_mode=self._effective.process.process_mode,
            wants_uv_grid=False,
            readback_metrics=False,
            cache_stages=cache_stages,
        )
        before = self.base_normalization_runs
        try:
            image = self._engine.process(source, self._effective, self._sha256, context)
        finally:
            _check_source(source, self._identity)
            self._validate()
        image = np.array(image, dtype=np.float32, order="C", copy=True)
        _require(
            image.shape == source.shape and np.isfinite(image).all(),
            "Invalid spatial output",
        )
        image.setflags(write=False)
        receipt = dict(
            schema=SCHEMA,
            scope=SCOPE,
            binding_sha256=self._sha256,
            meter_config_sha256=_digest(self._payload["meter_config"]),
            render_config_sha256=_digest(self._payload["render_config"]),
            effective_render_config_sha256=_digest(_config_document(self._effective)),
            frozen_metrics_sha256=_digest(self._payload["metrics"]),
            parent_type=self._payload["parent_type"],
            meter_packet_role="base",
            parent_binding_sha256=self._payload["parent_binding_sha256"],
            density_bundle_sha256=self._payload["density_bundle_sha256"],
            backend="cpu",
            cache_stages=cache_stages,
            base_stage_cache_hit=self.base_normalization_runs == before,
            base_normalization_runs=self.base_normalization_runs,
            original_normalization_called=False,
            old_auxiliary_meters_called_during_render=False,
            source_values_changed=False,
            output_crop_changed=False,
            spatial_input="entire original full source; bad pixels excluded from density meters only",
            defect_repair_claim=False,
            output_domain="working-OETF-encoded RGB; no ICC conversion",
        )
        return FrozenNativeCPUResult(image, deepcopy(context.metrics), receipt)
