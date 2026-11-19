"""Explicit, instance-bound Native CPU replay with frozen masked auxiliaries.

Only the original engine's base normalization callback is replaced. It never
invokes the original Normalization/geometry/source-base meters. The full image
still uses the original open log and normalization kernels, followed by the
unchanged supported CPU print stages. This module is not a default/UI adapter,
GPU fallback, defect repair, or arbitrary geometry/retouch/spatial-effects path.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .metering_snapshot import _check_source, _json_values, restore_zonal_metering

import negpy_core
from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.models import RenderIntent
from negpy.features.exposure.normalization import normalize_log_image
from negpy.features.geometry.models import GeometryConfig
from negpy.features.process.models import FADE_CARD_FIELDS, ProcessConfig, ProcessMode
from negpy.features.retouch.models import RetouchConfig
from negpy.kernel.system.config import render_scale_factor
from negpy.services.rendering.engine import DarkroomEngine


SCHEMA = "pure-rgb-frozen-native-cpu"
VERSION = 1
SCOPE = "masked-auxiliary-native-cpu-c41-print-v1"
REQUIRED_METRICS = {
    "log_bounds_base",
    "pre_trim_bounds",
    "final_bounds",
    "anchor_bounds",
    "norm_density_range",
    "neutral_axis_refs",
    "metered_anchor",
    "textural_range",
    "shadow_point",
    "highlight_point",
}


@dataclass(frozen=True)
class FrozenNativeCPUResult:
    image: np.ndarray
    metrics: dict
    receipt: dict


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _config_document(config):
    _require(isinstance(config, WorkspaceConfig), "Expected a Native WorkspaceConfig")
    document = config.to_dict()
    # StrEnum values in WorkspaceConfig are JSON strings. Round-trip severs all
    # mutable lists in otherwise frozen configuration dataclasses.
    try:
        text = _canonical(document)
        document = json.loads(text)
        _json_values(document)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "Native configuration must contain finite JSON values"
        ) from error
    return document


def _supported_config(config):
    _config_document(config)
    p, e, lab = config.process, config.exposure, config.lab
    _require(
        p.process_mode is ProcessMode.C41 and not p.positive_source,
        "Frozen Native CPU v1 requires C41 negative print",
    )
    _require(
        e.render_intent == RenderIntent.PRINT,
        "Frozen Native CPU v1 requires print intent",
    )
    _require(
        config.geometry == GeometryConfig(),
        "Frozen Native CPU v1 requires no-op geometry and full output",
    )
    _require(
        config.retouch == RetouchConfig(), "Frozen Native CPU v1 requires no-op retouch"
    )
    _require(
        not config.local.masks,
        "Frozen Native CPU v1 does not support local spatial masks",
    )
    _require(
        p.linear_raw is True,
        "Frozen Native CPU expects already prepared linear sensor bands",
    )
    _require(
        p.crosstalk_strength == 0 and p.sensor_matrix is None,
        "Frozen Native CPU v1 does not support capture/density unmix",
    )
    defaults = ProcessConfig()
    _require(
        all(
            getattr(p, field) == getattr(defaults, field) for field in FADE_CARD_FIELDS
        ),
        "Frozen Native CPU v1 does not support fade/gray estimation",
    )
    trims = (
        p.white_point_offset,
        p.black_point_offset,
        p.white_point_trim_red,
        p.white_point_trim_green,
        p.white_point_trim_blue,
        p.black_point_trim_red,
        p.black_point_trim_green,
        p.black_point_trim_blue,
    )
    _require(
        all(v == 0 for v in trims), "Frozen Native CPU v1 requires neutral point trims"
    )
    _require(
        not p.use_luma_average
        and not p.use_color_average
        and not p.use_cast_average
        and not p.lock_bounds
        and p.locked_neutral_axis is None
        and all(v == 0 for v in p.locked_floors + p.locked_ceils),
        "Frozen Native CPU v1 does not support roll averages/locked meters",
    )
    _require(
        p.manual_dmin is None and p.manual_dmax is None and p.film_base is None,
        "Frozen Native CPU v1 does not support manual endpoints/film-base diagnostics",
    )
    _require(
        e.contrast_mask_strength == 0 and e.diffusion == 0 and lab.clahe_strength == 0,
        "Frozen Native CPU v1 does not support contrast mask/diffusion/CLAHE",
    )
    _require(
        lab.sharpen == 0
        and lab.chroma_denoise == 0
        and lab.glow_amount == 0
        and lab.halation_strength == 0,
        "Frozen Native CPU v1 requires spatial Lab effects disabled",
    )
    _require(
        config.finish.carrier_width == 0 and config.finish.border_size == 0,
        "Frozen Native CPU v1 requires no carrier/paper border",
    )


class _FrozenEngine(DarkroomEngine):
    """Private original CPU engine with only the base-stage callback replaced."""

    def __init__(self, metrics, binding_sha256, *, log_domain=None):
        super().__init__()
        if log_domain is not None:
            from .film_density_domain_v45 import FilmDensityDomain

            _require(type(log_domain) is FilmDensityDomain, "Unknown calibrated log source")
            log_domain.validate()
        self._log_domain = log_domain
        self._frozen_metrics = deepcopy(metrics)
        self._binding_sha256 = binding_sha256
        self.base_normalization_runs = 0

    def _source_base_log(self, img_in, ctx):
        # Original preview-only rebate diagnostics must never sample excluded
        # pixels behind the frozen auxiliary route, even if later code asks.
        return None

    def _normalize(self, image, context):
        self.base_normalization_runs += 1
        context.metrics.update(deepcopy(self._frozen_metrics))
        # The only supported geometry is identity. Preserve its coordinate
        # metadata without invoking a geometric transform or an old meter.
        context.active_roi = None
        context.metrics.update(
            geometry_params={
                "rotation": 0,
                "fine_rotation": 0.0,
                "flip_horizontal": False,
                "flip_vertical": False,
                "converge_v": 0.0,
                "converge_h": 0.0,
            },
            distortion_k1=0.0,
            active_roi=None,
        )
        for key in (
            "base_log",
            "source_base_log",
            "base_source",
            "density_above_base",
            "scan_clip_fractions",
            "histogram_density",
            "density_parade",
            "scan_full_scale",
            "film_base_density",
        ):
            context.metrics[key] = None
        from .film_density_domain_v45 import film_domain_for

        domain = self._log_domain or film_domain_for(image)
        if domain is not None:
            domain.validate()
        image_log = (image if domain is not None else negpy_core.log_density_open(
            np.ascontiguousarray(image, dtype=np.float32)))
        normalized = normalize_log_image(image_log, context.metrics["final_bounds"])
        context.metrics["normalized_log"] = normalized
        context.metrics["frozen_native_binding_sha256"] = self._binding_sha256
        return normalized

    def _run_stage(
        self, img, config, cache_field, processor_fn, context, pipeline_changed
    ):
        if cache_field == "base":
            # Do not call the supplied original run_base closure: it performs
            # source-base reads, geometry, and original Normalization meters.
            config = (config, self._binding_sha256)
            processor_fn = self._normalize
        return super()._run_stage(
            img, config, cache_field, processor_fn, context, pipeline_changed
        )


class FrozenNativeCPU:
    """One source/config/bundle per instance; all other bindings are rejected.

    Construction measures the masked auxiliaries once. render validates current
    bytes and identities before any cache access, and never remeasures meters.
    A different density/mask bundle requires another independent instance.
    Output samples are already working-OETF encoded, with no ICC conversion.
    """

    def __init__(self, source, config, density_bundle_json, current_identity):
        from .native_masked_meters import (
            analyze_native_masked_meters,
            validate_native_masked_result,
        )

        _supported_config(config)
        restored = restore_zonal_metering(density_bundle_json, current_identity)
        _check_source(source, restored.source_identity)
        before_config = _config_document(config)
        measured = analyze_native_masked_meters(
            source, config, density_bundle_json, current_identity
        )
        validate_native_masked_result(
            measured, source, config, density_bundle_json, current_identity
        )
        _check_source(source, restored.source_identity)
        _require(
            _config_document(config) == before_config,
            "Native configuration changed during metering",
        )
        self._initialize(config, restored.source_identity, restored.sha256, measured)

    @classmethod
    def from_auxiliary_snapshot(
        cls,
        snapshot_json,
        source,
        config,
        density_bundle_json,
        current_identity,
        *,
        prefer_gpu=False,
    ):
        """Restore a durable auxiliary snapshot without invoking any meter."""
        from .metering_aux_snapshot import restore_native_auxiliary

        restored = restore_native_auxiliary(
            snapshot_json,
            source,
            config,
            density_bundle_json,
            current_identity,
            prefer_gpu=prefer_gpu,
        )
        instance = cls.__new__(cls)
        instance._initialize(
            config,
            restored.source_identity,
            restored.density_bundle_sha256,
            restored.result,
        )
        return instance

    def _initialize(self, config, identity, bundle_sha256, measured):
        from .native_masked_meters import native_masked_metrics_payload

        self._input_config_document = _config_document(config)
        self._input_config_sha256 = _digest(self._input_config_document)
        self._identity = deepcopy(identity)
        self._bundle_sha256 = bundle_sha256
        _require(
            REQUIRED_METRICS <= set(measured.metrics),
            "Frozen Native auxiliary metrics are incomplete",
        )
        self._effective_config = deepcopy(measured.effective_config)
        self._effective_config_document = _config_document(self._effective_config)
        self._metering_receipt = deepcopy(measured.receipt)
        self._metering_sha256 = measured.sha256
        self._metrics = deepcopy(measured.metrics)
        self._metrics_payload = deepcopy(native_masked_metrics_payload(self._metrics))
        self._binding = {
            "schema": SCHEMA,
            "version": VERSION,
            "scope": SCOPE,
            "source_identity": deepcopy(self._identity),
            "density_bundle_sha256": self._bundle_sha256,
            "input_config_sha256": self._input_config_sha256,
            "effective_config_sha256": _digest(self._effective_config_document),
            "native_masked_meters_sha256": self._metering_sha256,
            "metrics_sha256": _digest(self._metrics_payload),
        }
        self._binding_sha256 = _digest(self._binding)
        self._source_key = f"{self._identity['source_pixels_sha256']}|frozen-native-cpu-v1|{self._binding_sha256}"
        self._engine = _FrozenEngine(self._metrics, self._binding_sha256)

    @property
    def effective_config(self):
        """Independent frozen config for the original-CPU comparison oracle."""
        return deepcopy(self._effective_config)

    @property
    def metering_receipt(self):
        return deepcopy(self._metering_receipt)

    @property
    def binding_sha256(self):
        return self._binding_sha256

    @property
    def base_normalization_runs(self):
        return self._engine.base_normalization_runs

    def render(
        self,
        source,
        config,
        density_bundle_json,
        current_identity,
        *,
        prefer_gpu=False,
        cache_stages=True,
    ):
        _require(
            prefer_gpu is False,
            "Frozen Native supports CPU only; GPU is not a fallback",
        )
        _require(type(cache_stages) is bool, "cache_stages must be an explicit boolean")
        _supported_config(config)
        _require(
            _config_document(config) == self._input_config_document,
            "Frozen Native instance belongs to a different configuration",
        )
        restored = restore_zonal_metering(density_bundle_json, current_identity)
        _require(
            restored.sha256 == self._bundle_sha256
            and restored.source_identity == self._identity,
            "Frozen Native instance belongs to a different source/mask bundle",
        )
        _check_source(source, self._identity)
        context = PipelineContext(
            original_size=tuple(source.shape[:2]),
            scale_factor=render_scale_factor(source.shape),
            process_mode=ProcessMode.C41,
            wants_uv_grid=False,
            readback_metrics=False,
            cache_stages=cache_stages,
        )
        before = self._engine.base_normalization_runs
        try:
            image = self._engine.process(
                source, self._effective_config, self._source_key, context
            )
        finally:
            _check_source(source, self._identity)
        # Both copies are deliberate: vendor stage caches use shallow metric
        # copies and mutable bounds, so public results must not alias them.
        image = np.array(image, dtype=np.float32, order="C", copy=True)
        _require(
            image.shape == source.shape and np.isfinite(image).all(),
            "Frozen Native CPU produced invalid or changed-geometry output",
        )
        image.setflags(write=False)
        metrics = deepcopy(context.metrics)
        receipt = {
            **deepcopy(self._binding),
            "binding_sha256": self._binding_sha256,
            "backend": "cpu",
            "cache_stages": cache_stages,
            "base_stage_cache_hit": self._engine.base_normalization_runs == before,
            "base_normalization_runs": self._engine.base_normalization_runs,
            "original_normalization_called": False,
            "old_auxiliary_meters_called_during_render": False,
            "full_image_log_policy": "original log_density_open: low floor only; no upper clamp",
            "output_domain": "working-OETF-encoded RGB; no ICC conversion",
            "source_values_changed": False,
            "output_crop_changed": False,
            "unsupported_rebate_and_histogram_diagnostics": "Explicit None; no full-source meter fallback",
        }
        return FrozenNativeCPUResult(image, metrics, receipt)
