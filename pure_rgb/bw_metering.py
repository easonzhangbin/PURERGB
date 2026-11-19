"""Single-frame BW metering with original NegPy curves and frozen CPU auxiliaries.

The existing C41 providers are unchanged. HWC3 describes one decoded Bayer RAW,
not three captures. Only the base normalization callback is replaced: the BW
density collapse, print curve and original default Lab stages remain vendor code.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import negpy_core

from negpy.domain.interfaces import PipelineContext
from negpy.features.exposure import normalization as norm
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, RenderIntent
from negpy.features.geometry.models import GeometryConfig
from negpy.features.lab.models import LabConfig
from negpy.features.process.models import FADE_CARD_FIELDS, ProcessConfig, ProcessMode
from negpy.features.retouch.models import RetouchConfig
from negpy.kernel.system.config import render_scale_factor
from negpy.services.rendering.engine import DarkroomEngine

from .density_matrix import DensityWindow, check_density_domain, negpy_window
from .engines import Analysis
from .frozen_native_cpu import _config_document
from .metering_aux_snapshot import _restore_metrics
from .metering_masks import _shape, build_metering_mask, validate_metering_mask
from .metering_regions import _rectangle, metering_process, select_metering_roi
from .metering_snapshot import (
    COORDINATE_SPACE,
    SOURCE_DOMAIN,
    _check_source,
    _identity,
    freeze_zonal_metering,
    restore_zonal_metering,
)
from .native_masked_meters import (
    _luma,
    _packet,
    _spatial_activity,
    frozen_process_for_window,
    native_masked_metrics_payload,
)
from .zonal_metering import MAX_LEAVES

SCHEMA = "pure-rgb-single-bw-masked-native-state"
SCOPE = "single-frame-bw-print-frozen-meters-v1"
MAX_JSON_BYTES = 32 * 1024 * 1024
_CONTROL_KEYS = {
    "version",
    "enabled",
    "zones",
    "blocks_per_zone",
    "excluded_rois",
    "padding",
    "min_area_fraction",
    "include",
    "exclude",
}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _pixels(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def checked_bw_metering(document=None):
    """Strict JSON metadata only. Source-relative geometry is checked at prepare."""
    if document is None:
        return None
    _require(type(document) is dict, "BW metering requires a versioned object")
    _require(
        type(document.get("version")) is int and document["version"] == 1,
        "Unsupported BW metering version",
    )
    _require(type(document.get("enabled")) is bool, "BW enabled must be boolean")
    if not document["enabled"]:
        _require(
            set(document) == {"version", "enabled"},
            "Disabled BW metering cannot hide active fields",
        )
        return None
    _require(
        set(document) == _CONTROL_KEYS,
        "Active BW metering fields are incomplete or unknown",
    )
    try:
        _require(
            len(_canonical(document).encode()) <= 512 * 1024,
            "BW metering metadata exceeds size bound",
        )
    except (TypeError, OverflowError) as error:
        raise ValueError("BW metering metadata must be finite JSON") from error
    result = deepcopy(document)
    for key in ("zones", "blocks_per_zone"):
        pair = result[key]
        _require(
            type(pair) is list
            and len(pair) == 2
            and all(type(v) is int and 1 <= v <= MAX_LEAVES for v in pair),
            "BW zone/block pairs require positive integers",
        )
    _require(
        np.prod(result["zones"] + result["blocks_per_zone"], dtype=object)
        <= MAX_LEAVES,
        "BW metering exceeds leaf bound",
    )
    _require(
        type(result["padding"]) is int and 0 <= result["padding"] <= 2**31,
        "BW padding requires a nonnegative integer",
    )
    area = result["min_area_fraction"]
    _require(
        type(area) in (int, float) and np.isfinite(area) and 0 < area <= 1,
        "BW minimum usable area must lie in (0,1]",
    )
    result["min_area_fraction"] = float(area)
    # Reuse the vector schema without allocating a raster or reading source data.
    abstract_shape = (2**31, 2**31, 3)
    for key in ("include", "exclude"):
        _require(
            type(result[key]) is list and len(result[key]) <= 64,
            "BW metering supports at most64 vector shapes per role",
        )
        result[key] = [_shape(entry, abstract_shape) for entry in result[key]]
    boxes = result["excluded_rois"]
    _require(
        type(boxes) is list and len(boxes) <= 64,
        "BW metering supports at most64 confirmed exclusions",
    )
    result["excluded_rois"] = [
        list(box)
        for box in sorted(
            {_rectangle(box, abstract_shape, "BW exclusion") for box in boxes}
        )
    ]
    return result


def _configuration(config):
    p, e = config.process, config.exposure
    check_density_domain(p)
    _require(
        p.process_mode == ProcessMode.BW
        and p.source_color is False
        and not p.positive_source
        and not config.rgbscan.enabled,
        "BW meters require single-frame BW/source_color=False, without RGB scan",
    )
    _require(e.render_intent == RenderIntent.PRINT, "BW meters require PRINT")
    _require(
        config.geometry == GeometryConfig()
        and config.retouch == RetouchConfig()
        and not config.local.masks,
        "BW meters require identity geometry and no repair/local masks",
    )
    _require(
        e.contrast_mask_strength == e.diffusion == 0,
        "BW meter spatial print planes are not supported",
    )
    lab = LabConfig()
    for name in (
        "clahe_strength",
        "sharpen",
        "sharpen_method",
        "sharpen_radius",
        "sharpen_masking",
        "sharpen_auto_masking",
        "chroma_denoise",
        "glow_amount",
        "halation_strength",
    ):
        _require(
            getattr(config.lab, name) == getattr(lab, name),
            "BW frozen meters currently preserve original default Lab spatial settings only",
        )
    _require(
        p.crosstalk_strength == 0 and p.sensor_matrix is None,
        "BW meters do not apply density/sensor unmix",
    )
    defaults = ProcessConfig()
    _require(
        all(getattr(p, name) == getattr(defaults, name) for name in FADE_CARD_FIELDS),
        "BW meters do not support fade estimation",
    )
    _require(
        not p.use_luma_average
        and not p.use_color_average
        and not p.use_cast_average
        and not p.lock_bounds
        and p.locked_neutral_axis is None
        and p.manual_dmin is None
        and p.manual_dmax is None
        and p.film_base is None,
        "BW meters do not support pooled/manual base states",
    )
    _require(
        config.finish.carrier_width == config.finish.border_size == 0,
        "BW frozen meters require no carrier/paper border",
    )
    _require(
        not config.export.icc_input_path,
        "BW metering does not assume custom input ICC primaries",
    )
    # linear_raw deliberately remains the caller's original NegPy decode choice.
    _config_document(config)


def _runtime():
    files = [Path(__file__), Path(norm.__file__), Path(negpy_core.__file__)]
    for module in (
        "native_masked_meters",
        "metering_snapshot",
        "metering_masks",
        "metering_regions",
        "zonal_metering",
        "frozen_native_cpu",
        "metering_aux_snapshot",
        "density_matrix",
    ):
        files.append(Path(__file__).parent / (module + ".py"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def _source_identity(source):
    return _identity(
        dict(
            policy="single-frame-bw-prepared-v1",
            shape=list(source.shape),
            dtype=source.dtype.str,
            source_pixels_sha256=_pixels(source),
            geometry_token=_digest(
                {
                    "single_frame": True,
                    "shape": list(source.shape),
                    "orientation": "already-decoded",
                }
            ),
            coordinate_space=COORDINATE_SPACE,
            domain=SOURCE_DOMAIN,
            channel_order="RGB",
            source_read_only=True,
        )
    )


def _support(source, restored):
    x0, y0, x1, y1 = restored.config["roi"]
    support = np.ones((y1 - y0, x1 - x0), dtype=bool)
    if restored.selection_mask is not None:
        support &= validate_metering_mask(
            restored.selection_mask, source.shape, (x0, y0, x1, y1)
        )
    for bx0, by0, bx1, by1 in restored.receipt["effective_excluded_rois"]:
        support[by0 - y0 : by1 - y0, bx0 - x0 : bx1 - x0] = False
    return support


def _measure(
    source, config, window, roi, support, min_area_fraction, *, require_full_grid=False
):
    """Original BW grid/helpers; reference completeness is defined on that grid.

    The area-fraction policy belongs to base metering. A clean reference must
    retain every raw selection pixel and every available complete Native cell;
    the vendor's bottom/right grid tail is deliberately not a missing cell.
    """
    x0, y0, x1, y1 = roi
    rh, rw = y1 - y0, x1 - x0
    requested = int(np.ceil(max(rh, rw) / int(EXPOSURE_CONSTANTS["analysis_grid"])))
    b = requested if requested > 1 and min(rh, rw) >= requested else 1
    gh, gw = rh // b, rw // b
    hb, wb = gh * b, gw * b
    valid = support[:hb, :wb].reshape(gh, b, gw, b).all(axis=(1, 3))
    retained = int(np.count_nonzero(valid)) * b * b
    _require(
        type(require_full_grid) is bool, "BW grid-completeness policy must be boolean"
    )
    if require_full_grid:
        _require(
            support.all() and valid.size > 0 and valid.all(),
            "BW reference requires all available Native grid cells and raw support",
        )
    else:
        _require(
            retained and retained / (rh * rw) >= min_area_fraction,
            "Insufficient complete valid BW analysis-grid support",
        )
    grid = norm.prefilter_log_grid(source, (y0, y1, x0, x1), 0.0)
    _require(grid.shape == (gh, gw, 3), "BW analysis grid changed shape")
    process = frozen_process_for_window(config.process, window, source.shape)
    effective = replace(config, process=process)
    bounds, base = norm.resolve_bounds_detailed(
        process,
        lambda: (_ for _ in ()).throw(AssertionError("Frozen BW bounds remeasured")),
    )
    anchor = norm.luma_source_bounds(process, base)
    normalized, normalized_receipt = _spatial_activity(_luma(grid, anchor), valid)
    raw, raw_receipt = _spatial_activity(_luma(grid), valid)
    full = bool(support.all())
    meter_grid = grid if full else _packet(grid, normalized)
    range_grid = grid if full else _packet(grid, raw)
    metrics = dict(
        log_bounds_base=base,
        pre_trim_bounds=bounds,
        final_bounds=bounds,
        anchor_bounds=anchor,
        norm_density_range=float(norm.luminance_density_range(bounds)),
        neutral_axis_refs=None,
        metered_anchor=float(
            norm.measure_anchor_from_log(meter_grid, anchor, None, 0.0)
        ),
        textural_range=float(
            norm.measure_textural_range_from_log(range_grid, None, 0.0)
        ),
        shadow_point=float(
            norm.measure_shadow_point_from_log(meter_grid, anchor, None, 0.0)
        ),
        highlight_point=float(
            norm.measure_highlight_point_from_log(meter_grid, anchor, None, 0.0)
        ),
        density_above_base=None,
    )
    if full:
        clip = norm.measure_clip_fractions(source, (y0, y1, x0, x1), 0.0)
    else:
        pixel_support = np.repeat(np.repeat(valid, b, axis=0), b, axis=1)
        pixels = np.asarray(source[y0 : y0 + hb, x0 : x0 + wb], dtype=np.float32)
        clip = np.mean(
            pixels[pixel_support] >= EXPOSURE_CONSTANTS["scan_clip_level"], axis=0
        )
    metrics["scan_clip_fractions"] = tuple(float(v) for v in clip)
    receipt = dict(
        scope=SCOPE,
        process_mode="B&W",
        source_color=False,
        requested_roi=list(roi),
        grid_shape=[gh, gw],
        grid_block_size=b,
        grid_tail_pixel_area=rh * rw - hb * wb,
        require_full_native_grid=require_full_grid,
        raw_roi_support_all=bool(support.all()),
        native_grid_population_policy="Original complete bxb Native cells; bottom/right ROI tail discarded by original block-median grid",
        complete_valid_cells=int(np.count_nonzero(valid)),
        supported_pixel_area=retained,
        valid_cells_sha256=_pixels(valid),
        retained_grid_RGB_sha256=_pixels(grid[valid]),
        normalized_texture=normalized_receipt,
        raw_texture=raw_receipt,
        neutral_axis_refs=None,
        original_full_support_helpers=full,
        partial_mask_original_NegPy_equivalent=False,
        source_values_changed=False,
        RAW_decode_calls=0,
        renderer_calls=0,
    )
    return effective, metrics, receipt


def _window_record(window):
    return dict(
        roi=list(window.analysis.roi),
        floors=list(window.log_dense),
        ceils=list(window.log_thin),
    )


def _record_window(record):
    _require(
        type(record) is dict and set(record) == {"roi", "floors", "ceils"},
        "Malformed BW window",
    )
    roi = record["roi"]
    _require(
        type(roi) is list
        and len(roi) == 4
        and all(type(v) is int and v >= 0 for v in roi)
        and roi[0] < roi[2]
        and roi[1] < roi[3],
        "Malformed BW window ROI",
    )
    for key in ("floors", "ceils"):
        _require(
            type(record[key]) is list
            and len(record[key]) == 3
            and all(type(v) is float and np.isfinite(v) for v in record[key]),
            "Malformed BW bounds",
        )
    _require(
        all(a < b for a, b in zip(record["floors"], record["ceils"])),
        "Invalid BW bounds span",
    )
    return DensityWindow(
        "bw-frozen-reference-v1",
        Analysis(
            tuple(10 ** np.asarray(record["floors"])),
            tuple(10 ** np.asarray(record["ceils"])),
            tuple(record["roi"]),
            None,
        ),
        tuple(record["floors"]),
        tuple(record["ceils"]),
        {},
    )


class _BWEngine(DarkroomEngine):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.active = None
        self.base_normalization_runs = 0

    def _source_base_log(self, img_in, ctx):
        return None

    def _normalize(self, image, context):
        self.base_normalization_runs += 1
        context.metrics.update(deepcopy(self.active["metrics"]))
        context.active_roi = None
        context.metrics.update(
            geometry_params=dict(
                rotation=0,
                fine_rotation=0.0,
                flip_horizontal=False,
                flip_vertical=False,
                converge_v=0.0,
                converge_h=0.0,
            ),
            distortion_k1=0.0,
            active_roi=None,
        )
        for key in (
            "base_log",
            "source_base_log",
            "base_source",
            "density_above_base",
            "histogram_density",
            "density_parade",
            "scan_full_scale",
            "film_base_density",
        ):
            context.metrics[key] = None
        logged = negpy_core.log_density_open(
            np.ascontiguousarray(image, dtype=np.float32)
        )
        normalized = norm.normalize_log_image(logged, context.metrics["final_bounds"])
        context.metrics["normalized_log"] = normalized
        return normalized

    def _run_stage(
        self, img, config, cache_field, processor_fn, context, pipeline_changed
    ):
        if cache_field == "base":
            config = (config, self.active["sha256"])
            processor_fn = self._normalize
        return super()._run_stage(
            img, config, cache_field, processor_fn, context, pipeline_changed
        )

    def process(self, img, settings, source_hash, context=None):
        self.state._validate()
        _configuration(settings)
        _require(
            img.dtype == self.state.source.dtype
            and img.shape == self.state.source.shape
            and img.flags.c_contiguous
            and _pixels(img) == self.state.identity["source_pixels_sha256"],
            "BW renderer input differs from frozen source",
        )
        # SDK substitutes its output ExportConfig before invoking the engine.
        key = _digest(
            _config_document(replace(settings, export=self.state._config.export))
        )
        _require(
            key in self.state._records,
            "BW renderer configuration is not an explicitly frozen base/reference",
        )
        self.active = self.state._records[key]
        if context is None:
            context = PipelineContext(
                original_size=img.shape[:2],
                scale_factor=render_scale_factor(img.shape),
                process_mode=ProcessMode.BW,
            )
        _require(context.process_mode == ProcessMode.BW, "BW context mode mismatch")
        try:
            return super().process(
                img,
                settings,
                source_hash + "|bw-masked|" + self.active["sha256"],
                context,
            )
        finally:
            self.state._validate()


class BWMaskedState:
    """Freeze one single-frame source/recipe; restore without any auxiliary meters."""

    def __init__(self, source, config, roi, document, *, identity=None):
        _configuration(config)
        self.document = checked_bw_metering(document)
        _require(self.document is not None, "BWMaskedState requires active metering")
        _require(
            isinstance(source, np.ndarray) and source.dtype == np.float32,
            "BW metering source must be original float32 HWC",
        )
        self.source = source
        self.identity = (
            _identity(identity) if identity is not None else _source_identity(source)
        )
        _check_source(source, self.identity)
        self.roi = list(_rectangle(roi, source.shape, "BW analysis ROI"))
        self._config = deepcopy(config)
        self._input_config = _config_document(config)
        self._runtime = _runtime()
        mask = build_metering_mask(
            source.shape,
            self.roi,
            include=self.document["include"],
            exclude=self.document["exclude"],
        )
        self.bundle = freeze_zonal_metering(
            source,
            self.identity,
            self.roi,
            self.document["excluded_rois"],
            zones=self.document["zones"],
            blocks_per_zone=self.document["blocks_per_zone"],
            padding=self.document["padding"],
            min_area_fraction=self.document["min_area_fraction"],
            selection_mask=mask,
        )
        restored = restore_zonal_metering(self.bundle.to_json(), self.identity)
        self._support = _support(source, restored)
        effective, metrics, receipt = _measure(
            source,
            config,
            restored.window,
            self.roi,
            self._support,
            self.document["min_area_fraction"],
        )
        self._effective = effective
        self._records = {}
        self._register("base", effective, restored.window, metrics, receipt)
        self.engine = _BWEngine(self)
        self._validate()

    def _register(self, role, config, window, metrics, receipt):
        record = dict(
            role=role,
            config=_config_document(config),
            window=_window_record(window),
            metrics=native_masked_metrics_payload(metrics),
            receipt=deepcopy(receipt),
        )
        record["sha256"] = _digest(record)
        self._records[_digest(record["config"])] = {
            **record,
            "metrics": deepcopy(metrics),
            "serialized": record,
        }

    @property
    def effective_config(self):
        return deepcopy(self._effective)

    @property
    def receipt(self):
        return dict(
            schema=SCHEMA,
            scope=SCOPE,
            density_bundle_sha256=self.bundle.sha256,
            input_config_sha256=_digest(self._input_config),
            source_identity=deepcopy(self.identity),
            records=[deepcopy(r["serialized"]) for r in self._records.values()],
            original_default_Lab_preserved=True,
            native_BW_curves_unchanged=True,
        )

    def register_reference_roi(self, roi):
        self._validate()
        roi = list(_rectangle(roi, self.source.shape, "BW highlight reference ROI"))
        x0, y0, x1, y1 = roi
        bx0, by0, bx1, by1 = self.roi
        _require(
            bx0 <= x0 < x1 <= bx1
            and by0 <= y0 < y1 <= by1
            and self._support[y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0].all(),
            "BW reference ROI must lie entirely in effective metering support",
        )
        for record in self._records.values():
            if record["role"] == "reference" and record["window"]["roi"] == roi:
                return self._reference_config(record["window"])
        _require(
            len(self._records) < 65,
            "BW state supports at most64 explicit reference windows",
        )
        config = self.effective_config
        selected = np.ascontiguousarray(self.source[y0:y1, x0:x1])
        meter = replace(config, process=replace(config.process, meter_rect=None))
        window = negpy_window(selected, meter, (0, 0, x1 - x0, y1 - y0))
        window = replace(window, analysis=replace(window.analysis, roi=tuple(roi)))
        reference = self._reference_config(_window_record(window))
        _, metrics, receipt = _measure(
            self.source,
            reference,
            window,
            roi,
            np.ones((y1 - y0, x1 - x0), dtype=bool),
            1.0,
            require_full_grid=True,
        )
        self._register("reference", reference, window, metrics, receipt)
        self._validate()
        return deepcopy(reference)

    def _reference_config(self, record):
        window = _record_window(record)
        selection = select_metering_roi(
            self.source.shape, record["roi"], min_side=2, min_area_fraction=1.0
        )
        return replace(
            self._effective,
            process=replace(
                metering_process(self._effective.process, window, selection),
                meter_rect=None,
            ),
        )

    def _validate(self):
        _check_source(self.source, self.identity)
        _require(
            _runtime() == self._runtime, "BW metering implementation/runtime changed"
        )
        _require(
            _config_document(self._config) == self._input_config,
            "BW input configuration changed",
        )
        restored = restore_zonal_metering(self.bundle.to_json(), self.identity)
        _require(
            checked_bw_metering(self.document) == self.document
            and restored.config
            == {
                "roi": self.roi,
                **{
                    key: self.document[key]
                    for key in (
                        "zones",
                        "blocks_per_zone",
                        "excluded_rois",
                        "padding",
                        "min_area_fraction",
                    )
                },
            }
            and restored.selection_mask is not None
            and restored.selection_mask.spec["include"] == self.document["include"]
            and restored.selection_mask.spec["exclude"] == self.document["exclude"],
            "BW selection document/ROI differs from its frozen density bundle",
        )
        expected = replace(
            self._config,
            process=frozen_process_for_window(
                self._config.process, restored.window, self.source.shape
            ),
        )
        _require(
            _config_document(expected) == _config_document(self._effective),
            "BW effective configuration changed",
        )
        _require(
            np.array_equal(self._support, _support(self.source, restored)),
            "BW metering support changed",
        )
        for key, record in self._records.items():
            serialized = record["serialized"]
            _require(
                key == _digest(record["config"])
                and serialized["config"] == record["config"]
                and serialized["metrics"]
                == native_masked_metrics_payload(record["metrics"])
                and serialized["sha256"]
                == _digest({k: v for k, v in serialized.items() if k != "sha256"})
                and record["sha256"] == serialized["sha256"]
                and record["metrics"]["neutral_axis_refs"] is None,
                "BW frozen metric record changed",
            )

    def to_json(self):
        self._validate()
        payload = dict(
            schema=SCHEMA,
            version=1,
            scope=SCOPE,
            input_config=self._input_config,
            source_identity=self.identity,
            roi=self.roi,
            document=self.document,
            runtime=self._runtime,
            density_bundle=self.bundle.to_json(),
            records=[r["serialized"] for r in self._records.values()],
        )
        return _canonical({**payload, "sha256": _digest(payload)})

    @classmethod
    def restore(cls, text, source, config, roi, document, *, identity=None):
        _configuration(config)
        _require(
            type(text) is str and len(text.encode()) <= MAX_JSON_BYTES,
            "BW state JSON is invalid or oversized",
        )
        try:

            def unique(items):
                result = {}
                for key, value in items:
                    _require(key not in result, "Duplicate BW state JSON key")
                    result[key] = value
                return result

            parsed = json.loads(
                text,
                object_pairs_hook=unique,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError("Invalid BW state JSON") from error
        keys = {
            "schema",
            "version",
            "scope",
            "input_config",
            "source_identity",
            "roi",
            "document",
            "runtime",
            "density_bundle",
            "records",
            "sha256",
        }
        _require(
            type(parsed) is dict
            and set(parsed) == keys
            and parsed["schema"] == SCHEMA
            and type(parsed["version"]) is int
            and parsed["version"] == 1
            and parsed["scope"] == SCOPE,
            "Unsupported BW state schema/version/scope",
        )
        _require(
            parsed["sha256"]
            == _digest({k: v for k, v in parsed.items() if k != "sha256"}),
            "BW state checksum changed",
        )
        instance = cls.__new__(cls)
        instance.source = source
        instance.identity = (
            _identity(identity) if identity is not None else _source_identity(source)
        )
        _check_source(source, instance.identity)
        instance.document = checked_bw_metering(document)
        instance.roi = list(_rectangle(roi, source.shape, "BW analysis ROI"))
        instance._config = deepcopy(config)
        instance._input_config = _config_document(config)
        instance._runtime = _runtime()
        _require(
            parsed["source_identity"] == instance.identity
            and parsed["input_config"] == instance._input_config
            and parsed["roi"] == instance.roi
            and parsed["document"] == instance.document
            and parsed["runtime"] == instance._runtime,
            "BW state source/config/selection/runtime changed",
        )
        density = restore_zonal_metering(parsed["density_bundle"], instance.identity)
        from .metering_snapshot import FrozenMeteringBundle

        instance.bundle = FrozenMeteringBundle(parsed["density_bundle"], density.sha256)
        instance._support = _support(source, density)
        instance._effective = replace(
            config,
            process=frozen_process_for_window(
                config.process, density.window, source.shape
            ),
        )
        _require(
            type(parsed["records"]) is list and 1 <= len(parsed["records"]) <= 65,
            "Malformed BW records",
        )
        instance._records = {}
        for record in parsed["records"]:
            _require(
                type(record) is dict
                and set(record)
                == {"role", "config", "window", "metrics", "receipt", "sha256"}
                and record["role"] in ("base", "reference"),
                "Malformed BW metric record",
            )
            window = _record_window(record["window"])
            metrics = _restore_metrics(record["metrics"])
            _require(
                metrics["neutral_axis_refs"] is None,
                "BW state cannot contain a colour neutral axis",
            )
            if record["role"] == "base":
                expected = instance._effective
                _require(
                    _window_record(window) == _window_record(density.window),
                    "BW base bounds changed",
                )
            else:
                x0, y0, x1, y1 = _rectangle(
                    record["window"]["roi"], source.shape, "BW reference ROI"
                )
                bx0, by0, bx1, by1 = instance.roi
                _require(
                    bx0 <= x0 < x1 <= bx1
                    and by0 <= y0 < y1 <= by1
                    and instance._support[
                        y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0
                    ].all(),
                    "Restored BW reference outside effective support",
                )
                expected = instance._reference_config(record["window"])
            _require(
                _config_document(expected) == record["config"],
                "BW metric configuration changed",
            )
            key = _digest(record["config"])
            _require(key not in instance._records, "Duplicate BW metric configuration")
            instance._records[key] = {
                **deepcopy(record),
                "metrics": metrics,
                "serialized": deepcopy(record),
            }
        _require(
            sum(r["role"] == "base" for r in instance._records.values()) == 1,
            "BW state requires exactly one base",
        )
        instance.engine = _BWEngine(instance)
        instance._validate()
        return instance
