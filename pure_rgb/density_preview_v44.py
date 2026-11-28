"""Full-support frozen density, then exact point rendering of display pixels.

Saving still uses the existing complete renderers. This private preview route
never measures a reduced image: only the final point pipeline is sampled.
"""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math

import numpy as np

from .diagnostics import span
from .immutable_source_v44 import ImmutableSourceCertificate


def point_c41_supported(config):
    from negpy.features.geometry.models import GeometryConfig
    from negpy.features.retouch.models import RetouchConfig
    from negpy.features.process.models import ProcessMode
    from negpy.features.altprocess.models import AltProcess
    from negpy.features.exposure.models import RenderIntent

    lab, exposure, finish = config.lab, config.exposure, config.finish
    return bool(
        config.process.process_mode == ProcessMode.C41 and not config.process.positive_source
        and exposure.render_intent == RenderIntent.PRINT
        and config.geometry == GeometryConfig() and config.retouch == RetouchConfig()
        and not config.local.masks and config.altproc.alt_process == AltProcess.NONE
        and exposure.contrast_mask_strength == 0 and exposure.diffusion == 0
        and lab.clahe_strength == 0 and lab.chroma_denoise == 0 and lab.sharpen == 0
        and lab.glow_amount == 0 and lab.halation_strength == 0
        and finish.vignette_stops == 0 and finish.carrier_width == 0 and finish.border_size == 0
    )


def native_usm_supported(config):
    from negpy.features.lab.models import SharpenMethod

    lab = config.lab
    return bool(type(lab.sharpen) in (int, float) and math.isfinite(lab.sharpen) and 0 < lab.sharpen <= 2
                and type(lab.sharpen_radius) in (int, float) and math.isfinite(lab.sharpen_radius)
                and 0.1 <= lab.sharpen_radius <= 10
                and type(lab.sharpen_masking) in (int, float) and math.isfinite(lab.sharpen_masking)
                and 0 <= lab.sharpen_masking <= 1 and lab.sharpen_method == SharpenMethod.USM
                and not lab.sharpen_auto_masking
                and point_c41_supported(replace(config, lab=replace(lab, sharpen=0))))


def render_frozen_sample(runner, source, *, rectangle=None, sample_step=1, certificate=None,
                         render_config=None, float_output=False, sampling_policy="point",
                         print_ev=0.0):
    """Return exact RGB16 final-grid samples, or None for non-point finishing."""
    from negpy.domain.interfaces import PipelineContext
    from negpy.kernel.system.config import render_scale_factor
    from negpy.kernel.image.logic import _to_uint16_jit
    from .metering_snapshot import _check_source
    from .native_spatial_v30 import _SpatialEngine
    from .positive_float import working_icc_bytes
    from .film_density_domain_v45 import DOMAIN, film_domain_for

    if type(float_output) is not bool:
        raise ValueError("Frozen preview float output must be an explicit boolean")
    if type(sample_step) is not int or sample_step < 1:
        raise ValueError("Frozen preview sample step must be a positive integer")

    effective = runner._effective if render_config is None else render_config
    if render_config is not None and replace(render_config, exposure=runner._effective.exposure,
        lab=runner._effective.lab, toning=runner._effective.toning) != runner._effective:
        raise ValueError("Point preview may change only exposure, point Lab and toning after frozen measurement")
    if sampling_policy not in ("point", "screen-usm"):
        raise ValueError("Unknown Native sample policy")
    if not (point_c41_supported(effective) if sampling_policy == "point" else native_usm_supported(effective)):
        return None
    if sampling_policy == "screen-usm":
        effective = replace(effective, lab=replace(effective.lab,
                            sharpen_radius=effective.lab.sharpen_radius / sample_step))
    domain = film_domain_for(source)
    if (runner._identity["domain"] == DOMAIN) != (domain is not None):
        raise ValueError("Frozen point source does not belong to its declared density domain")
    if domain is not None:
        domain.validate()
    if certificate is not None:
        certificate.validate(source, runner._identity)
    else:
        _check_source(source, runner._identity)
    runner._validate()
    height, width = source.shape[:2]
    roi = [0, 0, width, height] if rectangle is None else list(rectangle)
    if (len(roi) != 4 or any(type(value) is not int for value in roi)
            or not 0 <= roi[0] < roi[2] <= width or not 0 <= roi[1] < roi[3] <= height
            or type(sample_step) is not int or sample_step < 1):
        raise ValueError("Frozen preview needs an exact source-pixel rectangle/step")
    x0, y0, x1, y1 = roi
    if ((x1 - x0 + sample_step - 1) // sample_step
            * ((y1 - y0 + sample_step - 1) // sample_step) > 4_194_304):
        raise ValueError("Frozen preview sample exceeds its pixel bound")
    sample = np.array(source[y0:y1:sample_step, x0:x1:sample_step], dtype=np.float32, order="C", copy=True)
    context = PipelineContext(original_size=source.shape[:2],
        scale_factor=render_scale_factor(source.shape), process_mode=effective.process.process_mode,
        wants_uv_grid=False, readback_metrics=False, cache_stages=False)
    # A fresh point engine cannot borrow a previous tile's stage arrays.
    engine = _SpatialEngine(runner._metrics, runner._sha256, log_domain=domain)
    key = json.dumps([runner._sha256, roi, sample_step, effective.to_dict()], sort_keys=True)
    image = engine.process(sample, effective, key, context, print_ev=print_ev)
    if not np.isfinite(image).all() or image.shape != sample.shape:
        raise ValueError("Frozen point renderer returned invalid pixels")
    pixels = (np.ascontiguousarray(image, dtype=np.float32) if float_output else
              _to_uint16_jit(np.ascontiguousarray(image, dtype=np.float32)))
    if certificate is not None:
        certificate.validate(source, runner._identity)
    else:
        _check_source(source, runner._identity)
    runner._validate()
    if domain is not None:
        domain.validate()
    return pixels, working_icc_bytes(), dict(
        full_source_shape=list(source.shape), source_rectangle=roi, sample_step=sample_step,
        full_support_measured=True, sample_remeasured=False,
        processed_pixels=int(pixels.shape[0] * pixels.shape[1]),
        original_SDK_point_engine=True, original_SDK_uint16_quantizer=not float_output,
        output_buffer_domain="working-OETF float32 before integer encoding" if float_output else "working RGB16",
        binding_sha256=runner._sha256,
        render_config_sha256=_key(effective.to_dict()),
        approximate=sampling_policy == "screen-usm", sampling_policy=sampling_policy,
    )


def _key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


class DensityPreviewCache:
    """One authenticated RAW source; a bounded LRU of full-support packets."""

    def __init__(self, owner):
        self.owner = owner
        self.session = self.certificate = None
        self.entries = OrderedDict()
        self.samples = OrderedDict()
        self.engine_proxy_samples = OrderedDict()
        self.original_base_samples = OrderedDict()
        self.original_meter_values = OrderedDict()
        self.original_highlight_measurements = OrderedDict()
        self.original_companion = None
        self.original_screen_base = None
        self.native_meter_proxy = None
        self.native_metal_tone = None
        self.interactive_metal_tone = None
        self.source_prepare_calls = self.measurement_calls = 0
        self.last_receipt = None
        self.references = OrderedDict()
        self.film_domains = OrderedDict()
        self.float_samples = OrderedDict()
        self.protection_samples = OrderedDict()
        self.reference_measurement_calls = 0
        self.usm_bands = OrderedDict()
        self.base_sample_render_calls = self.reference_sample_render_calls = 0
        self.native_sample_pixels = 0
        self.pristine_config = self.source_key = None

    @staticmethod
    def _source_descriptor(owner):
        from .workspace import source_recipe_identity

        return _key([
            source_recipe_identity(owner.recipe),
            owner.records, owner.manifest.get("calibration"), owner.manifest["source_shape"],
        ])

    def rebind(self, owner):
        if self._source_descriptor(owner) != self._source_descriptor(self.owner):
            return False
        if self.certificate is not None:
            self.certificate.validate()
        self.owner = owner
        return True

    def _progress(self, stage, fraction, message):
        from .live_worker import emit

        emit(type="progress", stage=stage, fraction=fraction, message=message)

    def remember_original_base(self, key, visible, metrics, roi):
        """Retain only the immutable display grid and full-grid readouts."""
        from .output_orientation_cache import cache_budget_bytes

        budget = min(64 * 1024**2, cache_budget_bytes())
        if visible.nbytes > budget:
            return np.array(visible, order="C", copy=True)
        while (len(self.original_base_samples) >= 3
               or visible.nbytes + sum(sample.nbytes for sample, _, _ in self.original_base_samples.values()) > budget):
            self.original_base_samples.popitem(last=False)
        sample = np.frombuffer(visible.tobytes(), dtype=visible.dtype).reshape(visible.shape)
        self.original_base_samples[key] = (sample, deepcopy(metrics), roi)
        return sample

    def prepare_source(self):
        decoded_certificate = None
        if self.session is None:
            from types import SimpleNamespace
            from . import workflow_render
            from .decoded_cache import render_decode_context, source_anchor
            from .flatfield import attach_calibration
            from .raw_import import is_standard_import, validate_frozen_import

            self._progress("source_prepare", 0.05, "正在读取原始素材，后续选区复用同一来源…")
            session = self.owner.saved_session
            if session is None or session.processor is None:
                recipe = self.owner.recipe
                manifest = self.owner.manifest
                preconfigured = None

                def configure_standard_source(candidate, initial_config, automatic_records):
                    # Standard-crop flatfields must pre-attach with the sensor
                    # reference shape, exactly like the renderer's own path;
                    # user-confirmed role changes keep the reorder/redecode
                    # fallback below.
                    nonlocal preconfigured
                    if automatic_records != self.owner.records:
                        return initial_config
                    initial_config, preconfigured = attach_calibration(
                        initial_config, manifest["calibration"],
                        self.owner.directory / "density-calibration", recipe["raw"],
                        strength=recipe.get("flatfield_strength", 1.0))
                    candidate.pure_flatfield_reference_shape = (
                        (preconfigured.get("provenance") or {})
                        .get("capture", {}).get("sensor_shape")
                    )
                    return initial_config

                # The saved run's import receipt anchors the decoded-RAW cache:
                # a cached source is only restored when it matches that proven
                # decode, otherwise this re-decodes exactly as before.
                decode_cache = render_decode_context(
                    {"inputs": self.owner.records,
                     "calibration": manifest.get("calibration")},
                    recipe, source_anchor(manifest), frozen=True,
                )
                open_cache = (
                    decode_cache
                    if decode_cache is not None
                    and (
                        not recipe.get("flatfield")
                        or recipe.get("raw_import") in ("standard-crop-v1", "standard-mono-v1")
                    )
                    else None
                )
                # The manifest authenticated by the draft already records
                # this exact automatic classification. Re-reading all three
                # mosaics just to classify the same hashes wastes time.
                automatic_roles = manifest.get("automatic_roles")
                verified_order = automatic_roles is not None and automatic_roles == self.owner.records
                processor, config, source, automatic = workflow_render.open_triplet(
                    recipe["raw"], raw_import=recipe.get("raw_import", "legacy-v1"),
                    **({"lens_correction": True} if recipe.get("lens_correction") else {}),
                    **({"chromatic_aberration": True} if recipe.get("chromatic_aberration") else {}),
                    **({"lens_profile": recipe["lens_profile"]} if recipe.get("lens_profile") else {}),
                    **({"registration_check": recipe["registration_check"]}
                       if recipe.get("registration_check") else {}),
                    **({"configure_source": configure_standard_source}
                       if recipe.get("flatfield")
                       and recipe.get("raw_import") in ("standard-crop-v1", "standard-mono-v1") else {}),
                    **({"decode_cache": open_cache} if open_cache is not None else {}),
                    **({"frozen_roles": automatic_roles} if verified_order else {}),
                )
                if [item["path"] for item in automatic] != recipe["raw"]:
                    config = replace(config, rgbscan=replace(config.rgbscan,
                        green_path=recipe["raw"][1], blue_path=recipe["raw"][2]))
                    source, _, _ = processor._load_source_f32(recipe["raw"][0], config)
                if recipe.get("flatfield") and preconfigured is None:
                    # The pre-attach document keys the decode cache; the attached
                    # profile identifier is process-local and must stay out of it.
                    config_document = config.to_dict()
                    config, applied = attach_calibration(
                        config, manifest["calibration"],
                        self.owner.directory / "density-calibration", recipe["raw"],
                        strength=recipe.get("flatfield_strength", 1.0))
                    if is_standard_import(recipe.get("raw_import")):
                        processor.pure_flatfield_reference_shape = (
                            (applied.get("provenance") or {})
                            .get("capture", {}).get("sensor_shape")
                        )
                    restored = (
                        decode_cache.restore(
                            processor, config_document, recipe["raw"][0], config
                        )
                        if decode_cache is not None
                        else None
                    )
                    if restored is not None:
                        source = restored
                    else:
                        source, ir, color_space = processor._load_source_f32(
                            recipe["raw"][0], config
                        )
                        if decode_cache is not None:
                            decode_cache.store(
                                processor, config_document, recipe["raw"][0],
                                source, ir, color_space,
                            )
                if hasattr(processor, "pure_raw_import_token"):
                    validate_frozen_import(processor, self.owner.manifest, source)
                if decode_cache is not None:
                    decoded_certificate = decode_cache.take_source_certificate(source)
                session = SimpleNamespace(processor=processor, config=config, source=source)
                self.source_prepare_calls += 1
            else:
                self.owner.saved_session = None
            self.session = session
        session = self.session
        if self.certificate is None:
            from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG as default

            self._progress("source_authenticate", 0.12, "正在认证完整来源并建立预览缓存…")
            # This session already loaded and validated its source above. Calling
            # _export_source_key here reloads the RAW when a restored flatfield
            # has a different process-local profile ID.
            self.source_key = hashlib.sha256(
                ("".join(record["sha256"] for record in self.owner.records)
                 + getattr(session.processor, "pure_raw_import_token", "")).encode()
            ).hexdigest()
            self.certificate = decoded_certificate or ImmutableSourceCertificate.create(session.source)
            original = session.source
            session.source = self.certificate.source
            # Retain the actual pinned processor's metadata/import cache while
            # replacing only its identical source raster with immutable bytes.
            cached = getattr(session.processor, "_source_cache_value", None)
            if cached is not None and cached[0] is original:
                session.processor._source_cache_value = (session.source, *cached[1:])
            # A decoded source is independent of a former saved density window
            # or Native look. Preserve only capture/import/flat-field settings.
            self.pristine_config = replace(
                session.config,
                process=replace(default.process, linear_raw=True, highlight_reconstruction=0),
                exposure=default.exposure, lab=default.lab, local=default.local,
                geometry=default.geometry, retouch=default.retouch,
                finish=default.finish, toning=default.toning, altproc=default.altproc,
            )
        self.certificate.validate()
        from .live_worker import preview_checkpoint

        self.certificate.checkpoint = preview_checkpoint()
        return self.session

    def _entry(self, recipe, mode):
        from .professional_controls import native_config
        from .shared_density_metering_v43 import spatial_document, window_from_document
        from .spatial_workflow_v31 import _prepare, _meter_config, _render_config

        session = self.prepare_source()
        doc = (deepcopy(recipe["native_spatial_controls"]) if mode == "native"
               else spatial_document(recipe["density_metering"]))
        wanted = replace(self.pristine_config,
                         exposure=replace(self.pristine_config.exposure, paper_profile=recipe["paper"]))
        if mode == "native":
            wanted = native_config(wanted, recipe.get("native_controls", dict(version=1, values={})))
        prospective = _render_config(_meter_config(wanted), doc)
        if not point_c41_supported(prospective) and not native_usm_supported(prospective):
            return None
        # Auxiliary meters read process/source/support, never PRINT grade,
        # CMY/WB, paper or point Lab. Freeze that neutral measurement once;
        # the private point renderer binds a separate strict creative config.
        config = self.pristine_config
        source, certificate, source_key = session.source, self.certificate, self.source_key
        domain = None
        film = recipe.get("native_film") if mode == "native" else None
        if film and film["strength"] == 0:
            film = None
        if film and film["strength"] != 0:
            from .film_density_domain_v45 import FilmDensityDomain

            film_key = _key([self.certificate.sha256, film])
            domain = self.film_domains.get(film_key)
            if domain is None:
                self._progress("film_density", 0.18, "正在建立胶片校正密度参考…")
                domain = FilmDensityDomain(session.source, self.pristine_config, film)
                self.film_domains[film_key] = domain
                while len(self.film_domains) > 2:
                    _, previous = self.film_domains.popitem(last=False)
                    for entry_key, old_entry in list(self.entries.items()):
                        if old_entry.get("domain") is previous:
                            self.entries.pop(entry_key)
                    previous.close()
            self.film_domains.move_to_end(film_key)
            domain.validate()
            domain.certificate.checkpoint = self.certificate.checkpoint
            source, certificate, config = domain.source, domain.certificate, domain.neutral_config
            source_key = _key([self.source_key, film, certificate.sha256])
        doc = {**doc, "lab": {}, "local_masks": []}
        roi = list(recipe.get("roi") or self.owner.manifest["roi"])
        key = _key([certificate.sha256, film, roi, doc, config.to_dict()])
        entry = self.entries.get(key)
        hit = entry is not None
        if entry is None:
            self._progress("density_support", 0.22, "正在测量当前选区与排除区域…")
            # All density and auxiliary meters retain the original full source
            # grid/support. Initial rectangular floors are replaced by this
            # spatial producer's own frozen window before any auxiliary meter.
            def checkpoint(stage):
                self._progress(stage, 0.32 if stage == "automatic_detection" else 0.42,
                               "异常区域检查已完成…" if stage == "automatic_detection" else "选区密度已测量，正在冻结测量参考…")

            runner, _, snapshot = _prepare(source, config, roi, doc, source_key, None,
                                          checkpoint=checkpoint)
            bundle = next((bundle for bundle in certificate.bundles.values()
                           if bundle.sha256 == runner._payload["density_bundle_sha256"]), None)
            if bundle is None:
                raise ValueError("Frozen preview has no authenticated base density support")
            entry = dict(key=key, runner=runner, window=window_from_document(snapshot["window"]),
                         snapshot=snapshot, config=config, document=doc,
                         source=source, certificate=certificate, domain=domain,
                         bundle_text=bundle.to_json(), references=OrderedDict())
            self.entries[key] = entry
            self.measurement_calls += 1
            while len(self.entries) > 4:
                self.entries.popitem(last=False)
            self._progress("auxiliary_freeze", 0.55, "当前选区测量已冻结，正在生成预览…")
        self.entries.move_to_end(key)
        entry["cache_hit"] = hit
        render_config = replace(entry["runner"]._effective, exposure=prospective.exposure,
                                lab=prospective.lab, toning=prospective.toning)
        return entry, render_config

    def _highlight_reference(self, entry, highlight):
        """Freeze the exact clean local reference, without rerendering/retesting base M."""
        from .metering_snapshot import restore_zonal_metering
        from .native_spatial_v30 import NativeSpatialCPU
        from .spatial_workflow_v31 import (
            _c41_packet, _check_highlight_reference_bounds, _meter_config, _render_config,
        )

        runner = entry["runner"]
        runner._validate()
        entry["certificate"].validate(entry["source"], runner._identity)
        roi = list(highlight["reference_roi"])
        key = _key([entry["key"], roi])
        found = entry["references"].get(key)
        if found is None:
            self._progress("highlight_reference", 0.60, "正在测量局部高光参考并冻结完整来源统计…")
            restored = restore_zonal_metering(entry["bundle_text"], runner._identity)
            if restored.sha256 != runner._payload["density_bundle_sha256"]:
                raise ValueError("Highlight reference belongs to a changed base support")
            _check_highlight_reference_bounds(roi, entry["source"].shape)
            document = deepcopy(entry["document"])
            document.pop("automatic_avoidance", None)
            document["metering"].update(include=[], exclude=[], excluded_rois=[], padding=0,
                                         min_area_fraction=1.0, zones=[2, 2], blocks_per_zone=[2, 2])
            meter = _meter_config(entry["config"])
            packet, bundle = _c41_packet(entry["source"], meter, roi, document, runner._identity)
            reference = NativeSpatialCPU.from_c41(packet, meter, _render_config(meter, document))
            found = dict(key=key, runner=reference, window=deepcopy(
                restore_zonal_metering(bundle.to_json(), runner._identity).window))
            entry["references"][key] = found
            self.reference_measurement_calls += 1
            while len(entry["references"]) > 2:
                entry["references"].popitem(last=False)
        entry["references"].move_to_end(key)
        found["runner"]._validate()
        return found

    def _exact_usm_float(self, entry, runner, render_config, rectangle, step, *, print_ev=0.0):
        from .native_usm_sample_v46 import render_original_usm_band

        height, width = entry["source"].shape[:2]
        x0, y0, x1, y1 = rectangle
        xs, ys = np.arange(x0, x1, step), np.arange(y0, y1, step)
        output = np.empty((len(ys), len(xs), 3), dtype=np.float32)
        rows = max(1, 1048576 // width)
        for start in sorted(set((ys // rows * rows).tolist())):
            band_key = _key([runner._sha256, render_config.to_dict(), start, print_ev])
            band = self.usm_bands.get(band_key)
            if band is None:
                image, receipt = render_original_usm_band(
                    runner, entry["source"], render_config, start, print_ev=print_ev
                )
                band = dict(image=np.frombuffer(image.tobytes(), dtype=np.float32).reshape(image.shape), receipt=receipt)
                self.usm_bands[band_key] = band
                self.native_sample_pixels += receipt["processed_pixels"]
                if runner is entry["runner"]:
                    self.base_sample_render_calls += 1
                else:
                    self.reference_sample_render_calls += 1
                while (sum(value["image"].nbytes for value in self.usm_bands.values()) > 128 * 1024**2
                       and len(self.usm_bands) > 1):
                    self.usm_bands.popitem(last=False)
            self.usm_bands.move_to_end(band_key)
            selected = np.flatnonzero((ys >= start) & (ys < min(height, start + rows)))
            low = band["receipt"]["source_rectangle"][1]
            output[selected] = band["image"][(ys[selected]-low)[:, None], xs[None, :]]
        from .positive_float import working_icc_bytes

        return output, working_icc_bytes()

    def _point_float(self, entry, runner, render_config, rectangle, step, key, *, exact=True,
                     print_ev=0.0):
        """Private immutable F32 samples, before the high/float one-quantizer join."""
        runner._validate()
        entry["certificate"].validate(entry["source"], runner._identity)
        if entry["domain"] is not None:
            entry["domain"].validate()
        usm = native_usm_supported(render_config)
        key = _key([key, render_config.to_dict(), rectangle, step, exact if usm else True, print_ev])
        found = self.float_samples.get(key)
        if found is None:
            if usm and exact:
                image, icc = self._exact_usm_float(
                    entry, runner, render_config, rectangle, step, print_ev=print_ev
                )
            else:
                image, icc, _ = render_frozen_sample(runner, entry["source"],
                    rectangle=rectangle, sample_step=step, certificate=entry["certificate"],
                    render_config=render_config, float_output=True,
                    sampling_policy="screen-usm" if usm else "point", print_ev=print_ev)
                self.native_sample_pixels += int(image.shape[0] * image.shape[1])
                if runner is entry["runner"]:
                    self.base_sample_render_calls += 1
                else:
                    self.reference_sample_render_calls += 1
            found = dict(image=np.frombuffer(image.tobytes(), dtype=np.float32).reshape(image.shape), icc=icc)
            self.float_samples[key] = found
            while len(self.float_samples) > 4:
                self.float_samples.popitem(last=False)
        self.float_samples.move_to_end(key)
        return found["image"], found["icc"]

    def _highlight_mask(self, source_key, shape, name, document, rectangle, step):
        if not document:
            return None
        from .highlight_protection import automatic_highlight_feather
        from .highlight_sample_v46 import sample_highlight_protection

        key = _key([source_key, name, document, rectangle, step])
        item = self.protection_samples.get(key)
        if item is None:
            item = sample_highlight_protection(
                list(shape), source_key, **document, rectangle=rectangle, sample_step=step, dtype="float64",
                automatic_feather_pixels=automatic_highlight_feather(shape))
            self.protection_samples[key] = item
            while len(self.protection_samples) > 3:
                self.protection_samples.popitem(last=False)
        self.protection_samples.move_to_end(key)
        return item

    def _native_join_sample(self, entry, render_config, reference, highlight, point, rectangle, step, *,
                            exact=True, print_ev=0.0):
        from negpy.kernel.image.logic import _to_uint16_jit
        from .highlight_sample_v46 import compose_native_sample

        image, icc = self._point_float(
            entry, entry["runner"], render_config, rectangle, step, entry["key"], exact=exact, print_ev=print_ev
        )
        ref = None
        if reference is not None:
            ref_config = replace(reference["runner"]._effective, exposure=render_config.exposure,
                                 lab=render_config.lab, toning=render_config.toning)
            ref, other = self._point_float(
                entry, reference["runner"], ref_config, rectangle, step, reference["key"],
                exact=exact, print_ev=print_ev
            )
            if other != icc:
                raise ValueError("Frozen highlight reference ICC differs from the Native base")
        mask_args = entry["key"], entry["source"].shape
        # Experimental spatial metering uses its authenticated reference
        # packet. The alternate pre-clip density render is for ordinary C41.
        if highlight and "reference_density_offset" in highlight:
            highlight = {key: value for key, value in highlight.items()
                         if key != "reference_density_offset"}
        controls = highlight or {}
        protection = self._highlight_mask(*mask_args, "protection", controls.get("protection"), rectangle, step)
        effect = self._highlight_mask(*mask_args, "effect_area", controls.get("effect_area"), rectangle, step)
        image, receipt = compose_native_sample(image, icc=icc, reference_encoded=ref,
                                               highlight=highlight, point=point,
                                               protection_mask=protection.mask if protection else None,
                                               effect_mask=effect.mask if effect else None)
        if receipt is not None:
            receipt["protection_sampling"] = protection.receipt if protection else None
            if effect:
                receipt["effect_sampling"] = effect.receipt
        return _to_uint16_jit(image), icc, receipt

    def _reference(self, recipe, mode, rectangle, step, direction, fallback, icc, *, share=False, bounded=False):
        from .draft_preview import _geometry_document, _identity, defer_comparison
        from .output_orientation import orient_reference, orient_codes
        from .preview_data import verified_pixels
        from .workspace import output_rectangle
        from .native_draft_preview_v40 import resolved_saved_recipe

        if defer_comparison(self.owner):
            return fallback
        record = self.owner.manifest["results"].get(mode)
        if record is None or _geometry_document(self.owner.recipe, mode):
            self.owner.reference_label = "调整前（当前测量）"
            return fallback.copy()
        saved_recipe = resolved_saved_recipe(self.owner.manifest)
        a, b, c, d = output_rectangle(saved_recipe, self.owner.manifest["roi"], self.session.source.shape)
        x0, y0, x1, y1 = rectangle
        if not (a <= x0 < x1 <= c and b <= y0 < y1 <= d):
            self.owner.reference_label = "调整前（当前测量）"
            return fallback.copy()
        identity = _identity(record["path"])
        key = _key([record, rectangle, step, direction])
        cached = self.references.get(key)
        if cached is not None and cached["identity"] == identity:
            self.owner.reference_label = "已保存版本"
            return cached["pixels"] if share else cached["pixels"].copy()
        # At 1:1, the A/B pane needs only its matching saved strips.
        # A sampled fit has a different phase contract and keeps its own route.
        if bounded and step == 1:
            from .viewport_preview_v43 import _saved_strip_tile

            tile = _saved_strip_tile(self.owner, record, identity, (a, b, c, d), rectangle, direction, step)
            if tile is not None:
                pixels, other = tile
                if other != icc:
                    raise ValueError("已保存预览参考ICC或身份不一致")
                self.owner.reference_label = "已保存版本"
                self.references[key] = dict(
                    identity=identity,
                    pixels=np.frombuffer(pixels.tobytes(), dtype=np.uint16).reshape(pixels.shape),
                )
                while len(self.references) > 2:
                    self.references.popitem(last=False)
                return pixels
        anchor = getattr(self.owner, "anchor", None)
        if (anchor is not None and anchor.get("owned") is False
                and anchor["record"] == record and anchor["identity"] == identity
                and anchor.get("pixels") is not None):
            # The matching saved anchor already verified this immutable buffer.
            pixels, other = anchor["pixels"], anchor["icc"]
        else:
            pixels, other = verified_pixels(record)
        after = _identity(record["path"])
        if after != identity:
            identity = after
            pixels, other = verified_pixels(record)
        if other != icc or _identity(record["path"]) != identity:
            raise ValueError("已保存预览参考ICC或身份不一致")
        pixels = orient_reference(pixels, self.owner.recipe.get("output_orientation"), None)
        reference = orient_codes(pixels[y0-b:y1-b:step, x0-a:x1-a:step], direction)
        self.references[key] = dict(identity=identity,
            pixels=np.frombuffer(reference.tobytes(), dtype=np.uint16).reshape(reference.shape))
        while len(self.references) > 2:
            self.references.popitem(last=False)
        self.owner.reference_label = "已保存版本"
        return self.references[key]["pixels"] if share else reference.copy()

    def try_render(self, request, recipe, mode):
        from .native_headroom import native_local_highlight, print_exposure_receipt
        from .preview_grid import post_requires_full_grid
        from .professional_controls import native_print_ev
        from .workspace import output_rectangle
        from .viewport_preview_v43 import output_size_for_recipe, checked_viewport, source_rectangle_for_viewport
        from .flex_draft_preview_v40 import oriented_source_sample_roi
        from .output_orientation import orient_codes

        if (request.get("full_resolution") or recipe.get("source_mode")
                or recipe["density"] != "negpy"
                or post_requires_full_grid(recipe.get("post_controls", {}).get(mode))
                or (mode == "native" and not recipe.get("native_spatial_controls"))
                or (mode != "native" and not recipe.get("density_metering"))
                or (mode == "cool-shared-rgb" and not recipe.get("nikon_research_controls"))
                or (mode == "flex-endpoint" and not recipe.get("flex_research_controls"))):
            return None
        size = output_size_for_recipe(recipe, self.owner.manifest["source_shape"], self.owner.manifest["roi"], mode)
        if size is None:
            return None
        viewport = checked_viewport(request["viewport"], size) if "viewport" in request else None
        rectangle = output_rectangle(recipe, self.owner.manifest["roi"], self.owner.manifest["source_shape"])
        direction = recipe.get("output_orientation")
        if viewport:
            rectangle = source_rectangle_for_viewport(rectangle, direction, viewport)
            step = viewport["sample_step"]
        else:
            fit_edge = request.get("fit_edge", 1400) if mode == "native" else 1400
            step = max(1, (max(size) + fit_edge - 1) // fit_edge)
        with span("preview_density_entry", mode=mode, sample_step=step):
            selected = self._entry(recipe, mode)
        if selected is None:
            return None
        entry, render_config = selected
        interactive = bool(viewport is None and request.get("interactive_edge")
                           and entry["cache_hit"])
        if interactive:
            edge = request["interactive_edge"]
            factor = max(1, (max(size) + edge * step - 1) // (edge * step))
            step *= factor
        phase_roi = oriented_source_sample_roi(rectangle, direction, step)
        highlight = native_local_highlight(recipe) if mode == "native" else None
        point = recipe.get("native_float_controls") if mode == "native" else None
        print_ev = native_print_ev(recipe) if mode == "native" else 0.0
        if highlight and highlight.get("automatic_reference"):
            from .highlight_workflow import resolve_spatial_highlight_reference
            from .metering_snapshot import restore_zonal_metering

            restored = restore_zonal_metering(entry["bundle_text"], entry["runner"]._identity)
            highlight = resolve_spatial_highlight_reference(highlight, restored)
        reference_entry = None
        if highlight:
            with span("preview_highlight_reference", sample_step=step):
                reference_entry = self._highlight_reference(entry, highlight)
        usm = mode == "native" and native_usm_supported(render_config)
        exact_neighborhood = viewport is not None or step == 1
        if usm and not exact_neighborhood:
            self.owner.preview_quality = dict(
                mode="interactive", approximate=True, label="交互预览 · 100%查看精细效果",
                full_support_measurement=True, sample_step=step, full_output_size=size,
                sampling_scope="Native fixed-mask USM runs on the screen grid with radius/step; base and highlight reference meters remain full-source",
                full_resolution_and_saved_output_unchanged=True,
                native_sharpen_amount=render_config.lab.sharpen,
                native_source_radius=render_config.lab.sharpen_radius,
                native_screen_radius=render_config.lab.sharpen_radius / step)
        self.owner.processor = self.session.processor
        self._progress("sampled_native", 0.72, "正在生成当前画面的清晰预览…")
        color_doc = (recipe.get("nikon_research_controls") if mode == "cool-shared-rgb"
                     else recipe.get("flex_research_controls") if mode == "flex-endpoint" else None)
        sample_key = _key([entry["key"], mode, color_doc,
                          render_config.to_dict() if mode != "flex-endpoint" else None,
                          phase_roi, step, direction, highlight, point, exact_neighborhood if usm else True, print_ev])
        cached = self.samples.get(sample_key)
        icc = None
        if cached is None:
            if mode == "flex-endpoint":
                from .flex_research_workflow import render_flex_research_workflow
                from .positive_float import working_icc_bytes

                x0, y0, x1, y1 = phase_roi
                sample = np.array(self.session.source[y0:y1:step, x0:x1:step], copy=True, order="C")
                sample.setflags(write=False)
                self._progress("sampled_color", 0.86, "正在转换当前预览的色彩…")
                pixels, icc, _ = render_flex_research_workflow(
                    sample, entry["window"].analysis, color_doc, working_icc=working_icc_bytes(),
                    source_key=self.source_key, source_geometry=dict(
                        full_source_shape=list(self.session.source.shape), view_roi=phase_roi, sample_step=step))
            elif mode == "native" and (highlight or point or usm):
                pixels, icc, joined = self._native_join_sample(
                    entry, render_config, reference_entry, highlight, point, phase_roi, step,
                    exact=exact_neighborhood, print_ev=print_ev)
            else:
                result = render_frozen_sample(entry["runner"], entry["source"],
                    rectangle=phase_roi, sample_step=step, certificate=entry["certificate"],
                    render_config=render_config, print_ev=print_ev)
                if result is None:
                    return None
                pixels, icc, _ = result
                self.base_sample_render_calls += 1
                self.native_sample_pixels += int(pixels.shape[0] * pixels.shape[1])
                if mode == "cool-shared-rgb":
                    from .nikon_research_workflow import render_nikon_research_workflow

                    pixels.setflags(write=False)
                    self._progress("sampled_color", 0.86, "正在转换当前预览的色彩…")
                    pixels, icc, _ = render_nikon_research_workflow(
                        pixels, color_doc, working_icc=icc, source_key=self.source_key,
                        reference_config=render_config.to_dict(), source_geometry=dict(
                            full_source_shape=list(self.session.source.shape), view_roi=phase_roi, sample_step=step))
            pixels = orient_codes(pixels, direction)
            cached = dict(pixels=np.frombuffer(pixels.tobytes(), dtype=np.uint16).reshape(pixels.shape), icc=icc,
                          native_join_receipt=joined if mode == "native" and (highlight or point or usm) else None)
            self.samples[sample_key] = cached
            while len(self.samples) > 3:
                self.samples.popitem(last=False)
        pixels, icc = cached["pixels"], cached["icc"]
        current = self.owner._finish(pixels, recipe, mode)
        if interactive:
            self.owner.reference_label = ""
            reference = current
        else:
            reference = self._reference(recipe, mode, phase_roi, step, direction, pixels, icc, bounded=viewport is not None)
        self.certificate.validate()
        if entry["domain"] is not None:
            entry["domain"].validate()
        self.owner._validate(request)
        self.owner.output_size, self.owner.sample_step = size, step
        self.owner.viewport_result = viewport
        doc = entry["document"]
        from .metering_hover_v45 import metering_zone_records

        self.owner.metering_receipt = deepcopy(dict(roi=list(entry["window"].analysis.roi),
            metering=doc["metering"], automatic=doc.get("automatic_avoidance"),
            report=entry["snapshot"]["automatic_result"] or {},
            zones=metering_zone_records(entry["snapshot"]["window"]),
            **({"highlight_reference_roi": highlight["reference_roi"]} if highlight else {})))
        if interactive:
            self.owner.preview_quality = {
                **(self.owner.preview_quality or {}),
                "mode": "density-screen",
                "approximate": True,
                "interactive_frame": True,
                "label": "拖动预览",
                "sample_step": step,
                "full_output_size": size,
                "full_support_measurement": True,
            }
        self.last_receipt = dict(route="full-support-frozen-then-point-grid", full_support_measured=True,
            measurement_cache_hit=entry["cache_hit"], source_prepare_calls=self.source_prepare_calls,
            measurement_calls=self.measurement_calls, sample_pixels=int(current.shape[0] * current.shape[1]),
            reference_measurement_calls=self.reference_measurement_calls,
            native_base_sample_calls=self.base_sample_render_calls,
            native_reference_sample_calls=self.reference_sample_render_calls,
            native_evaluated_pixels=self.native_sample_pixels,
            native_neighborhood="original-width-original-band-USM" if usm and exact_neighborhood else "screen-grid-USM" if usm else None,
            approximate=bool(interactive or (usm and not exact_neighborhood)),
            highlight_reference_roi=deepcopy(highlight["reference_roi"]) if highlight else None,
            highlight_reference_window=(asdict(reference_entry["window"]) if reference_entry else None),
            native_join_processing=deepcopy(cached["native_join_receipt"]),
            full_Native_exports=0, full_CMS_pixels=0, full_source_shape=list(self.session.source.shape),
            retained_source_bytes=int(self.session.source.nbytes),
            retained_derived_bytes=self.cache_report["derived_bytes"],
            auxiliary_grid_hits=entry["certificate"].grid_hits, validated_bundle_hits=entry["certificate"].restore_hits,
            automatic_detection_hits=entry["certificate"].automatic_hits,
            exact_leaf_hits=entry["certificate"].leaf_hits, exact_leaf_misses=entry["certificate"].leaf_misses,
            density_domain="native-c41-sdk-unmixed-log-density-v45" if entry["domain"] else "linear-selected-sensor-bands",
            measurement_window=deepcopy(entry["snapshot"]["window"]))
        if print_ev:
            self.last_receipt["native_print_exposure"] = print_exposure_receipt(print_ev)
        self._progress("sampled_encode", 0.94, "预览处理完成，正在准备屏幕显示…")
        if interactive:
            return current, current, icc
        return np.array(current, copy=True), np.array(reference, copy=True), icc

    def close(self):
        self._close_native_metal()
        self.native_meter_proxy = None
        self.entries.clear()
        self.float_samples.clear()
        self.protection_samples.clear()
        self.usm_bands.clear()
        for domain in self.film_domains.values():
            domain.close()
        self.film_domains.clear()
        self.samples.clear()
        self.engine_proxy_samples.clear()
        self.original_base_samples.clear()
        self.original_meter_values.clear()
        self.original_highlight_measurements.clear()
        self.original_companion = None
        self.references.clear()
        if self.session is not None and self.session.processor is not None:
            self.session.processor.cleanup()
            self.session.processor.destroy_all()
        self.session = None
        if self.certificate is not None:
            self.certificate.close()
        self.certificate = None

    def _close_native_metal(self):
        for metal in (self.native_metal_tone, self.interactive_metal_tone):
            if metal is not None:
                metal.close()
        self.native_metal_tone = self.interactive_metal_tone = None
        self.original_screen_base = None

    @property
    def cache_report(self):
        samples = (*self.samples.values(), *self.engine_proxy_samples.values())
        return dict(retained_source_bytes=int(self.session.source.nbytes) if self.certificate else 0,
                    derived_bytes=sum(item["pixels"].nbytes for item in samples
                                      if "pixels" in item)
                        + sum(item["encoded"].nbytes for item in samples
                              if "encoded" in item)
                        + sum(item["drag_finish"][1].nbytes for item in samples
                              if "drag_finish" in item)
                        + sum(sample.nbytes for sample, _, _ in self.original_base_samples.values())
                        + (self.original_companion[1]["encoded"].nbytes if self.original_companion else 0)
                        + sum(value.nbytes for metrics, _ in self.original_meter_values.values()
                              for value in metrics.values() if isinstance(value, np.ndarray))
                        + (self.native_meter_proxy[2].nbytes if self.native_meter_proxy is not None else 0)
                        + sum(metal.retained_cpu_bytes for metal in (self.native_metal_tone, self.interactive_metal_tone)
                              if metal is not None)
                        + sum(reference["encoded"].nbytes
                              for item in samples
                              if (reference := item.get("highlight_reference")) is not None)
                        + sum(result["pixels"].nbytes for item in samples
                              if (result := item.get("highlight_reference", {}).get("result")) is not None)
                        + sum(item["nikon_cms"]["pixels"].nbytes for item in samples
                              if "nikon_cms" in item)
                        + sum(item["flex_legacy"]["pixels"].nbytes for item in samples
                              if "pixels" in item.get("flex_legacy", {}))
                        + sum(item["pixels"].nbytes for item in self.references.values())
                        + sum(item["image"].nbytes for item in self.float_samples.values())
                        + sum(item.mask.nbytes for item in self.protection_samples.values())
                        + sum(item["image"].nbytes for item in self.usm_bands.values())
                        + sum(domain.source.nbytes + domain.certificate.derived_bytes for domain in self.film_domains.values())
                        + (self.certificate.derived_bytes if self.certificate else 0))

    def clear_derived(self):
        self._close_native_metal()
        self.native_meter_proxy = None
        self.entries.clear()
        self.float_samples.clear()
        self.protection_samples.clear()
        self.usm_bands.clear()
        for domain in self.film_domains.values():
            domain.close()
        self.film_domains.clear()
        self.samples.clear()
        self.engine_proxy_samples.clear()
        self.original_base_samples.clear()
        self.original_meter_values.clear()
        self.original_companion = None
        self.references.clear()
        if self.certificate is not None:
            self.certificate.grids.clear()
            self.certificate.restored.clear()
            self.certificate.automatic.clear()
            self.certificate.bundles.clear()
            self.certificate.leaves.clear()
