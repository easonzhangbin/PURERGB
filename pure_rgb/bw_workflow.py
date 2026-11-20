"""Native-only single-RAW B&W jobs and frozen-source preview replay.

The vendor B&W density collapse, print curves, default Lab processing and SDK
encoder remain in use. Zonal auxiliary meters and local highlight references
are explicit additions; no triplet, Nikon/Flex fit or colour CMS path is used.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import ImageCms
import tifffile

from .bw_recipe import is_bw_recipe
from .bw_source import open_bw_source, validate_bw_source
from .cli import inspect_tiff
from .density_matrix import DensityWindow, negpy_window
from .engines import Analysis
from .metering_regions import metering_process, select_metering_roi
from .native import check_baseline, native_export, sha256
from .preview_data import verified_pixels
from .professional_controls import native_config
from .raw_import import create_processor
from .study import preview


def _roi(source, config, requested):
    if requested is not None:
        select_metering_roi(source.shape, requested, min_side=16, min_area_fraction=1)
        return list(requested)
    from negpy.features.exposure.normalization import resolve_analysis_region

    h, w = source.shape[:2]
    region, buffer = resolve_analysis_region(
        source.shape,
        None,
        config.process.analysis_buffer,
        config.process.analysis_rect,
        config.process.meter_rect,
    )
    y0, y1, x0, x1 = region if region else (0, h, 0, w)
    ratio = min(max(buffer, 0), 0.3)
    dy, dx = int((y1 - y0) * ratio), int((x1 - x0) * ratio)
    result = [x0 + dx, y0 + dy, x1 - dx, y1 - dy]
    select_metering_roi(source.shape, result, min_side=16, min_area_fraction=1)
    return result


def _window(source, config, roi):
    # Match Native's exact integer population, without normalized-edge rounding.
    x0, y0, x1, y1 = roi
    measured = negpy_window(source[y0:y1, x0:x1], config, (0, 0, x1 - x0, y1 - y0))
    return replace(measured, analysis=replace(measured.analysis, roi=tuple(roi)))


def _restore_window(document):
    return DensityWindow(
        document["provider"],
        Analysis(**deepcopy(document["analysis"])),
        tuple(document["log_dense"]),
        tuple(document["log_thin"]),
        deepcopy(document["metadata"]),
    )


def _configured(base, recipe):
    config = replace(
        base, exposure=replace(base.exposure, paper_profile=recipe["paper"])
    )
    return native_config(
        config, recipe.get("native_controls", {"version": 1, "values": {}})
    )


def _prepare_metering(source, config, roi, recipe, *, snapshot=None):
    if not recipe.get("bw_metering") or recipe.get("native_spatial_controls"):
        return None
    from .bw_metering import BWMaskedState

    state = (
        BWMaskedState.restore(snapshot, source, config, roi, recipe["bw_metering"])
        if snapshot is not None
        else BWMaskedState(source, config, roi, recipe["bw_metering"])
    )
    high = recipe.get("highlight_controls")
    if high:
        from .spatial_workflow_v31 import _check_highlight_reference_bounds

        reference = high["reference_roi"]
        _check_highlight_reference_bounds(reference, source.shape)
        a, b, c, d = reference
        x0, y0, x1, y1 = state.roi
        inside = (x0 <= a < c <= x1 and y0 <= b < d <= y1
                  and state._support[b - y0 : d - y0, a - x0 : c - x0].all())
        if inside or high.get("automatic_reference"):
            state.register_reference_roi(reference)
        else:
            # The legacy BW snapshot authenticates the density support. Keep
            # it unchanged and register the independently selected highlight
            # reference only for this render; replay can rebuild it from the
            # saved recipe while retaining the frozen base measurement.
            from .bw_metering import _measure, _window_record

            state._independent_snapshot = state.to_json()
            selected = np.ascontiguousarray(source[b:d, a:c])
            meter = replace(state.effective_config, process=replace(
                state.effective_config.process, meter_rect=None,
            ))
            window = negpy_window(selected, meter, (0, 0, c - a, d - b))
            window = replace(window, analysis=replace(window.analysis, roi=tuple(reference)))
            ref_config = state._reference_config(_window_record(window))
            _, metrics, receipt = _measure(
                source, ref_config, window, reference,
                np.ones((d - b, c - a), dtype=bool), 1.0,
                require_full_grid=True,
            )
            state._register("reference", ref_config, window, metrics, receipt)
            state._validate()
    return state


def _encode(
    processor, config, source, records, recipe, state, *, spatial_snapshot=None
):
    from .workflow_render import _native_highlight_export, _export_source_key

    if recipe.get("native_spatial_controls"):
        from .spatial_workflow_v31 import encode_spatial_workflow

        payload, _, info, snapshot = encode_spatial_workflow(
            processor,
            source,
            config,
            records,
            recipe["roi"],
            recipe["native_spatial_controls"],
            source_key=_export_source_key(processor, config, records),
            snapshot=spatial_snapshot,
            highlight=recipe.get("highlight_controls"),
            point=recipe.get("native_float_controls"),
        )
        processor.pure_spatial_processing = info
        processor.pure_spatial_snapshot = snapshot
        return payload, info["high_processing"]

    if state is not None:
        processor.engine_cpu = state.engine
        config = state.effective_config
    if recipe.get("highlight_controls"):
        # Match native_export's encoder contract without an unnecessary baseline.
        from negpy.domain.models import ColorSpace, ExportFormat, TiffCompression

        export = replace(
            config.export,
            export_fmt=ExportFormat.TIFF,
            export_bit_depth=16,
            export_color_space=ColorSpace.ADOBE_RGB.value,
            tiff_compression=TiffCompression.ZIP,
        )
        payload, high = _native_highlight_export(
            processor, config, records, source, export, recipe["highlight_controls"]
        )
        return payload, high
    payload, _ = native_export(processor, config, records)
    return payload, None


def _cropped(pixels, recipe, roi, *, apply_orientation=True, preview=False):
    from .workspace import output_rectangle
    from .workflow_render import exposure_codes

    x0, y0, x1, y1 = output_rectangle(recipe, roi, pixels.shape)
    cropped = pixels[y0:y1, x0:x1]
    from .common_positive import (
        apply_common_positive,
        checked_common_positive_controls,
    )

    common = checked_common_positive_controls(
        recipe.get("common_positive_controls")
    )
    if common:
        cropped = apply_common_positive(cropped, common)
    if preview:
        from .gpu_positive_preview import preview_output_codes

        result, _ = preview_output_codes(
            cropped, common_ev=recipe["exposure_ev"]
        )
    else:
        result = exposure_codes(cropped, recipe["exposure_ev"])
    document = recipe.get("post_controls", {}).get("native")
    if document:
        from .positive_post import post_codes

        result, _ = post_codes(result, document, sample_step=1)
    if apply_orientation and recipe.get("output_orientation"):
        from .output_orientation import orient_codes

        result = orient_codes(result, recipe["output_orientation"])
    return result


def render_bw(request, *, emit_callback=None):
    from .workspace import checked_recipe, output_rectangle, write_json
    from .workflow_render import _study_picture_region, emit

    emit_event = emit_callback or emit
    recipe = checked_recipe(request["recipe"])
    if not is_bw_recipe(recipe):
        raise ValueError("Explicit single B&W source mode required")
    records = request["inputs"]
    if (
        len(records) != 1
        or records[0]["role"] != "BW"
        or [records[0]["path"]] != recipe["raw"]
    ):
        raise ValueError("B&W recipe and frozen single input differ")
    if sha256(records[0]["path"]) != records[0]["sha256"]:
        raise ValueError("B&W RAW changed since enqueue")
    out = Path(request["directory"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["NEGPY_USER_DIR"] = str(out / "negpy-state")
    processor = None
    report = dict(
        state="running",
        recipe=recipe,
        inputs=deepcopy(records),
        results={},
        scope="Original single-frame NegPy B&W plus explicit zonal density/highlight; no RGB capture assembly or C41 research engine",
    )
    try:
        emit_event(type="progress", fraction=0.05, message="准备单张黑白 RAW")
        processor, base, source, opened = open_bw_source(recipe["raw"][0])
        if opened != records:
            raise ValueError("B&W source identity differs from queued input")
        report["bw_source_receipt"] = deepcopy(processor.pure_bw_source_receipt)
        report["bw_input_config"] = base.to_dict()
        config = _configured(base, recipe)
        if recipe["roi"] is None and type(request.get("import_stage")) is int:
            from .workflow_geometry_preview import write_geometry_previews

            locators = write_geometry_previews(source, out, monochrome=True)
            region = _study_picture_region(source, None, locators)
            region["coordinate_space"] = "EXIF-oriented single RAW linear source; half-open xyxy"
            report["picture_region"] = region
            roi = list(region["analysis_roi"])
        else:
            roi = _roi(source, config, recipe["roi"])
        # The spatial helper requires an exact source ROI, including auto-resolved BW jobs.
        effective_recipe = {**recipe, "roi": roi}
        state = _prepare_metering(source, config, roi, recipe)
        if state is None:
            window = _window(source, config, roi)
            selection = select_metering_roi(
                source.shape, roi, min_side=2, min_area_fraction=1
            )
            config = replace(
                config, process=metering_process(config.process, window, selection)
            )
        else:
            config = state.effective_config
            # Use the state's restored density-window receipt as the canonical window.
            from .metering_snapshot import restore_zonal_metering

            restored = restore_zonal_metering(state.bundle.to_json(), state.identity)
            window = restored.window
        emit_event(type="progress", fraction=0.4, message="生成 NegPy 原生黑白正像")
        payload, high = _encode(
            processor, config, source, records, effective_recipe, state
        )
        pixels, icc = inspect_tiff(payload)
        if list(pixels.shape) != list(source.shape):
            raise ValueError("B&W Native output geometry changed")
        with tifffile.TiffFile(io.BytesIO(payload)) as tif:
            page = tif.pages[0]
            dpi = dict(
                resolution=(page.tags[282].value, page.tags[283].value),
                resolutionunit=int(page.tags[296].value),
            )
        pixels = _cropped(pixels, recipe, roi)
        path = out / "native.tiff"
        tifffile.imwrite(
            path,
            pixels,
            photometric="rgb",
            compression="deflate",
            metadata=None,
            extratags=[(34675, 7, len(icc), icc, False), (274, 3, 1, 1, False)],
            **dpi,
        )
        record = dict(
            path=str(path),
            sha256=sha256(path),
            shape=list(pixels.shape),
            icc_sha256=hashlib.sha256(icc).hexdigest(),
            kind="Unchanged NegPy B&W print stages with explicit optional density/highlight additions",
        )
        actual, actual_icc = verified_pixels(record)
        if not np.array_equal(actual, pixels) or actual_icc != icc:
            raise ValueError("B&W TIFF pixel/ICC verification failed")
        preview(pixels, icc, (640, 480)).save(
            out / "native-preview.png",
            icc_profile=ImageCms.ImageCmsProfile(
                ImageCms.createProfile("sRGB")
            ).tobytes(),
        )
        report.update(
            roi=roi,
            output_roi=output_rectangle(recipe, roi, source.shape),
            source_shape=list(source.shape),
            density_window=asdict(window),
            native_config=config.to_dict(),
            icc_sha256=record["icc_sha256"],
            bw_metering_snapshot=(getattr(state, "_independent_snapshot", None)
                                  or state.to_json()) if state else None,
            bw_metering_processing=deepcopy(state.receipt) if state else None,
            highlight_processing=high,
        )
        report["results"]["native"] = record
        if recipe.get("native_spatial_controls"):
            report["native_spatial_snapshot"] = processor.pure_spatial_snapshot
            report["native_spatial_processing"] = deepcopy(
                processor.pure_spatial_processing
            )
        if sha256(records[0]["path"]) != records[0]["sha256"]:
            raise ValueError("B&W RAW changed during processing")
        report.update(state="complete", raw_hashes_unchanged=True)
    except BaseException as error:
        report.update(state="failed", error=str(error))
        raise
    finally:
        write_json(out / "manifest.json", report)
        if processor is not None:
            processor.cleanup()
            processor.destroy_all()
    emit_event(type="result", directory=str(out))
    return report


class BWSession:
    """Full Native processing is cached; display sampling happens afterwards."""

    def __init__(self, path, digest, state_directory, *, manifest=None, recipe=None):
        from .workspace import checked_recipe

        self.path, self.digest = Path(path).resolve(), digest
        if sha256(self.path) != digest:
            raise ValueError("B&W manifest changed")
        self.manifest = (
            deepcopy(manifest)
            if manifest is not None
            else json.loads(self.path.read_text())
        )
        self.recipe = (
            checked_recipe(self.manifest["recipe"])
            if recipe is None
            else deepcopy(recipe)
        )
        if (
            not is_bw_recipe(self.recipe)
            or self.manifest["state"] != "complete"
            or set(self.manifest["results"]) != {"native"}
        ):
            raise ValueError("Completed single B&W Native manifest required")
        if (
            self.recipe.get("bw_metering")
            and not self.recipe.get("native_spatial_controls")
            and (
                type(self.manifest.get("bw_metering_snapshot")) is not str
                or not self.manifest["bw_metering_snapshot"]
            )
        ):
            raise ValueError(
                "Completed zonal B&W job requires its frozen metering snapshot"
            )
        self.spatial_snapshot = self.manifest.get("native_spatial_snapshot")
        if (
            self.recipe.get("native_spatial_controls")
            and type(self.spatial_snapshot) is not str
        ):
            raise ValueError("Completed spatial BW job requires its frozen snapshot")
        records = self.manifest["inputs"]
        if (
            len(records) != 1
            or records[0]["role"] != "BW"
            or [records[0]["path"]] != self.recipe["raw"]
            or sha256(records[0]["path"]) != records[0]["sha256"]
        ):
            raise ValueError("B&W frozen source differs")
        self.identities = {r["path"]: self.identity(r["path"]) for r in records}
        self.state_directory = Path(state_directory)
        self.processor = self.source = self.base = None
        self.cache = {}
        self.sample_step = 1
        check_baseline()

    @staticmethod
    def identity(path):
        s = Path(path).stat()
        return s.st_size, s.st_mtime_ns, s.st_ctime_ns

    def prepare_raw(self):
        if self.processor is not None:
            return
        from negpy.domain.models import WorkspaceConfig

        os.environ["NEGPY_USER_DIR"] = str(self.state_directory / "native-state")
        base = WorkspaceConfig.from_flat_dict(
            deepcopy(self.manifest["bw_input_config"])
        )
        from .bw_source import _loader_contract

        if (
            _loader_contract(base)
            != self.manifest["bw_source_receipt"]["loader_contract"]
        ):
            raise ValueError("B&W decode parameters changed before source replay")
        processor = create_processor("legacy-v1")
        try:
            pixels, _, label = processor._load_source_f32(self.recipe["raw"][0], base)
            if (
                not isinstance(pixels, np.ndarray)
                or pixels.dtype != np.float32
                or pixels.ndim != 3
                or pixels.shape[-1] != 3
            ):
                raise ValueError(
                    "B&W loader must reproduce original HWC float32 source"
                )
            source = np.array(pixels, order="C", copy=True)
            source.setflags(write=False)
            validate_bw_source(
                self.manifest,
                base,
                source,
                self.manifest["inputs"],
                source_color_space=str(label),
            )
        except BaseException:
            processor.cleanup()
            processor.destroy_all()
            raise
        self.processor, self.base, self.source = processor, base, source
        self.source_color_space = str(label)

    def render(self, request):
        from .workspace import checked_recipe

        if (
            request.get("mode") != "native"
            or type(request.get("full_resolution", False)) is not bool
        ):
            raise ValueError("Single B&W supports explicit Native preview only")
        controls = request.get("controls")
        if type(controls) is not dict or set(controls) - {
            "native_controls",
            "highlight_controls",
            "exposure_ev",
            "native_spatial_controls",
            "native_float_controls",
            "output_orientation",
            "post_controls",
        }:
            raise ValueError(
                "B&W preview accepts Native/highlight/exposure edits; density changes require a new job"
            )
        base_recipe = deepcopy(self.recipe)
        for key in (
            "native_controls",
            "highlight_controls",
            "native_spatial_controls",
            "native_float_controls",
            "output_orientation",
            "post_controls",
        ):
            base_recipe.pop(key, None)
        recipe = checked_recipe({**base_recipe, **controls})
        for path, identity in self.identities.items():
            if self.identity(path) != identity:
                raise ValueError("B&W RAW identity changed")
        from .output_orientation_cache import cached_preview, remember_completed

        cached = cached_preview(
            self, recipe, "native", request.get("full_resolution", False)
        )
        if cached is not None:
            return cached
        self.prepare_raw()
        validate_bw_source(
            self.manifest,
            self.base,
            self.source,
            self.manifest["inputs"],
            source_color_space=self.source_color_space,
        )
        if recipe.get("native_spatial_controls") and self.spatial_snapshot is None:
            raise ValueError("Spatial BW preview requires a saved frozen-meter job")
        if recipe.get("native_spatial_controls") and self.spatial_snapshot is not None:
            from .spatial_workflow_v31 import validate_spatial_replay
            from .workflow_render import _export_source_key

            if self.manifest.get("native_spatial_snapshot") != self.spatial_snapshot:
                raise ValueError("Frozen spatial BW snapshot changed")
            spatial_config = _configured(self.base, recipe)
            selection = select_metering_roi(
                self.source.shape, self.manifest["roi"], min_side=2, min_area_fraction=1
            )
            spatial_config = replace(
                spatial_config,
                process=metering_process(
                    spatial_config.process,
                    _restore_window(self.manifest["density_window"]),
                    selection,
                ),
            )
            validate_spatial_replay(
                self.spatial_snapshot,
                self.source,
                spatial_config,
                self.manifest["roi"],
                recipe["native_spatial_controls"],
                _export_source_key(
                    self.processor, spatial_config, self.manifest["inputs"]
                ),
                recipe.get("highlight_controls"),
            )
        key = json.dumps(
            {
                k: recipe.get(k)
                for k in (
                    "native_controls",
                    "highlight_controls",
                    "native_spatial_controls",
                    "native_float_controls",
                )
            },
            sort_keys=True,
        )
        if key not in self.cache:
            from negpy.services.rendering.engine import DarkroomEngine

            self.processor.engine_cpu = DarkroomEngine()
            config = _configured(self.base, recipe)
            same_controls = recipe.get("native_controls") == self.recipe.get(
                "native_controls"
            )
            snapshot = (
                self.manifest.get("bw_metering_snapshot") if same_controls else None
            )
            state = _prepare_metering(
                self.source, config, self.manifest["roi"], recipe, snapshot=snapshot
            )
            if state is None:
                window = _restore_window(self.manifest["density_window"])
                selection = select_metering_roi(
                    self.source.shape,
                    self.manifest["roi"],
                    min_side=2,
                    min_area_fraction=1,
                )
                config = replace(
                    config, process=metering_process(config.process, window, selection)
                )
            else:
                config = state.effective_config
            payload, _ = _encode(
                self.processor,
                config,
                self.source,
                self.manifest["inputs"],
                {**recipe, "roi": self.manifest["roi"]},
                state,
                spatial_snapshot=self.spatial_snapshot,
            )
            pixels, icc = inspect_tiff(payload)
            if (
                list(pixels.shape) != self.manifest["source_shape"]
                or hashlib.sha256(icc).hexdigest() != self.manifest["icc_sha256"]
            ):
                raise ValueError("B&W replay output geometry/ICC changed")
            pixels.setflags(write=False)
            self.cache = {key: (pixels, icc)}
        source_pixels, icc = self.cache[key]
        pixels = _cropped(
            source_pixels, recipe, self.manifest["roi"], apply_orientation=False, preview=True
        )
        remember_completed(self, recipe, "native", pixels, icc)
        reference, saved_icc = verified_pixels(self.manifest["results"]["native"])
        if saved_icc != icc:
            raise ValueError("B&W preview working ICC differs")
        self.sample_step = (
            1
            if request.get("full_resolution", False)
            else max(1, int(np.ceil(max(reference.shape[:2]) / 1400)))
        )
        step = self.sample_step
        from .output_orientation import sample_oriented

        reference_orientation = self.recipe.get("output_orientation")
        if recipe.get("post_controls") != self.recipe.get("post_controls"):
            reference = _cropped(
                source_pixels, {**recipe, "exposure_ev": 0.0}, self.manifest["roi"],
                apply_orientation=False,
            )
            reference_orientation = None
            self.reference_label = "调整前（当前草稿）"
        return (
            sample_oriented(pixels, None, recipe.get("output_orientation"), step),
            sample_oriented(
                reference,
                reference_orientation,
                recipe.get("output_orientation"),
                step,
            ),
            icc,
        )
