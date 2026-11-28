"""Persistent preview process: cache decoded RAW, use frozen analysis, never edit jobs."""

import json
import hashlib
import os
from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import tifffile

from .cli import inspect_tiff
from .cool_whitepoint_study import lift_codes
from .editing_render import experimental_positive
from .endpoint_study import EndpointFit
from .engines import Analysis, density_auto, flexcolor_research, check_reference_sources
from .native import check_baseline, encode_experiment, native_export, sha256
from .native_headroom import native_headroom_active, native_headroom_strength, native_local_highlight
from .preview_data import verified_pixels
from .preview_fit_transport import MAX_FIT_EDGE, FitReferenceCache, can_transport_fit, write_fit_records
from .preview_grid import post_requires_full_grid
from .professional_controls import native_config, native_print_ev
from .rgb_curve_study import SharedCurveFit
from .tone_study import GAMMA, ICC_SHA256
from .workflow_render import (
    _native_float_export,
    _native_headroom_export,
    _native_highlight_export,
    _cool_headroom_export,
    _flex_research_export,
    _nikon_research_export,
    _nikon_reference_config,
    _nikon_reference_guard,
    _export_source_key,
)
from .workspace import (
    checked_recipe,
    output_rectangle,
    checked_native_float_controls,
    checked_highlight_controls,
    checked_cool_headroom_controls,
    checked_flex_research_controls,
    checked_nikon_research_controls,
)
from .flatfield import attach_calibration
from .diagnostics import setup, event, failure, span, MemorySampler
from .positive_post import post_codes
from .output_orientation_cache import cached_preview, remember_completed
from .output_orientation import (
    checked_document as checked_output_orientation,
    orient_codes,
    sample_oriented,
)
from .film_profiles import native_film_context
from .raw_import import create_processor, validate_frozen_import


EDITING_DOCUMENT_KEYS = (
    "measurement_method",
    "density_metering",
    "native_controls",
    "native_float_controls",
    "native_spatial_controls",
    "highlight_controls",
    "cool_headroom_controls",
    "flex_research_controls",
    "nikon_research_controls",
    "output_controls",
    "engine_controls",
    "post_controls",
    "output_orientation",
    "flex_profile",
    "native_film",
    "common_positive_controls",
    "experimental_film_controls",
)
PREVIEW_CONTROL_KEYS = frozenset((*EDITING_DOCUMENT_KEYS, "exposure_ev", "cool_tone"))
_REQUEST_STATE = None


def check_preview_superseded():
    if _REQUEST_STATE is not None:
        _REQUEST_STATE.check()


def preview_checkpoint():
    return _REQUEST_STATE.checkpoint() if _REQUEST_STATE is not None else lambda: None


def _checked_preview_request(request):
    """Reject unsupported request shape before a worker opens/replaces a session."""
    if not isinstance(request, Mapping):
        raise ValueError("预览请求必须为对象")
    controls = request.get("controls")
    if not isinstance(controls, Mapping):
        raise ValueError("预览调色参数必须为对象")
    if set(controls) - PREVIEW_CONTROL_KEYS:
        raise ValueError("预览仅接受调色参数；输入、密度和区域改变后需创建新任务重算")
    if "native_float_controls" in controls:
        checked_native_float_controls(controls["native_float_controls"])
    if "density_metering" in controls:
        from .shared_density_metering_v43 import checked_density_metering

        checked_density_metering(controls["density_metering"])
    if "output_orientation" in controls:
        checked_output_orientation(controls["output_orientation"])
    if "native_spatial_controls" in controls:
        from .spatial_workflow_v31 import checked_spatial_controls

        checked_spatial_controls(controls["native_spatial_controls"])
    if "highlight_controls" in controls:
        checked_highlight_controls(controls["highlight_controls"])
    if "cool_headroom_controls" in controls:
        checked_cool_headroom_controls(controls["cool_headroom_controls"])
    if "flex_research_controls" in controls:
        checked_flex_research_controls(controls["flex_research_controls"])
    if "nikon_research_controls" in controls:
        checked_nikon_research_controls(controls["nikon_research_controls"])
    mode = request.get("mode")
    if not isinstance(mode, str) or mode not in (
        "native",
        "cool-shared-rgb",
        "flex-endpoint",
    ):
        raise ValueError("未知预览呈现模式")
    if mode != "native" and "native_float_controls" in controls:
        raise ValueError("Native 浮点编辑参数仅能用于 Native 预览")
    if mode != "native" and "native_spatial_controls" in controls:
        raise ValueError("分区空间编辑参数仅能用于 Native 预览")
    if mode != "native" and "highlight_controls" in controls:
        raise ValueError("参考高光编辑参数当前仅能用于 Native 预览")
    if mode != "cool-shared-rgb" and "cool_headroom_controls" in controls:
        raise ValueError("Cool 保留余量参数仅能用于 Cool 预览")
    if mode != "flex-endpoint" and "flex_research_controls" in controls:
        raise ValueError("Flex 研究链参数仅能用于 Flex 预览")
    if mode != "cool-shared-rgb" and "nikon_research_controls" in controls:
        raise ValueError("Nikon 研究链参数仅能用于 Nikon／Cool 预览")
    if "common_positive_controls" in controls:
        from .common_positive import checked_common_positive_controls

        checked_common_positive_controls(controls["common_positive_controls"])
    if "experimental_film_controls" in controls:
        from .experimental_film import checked_experimental_film_controls

        checked_experimental_film_controls(
            controls["experimental_film_controls"]
        )
    full_resolution = request.get("full_resolution", False)
    if type(full_resolution) is not bool:
        raise ValueError("预览分辨率参数必须为布尔值")
    if type(request.get("tone_edit", False)) is not bool:
        raise ValueError("交互预览参数必须为布尔值")
    if "metering_edit" in request and (
        request["metering_edit"] is not True or mode != "native" or "interactive_edge" not in request
    ):
        raise ValueError("交互测量只适用于原生拖动预览")
    if "engine_proxy" in request and (
        request["engine_proxy"] is not True or "interactive_edge" not in request
        or "metering_edit" in request or "fit_edge" in request
        or request.get("release_priority")
    ):
        raise ValueError("引擎快速预览仅支持独立的小尺寸适应窗口请求")
    if "defer_reference" in request and (
        request["defer_reference"] is not True or "draft_recipe" not in request
        or full_resolution or "viewport" in request or request.get("crop_locator")
        or request.get("repair_plan") is not None or "interactive_edge" in request
    ):
        raise ValueError("延后对比参考仅支持无修补的普通适应窗口预览")
    for field, minimum, label in (
        ("fit_edge", 1400, "适应窗口预览"),
        ("interactive_edge", 256, "拖动预览"),
    ):
        if field not in request:
            continue
        edge = request[field]
        if type(edge) is not int or not minimum <= edge <= MAX_FIT_EDGE:
            raise ValueError(f"{label}长边必须为 {minimum} 到 {MAX_FIT_EDGE} 像素的整数")
        if ((field == "fit_edge" and mode != "native")
                or not request.get("tone_edit") or "draft_recipe" not in request
                or full_resolution or "viewport" in request or request.get("crop_locator")
                or request.get("repair_plan") is not None):
            raise ValueError(f"{label}尺寸只适用于无修补的适应窗口调色")
    return controls, mode, full_resolution


def emit(**data):
    if data.get("type") == "progress" and _REQUEST_STATE is not None:
        _REQUEST_STATE.check()
    print(json.dumps(data, ensure_ascii=False), flush=True)


def _legacy_flatfield_binding(manifest, processor, source, config, snapshot):
    """Rebind a historical profile only when its complete source is proven identical."""
    from dataclasses import replace

    saved = manifest.get("raw_import_receipt")
    actual = getattr(processor, "pure_raw_import_receipt", None)
    calibration = manifest.get("calibration") or {}
    if not isinstance(saved, dict) or not isinstance(actual, dict):
        return None
    try:
        packet = json.loads(snapshot)
        old_flat = saved["flatfield"]
        new_flat = actual["flatfield"]
        old_id, new_id = old_flat["profile_id"], new_flat["profile_id"]
        if not (old_id and new_id and old_flat["applied"] and new_flat["applied"]
                and old_flat["gain_token"] and old_flat["gain_token"] == new_flat["gain_token"]
                and packet["input_config"]["profile_id"] == old_id
                and config.flatfield.profile_id
                and calibration.get("gain_path") and calibration.get("gain_sha256")
                and sha256(calibration["gain_path"]) == calibration["gain_sha256"]):
            return None
        old_copy, new_copy = deepcopy(saved), deepcopy(actual)
        old_copy["flatfield"].pop("profile_id")
        new_copy["flatfield"].pop("profile_id")
        if json.dumps(old_copy, sort_keys=True, allow_nan=False) != json.dumps(
            new_copy, sort_keys=True, allow_nan=False
        ):
            return None
        pixels = hashlib.sha256(memoryview(source).cast("B")).hexdigest()
        packet_pixels = packet.get("source_pixels_sha256") or packet["source_identity"]["source_pixels_sha256"]
        if not (pixels == saved["source_pixels_sha256"] == packet_pixels
                and list(source.shape) == manifest["source_shape"] == saved["source_shape"]):
            return None
        token = hashlib.sha256(json.dumps(saved, sort_keys=True, allow_nan=False).encode()).hexdigest()
        source_key = hashlib.sha256(("".join(r["sha256"] for r in manifest["inputs"]) + token).encode()).hexdigest()
        if source_key != packet["source_key"]:
            return None
        return replace(config, flatfield=replace(config.flatfield, profile_id=old_id)), source_key
    except (KeyError, TypeError, ValueError, OSError):
        return None


class Session:
    def __init__(self, path, digest, state_directory):
        path = Path(path).resolve()
        if sha256(path) != digest:
            raise ValueError("预览来源清单已改变")
        self.path, self.digest = path, digest
        self.manifest = json.loads(path.read_text())
        self.recipe = checked_recipe(self.manifest["recipe"])
        if self.manifest["state"] != "complete":
            raise ValueError("仅可编辑已完成的照片")
        if self.recipe.get("source_mode") == "bw-single-v1":
            from .bw_workflow import BWSession

            self.bw_session = BWSession(
                path,
                digest,
                state_directory,
                manifest=self.manifest,
                recipe=self.recipe,
            )
            self.processor = None
            self.sample_step = 1
            return
        os.environ["NEGPY_USER_DIR"] = str(Path(state_directory) / "native-state")
        check_baseline()
        check_reference_sources()
        for r in self.manifest["inputs"]:
            if sha256(r["path"]) != r["sha256"]:
                raise ValueError("原 RAW 已改变，不能使用旧分析预览")
        self.identities = {
            r["path"]: self.identity(r["path"]) for r in self.manifest["inputs"]
        }
        if self.recipe["flatfield"]:
            cal = self.manifest["calibration"]
            if sha256(cal["path"]) != cal["sha256"]:
                raise ValueError("平场源文件已改变")
            self.identities[cal["path"]] = self.identity(cal["path"])
            if cal.get("gain_path") and cal.get("gain_sha256"):
                if sha256(cal["gain_path"]) != cal["gain_sha256"]:
                    raise ValueError("平场增益文件已改变")
                self.identities[cal["gain_path"]] = self.identity(cal["gain_path"])
        self.processor = self.config = self.source = self.export = None
        self.experimental, self.native_cache, self.references = {}, {}, {}
        self.native_print_cache = {}
        self.native_print = None
        self._spatial_context_cache = {}
        self.headroom_cache = {}
        self.flex_research_cache = {}
        self.nikon_reference_cache = None
        self.nikon_research_cache = {}
        self.shared_measurement_cache = None
        self.shared_measurement_snapshot = self.manifest.get("shared_measurement_snapshot")
        self.shared_measurement_processing = deepcopy(self.manifest.get("shared_measurement_processing"))
        self._shared_measurement_binding = json.dumps(
            [self.recipe.get("density_metering"), self.shared_measurement_snapshot,
             self.shared_measurement_processing, self.manifest.get("density_window")],
            sort_keys=True, allow_nan=False,
        )
        if self.recipe.get("density_metering"):
            self._check_shared_measurement()
        self.spatial_snapshot = self.manifest.get("native_spatial_snapshot")
        if (
            self.recipe.get("native_spatial_controls")
            and type(self.spatial_snapshot) is not str
        ):
            raise ValueError("已保存的分区空间结果缺少冻结测量快照")

    @staticmethod
    def identity(path):
        s = Path(path).stat()
        return s.st_size, s.st_mtime_ns, s.st_ctime_ns

    def prepare_raw(self):
        if self.processor is not None:
            return
        from negpy.domain.models import WorkspaceConfig
        from negpy.features.flatfield.logic import set_gain_provider
        from negpy.services.assets.flatfield import FlatFieldProfiles

        emit(type="progress", message="首次准备 RAW 预览；画面保持可见…")
        set_gain_provider(FlatFieldProfiles.load_gain)
        config = self.manifest.get(
            "native_reference_config", self.manifest["native_config"]
        )
        self.config = WorkspaceConfig.from_flat_dict(deepcopy(config))
        if self.recipe["flatfield"]:
            calibration = self.manifest["calibration"]
            self.config, applied = attach_calibration(
                self.config,
                calibration,
                Path(os.environ["NEGPY_USER_DIR"]) / "calibration",
                self.recipe["raw"],
                strength=self.recipe.get("flatfield_strength", 1.0),
            )
        reference_shape = (
            (applied.get("provenance") or {}).get("capture", {}).get("sensor_shape")
            if self.recipe["flatfield"]
            else None
        )
        processor = create_processor(
            self.recipe.get("raw_import", "legacy-v1"),
            flatfield_reference_shape=reference_shape,
            **({"lens_correction": True} if self.recipe.get("lens_correction") else {}),
            **({"chromatic_aberration": True} if self.recipe.get("chromatic_aberration") else {}),
            **({"lens_profile": self.recipe["lens_profile"]} if self.recipe.get("lens_profile") else {}),
            **(
                {"registration_check": self.recipe["registration_check"]}
                if self.recipe.get("registration_check")
                else {}
            ),
        )
        from .decoded_cache import restore_source, store_source

        source = restore_source(
            processor,
            self.manifest,
            self.recipe,
            config,
            reference_shape,
            self.recipe["raw"][0],
            self.config,
        )
        if source is None:
            source, ir, color_space = processor._load_source_f32(
                self.recipe["raw"][0], self.config
            )
            validate_frozen_import(processor, self.manifest, source)
            store_source(
                processor,
                self.manifest,
                self.recipe,
                config,
                reference_shape,
                self.recipe["raw"][0],
                source,
                ir,
                color_space,
            )
        else:
            validate_frozen_import(processor, self.manifest, source)
        self.processor, self.source = processor, source
        # Match the actual encoder contract, without rendering an unnecessary baseline.
        from dataclasses import replace
        from negpy.domain.models import ExportFormat, ColorSpace, TiffCompression

        self.export = replace(
            self.config.export,
            export_fmt=ExportFormat.TIFF,
            export_bit_depth=16,
            export_color_space=ColorSpace.ADOBE_RGB.value,
            tiff_compression=TiffCompression.ZIP,
        )

    def source_view(self, pixels):
        x0, y0, x1, y1 = output_rectangle(
            self.recipe, self.manifest["roi"], pixels.shape
        )
        pixels = pixels[y0:y1, x0:x1]
        step = (
            1
            if self.full_resolution
            else max(1, int(np.ceil(max(pixels.shape[:2]) / 1400)))
        )
        return np.ascontiguousarray(pixels[::step, ::step]), step

    def prepare_nikon_reference(self):
        """Keep the initial pre-edit Native reference separate from Native edits."""
        config_doc = _nikon_reference_config(self.manifest)
        expected = self.manifest.get(
            "native_reference_config", self.manifest["native_config"]
        )
        if config_doc != expected:
            raise ValueError("Nikon 参考配置与已冻结的初次 Native 配置不一致")
        _nikon_reference_guard(self.config)
        source_key = _export_source_key(
            self.processor, self.config, self.manifest["inputs"]
        )
        identity = json.dumps(
            [config_doc, self.config.to_dict(), source_key],
            sort_keys=True,
            allow_nan=False,
        )
        if self.recipe.get("density_metering"):
            identity += hashlib.sha256(self.shared_measurement_snapshot.encode()).hexdigest()
        if (
            self.nikon_reference_cache is None
            or self.nikon_reference_cache["identity"] != identity
        ):
            if self.recipe.get("density_metering"):
                pixels, icc = self.prepare_shared_measurement()
            else:
                payload, _ = native_export(
                    self.processor, self.config, self.manifest["inputs"]
                )
                pixels, icc = inspect_tiff(payload)
            if pixels.shape != self.source.shape:
                raise ValueError("Nikon 参考正像与完整来源几何不一致")
            pixels.setflags(write=False)
            self.nikon_reference_cache = {
                "identity": identity,
                "pixels": pixels,
                "icc": icc,
                "pixels_sha256": hashlib.sha256(
                    memoryview(pixels).cast("B")
                ).hexdigest(),
            }
        reference = self.nikon_reference_cache
        if (
            reference["pixels"].dtype != np.uint16
            or reference["pixels"].shape != self.source.shape
            or not reference["pixels"].flags.c_contiguous
            or reference["pixels"].flags.writeable
            or hashlib.sha256(reference["icc"]).hexdigest() != ICC_SHA256
            or hashlib.sha256(memoryview(reference["pixels"]).cast("B")).hexdigest()
            != reference["pixels_sha256"]
        ):
            raise ValueError("Nikon 中性参考缓存发生变化")
        return reference

    def _check_shared_measurement(self):
        from .shared_density_metering_v43 import validate_saved_shared_measurement, spatial_document

        snapshot, info = self.shared_measurement_snapshot, self.shared_measurement_processing
        if (type(snapshot) is not str or not snapshot or type(info) is not dict
                or self.manifest.get("shared_measurement_snapshot") != snapshot
                or self.manifest.get("shared_measurement_processing") != info
                or info.get("contract_document") != self.recipe["density_metering"]
                or info.get("spatial_controls") != spatial_document(self.recipe["density_metering"])
                or info.get("measurement_only") is not True or info.get("native_reference") is not True
                or info.get("metering_window") != self.manifest.get("density_window")):
            raise ValueError("共享密度测量的冻结参考或快照不一致")
        if (sha256(self.path) != self.digest or self._shared_measurement_binding != json.dumps(
            [self.recipe.get("density_metering"), snapshot, info, self.manifest.get("density_window")],
            sort_keys=True, allow_nan=False,
        )):
            raise ValueError("共享密度测量的原始清单或冻结绑定已改变")
        validate_saved_shared_measurement(self.manifest)

    def prepare_shared_measurement(self):
        """Replay the authenticated neutral masked meter, never a Native edit."""
        from .shared_density_metering_v43 import render_shared_measurement

        self._check_shared_measurement()
        if self.shared_measurement_cache is None:
            self.prepare_raw()
            payload, _, info, _, _ = render_shared_measurement(
                self.processor, self.source, self.config, self.manifest["inputs"],
                self.manifest["roi"], self.recipe["density_metering"],
                source_key=_export_source_key(self.processor, self.config, self.manifest["inputs"]),
                snapshot=self.shared_measurement_snapshot,
            )
            if info["metering_window"] != self.shared_measurement_processing["metering_window"]:
                raise ValueError("共享密度重放窗口与保存结果不一致")
            pixels, icc = inspect_tiff(payload)
            pixels.setflags(write=False)
            self.shared_measurement_cache = dict(
                pixels=pixels, icc=icc,
                sha256=hashlib.sha256(memoryview(pixels).cast("B")).hexdigest(),
            )
        cache = self.shared_measurement_cache
        if (cache["pixels"].shape != self.source.shape or cache["pixels"].flags.writeable
                or hashlib.sha256(memoryview(cache["pixels"]).cast("B")).hexdigest() != cache["sha256"]
                or hashlib.sha256(cache["icc"]).hexdigest() != ICC_SHA256):
            raise ValueError("共享密度中性参考缓存发生变化")
        return cache["pixels"], cache["icc"]

    def render(self, request):
        self.native_print = None
        if hasattr(self, "bw_session"):
            result = self.bw_session.render(request)
            self.processor = self.bw_session.processor
            self.sample_step = self.bw_session.sample_step
            return result
        controls, mode, full_resolution = _checked_preview_request(request)
        if mode not in self.manifest["results"]:
            raise ValueError("预览呈现模式不在已完成结果中")
        base = dict(self.recipe)
        for key in EDITING_DOCUMENT_KEYS:
            base.pop(key, None)
        recipe = checked_recipe({**base, **controls})
        if recipe.get("measurement_method") != self.recipe.get("measurement_method"):
            raise ValueError("密度测量算法已改变，需要重新创建测量参考")
        if recipe.get("density_metering") != self.recipe.get("density_metering"):
            raise ValueError("共享密度支持已改变，需要重新创建测量参考")
        if self.recipe.get("density_metering"):
            self._check_shared_measurement()
        if mode == "cool-shared-rgb" and recipe.get("nikon_research_controls"):
            from .nikon_research_workflow import checked_nikon_resources

            _nikon_reference_config(self.manifest)
            checked_nikon_resources()
        for path, identity in self.identities.items():
            if self.identity(path) != identity:
                raise ValueError("RAW 状态发生变化，请重新导入")
        cached = cached_preview(self, recipe, mode, full_resolution)
        if cached is not None:
            return cached
        # Geometry, neighbourhood effects and position-dependent finishing
        # must see the saved output canvas before display subsampling.
        # Reference sampling and source-processing resolution are separate.
        orientation_preview = bool(
            recipe.get("output_orientation") or self.recipe.get("output_orientation")
        )
        full_grid_preview = not full_resolution and (
            post_requires_full_grid(recipe.get("post_controls", {}).get(mode))
            or (mode == "native" and bool(recipe.get("native_spatial_controls")))
            or orientation_preview
        )
        self.full_resolution = full_resolution or full_grid_preview
        view_key = (mode, full_resolution, orientation_preview)
        processing_view_key = (mode, self.full_resolution)
        if view_key not in self.references:
            original, icc = verified_pixels(self.manifest["results"][mode])
            step = (
                1
                if full_resolution
                else max(1, int(np.ceil(max(original.shape[:2]) / 1400)))
            )
            self.references = {
                view_key: (
                    np.ascontiguousarray(
                        original if orientation_preview else original[::step, ::step]
                    ),
                    icc,
                    step,
                )
            }
        reference, icc, self.sample_step = self.references[view_key]
        if orientation_preview:
            reference = sample_oriented(
                reference,
                self.recipe.get("output_orientation"),
                recipe.get("output_orientation"),
                self.sample_step,
            )
        with span("preview_prepare_raw"):
            self.prepare_raw()
        if mode == "native":
            document = recipe.get("native_controls", {"version": 1, "values": {}})
            film = recipe.get("native_film")
            float_controls = recipe.get("native_float_controls")
            headroom = native_headroom_active(recipe)
            print_ev = native_print_ev(recipe)
            highlight_controls = native_local_highlight(recipe)
            spatial_controls = recipe.get("native_spatial_controls")
            spatial_config = spatial_source_key = None
            cache_documents = [document, film, recipe.get("measurement_method")]
            if float_controls:
                cache_documents.append(float_controls)
            if highlight_controls:
                cache_documents.append(highlight_controls)
            if spatial_controls:
                cache_documents.append(spatial_controls)
                if self.spatial_snapshot is None:
                    raise ValueError("启用分区空间处理需要先保存具有冻结测量的新任务")
                if self.spatial_snapshot is not None:
                    from .spatial_workflow_v31 import validate_spatial_replay

                    if (
                        self.manifest.get("native_spatial_snapshot")
                        != self.spatial_snapshot
                    ):
                        raise ValueError("冻结的分区空间快照发生变化")
                    context_key = json.dumps([film, document], sort_keys=True, allow_nan=False)
                    if context_key not in self._spatial_context_cache:
                        with span("preview_native_context"):
                            spatial_config, _ = native_film_context(
                                self.config, self.source, self.manifest["roi"], recipe["density"], film
                            )
                            spatial_config = native_config(spatial_config, document)
                        bound = _legacy_flatfield_binding(
                            self.manifest, self.processor, self.source, spatial_config, self.spatial_snapshot
                        ) if self.recipe.get("flatfield") else None
                        if bound:
                            spatial_config, spatial_source_key = bound
                        else:
                            spatial_source_key = _export_source_key(
                                self.processor, spatial_config, self.manifest["inputs"]
                            )
                        self._spatial_context_cache = {context_key: (spatial_config, spatial_source_key)}
                    spatial_config, spatial_source_key = self._spatial_context_cache[context_key]
                    validate_spatial_replay(
                        self.spatial_snapshot,
                        self.source,
                        spatial_config,
                        self.manifest["roi"],
                        spatial_controls,
                        spatial_source_key,
                        highlight_controls,
                        film=film,
                    )
            key = json.dumps(cache_documents, sort_keys=True) + str(
                self.full_resolution
            )
            if key not in self.native_cache:
                if spatial_controls:
                    config = spatial_config
                else:
                    with span("preview_native_context"):
                        config, _ = native_film_context(
                            self.config,
                            self.source,
                            self.manifest["roi"],
                            recipe["density"],
                            film,
                        )
                        config = native_config(config, document)
                        if recipe.get("measurement_method") == "negpy":
                            from .measurement_method import original_config

                            config = original_config(config, self.manifest["roi"], self.source.shape)
                with span(
                    "preview_native_positive",
                    spatial=bool(spatial_controls),
                    film=bool(film),
                    highlight=bool(highlight_controls),
                    shared_density=bool(recipe.get("density_metering")),
                    full_grid=self.full_resolution,
                    source_shape=self.manifest.get("source_shape"),
                ):
                    if spatial_controls:
                        from .spatial_workflow_v31 import encode_spatial_workflow

                        payload, _, _, _ = encode_spatial_workflow(
                            self.processor,
                            self.source,
                            config,
                            self.manifest["inputs"],
                            self.manifest["roi"],
                            spatial_controls,
                            source_key=spatial_source_key,
                            snapshot=self.spatial_snapshot,
                            highlight=highlight_controls,
                            point=float_controls,
                            film=film,
                            print_ev=print_ev,
                        )
                    elif headroom or (print_ev and not highlight_controls):
                        payload, _, _ = _native_headroom_export(
                            self.processor, config, self.manifest["inputs"], self.export, float_controls,
                            strength=native_headroom_strength(recipe) if headroom else 0,
                            print_ev=print_ev,
                        )
                        self.native_print_cache = {key: deepcopy(self.processor.pure_native_print)}
                    elif highlight_controls:
                        payload, _ = _native_highlight_export(
                            self.processor,
                            config,
                            self.manifest["inputs"],
                            self.source,
                            self.export,
                            highlight_controls,
                            float_controls,
                            film=film,
                            print_ev=print_ev,
                        )
                    elif float_controls:
                        payload, _ = _native_float_export(
                            self.processor,
                            config,
                            self.manifest["inputs"],
                            self.export,
                            float_controls,
                        )
                    elif recipe.get("density_metering"):
                        pixels, other = self.prepare_shared_measurement()
                        if other != icc:
                            raise ValueError("共享密度Native中性参考ICC不一致")
                        pixels, _ = self.source_view(pixels)
                        self.native_cache = {key: pixels}
                        payload = None
                    else:
                        payload, _ = native_export(
                            self.processor, config, self.manifest["inputs"]
                        )
                        self.native_print_cache = {key: deepcopy(self.processor.pure_native_print)}
                    if payload is not None:
                        pixels, other = inspect_tiff(payload)
                        if other != icc:
                            raise ValueError("预览工作 ICC 与原图不一致")
                        pixels, _ = self.source_view(pixels)
                        self.native_cache = {key: pixels}
            pixels = self.native_cache[key]
            self.native_print = deepcopy(self.native_print_cache.get(key))
        elif mode == "cool-shared-rgb" and recipe.get("nikon_research_controls"):
            original = self.prepare_nikon_reference()
            if original["icc"] != icc:
                raise ValueError("Nikon 参考正像 ICC 与共同工作 ICC 不一致")
            document = recipe["nikon_research_controls"]
            key = json.dumps(
                [
                    document,
                    original["identity"],
                    original["pixels_sha256"],
                    self.full_resolution,
                    recipe.get("experimental_film_controls"),
                ],
                sort_keys=True,
            )
            if key not in self.nikon_research_cache:
                small, step = self.source_view(original["pixels"])
                small.setflags(write=False)
                geometry = {
                    "full_source_shape": list(self.source.shape),
                    "view_roi": output_rectangle(
                        recipe, self.manifest["roi"], self.source.shape
                    ),
                    "sample_step": step,
                }
                pixels, other, _ = _nikon_research_export(
                    self.processor,
                    self.config,
                    self.manifest["inputs"],
                    small,
                    icc,
                    document,
                    geometry,
                    experimental_film=recipe.get("experimental_film_controls"),
                )
                if other != icc:
                    raise ValueError("Nikon 研究链预览 ICC 与保存图不一致")
                self.nikon_research_cache = {key: pixels}
            pixels = self.nikon_research_cache[key]
        elif mode == "flex-endpoint" and recipe.get("flex_research_controls"):
            if recipe.get("density_metering"):
                self.prepare_shared_measurement()
            document = recipe["flex_research_controls"]
            key = (
                json.dumps(document, sort_keys=True)
                + json.dumps(recipe.get("experimental_film_controls"), sort_keys=True)
                + str(self.full_resolution)
            )
            if key not in self.flex_research_cache:
                small, step = self.source_view(self.source)
                analysis = Analysis(**self.manifest["density_window"]["analysis"])
                geometry = {
                    "full_source_shape": list(self.source.shape),
                    "view_roi": list(
                        output_rectangle(
                            recipe, self.manifest["roi"], self.source.shape
                        )
                    ),
                    "sample_step": step,
                }
                pixels, other, _ = _flex_research_export(
                    self.processor,
                    self.config,
                    self.manifest["inputs"],
                    small,
                    analysis,
                    icc,
                    document,
                    geometry,
                    experimental_film=recipe.get("experimental_film_controls"),
                )
                if other != icc:
                    raise ValueError("Flex 研究链预览 ICC 与保存图不一致")
                self.flex_research_cache = {key: pixels}
            pixels = self.flex_research_cache[key]
        elif mode == "cool-shared-rgb" and recipe.get("cool_headroom_controls"):
            document = recipe["cool_headroom_controls"]
            key = json.dumps(document, sort_keys=True) + str(self.full_resolution)
            if key not in self.headroom_cache:
                small, step = self.source_view(self.source)
                analysis = Analysis(**self.manifest["density_window"]["analysis"])
                fit = SharedCurveFit(**self.manifest["results"][mode]["fit"])
                geometry = {
                    "full_source_shape": list(self.source.shape),
                    "view_roi": list(
                        output_rectangle(
                            recipe, self.manifest["roi"], self.source.shape
                        )
                    ),
                    "sample_step": step,
                }
                pixels, other, _ = _cool_headroom_export(
                    self.processor,
                    self.config,
                    self.manifest["inputs"],
                    small,
                    analysis,
                    fit,
                    self.export,
                    document,
                    geometry,
                )
                if other != icc:
                    raise ValueError("Cool 保留余量预览 ICC 与保存图不一致")
                self.headroom_cache = {key: pixels}
            pixels = self.headroom_cache[key]
        else:
            if processing_view_key not in self.experimental:
                small, _ = self.source_view(self.source)
                analysis = Analysis(**self.manifest["density_window"]["analysis"])
                if mode == "cool-shared-rgb":
                    samples = density_auto(np.maximum(small, 0), analysis)
                    payload = encode_experiment(self.processor, samples, self.export)
                else:
                    samples, _ = flexcolor_research(np.maximum(small, 0), analysis)
                    payload = encode_experiment(
                        self.processor, samples, self.export, transfer="working-encoded"
                    )
                codes, other = inspect_tiff(payload)
                if other != icc:
                    raise ValueError("实验预览工作 ICC 不一致")
                fit_cls = SharedCurveFit if mode == "cool-shared-rgb" else EndpointFit
                fit = fit_cls(**self.manifest["results"][mode]["fit"])
                self.experimental = {
                    processing_view_key: (
                        (codes.astype(float) / 65535) ** GAMMA,
                        small,
                        analysis,
                        fit,
                    )
                }
            linear, small, analysis, fit = self.experimental[processing_view_key]
            pixels, _ = experimental_positive(
                mode,
                linear,
                fit,
                recipe.get("engine_controls", {}).get(mode),
                small,
                analysis,
                self.processor,
                self.export,
                profile=recipe.get("flex_profile") if mode == "flex-endpoint" else None,
                expected_icc=icc,
            )
        controls = recipe.get("output_controls", {}).get(mode)
        from .gpu_positive_preview import preview_output_codes

        # Engine-agnostic positive curve must precede output controls, matching
        # the save() order in workflow_render; without this the sidebar edits
        # validate and cache correctly but never reach preview pixels.
        from .common_positive import (
            apply_common_positive,
            checked_common_positive_controls,
        )

        common = checked_common_positive_controls(
            recipe.get("common_positive_controls")
        )
        if common:
            pixels = apply_common_positive(pixels, common)
        pixels, acceleration = preview_output_codes(
            pixels, controls, recipe["exposure_ev"]
        )
        event("preview_positive_backend", **acceleration)
        if (
            mode == "cool-shared-rgb"
            and not recipe.get("nikon_research_controls")
            and recipe["cool_tone"] == "soft-shoulder"
        ):
            pixels = lift_codes(pixels, soft=True)
        pixels, _ = post_codes(
            pixels,
            recipe.get("post_controls", {}).get(mode),
            sample_step=1 if full_grid_preview else self.sample_step,
            collect_ranges=False,
        )
        if self.full_resolution:
            remember_completed(self, recipe, mode, pixels, icc)
        if full_grid_preview:
            pixels = sample_oriented(
                pixels, None, recipe.get("output_orientation"), self.sample_step
            )
        elif recipe.get("output_orientation"):
            pixels = orient_codes(pixels, recipe["output_orientation"])
        return pixels, reference, icc


def _write_preview_records(directory, token, pixels, reference, icc):
    """Write both protocol paths, reusing the encoding for identical images."""
    records = []
    paths = []
    same_image = pixels is reference or (
        pixels.nbytes <= 16 * 1024**2
        and pixels.shape == reference.shape
        and pixels.dtype == reference.dtype
        and np.array_equal(pixels[::64, ::64], reference[::64, ::64])
        and np.array_equal(pixels, reference)
    )
    try:
        for suffix, image in (("current", pixels), ("reference", reference)):
            path = Path(directory) / f"{token}-{suffix}.tiff"
            paths.append(path)
            copied = False
            if suffix == "reference" and same_image:
                try:
                    shutil.copyfile(records[0]["path"], path)
                    copied = True
                except OSError:
                    # Preserve the independent-encoding path when copying fails.
                    pass
            if copied:
                digest = records[0]["sha256"]
            else:
                tifffile.imwrite(
                    path,
                    image,
                    photometric="rgb",
                    metadata=None,
                    # Screen-sized previews are short-lived. Avoid compression
                    # latency while keeping large, full-resolution views bounded.
                    compression=None if image.nbytes <= 16 * 1024**2 else "deflate",
                    maxworkers=max(1, int(os.environ.get("PURE_RGB_IO_THREADS", "4"))),
                    extratags=[(34675, 7, len(icc), icc, False)],
                )
                digest = sha256(path)
            records.append({"path": str(path), "sha256": digest})
    except BaseException:
        for path in paths:
            path.unlink(missing_ok=True)
        raise
    return records


def main():
    global _REQUEST_STATE
    from .preview_requests_v44 import IncomingRequests, PreviewSuperseded, preview_metering_path

    setup("preview")
    if os.name == "posix" and os.getpgrp() != os.getpid():
        os.setsid()
    session = None
    retained_density = None
    fit_reference = FitReferenceCache()
    incoming = _REQUEST_STATE = IncomingRequests(sys.stdin)
    for line in incoming:
        worker_received_ns = time.monotonic_ns()
        request = None
        records = []
        metering_path = None
        delivered = False
        try:
            request = json.loads(line)
            if isinstance(request, dict) and request.get("action") == "clear_derived":
                fit_reference.clear()
                cache = getattr(session, "density_preview", None)
                if cache is not None:
                    cache.clear_derived()
                continue
            _checked_preview_request(request)
            incoming.begin(request)
            event(
                "preview_start",
                token=request.get("token"),
                mode=request.get("mode"),
                full_resolution=request.get("full_resolution", False),
            )
            started = time.monotonic()
            from .draft_preview import DraftPreview

            session_type = DraftPreview if "draft_recipe" in request else Session
            # Compare the resolved manifest identity: project runs are normally
            # symlinked, so an unresolved request path would recreate the session
            # (and drop its transient anchor) on every preview.
            manifest_path = Path(request["manifest"]).resolve()
            reused = (
                session is not None
                and type(session) is session_type
                and session.path == manifest_path
                and session.digest == request["manifest_sha256"]
            )
            if not reused:
                # Release the old RAW and native caches before decoding another photo.
                fit_reference.clear()
                verified_inputs = getattr(session, "authenticated_inputs", None)
                if getattr(session, "density_preview", None) is not None:
                    retained_density = session.density_preview
                    session.density_preview = None
                if session is not None and hasattr(session, "close"):
                    session.close()
                session = None
                session = session_type(
                    request["manifest"],
                    request["manifest_sha256"],
                    request["directory"],
                    **({"authenticated_inputs": verified_inputs} if "draft_recipe" in request and verified_inputs else {}),
                )
                if retained_density is not None:
                    if hasattr(session, "density_preview") and retained_density.rebind(session):
                        session.density_preview = retained_density
                    else:
                        retained_density.close()
                    retained_density = None
            # A slider gesture can dispatch many tiny frames per second. Skip
            # starting and joining a resource polling thread on each frame.
            measure_resources = "interactive_edge" not in request or os.environ.get("PURE_RGB_DEBUG") == "1"
            sampler = MemorySampler() if measure_resources else nullcontext()
            with (
                sampler as resources,
                span("preview_render", token=request["token"], mode=request["mode"]),
            ):
                pixels, reference, icc = session.render(request)
                repair_provenance = None
                if request.get("repair_plan") is not None:
                    from .repair_pipeline import preview_repair

                    with span("confirmed_repair_preview", token=request["token"]):
                        pixels, reference, repair_provenance = preview_repair(
                            request, session, pixels, reference, icc,
                            on_progress=lambda progress: emit(**progress),
                        )
            incoming.check()
            event(
                "preview_rendered",
                token=request["token"],
                seconds=time.monotonic() - started,
                shape=list(pixels.shape),
                **({"resources": resources.report()} if resources is not None else {}),
            )
            quality = getattr(session, "preview_quality", None)
            interactive = "interactive_edge" in request and (quality or {}).get("interactive_frame") is True
            reference_deferred = request.get("defer_reference") is True and getattr(session, "reference_deferred", False)
            current_only = interactive or reference_deferred
            if current_only:
                reference = pixels
            if interactive and max(pixels.shape[:2]) > request["interactive_edge"]:
                raise ValueError("拖动预览超出请求的采样尺寸")
            if can_transport_fit(request, pixels, reference, icc):
                try:
                    records = write_fit_records(
                        request["directory"], request["token"], pixels, reference, icc,
                        current_only=current_only, reference_cache=fit_reference,
                    )
                except OSError as error:
                    if interactive:
                        raise
                    event("preview_fit_transport_fallback", token=request["token"], reason=str(error))
            if not records:
                if interactive:
                    raise ValueError("拖动预览未生成可交接的 RGB16 像素")
                records = _write_preview_records(
                    request["directory"], request["token"], pixels, reference, icc
                )
            if repair_provenance is not None:
                records[0].update(outside_mask_changed_pixels=0, repair_replay=repair_provenance)
                quality = {
                    **(quality or {}),
                    "approximate": bool((quality or {}).get("approximate"))
                        or repair_provenance["quality"] != "full-resolution",
                    "label": "已保留确认除尘的预览；完整导出按原像素修补",
                    "repair_replay": repair_provenance,
                }
            metering = getattr(session, "metering_receipt", None)
            desktop = request.get("metering_transport") == "sidecar"
            metering_fields = {"metering_receipt": metering}
            if interactive:
                metering_fields = {}
            elif desktop:
                metering_fields = {}
                if metering and not request.get("viewport"):
                    metering_path = preview_metering_path(records[0])
                    metering_path.write_text(
                        json.dumps(metering, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8",
                    )
                    metering_fields["metering_sidecar"] = sha256(metering_path)
                    event("preview_metering_sidecar", token=request["token"],
                          bytes=metering_path.stat().st_size, zones=len(metering.get("zones", ())))
            incoming.check()
            visible_binding = None
            if (desktop and not interactive and not (quality or {}).get("approximate")
                    and hasattr(session, "visible_engine_binding")):
                visible_binding = session.visible_engine_binding(request)
            emit(
                type="result",
                token=request["token"],
                records=records,
                **({"interactive_frame": True} if interactive else {}),
                **({"reference_deferred": True} if reference_deferred else {}),
                **({"visible_engine_binding": visible_binding} if visible_binding else {}),
                seconds=time.monotonic() - started,
                preview_long_edge=max(pixels.shape[:2]),
                sample_step=session.sample_step,
                output_size=getattr(session, "output_size", None),
                **(
                    dict(viewport=session.viewport_result,
                         detail_id=request.get("detail_id"))
                    if getattr(session, "viewport_result", None) is not None else {}
                ),
                reference_label=getattr(session, "reference_label", "已保存版本"),
                preview_quality=quality,
                native_print=getattr(session, "native_print", None),
                **metering_fields,
                # A retained anchor or original Native source makes the next
                # edit cheaper, so the UI can shorten its debounce.
                preview_reusable=bool(
                    getattr(session, "anchor", None)
                    or getattr(session, "native_cache", None)
                    or (quality or {}).get("mode") in {
                        "native-original-screen", "nikon-original-screen", "flex-original-screen",
                        "nikon-screen", "flex-screen",
                    }
                ),
                preview_performance=(
                    getattr(session.density_preview, "last_receipt", None)
                    if not desktop and getattr(session, "density_preview", None) is not None else None
                ),
                preview_cache=(session.density_preview.cache_report
                    if getattr(session, "density_preview", None) is not None else None),
                worker_received_ns=worker_received_ns,
                worker_finished_ns=time.monotonic_ns(),
            )
            delivered = True
        except PreviewSuperseded as error:
            event("preview_superseded", token=request.get("token"), message=str(error))
            emit(type="error", token=request.get("token"), message=str(error), superseded=True)
        except Exception as error:
            token = request.get("token") if isinstance(request, Mapping) else None
            failure(
                "preview_error",
                error,
                token=token,
            )
            emit(
                type="error",
                token=token,
                message=str(error),
            )
        finally:
            if not delivered:
                for record in records:
                    Path(record["path"]).unlink(missing_ok=True)
                if metering_path is not None:
                    metering_path.unlink(missing_ok=True)
            incoming.finish()

    if session is not None and hasattr(session, "close"):
        session.close()
    if retained_density is not None:
        retained_density.close()
    fit_reference.clear()
    _REQUEST_STATE = None


if __name__ == "__main__":
    main()
