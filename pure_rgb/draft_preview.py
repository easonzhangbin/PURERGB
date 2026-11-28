"""Transient automatic previews, separate from immutable workspace versions.

Existing frozen sessions remain the fast path.  Changes to analysis inputs
produce one private, full-grid color anchor for the selected engine.  Crop,
positive controls, finishing and output direction then reuse that anchor.
No workspace, saved manifest, source RAW or export is modified here.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from .native import sha256
from .output_orientation import orient_reference, sample_oriented
from .output_orientation_cache import _color_recipe, cache_budget_bytes
from .preview_data import verified_pixels
from .workspace import checked_recipe, output_rectangle, source_recipe_identity


_FINISH_KEYS = frozenset(
    (
        "crop",
        "output_roi",
        "exposure_ev",
        "output_controls",
        "post_controls",
        "output_orientation",
        "cool_tone",
        "common_positive_controls",
    )
)
def _canonical(document):
    return json.dumps(document, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _identity(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _saved_measurement_valid(manifest):
    """Shared metering TIFFs/windows require the complete frozen receipt.

    A recipe label alone cannot prove that an older Native TIFF or rectangular
    density window contains a new mask. The save pipeline binds its effective
    window, strict measurement document and spatial snapshot together.
    """
    document = manifest.get("recipe", {}).get("density_metering")
    if not document:
        return True
    from .shared_density_metering_v43 import validate_saved_shared_measurement

    try:
        validate_saved_shared_measurement(manifest)
    except (ValueError, TypeError, KeyError, OverflowError):
        return False
    return True


def uses_direct_highlight_recovery(recipe):
    """Spatial and monochrome rendering do not use the darker alternate."""
    from .native_headroom import native_local_highlight

    return (not recipe.get("source_mode") and not recipe.get("native_spatial_controls")
            and bool((native_local_highlight(recipe) or {}).get("reference_density_offset")))


def saved_highlight_recovery_current(manifest):
    from .highlight_reference_guarded import RECOVERY_POLICY

    direct = manifest.get("highlight_processing")
    processing = direct.get("highlight_processing") if isinstance(direct, dict) else None
    return (isinstance(processing, dict)
            and processing.get("reference_recovery_policy") == RECOVERY_POLICY)


def saved_print_exposure_current(manifest):
    from .native_headroom import PRINT_EXPOSURE_POLICY
    from .professional_controls import native_print_ev

    ev = native_print_ev(manifest.get("recipe") or {})
    processing = manifest.get("native_print_exposure_processing")
    return not ev or (isinstance(processing, dict)
                      and processing.get("policy") == PRINT_EXPOSURE_POLICY
                      and type(processing.get("ev")) in (int, float)
                      and processing["ev"] == ev)


def _saved_color_valid(manifest, mode):
    if mode == "native":
        if uses_direct_highlight_recovery(manifest.get("recipe") or {}) and not saved_highlight_recovery_current(manifest):
            return False
        if not saved_print_exposure_current(manifest):
            return False
    return _saved_measurement_valid(manifest)


def _anchor_grid_valid(anchor, pixels, icc):
    record, manifest = anchor["record"], anchor["manifest"]
    if (pixels.dtype != np.uint16 or list(pixels.shape) != manifest["source_shape"]
            or hashlib.sha256(icc).hexdigest() != manifest["icc_sha256"]
            or (record.get("shape") is not None and list(pixels.shape) != list(record["shape"]))
            or (record.get("icc_sha256") is not None
                and hashlib.sha256(icc).hexdigest() != record["icc_sha256"])):
        raise ValueError("草稿锚点的完整网格或工作ICC发生变化")


def _reauthenticate_owned_anchor(anchor):
    """Recheck bytes after APFS updates only a transient TIFF's ctime.

    Source and saved-result guards remain strict. Each attempt authenticates
    the complete TIFF and its decoded grid between two identical stat values;
    a changed inode, size or mtime never enters this recovery route.
    """
    expected, record = anchor["identity"], anchor["record"]
    if anchor.get("owned") is not True:
        raise ValueError("已保存草稿锚点发生变化")
    for _ in range(3):
        before = _identity(record["path"])
        if before[:4] != expected[:4]:
            raise ValueError("临时草稿锚点发生变化")
        pixels, icc = verified_pixels(record)
        _anchor_grid_valid(anchor, pixels, icc)
        after = _identity(record["path"])
        if after[:4] != expected[:4]:
            raise ValueError("临时草稿锚点认证期间发生变化")
        if before == after:
            anchor.pop("pixels", None)
            if pixels.nbytes <= cache_budget_bytes():
                pixels = np.frombuffer(pixels.tobytes(), dtype=np.uint16).reshape(pixels.shape)
                anchor["pixels"] = pixels
            anchor["identity"], anchor["icc"] = after, icc
            return pixels, icc
    raise ValueError("临时草稿锚点认证期间持续发生变化")


def _verified_anchor_pixels(anchor, *, check_only=False):
    """Read/check one anchor, with bounded owned-ctime reauthentication."""
    expected, record = anchor["identity"], anchor["record"]
    before = _identity(record["path"])
    if before != expected:
        if before[:4] != expected[:4] or anchor.get("owned") is not True:
            raise ValueError("临时草稿锚点发生变化")
        return _reauthenticate_owned_anchor(anchor)
    if check_only:
        return None, anchor["icc"]
    pixels = anchor.get("pixels")
    if pixels is not None:
        return pixels, anchor["icc"]
    pixels, icc = verified_pixels(record)
    _anchor_grid_valid(anchor, pixels, icc)
    after = _identity(record["path"])
    if after != expected:
        if after[:4] != expected[:4] or anchor.get("owned") is not True:
            raise ValueError("草稿锚点读取期间发生变化")
        return _reauthenticate_owned_anchor(anchor)
    return pixels, icc


def _controls(recipe, mode):
    from .live_worker import EDITING_DOCUMENT_KEYS

    relevant = _color_recipe(recipe, mode)
    result = {
        key: deepcopy(relevant[key]) for key in EDITING_DOCUMENT_KEYS if key in relevant
    }
    if "output_orientation" in recipe:
        result["output_orientation"] = deepcopy(recipe["output_orientation"])
    result["exposure_ev"] = recipe["exposure_ev"]
    if recipe.get("source_mode") != "bw-single-v1":
        result["cool_tone"] = recipe["cool_tone"]
    return result


def _replay_setup(recipe, mode):
    """Conservative frozen-session compatibility; never invent a snapshot."""
    from .live_worker import PREVIEW_CONTROL_KEYS
    from .native_headroom import native_local_highlight

    relevant = _color_recipe(recipe, mode)
    result = {
        key: value for key, value in relevant.items() if key not in PREVIEW_CONTROL_KEYS
    }
    # Shared measurement belongs to every engine's analysis context. Keep it
    # even if a future editing contract admits it as a preview control.
    if recipe.get("density_metering"):
        result["density_metering"] = deepcopy(recipe["density_metering"])
    spatial = recipe.get("native_spatial_controls") if mode == "native" else None
    if spatial:
        result["spatial_analysis"] = {
            key: deepcopy(spatial.get(key))
            for key in ("metering", "automatic_avoidance")
        }
        # Spatial snapshots bind the input configuration, not only the mask.
        result["spatial_native_config"] = deepcopy(recipe.get("native_controls"))
        result["spatial_film"] = deepcopy(recipe.get("native_film"))
    local = native_local_highlight(recipe) if mode == "native" else None
    if local:
        result["highlight_reference"] = deepcopy(
            (local.get("automatic_reference", False), local["reference_roi"])
        )
    return result


def _anchor_recipe(recipe, mode):
    """Full source output before output-only adjustments and geometry."""
    result = _color_recipe(recipe, mode)
    for key in _FINISH_KEYS:
        result.pop(key, None)
    result.update(crop=False, exposure_ev=0.0)
    if recipe.get("source_mode") != "bw-single-v1":
        result["cool_tone"] = "conservative"
    return checked_recipe(result)


def _geometry_document(recipe, mode):
    values = recipe.get("post_controls", {}).get(mode, {}).get("values", {})
    selected = {
        key: value for key, value in values.items() if key.startswith("geometry.")
    }
    return {"version": 1, "values": selected} if selected else None


def defer_comparison(owner):
    """Mark a draft that skipped only its secondary comparison input."""
    if not getattr(owner, "_defer_reference", False):
        return False
    owner.reference_deferred = True
    owner.reference_label = ""
    return True


class DraftPreview:
    def __init__(self, path, digest, state_directory, *, authenticated_inputs=None):
        self.path = Path(path).resolve()
        self.digest = digest
        self.manifest_identity = _identity(self.path)
        payload = self.path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest or _identity(self.path) != self.manifest_identity:
            raise ValueError("草稿预览来源清单已改变")
        self.manifest = json.loads(payload)
        if self.manifest.get("state") != "complete":
            raise ValueError("草稿预览需要已完成的原始导入")
        from .native_draft_preview_v40 import resolved_saved_recipe

        self.recipe = resolved_saved_recipe(self.manifest)
        self.directory = Path(state_directory).resolve()
        self.anchor = None
        self.saved_session = None
        self.nikon_preview = None
        self.flex_preview = None
        self.sample_step = 1
        self.reference_label = "已保存版本"
        self.metering_receipt = {}
        self.processor = None
        self.output_size = None
        self.viewport_result = None
        self.density_preview = None
        self.identities = {}
        self.authenticated_inputs = {}
        self.records = deepcopy(self.manifest["inputs"])
        expected_roles = (
            ["BW"] if self.recipe.get("source_mode") == "bw-single-v1" else list("RGB")
        )
        if [record["path"] for record in self.records] != self.recipe["raw"] or [
            record["role"] for record in self.records
        ] != expected_roles:
            raise ValueError("草稿来源与已保存输入通道不一致")
        source_records = self.records + (
            [self.manifest["calibration"]] if self.recipe["flatfield"] else []
        )
        previous = authenticated_inputs or {}
        for record in source_records:
            path, digest = record["path"], record["sha256"]
            before = _identity(path)
            if (
                (previous.get((path, digest)) != before and sha256(path) != digest)
                or _identity(path) != before
            ):
                raise ValueError("草稿预览的原始来源发生变化")
            self.identities[path] = before
            self.authenticated_inputs[path, digest] = before

    def visible_engine_binding(self, request):
        recipe = request["draft_recipe"]
        if (request["mode"] != "native" or request.get("full_resolution") or request.get("viewport")
                or request.get("crop_locator") or request.get("repair_plan") is not None
                or any(recipe.get(key) for key in ("lens_profile", "native_film", "source_mode"))):
            return None
        return {"manifest": str(self.path), "identity": self.manifest_identity,
                "sources": self.identities}

    def _validate(self, request):
        from .live_worker import _checked_preview_request

        _, mode, full = _checked_preview_request(request)
        recipe = checked_recipe(request["draft_recipe"])
        if mode == "cool-shared-rgb" and recipe.get("nikon_research_controls"):
            from .nikon_research_workflow import checked_nikon_resources

            checked_nikon_resources()
        if sha256(self.path) != self.digest:
            raise ValueError("草稿预览来源清单已改变")
        if source_recipe_identity(recipe) != source_recipe_identity(self.recipe):
            raise ValueError("草稿不可替换已认证的 RAW、导入或平场来源；请重新导入")
        if request.get("draft_inputs", self.records) != self.records:
            raise ValueError("草稿输入记录与已认证清单不一致")
        if request.get(
            "draft_calibration", self.manifest.get("calibration")
        ) != self.manifest.get("calibration"):
            raise ValueError("草稿平场记录与已认证清单不一致")
        for path, identity in self.identities.items():
            if _identity(path) != identity:
                raise ValueError("原始来源状态发生变化，请重新导入")
        if recipe.get("source_mode") == "bw-single-v1" and mode != "native":
            raise ValueError("黑白草稿仅支持原生预览")
        return recipe, mode, full

    def render(self, request):
        from .diagnostics import span
        from .viewport_preview_v43 import (
            output_size_for_recipe, render_viewport, sample_full_viewport,
        )

        recipe, mode, _ = self._validate(request)
        self.preview_quality = None
        self.native_print = None
        self.metering_receipt = {}
        self._defer_reference = request.get("defer_reference") is True
        self.reference_deferred = False
        if self.density_preview is not None:
            self.density_preview.last_receipt = None
        self.viewport_result = None
        self.output_size = output_size_for_recipe(
            recipe, self.manifest["source_shape"], self.manifest["roi"], mode,
        )
        # A certified saved color frame is already final: do not decode RAW
        # just to show an unchanged crop or detail from that frame.
        if "viewport" in request:
            saved = render_viewport(self, request, recipe, mode)
            if saved is not None:
                self._validate(request)
                return saved
        elif (not request.get("full_resolution")
              and (recipe.get("density_metering") or recipe.get("native_spatial_controls"))):
            saved = self._saved_crop(recipe, mode, False)
            if saved is not None:
                self.metering_receipt = self._metering_receipt(self.manifest, mode)
                self._validate(request)
                return saved
        from .density_preview_v44 import DensityPreviewCache

        if self.density_preview is None:
            self.density_preview = DensityPreviewCache(self)
        from .interactive_post_preview_v45 import render_interactive_post

        interactive = render_interactive_post(self, request, recipe, mode)
        if interactive is not None:
            return interactive
        if recipe.get("measurement_method") == "negpy":
            from .native_original_preview import try_original_density_screen

            native_sample = try_original_density_screen(self, request, recipe, mode)
            if native_sample is not None:
                if self.anchor is not None:
                    self.anchor.pop("pixels", None)
                return native_sample
        with span("preview_sampled_density", mode=mode):
            sampled = self.density_preview.try_render(request, recipe, mode)
        if sampled is not None:
            if mode != "native" and "viewport" not in request:
                return self._alternate_fit_frame(request, mode, *sampled)
            return sampled
        if "viewport" in request:
            from .viewport_preview_v43 import checked_viewport

            viewport = request["viewport"]
            checked_viewport(viewport, viewport.get("output_size") if isinstance(viewport, dict) else None)
            result = render_viewport(self, request, recipe, mode)
            if result is None:
                # Spatial/geometry operations keep their full-grid contract.
                full_request = {key: value for key, value in request.items() if key != "viewport"}
                full_request["full_resolution"] = True
                result, viewport = sample_full_viewport(self._render_frame(full_request), request["viewport"])
                self.viewport_result = viewport
                self.output_size = viewport["output_size"]
                self.sample_step = viewport["sample_step"]
            self._validate(request)
            return result
        result = self._render_frame(request)
        if request.get("full_resolution", False):
            self.output_size = [result[0].shape[1], result[0].shape[0]]
        return result

    def _alternate_fit_frame(self, request, mode, current, reference, icc):
        self.preview_quality = dict(
            mode="nikon-screen" if mode == "cool-shared-rgb" else "flex-screen",
            approximate=False,
        )
        edge = request.get("interactive_edge")
        if edge and request.get("repair_plan") is None and max(current.shape[:2]) <= edge:
            label = "引擎快速预览" if request.get("engine_proxy") else "拖动预览"
            self.preview_quality.update(approximate=edge < 1400, interactive_frame=True, label=label)
            return current, current, icc
        return current, reference, icc

    def _render_frame(self, request):
        from .live_worker import Session

        recipe, mode, full = self._validate(request)
        if mode == "cool-shared-rgb" and recipe.get("nikon_research_controls"):
            if self.nikon_preview is None:
                from .nikon_draft_preview_v40 import NikonDraftPreview

                self.nikon_preview = NikonDraftPreview(self.manifest, self.digest)
            result = self.nikon_preview.render(recipe, full=full, interactive_edge=request.get("interactive_edge"))
            if result is not None:
                self.sample_step = result.sample_step
                self.reference_label = result.reference_label
                self.metering_receipt = {}
                self._validate(request)
                if not full:
                    return self._alternate_fit_frame(request, mode, result.pixels, result.reference, result.icc)
                return result.pixels, result.reference, result.icc
        if mode == "flex-endpoint" and recipe.get("flex_research_controls"):
            if self.flex_preview is None:
                from .flex_draft_preview_v40 import FlexDraftPreview

                self.flex_preview = FlexDraftPreview(self)
            result = self.flex_preview.try_render(recipe, full=full)
            if result is not None:
                pixels, reference, icc, step, receipt = result
                self.sample_step = step
                self.reference_label = receipt["preview"]["reference_label"]
                self.metering_receipt = {}
                self._validate(request)
                if not full:
                    return self._alternate_fit_frame(request, mode, pixels, reference, icc)
                return pixels, reference, icc
        saved_crop = self._saved_crop(recipe, mode, full)
        if saved_crop is not None:
            self.metering_receipt = self._metering_receipt(self.manifest, mode)
            self._validate(request)
            return saved_crop
        anchor_recipe = _anchor_recipe(recipe, mode)
        saved_anchor = self._saved_neutral_eligible(anchor_recipe, mode)
        if (
            not saved_anchor
            and mode in self.manifest["results"]
            and _saved_color_valid(self.manifest, mode)
            and _replay_setup(recipe, mode) == _replay_setup(self.recipe, mode)
            and not (recipe.get("source_mode") == "bw-single-v1"
                     and recipe.get("post_controls") != self.recipe.get("post_controls"))
        ):
            if self.anchor is not None:
                self.anchor.pop("pixels", None)
            if self.saved_session is None:
                self.saved_session = Session(self.path, self.digest, self.directory)
            self.reference_label = "已保存版本"
            self.metering_receipt = self._metering_receipt(self.manifest, mode)
            result = self.saved_session.render(
                {**request, "controls": _controls(recipe, mode)}
            )
            self.sample_step = self.saved_session.sample_step
            self.processor = self.saved_session.processor
            self._validate(request)
            return result
        key = hashlib.sha256(_canonical([mode, anchor_recipe]).encode()).hexdigest()
        from .diagnostics import event as _audit_event

        _audit_event(
            "draft_anchor_probe",
            session=id(self),
            key=key[:12],
            anchor_key=(self.anchor or {}).get("key", "-")[:12],
            anchor_owned=(self.anchor or {}).get("owned"),
        )
        warm_anchor = self.anchor is not None and self.anchor["key"] == key
        if not warm_anchor:
            if saved_anchor:
                self._use_saved_anchor(mode, key)
            else:
                self._build_anchor(anchor_recipe, mode, key)
        anchor = self.anchor
        self.metering_receipt = self._metering_receipt(anchor["manifest"], mode)
        pixels, icc = _verified_anchor_pixels(anchor)
        rectangle = output_rectangle(recipe, anchor["manifest"]["roi"], pixels.shape)
        x0, y0, x1, y1 = rectangle
        before = pixels[y0:y1, x0:x1]
        from .preview_grid import post_requires_full_grid

        direction = recipe.get("output_orientation")
        point_sample = not full and not post_requires_full_grid(
            recipe.get("post_controls", {}).get(mode)
        )
        if point_sample:
            # Point operations commute with this exact source-coordinate sample.
            # Avoid adjusting a full 96MP frame only to display ~1MP of it.
            native_tone = mode == "native" and request.get("tone_edit") is True
            fit_edge = request.get("fit_edge", 1400) if native_tone else 1400
            longest = max(before.shape[:2])
            step = max(1, (longest + fit_edge - 1) // fit_edge)
            edge = request.get("interactive_edge")
            drag_ready = bool(edge and warm_anchor and anchor.get("pixels") is not None)
            factor = (longest + edge * step - 1) // (edge * step) if drag_ready else 1
            step *= factor
            self.sample_step = step
            current = self._finish(
                np.ascontiguousarray(sample_oriented(before, None, direction, step)),
                recipe, mode, fit=True,
            )
            if factor > 1 or (native_tone and drag_ready):
                self.reference_label = ""
                self.preview_quality = dict(mode="draft-anchor-screen", approximate=True,
                    interactive_frame=True, label="拖动预览", sample_step=step, full_output_size=self.output_size)
                self._validate(request)
                _verified_anchor_pixels(anchor, check_only=True)
                return current, current, icc
            if defer_comparison(self):
                reference = current
            else:
                reference = self._reference(before, rectangle, recipe, mode, icc)
                reference = sample_oriented(reference, None, direction, step)
            self._validate(request)
            _verified_anchor_pixels(anchor, check_only=True)
            return np.array(current, copy=True), np.array(reference, copy=True), icc
        current = self._finish(before, recipe, mode)
        reference = self._reference(before, rectangle, recipe, mode, icc)
        if current.shape != reference.shape:
            raise ValueError("草稿与调整前预览几何不一致")
        step = 1 if full else max(1, int(np.ceil(max(current.shape[:2]) / 1400)))
        self.sample_step = step
        # Return owned samples, so callers cannot mutate a retained anchor.
        result = tuple(
            np.array(sample_oriented(image, None, direction, step), copy=True)
            for image in (current, reference)
        )
        self._validate(request)
        _verified_anchor_pixels(anchor, check_only=True)
        return result[0], result[1], icc

    def _saved_neutral_eligible(self, anchor_recipe, mode):
        """A full saved output with identity finishing is already the anchor."""
        if mode not in self.manifest["results"] or not _saved_color_valid(self.manifest, mode):
            return False
        if (
            self.recipe["exposure_ev"] != 0
            or self.recipe.get("output_controls", {}).get(mode)
            or self.recipe.get("post_controls", {}).get(mode)
            or self.recipe.get("output_orientation")
        ):
            return False
        if (
            mode == "cool-shared-rgb"
            and not self.recipe.get("nikon_research_controls")
            and self.recipe["cool_tone"] != "conservative"
        ):
            return False
        shape = self.manifest["source_shape"]
        full = [0, 0, shape[1], shape[0]]
        return output_rectangle(
            self.recipe, self.manifest["roi"], shape
        ) == full and anchor_recipe == _anchor_recipe(self.recipe, mode)

    def _use_saved_anchor(self, mode, key):
        from .engines import check_reference_sources
        from .native import check_baseline

        check_baseline()
        check_reference_sources()
        self._release_saved_session()
        if self.anchor is not None:
            self.anchor.pop("pixels", None)
        record = deepcopy(self.manifest["results"][mode])
        identity = _identity(record["path"])
        pixels, icc = verified_pixels(record)
        if (
            list(pixels.shape) != self.manifest["source_shape"]
            or hashlib.sha256(icc).hexdigest() != self.manifest["icc_sha256"]
            or _identity(record["path"]) != identity
        ):
            raise ValueError("已保存成片不符合完整源网格／ICC锚点契约")
        old = self.anchor
        self.anchor = dict(
            key=key,
            directory=None,
            owned=False,
            manifest=self.manifest,
            record=record,
            identity=identity,
            icc=icc,
        )
        if pixels.nbytes <= cache_budget_bytes():
            self.anchor["pixels"] = np.frombuffer(
                pixels.tobytes(), dtype=np.uint16
            ).reshape(pixels.shape)
        if old is not None and old.get("owned", True):
            shutil.rmtree(old["directory"], ignore_errors=True)

    def _saved_crop(self, recipe, mode, full):
        """Crop-only edits inside a saved canvas need no RAW/color preparation."""
        from .preview_grid import post_requires_full_grid

        if mode not in self.manifest["results"] or not _saved_color_valid(self.manifest, mode):
            return None
        old = _color_recipe(self.recipe, mode)
        new = _color_recipe(recipe, mode)
        for document in (old, new):
            document.pop("crop", None)
            document.pop("output_roi", None)
        if old != new or post_requires_full_grid(
            recipe.get("post_controls", {}).get(mode)
        ):
            return None
        shape, roi = self.manifest["source_shape"], self.manifest["roi"]
        previous = output_rectangle(self.recipe, roi, shape)
        rectangle = output_rectangle(recipe, roi, shape)
        if rectangle == previous:
            return None
        a, b, c, d = previous
        x0, y0, x1, y1 = rectangle
        if not (a <= x0 < x1 <= c and b <= y0 < y1 <= d):
            return None
        pixels, icc = verified_pixels(self.manifest["results"][mode])
        pixels = orient_reference(pixels, self.recipe.get("output_orientation"), None)
        if pixels.shape[:2] != (d - b, c - a):
            raise ValueError("已保存裁切画布与冻结来源不一致")
        view = pixels[y0 - b : y1 - b, x0 - a : x1 - a]
        self.sample_step = (
            1 if full else max(1, int(np.ceil(max(view.shape[:2]) / 1400)))
        )
        current = sample_oriented(
            view, None, recipe.get("output_orientation"), self.sample_step
        )
        self.reference_label = "已保存版本"
        return np.array(current, copy=True), np.array(current, copy=True), icc

    @staticmethod
    def _metering_receipt(manifest, mode):
        from .metering_hover_v45 import metering_zone_records

        def _zone_receipt(processing):
            records = metering_zone_records(processing.get("metering_window") or {})
            return {"zones": records} if records else {}

        shared = manifest["recipe"].get("density_metering")
        if shared:
            processing = manifest.get("shared_measurement_processing", {})
            return deepcopy(dict(
                roi=manifest["roi"], metering=shared.get("metering"),
                automatic=shared.get("automatic_avoidance"),
                report=processing.get("automatic_avoidance") or {},
                **_zone_receipt(processing),
            ))
        if mode != "native":
            return {}
        spatial = manifest["recipe"].get("native_spatial_controls", {})
        processing = manifest.get("native_spatial_processing", {})
        reference_roi = processing.get("highlight_reference_roi")
        return deepcopy(dict(
            roi=manifest["roi"], metering=spatial.get("metering"),
            automatic=spatial.get("automatic_avoidance"),
            report=processing.get("automatic_avoidance") or {},
            **_zone_receipt(processing),
            **({"highlight_reference_roi": reference_roi} if reference_roi else {}),
        ))

    def _release_saved_session(self):
        session, self.saved_session = self.saved_session, None
        self.processor = None
        if session is not None and session.processor is not None:
            session.processor.cleanup()
            session.processor.destroy_all()

    def close(self):
        if self.density_preview is not None:
            self.density_preview.close()
            self.density_preview = None
        for preview in (self.nikon_preview, self.flex_preview):
            if preview is not None:
                preview.close()
        self.nikon_preview = self.flex_preview = None
        self._release_saved_session()
        if self.anchor is not None:
            self.anchor.pop("pixels", None)
            if self.anchor.get("owned", True):
                shutil.rmtree(self.anchor["directory"], ignore_errors=True)
            self.anchor = None

    def _build_anchor(self, recipe, mode, key):
        from .workflow_render import render
        from .live_worker import emit

        self._release_saved_session()
        if self.anchor is not None:
            self.anchor.pop("pixels", None)
        emit(type="progress", message="正在更新草稿密度与正像；当前画面保持可见…")
        path = self.directory / "draft-analysis" / uuid.uuid4().hex

        def progress(**data):
            if data.get("type") == "progress":
                emit(type="progress", message=data["message"])

        try:
            prepared = None
            density = self.density_preview
            if (density is not None and density.certificate is not None
                    and not recipe.get("flatfield")):
                density.certificate.validate()
                prepared = (density.session.processor, density.pristine_config,
                            density.session.source, deepcopy(self.records))
            request = {
                "recipe": recipe,
                "inputs": deepcopy(self.records),
                "calibration": deepcopy(self.manifest.get("calibration")),
                "directory": str(path),
                "only_mode": mode,
                "transient_preview": True,
            }
            if prepared is None:
                # _validate already bound the draft's decode identity and RAW
                # records to this completed manifest. The render cache also
                # checks the frozen import receipt before restoring pixels.
                from .decoded_cache import source_anchor

                request["decode_anchor"] = deepcopy(source_anchor(self.manifest))
            report = render(
                request,
                emit_callback=progress,
                **({"prepared_source": prepared} if prepared is not None else {}),
            )
            record = report["results"][mode]
            identity = _identity(record["path"])
            pixels, icc = verified_pixels(record)
            if (
                list(pixels.shape) != report["source_shape"]
                or _identity(record["path"]) != identity
            ):
                raise ValueError("草稿锚点必须保留完整源网格")
        except BaseException as error:
            from .workspace import write_json

            try:
                write_json(
                    self.directory / "draft-failures" / (path.name + ".json"),
                    dict(
                        state="failed",
                        mode=mode,
                        anchor_key=key,
                        source_manifest_sha256=self.digest,
                        error_type=type(error).__name__,
                        error=str(error)[:2000],
                    ),
                )
            except OSError:
                pass
            finally:
                shutil.rmtree(path, ignore_errors=True)
            raise
        old = self.anchor
        self.anchor = dict(
            key=key,
            directory=path,
            owned=True,
            manifest=report,
            record=record,
            identity=identity,
            icc=icc,
        )
        if pixels.nbytes <= cache_budget_bytes():
            # Bytes backing prevents accidental writeability changes through views.
            self.anchor["pixels"] = np.frombuffer(
                pixels.tobytes(), dtype=np.uint16
            ).reshape(pixels.shape)
        if old is not None and old.get("owned", True):
            shutil.rmtree(old["directory"], ignore_errors=True)

    @staticmethod
    def _finish(pixels, recipe, mode, *, fit=False):
        from .common_positive import (
            apply_common_positive,
            checked_common_positive_controls,
        )
        from .cool_whitepoint_study import lift_codes
        from .gpu_positive_preview import preview_output_codes
        from .positive_post import post_codes

        # Same order as workflow_render.save(): common positive curve first,
        # then output controls, cool tone, then post controls.
        common = checked_common_positive_controls(
            recipe.get("common_positive_controls")
        )
        if common:
            pixels = apply_common_positive(pixels, common)
        output = recipe.get("output_controls", {}).get(mode)
        post = recipe.get("post_controls", {}).get(mode)
        cool_lift = (
            mode == "cool-shared-rgb"
            and not recipe.get("nikon_research_controls")
            and recipe["cool_tone"] == "soft-shoulder"
        )
        if (fit and not cool_lift
                and os.environ.get("PURE_RGB_COMPUTE_BACKEND", "auto") in {"auto", "cpu"}):
            from .positive_lut import fit_point_codes

            adjusted = fit_point_codes(pixels, output, recipe["exposure_ev"], post)
            if adjusted is not None:
                return adjusted
        pixels, _ = preview_output_codes(
            pixels, output, recipe["exposure_ev"]
        )
        if cool_lift:
            pixels = lift_codes(pixels, soft=True)
        pixels, _ = post_codes(
            pixels, post, sample_step=1, collect_ranges=False,
        )
        return pixels

    def _reference(self, before, rectangle, recipe, mode, icc):
        from .positive_post import post_codes

        self.reference_label = "调整前（当前草稿）"
        reference = before
        record = self.manifest["results"].get(mode)
        saved_geometry = _geometry_document(self.recipe, mode)
        if (
            self.anchor is not None
            and not self.anchor.get("owned", True)
            and self.anchor["record"] == record
        ):
            self.reference_label = "已保存版本"
        elif record is not None and not saved_geometry:
            saved_roi = output_rectangle(
                self.recipe, self.manifest["roi"], self.manifest["source_shape"]
            )
            a, b, c, d = saved_roi
            x0, y0, x1, y1 = rectangle
            if a <= x0 < x1 <= c and b <= y0 < y1 <= d:
                saved, other = verified_pixels(record)
                if other != icc:
                    raise ValueError("草稿参考图 ICC 不一致")
                saved = orient_reference(
                    saved, self.recipe.get("output_orientation"), None
                )
                if saved.shape[:2] != (d - b, c - a):
                    raise ValueError("已保存参考图与源裁切不一致")
                reference = saved[y0 - b : y1 - b, x0 - a : x1 - a]
                self.reference_label = "已保存版本"
        reference, _ = post_codes(
            reference, _geometry_document(recipe, mode), sample_step=1,
            collect_ranges=False,
        )
        return reference
