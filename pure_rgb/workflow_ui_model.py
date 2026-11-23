"""Photo/engine drafts for the staged desktop, using the existing recipe contracts."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import uuid

from .highlight_workflow import (
    align_highlight_reference as align_highlight_reference,
    highlight_support,
    select_highlight_reference,
)
from .preview_data import ROOT
from .output_orientation import checked_document as checked_output_orientation
from .workspace import Workspace, checked_recipe, freeze_render_request, source_recipe_identity, write_json
from .workflow_edit_history import PhotoEditHistory

# Persisted stage IDs remain stable for existing projects and CLI arguments.
STAGES = ("导入归组", "画幅与对齐", "画幅与测量", "转换调色", "修整审核", "导出", "观片台")
LIGHT_TABLE_STAGE = 6
VISIBLE_STAGE_INDICES = (0, 2, 3, 4, LIGHT_TABLE_STAGE, 5)
VISIBLE_STAGE_LABELS = tuple(STAGES[index] for index in VISIBLE_STAGE_INDICES)


def canonical_stage(index):
    """The former frame-only step now opens the combined measurement step."""
    if type(index) is not int or not 0 <= index < len(STAGES):
        raise ValueError("未知工作流程阶段")
    return 2 if index == 1 else index


def stage_presentation(index):
    index = canonical_stage(index)
    return VISIBLE_STAGE_INDICES.index(index) + 1, STAGES[index]


ENGINES = {
    "native": "NegPy · 原生",
    "nikon": "Nikon CMS · 仅 CMS",
    "flex": "FlexColor · 演算",
}
RESULT_MODES = {"native": "native", "nikon": "cool-shared-rgb", "flex": "flex-endpoint"}
STRAIGHTEN_KEYS = ("geometry.fine_rotation", "geometry.crop_to_valid")


def straighten_values(recipe, engine):
    """Read the non-default straightening settings from a checked recipe."""
    values = recipe.get("post_controls", {}).get(RESULT_MODES[engine], {}).get("values", {})
    return {key: values[key] for key in STRAIGHTEN_KEYS if key in values}


def with_straighten(recipe, engine, settings):
    """Replace only the straightening keys of the requested render mode."""
    if straighten_values(recipe, engine) == settings:
        return recipe
    mode = RESULT_MODES[engine]
    post = dict(recipe.get("post_controls", {}))
    document = post.pop(mode, dict(version=1, values={}))
    values = {key: value for key, value in document["values"].items() if key not in STRAIGHTEN_KEYS}
    values.update(settings)
    if values:
        post[mode] = {**document, "values": values}
    result = dict(recipe)
    if post:
        result["post_controls"] = post
    else:
        result.pop("post_controls", None)
    return result


def photo_key(paths, source_mode):
    return hashlib.sha256(
        json.dumps([source_mode, [str(Path(p).resolve()) for p in paths]]).encode()
    ).hexdigest()[:20]


def base_recipe(paths, source_mode):
    raw = [str(Path(p).expanduser().resolve()) for p in paths]
    recipe = dict(
        raw=raw,
        roi=None,
        density="negpy",
        measurement_method="negpy",
        paper="neutral",
        exposure_ev=0,
        crop=False,
        flatfield=None,
    )
    if source_mode == "bw":
        recipe.update(source_mode="bw-single-v1", raw_import="legacy-v1")
    elif source_mode in ("rgb", "mono_rgb"):
        recipe.update(
            raw_import=(
                "standard-mono-v1" if source_mode == "mono_rgb"
                else "standard-cfa-dng-v1" if all(Path(p).suffix.lower() == ".dng" for p in raw)
                else "standard-crop-v1"
            ),
            cool_tone="conservative",
        )
    else:
        raise ValueError("未知输入方式")
    return checked_recipe(recipe)


def nikon_threads():
    """Bounded CMS workers; the pinned transform is thread-invariant."""
    return max(1, min(8, os.cpu_count() or 1))


def common_positive_controls(highlight=0.0, shadow=0.0, tone=0.0):
    from .common_positive import checked_common_positive_controls

    return checked_common_positive_controls(
        dict(version=1, enabled=True, highlight=highlight, shadow=shadow, tone=tone)
    )


def experimental_film_controls(*, flex=None, nikon=None, enabled=True):
    """Build the explicitly-uncalibrated experimental film style document."""
    from .experimental_film import (
        CALIBRATION,
        VERSION,
        checked_experimental_film_controls,
    )

    return checked_experimental_film_controls(
        dict(
            version=VERSION,
            calibration=CALIBRATION,
            enabled=enabled,
            flex=flex,
            nikon=nikon,
        )
    )


def experimental_repair_block_reason(recipe, engine):
    """Return actionable guidance when dust repair cannot use this engine's raster.

    PURE Neg only accepts RGB16 positives and never inverts on its side.
    Explicitly-uncalibrated experimental chains that insert a negative-inversion
    stage do not produce a legal positive source, so the desktop must refuse
    instead of feeding a negative-like raster into the repair contract.
    CMS-only styles without an inversion stage remain valid positives.
    """
    from .experimental_film import checked_experimental_film_controls

    doc = checked_experimental_film_controls(
        (recipe or {}).get("experimental_film_controls")
    )
    if not doc:
        return None
    if engine == "nikon":
        nikon = doc.get("nikon")
        if nikon and nikon["mode"] == "parameterized":
            return (
                "Nikon 引擎当前启用了未标定的「参数化 analyzer/LUT 反演」实验链，"
                "其输出是研究用反演结果、不是正像；PURE Neg 只接收 RGB16 正片且不会再次反相。\n\n"
                "请在第 3 阶段「高级参数 → 实验胶片（Nikon）」把模式切回"
                "「仅 CMS（生产同构）」并应用，再进入除尘。"
            )
    elif engine == "flex" and doc.get("flex"):
        return (
            "Flex 引擎当前启用了未标定实验负片曲线（GammaNeg／通道 Negative），"
            "其输出不是可用于除尘的正像。\n\n"
            "请在第 3 阶段「高级参数 → 实验胶片（Flex）」关闭负片反演开关并应用，再进入除尘。"
        )
    return None


def nikon_controls(strength=0):
    from .nikon_research_workflow import checked_nikon_research_controls
    from .tone_study import ICC_SHA256

    return checked_nikon_research_controls(
        dict(
            version=1,
            enabled=True,
            source="neutral-native-working-positive",
            input_hypothesis="working-rgb-positive-rgb16",
            fixed_strength=strength,
            threads=nikon_threads(),
            output_icc_sha256=ICC_SHA256,
            assumption="Explicit uncalibrated camera-positive adapter; not scanner acquisition or physical calibration",
        )
    )


def flex_controls():
    """Use a verified local Flex recipe if installed, or the neutral curve route."""
    from .flex_research_workflow import checked_flex_research_controls

    path = ROOT / "outputs/flex-color-workflow-v26/plan.json"
    if not path.exists() and not path.is_symlink():
        return None
    if not path.is_file():
        raise ValueError("Flex 已验证资源配方不可用，请先恢复本机资源")
    document = checked_flex_research_controls(
        json.loads(path.read_text())["recipe"]["flex_research_controls"]
    )
    if document is None:
        raise ValueError("本机 Flex 研究配方必须包含启用的资源配置")
    return document


def metering_document(include=(), exclude=(), excluded_rois=()):
    return dict(
        version=1,
        enabled=True,
        zones=[8, 12],
        blocks_per_zone=[2, 2],
        excluded_rois=deepcopy(list(excluded_rois)),
        padding=0,
        min_area_fraction=0.25,
        include=deepcopy(list(include)),
        exclude=deepcopy(list(exclude)),
    )


def spatial_document(meter, action="suggest", protected_rois=(), previous=None):
    from .spatial_workflow_v31 import checked_spatial_controls
    from .defect_reliability_v32 import MODE

    result = deepcopy(previous or dict(version=1, enabled=True, lab={}, local_masks=[]))
    result["metering"] = deepcopy(meter)
    result.pop("automatic_avoidance", None)
    if action != "off":
        if action not in ("suggest", "exclude"):
            raise ValueError("未知异常处理方式")
        result["automatic_avoidance"] = dict(
            version=2,
            enabled=True,
            mode=MODE,
            max_fraction=(previous or {})
            .get("automatic_avoidance", {})
            .get("max_fraction", 0.12),
            protected_rois=deepcopy(list(protected_rois)),
            action=action,
        )
    return checked_spatial_controls(result)


def shape_bounds(shape):
    if shape["type"] == "polygon":
        xs, ys = zip(*shape["points"])
        import math

        return [
            int(math.floor(min(xs))),
            int(math.floor(min(ys))),
            int(math.ceil(max(xs))),
            int(math.ceil(max(ys))),
        ]
    return [int(v) for v in shape["bounds"]]


def suggest_highlight_reference(recipe, source_shape, additional_exclusions=()):
    return select_highlight_reference(
        recipe, highlight_support(recipe, source_shape, additional_exclusions)
    )


def fit_manual_highlight_reference(recipe, source_shape, bounds, receipt):
    """Align a manual reference within the original source image."""
    del receipt  # The density selection does not constrain a manual reference.
    height, width = source_shape[:2]
    a, b, c, d = bounds
    if not (0 <= a < c <= width and 0 <= b < d <= height):
        raise ValueError("高光参考区超出原图范围，请重新绘制")
    if recipe.get("native_spatial_controls") and min(c - a, d - b) < 4:
        raise ValueError("高光参考至少需要 4 × 4 源像素")
    return align_highlight_reference(recipe, bounds)


def validate_highlight_reference(
    recipe, source_shape, additional_exclusions=(), *, require_metering_support=True
):
    a, b, c, d = recipe["highlight_controls"]["reference_roi"]
    height, width = source_shape[:2]
    if not (0 <= a < c <= width and 0 <= b < d <= height) or min(c - a, d - b) < 4:
        raise ValueError("高光参考区超出原图范围或过小，请重新绘制")
    if not require_metering_support:
        return
    allowed = highlight_support(recipe, source_shape, additional_exclusions)
    x0, y0, x1, y1 = recipe["roi"]
    if (
        not (x0 <= a < c <= x1 and y0 <= b < d <= y1)
        or not allowed[b - y0 : d - y0, a - x0 : c - x0].all()
    ):
        raise ValueError("高光参考区跨越已排除的测量支持，请在有效区域内重选")


class WorkflowProject:
    """Photo-wide output geometry, separate engine color drafts and immutable jobs."""

    def __init__(self, directory, *, create=False):
        directory = Path(directory)
        if (directory / "workspace.json").exists() and not (
            directory / "ui-workflow.json"
        ).exists():
            old = json.loads((directory / "workspace.json").read_text())
            if old.get("jobs"):
                raise ValueError(
                    "此入口使用独立流程测试项目；旧工作区请继续使用原工作台，不自动迁移"
                )
        self.workspace = Workspace(directory, create=create)
        self.path = self.workspace.root / "ui-workflow.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if (
                self.data.get("schema") != "pure-rgb-staged-ui"
                or self.data.get("version") != 1
            ):
                raise ValueError("未知界面项目版本")
        else:
            self.data = dict(
                schema="pure-rgb-staged-ui",
                version=1,
                photos=[],
                jobs={},
                stage=0,
                current=None,
                checked=[],
                selection_version=1,
            )
        # Opening a historical project resolves the former frame step in
        # memory; jobs, image manifests and UI metadata stay untouched here.
        self.data["stage"] = canonical_stage(self.data.get("stage", 0))
        self.photos = self.data["photos"]
        # Existing workspaces used an empty checkbox set to mean the current
        # photo. Resolve that once when adopting explicit thumbnail selection.
        if self.data.get("selection_version") != 1:
            self.data["checked"] = self.data.get("checked") or (
                [self.data["current"]] if self.data.get("current") else []
            )
            self.data["selection_version"] = 1
        self.preview_metering = {}
        self._highlight_policy_cache = {}
        self._repair_documents = {}
        self._repair_candidates = {}
        self._repair_frames = {}
        self.orientation_migrations = []
        # Resolve old per-engine directions in memory only. Opening a project
        # does not rewrite its UI metadata, saved jobs or image manifests.
        for photo in self.photos:
            # A short-lived development build required an extra drawn density
            # selection after successful automatic measurement. It must not
            # block existing projects from continuing with that measurement.
            photo.pop("requires_drawn_density", None)
            self._shared_orientation(photo)
            self._color_engine(photo)
            self._restore_shared_crop(photo)
            self._restore_shared_straighten(photo)
            self._shared_measurement(photo, normalize=True)
            self._shared_flatfield(photo)
        self.edit_histories = {
            photo["id"]: PhotoEditHistory(self._edit_snapshot(photo))
            for photo in self.photos
        }
        if self.orientation_migrations:
            from .diagnostics import event

            event(
                "shared_output_orientation_resolved",
                migrated_photo_count=len(self.orientation_migrations),
                assignments=self.orientation_migrations[:32],
                assignments_omitted=max(0, len(self.orientation_migrations) - 32),
                persistence="in memory; next normal project save",
            )

    @staticmethod
    def _with_orientation(recipe, orientation):
        result = deepcopy(recipe)
        if orientation is None:
            result.pop("output_orientation", None)
        else:
            result["output_orientation"] = deepcopy(orientation)
        return result

    @staticmethod
    def _with_crop(recipe, source):
        result = {**recipe, "crop": source.get("crop", False)}
        result.pop("output_roi", None)
        if source.get("output_roi") is not None:
            result["output_roi"] = list(source["output_roi"])
        return result

    def _set_shared_crop(self, photo, source):
        for engine, recipe in photo["drafts"].items():
            photo["drafts"][engine] = self._with_crop(recipe, source)

    def _restore_shared_crop(self, photo):
        drafts = photo["drafts"]
        active = drafts.get(photo["engine"], photo["base"])
        # Legacy projects may store a drawn rectangle on only one engine.
        # Prefer the current or remembered engine when rectangles disagree.
        candidates = (active, drafts.get(photo.get("color_engine"), {}), *drafts.values())
        source = next((recipe for recipe in candidates if recipe.get("output_roi")), active)
        self._set_shared_crop(photo, source)

    @staticmethod
    def _set_shared_straighten(photo, settings):
        for engine, recipe in photo["drafts"].items():
            photo["drafts"][engine] = with_straighten(recipe, engine, settings)

    def _restore_shared_straighten(self, photo):
        drafts = photo["drafts"]
        # Old projects stored only non-default values, independently per engine.
        # Recover the remembered color route's settings first, then fall back
        # to a draft with settings when the selected route had none.
        engines = dict.fromkeys((self._color_engine(photo), photo["engine"], *ENGINES))
        settings = next((
            values for engine in engines
            if (values := straighten_values(drafts.get(engine, {}), engine))
        ), {})
        self._set_shared_straighten(photo, settings)

    def _shared_orientation(self, photo):
        """One output permutation per photo; engine color drafts stay separate."""
        if "output_orientation" not in photo:
            engine = photo["engine"]
            selected = photo["drafts"].get(engine, {})
            orientation = checked_output_orientation(selected.get("output_orientation"))
            self.orientation_migrations.append(dict(
                photo=photo["id"], selected_engine=engine,
                adopted_orientation=deepcopy(orientation),
                policy="selected-engine-only; no disk write until normal save",
            ))
        else:
            orientation = checked_output_orientation(photo["output_orientation"])
        photo["output_orientation"] = deepcopy(orientation)
        for engine, recipe in photo["drafts"].items():
            if (orientation is None and "output_orientation" in recipe) or (
                orientation is not None and recipe.get("output_orientation") != orientation
            ):
                photo["drafts"][engine] = self._with_orientation(recipe, orientation)
        return orientation

    @staticmethod
    def _measurement_from_recipe(recipe, engine, previous=None):
        legacy_spatial = recipe.get("native_spatial_controls", {})
        keep_legacy = bool(legacy_spatial.get("local_masks") or (
            recipe.get("source_mode") and legacy_spatial
        ))
        result = dict(version=1, roi=deepcopy(recipe.get("roi")), density="negpy",
                      method=recipe.get("measurement_method", "zonal" if keep_legacy else "negpy"))
        if engine == "native":
            spatial = recipe.get("native_spatial_controls")
            if spatial:
                result["spatial_controls"] = deepcopy(spatial)
                result["density_metering"] = dict(
                    version=1, metering=deepcopy(spatial["metering"]),
                )
                if spatial.get("automatic_avoidance"):
                    result["density_metering"]["automatic_avoidance"] = deepcopy(
                        spatial["automatic_avoidance"]
                    )
        elif recipe.get("density_metering"):
            result["density_metering"] = deepcopy(recipe["density_metering"])
        if "density_metering" not in result and previous and previous.get("density_metering"):
            result["density_metering"] = deepcopy(previous["density_metering"])
        if "spatial_controls" not in result and previous and previous.get("spatial_controls"):
            result["spatial_controls"] = deepcopy(previous["spatial_controls"])
        return result

    @staticmethod
    def _with_measurement(recipe, engine, measurement):
        from .measurement_method import native_recipe

        method = measurement.get("method", "negpy")
        result = native_recipe(recipe) if method == "negpy" else deepcopy(recipe)
        result["measurement_method"] = method
        result.update(roi=deepcopy(measurement["roi"]), density="negpy")
        document = measurement.get("density_metering") if method == "zonal" else None
        if engine == "native":
            result.pop("density_metering", None)
            if document:
                spatial = result.setdefault("native_spatial_controls", deepcopy(
                    measurement.get("spatial_controls") or dict(
                        version=1, enabled=True, lab={}, local_masks=[],
                    )
                ))
                from .native_spatial_v30 import LAB_FIELDS

                values = result.get("native_controls", {}).get("values", {})
                for key in list(values):
                    if key.startswith("lab.") and key[4:] in LAB_FIELDS:
                        spatial["lab"][key[4:]] = values.pop(key)
                spatial["metering"] = deepcopy(document["metering"])
                spatial.pop("automatic_avoidance", None)
                if document.get("automatic_avoidance"):
                    spatial["automatic_avoidance"] = deepcopy(document["automatic_avoidance"])
            else:
                result.pop("native_spatial_controls", None)
        elif document:
            result["density_metering"] = deepcopy(document)
        else:
            result.pop("density_metering", None)
        return checked_recipe(result)

    def _shared_measurement(self, photo, *, normalize=False):
        """One source measurement; positive color controls remain per engine."""
        if "measurement" not in photo:
            # Native is the front-end measurement authority, including projects
            # last opened on another color route. Resolve in memory until save.
            native = photo["drafts"].get("native", photo["base"])
            photo["measurement"] = self._measurement_from_recipe(native, "native")
            if photo["measurement"]["roi"] is None and photo.get("analysis_roi"):
                photo["measurement"]["roi"] = list(photo["analysis_roi"])
        measurement = photo["measurement"]
        if (type(measurement) is not dict or type(measurement.get("version")) is not int
                or measurement["version"] != 1
                or set(measurement) - {"version", "roi", "density", "density_metering", "method", "spatial_controls"}
                or measurement.get("density") != "negpy" or "roi" not in measurement):
            raise ValueError("未知照片前置测量配置")
        if "method" not in measurement:
            native = photo["drafts"].get("native", {})
            spatial = native.get("native_spatial_controls", {})
            # Existing explicit local exposure masks keep their original
            # spatial renderer. Ordinary grid-only drafts adopt the native default.
            measurement["method"] = "zonal" if spatial.get("local_masks") else "negpy"
            if spatial and "spatial_controls" not in measurement:
                measurement["spatial_controls"] = deepcopy(spatial)
        if measurement["method"] not in ("negpy", "zonal"):
            raise ValueError("未知密度测量算法")
        document = measurement.get("density_metering") if measurement["method"] == "zonal" else None
        for engine, draft in photo["drafts"].items():
            if normalize:
                photo["drafts"][engine] = self._with_measurement(draft, engine, measurement)
                continue
            if engine == "native":
                spatial = draft.get("native_spatial_controls")
                matches = bool(document) == bool(spatial) and "density_metering" not in draft
                if matches and document:
                    matches = (spatial.get("metering") == document["metering"]
                               and spatial.get("automatic_avoidance") == document.get("automatic_avoidance"))
            else:
                matches = draft.get("density_metering") == document
            if (not matches or draft.get("roi") != measurement["roi"]
                    or draft.get("density") != "negpy"
                    or draft.get("measurement_method") != measurement["method"]):
                photo["drafts"][engine] = self._with_measurement(draft, engine, measurement)
        return measurement

    def save(self):
        write_json(self.path, self.data)

    def _shared_flatfield(self, photo, *, strength=None):
        """One source gain per photo, without copying engine color controls."""
        if "flatfield_input" not in photo:
            photo["flatfield_input"] = photo["drafts"].get(photo["engine"], photo["base"]).get("flatfield")
        path = photo["flatfield_input"]
        if path is not None and (type(path) is not str or not path):
            raise ValueError("平场源需要增益配置路径")
        drafts = photo["drafts"]
        if strength is None:
            # Older projects could create an engine draft without copying the
            # strength; the original Native draft carries the applied value.
            preferred = (drafts.get("native", {}), *drafts.values())
            strength = next((draft["flatfield_strength"] for draft in preferred
                             if "flatfield_strength" in draft), 1.0)
        for engine, recipe in drafts.items():
            strength_changed = bool(path and recipe.get("flatfield_strength", 1.0) != strength)
            if recipe.get("flatfield") != path or strength_changed:
                fields = {"flatfield_strength": strength} if strength_changed else {}
                drafts[engine] = checked_recipe({**recipe, "flatfield": path, **fields})
        return path

    def set_flatfield(self, path, *, strength=None):
        photo = self.photo()
        path = str(Path(path).resolve(strict=True)) if path is not None else None
        label = "应用平场矫正" if path else "移除平场矫正"
        self._set_flatfield(photo, path, strength, label)

    def flatfield_default(self):
        """The roll's import default, including an explicitly disabled gain."""
        if "flatfield_default" in self.data:
            return deepcopy(self.data["flatfield_default"])
        current = next((p for p in self.photos if p["id"] == self.data["current"]), None)
        candidates = ([current] if current else []) + list(reversed(self.photos))
        for photo in candidates:
            recipe = photo["drafts"].get(photo["engine"], photo["base"])
            path = photo.get("flatfield_input", recipe.get("flatfield"))
            if path:
                return dict(path=path, strength=recipe.get("flatfield_strength", 1.0), enabled=True,
                            name=self.flatfield_name(path))
        return None

    def flatfield_name(self, path):
        for entry in self.data.get("flatfield_library", []):
            if entry.get("gain_path") == path:
                return entry.get("name") or Path(path).name
        setting = self.data.get("flatfield_default") or {}
        return setting.get("name") if setting.get("path") == path and setting.get("name") else Path(path).name

    def apply_flatfield_default(self, path, *, strength=1.0):
        """Apply one source choice to every supported group in one saved edit."""
        if type(strength) not in (int, float) or not 0 <= strength <= 2:
            raise ValueError("平场强度需为 0..2 的数")
        path = str(Path(path).resolve(strict=True)) if path is not None else None
        previous_default = self.data.get("flatfield_default")
        had_default = "flatfield_default" in self.data
        updates = []
        for photo in self.photos:
            if photo["source_mode"] == "bw" or photo["base"].get("raw_import") == "standard-cfa-dng-v1":
                continue
            drafts = {engine: checked_recipe({**recipe, "flatfield": path, "flatfield_strength": strength})
                      for engine, recipe in photo["drafts"].items()}
            updates.append((photo, dict(photo), drafts, self._edit_history(photo)))
        self.data["flatfield_default"] = dict(path=path, strength=strength, enabled=path is not None,
                                             name=self.flatfield_name(path) if path else None)
        for photo, _, drafts, _ in updates:
            photo.update(drafts=drafts, flatfield_input=path)
            photo.pop("geometry", None)
            photo.pop("flatfield_inheritance_warning", None)
        try:
            self.save()
        except OSError:
            if had_default:
                self.data["flatfield_default"] = previous_default
            else:
                self.data.pop("flatfield_default", None)
            for photo, previous, _, _ in updates:
                photo.clear()
                photo.update(previous)
            raise
        for photo, _, _, history in updates:
            self.preview_metering.pop(photo["id"], None)
            history.record(self._edit_snapshot(photo), "整卷平场矫正")
        return len(updates)

    def inherit_flatfield(self, photo):
        setting = self.flatfield_default()
        if not setting or not setting.get("enabled") or not setting.get("path"):
            return
        path = setting["path"]
        if photo["source_mode"] == "bw" or photo["base"].get("raw_import") == "standard-cfa-dng-v1":
            photo["flatfield_inheritance_warning"] = "当前输入格式不支持卷默认平场，请为此照片核对校准。"
            return
        if not Path(path).is_file():
            photo["flatfield_inheritance_warning"] = "默认平场文件已不存在，请重新选择参考后再导入。"
            return
        photo["flatfield_input"] = path
        photo["drafts"] = {engine: checked_recipe({**draft, "flatfield": path,
                                                  "flatfield_strength": setting.get("strength", 1.0)})
                           for engine, draft in photo["drafts"].items()}

    def set_flatfield_strength(self, strength):
        if strength is None:
            raise ValueError("平场强度需为 0..2 的数")
        photo = self.photo()
        self._set_flatfield(photo, self._shared_flatfield(photo), strength, "调整平场强度")

    def _set_flatfield(self, photo, path, strength, label):
        history = self._edit_history(photo)
        fields = {"flatfield_strength": strength} if strength is not None else {}
        drafts = {engine: checked_recipe({**recipe, "flatfield": path, **fields})
                  for engine, recipe in photo["drafts"].items()}
        prior = dict(photo)
        photo.update(drafts=drafts, flatfield_input=path)
        photo.pop("geometry", None)
        try:
            self.save()
        except OSError:
            photo.clear()
            photo.update(prior)
            raise
        self.preview_metering.pop(photo["id"], None)
        history.record(self._edit_snapshot(photo), label)

    def set_lens_correction(self, enabled):
        photo = self.photo()
        history = self._edit_history(photo)
        photo["drafts"] = {engine: checked_recipe({**recipe, "lens_correction": enabled})
                          for engine, recipe in photo["drafts"].items()}
        photo.pop("geometry", None)
        self.preview_metering.pop(photo["id"], None)
        history.record(self._edit_snapshot(photo), "RAW 内嵌镜头矫正")
        self.save()

    def register_flatfield(self, report):
        """Index a computed gain for later reuse; one entry per gain content."""
        gain_path = report.get("gain_path")
        if type(gain_path) is not str or not gain_path:
            return None
        digest = report.get("gain_sha256")
        library = self.data.setdefault("flatfield_library", [])
        for entry in library:
            if entry.get("gain_sha256") == digest and entry.get("gain_path") == gain_path:
                return entry
        capture = (report.get("provenance") or {}).get("capture") or {}
        entry = {
            "id": uuid.uuid4().hex,
            "gain_path": gain_path,
            "gain_sha256": digest,
            "name": Path(report.get("path") or gain_path).name,
            "camera": [capture.get("make"), capture.get("model")],
            "sensor_shape": capture.get("sensor_shape"),
        }
        library.append(entry)
        self.save()
        return entry

    def remove_flatfield_entry(self, entry_id):
        """Drop a reusable entry; gain files stay for photos that use them."""
        library = self.data.get("flatfield_library") or []
        remaining = [entry for entry in library if entry.get("id") != entry_id]
        if len(remaining) == len(library):
            return False
        self.data["flatfield_library"] = remaining
        self.save()
        return True

    _EDIT_KEYS = (
        "drafts", "color_engine", "output_orientation", "measurement",
        "protected_rois", "fixed_last_strength", "highlight_reference_origin",
        "highlight_reference_drawn_roi", "highlight_reference_aligned",
        "suspended_highlight_controls",
        "flatfield_input", "repair_reset_index",
    )

    def _edit_snapshot(self, photo):
        return deepcopy({key: photo[key] for key in self._EDIT_KEYS if key in photo})

    def _edit_history(self, photo=None):
        photo = photo or self.photo()
        if photo["id"] not in self.edit_histories:
            self.edit_histories[photo["id"]] = PhotoEditHistory(self._edit_snapshot(photo))
        return self.edit_histories[photo["id"]]

    def begin_edit(self, label="调整参数"):
        if self.data["current"]:
            self._edit_history().begin(label)

    def end_edit(self):
        if self.data["current"]:
            self._edit_history().end()

    def can_undo(self):
        return bool(self.data["current"] and self._edit_history().can_undo())

    def can_redo(self):
        return bool(self.data["current"] and self._edit_history().can_redo())

    def undo_label(self):
        return self._edit_history().undo_label() if self.data["current"] else ""

    def redo_label(self):
        return self._edit_history().redo_label() if self.data["current"] else ""

    def _move_edit_history(self, redo=False):
        if not self.data["current"]:
            return False
        photo = self.photo()
        previous_source = source_recipe_identity(self.recipe(photo))
        history = self._edit_history(photo)
        state = history.move(redo)
        if state is None:
            return False
        for key in self._EDIT_KEYS:
            photo.pop(key, None)
        photo.update(state)
        # A meter workspace always displays Native, even if the undone command
        # restores another remembered color route. Navigation is not an edit.
        engine = "native" if self.data.get("stage") == 2 else self._color_engine(photo)
        self._activate_engine(photo, engine)
        if source_recipe_identity(self.recipe(photo)) != previous_source:
            photo.pop("geometry", None)
        history.current = self._edit_snapshot(photo)
        self.preview_metering.pop(photo["id"], None)
        self.save()
        return True

    def undo(self):
        return self._move_edit_history()

    def redo(self):
        return self._move_edit_history(redo=True)

    def photo(self, key=None):
        key = key or self.data["current"]
        return next(p for p in self.photos if p["id"] == key)

    def add_photo(self, paths, source_mode):
        recipe = base_recipe(paths, source_mode)
        key = photo_key(recipe["raw"], source_mode)
        old = next((p for p in self.photos if p["id"] == key), None)
        if old is None:
            stashed = (self.data.get("removed") or {}).pop(key, None)
            # completed() records the measured ROI in base. It is derived
            # geometry, not a different input when importing the same RAW.
            restore = (isinstance(stashed, dict) and isinstance(stashed.get("base"), dict)
                       and {**stashed["base"], "roi": None} == recipe)
            if restore:
                # Re-importing an unloaded photo restores its saved versions.
                old = stashed
                self._restore_shared_crop(old)
                self._restore_shared_straighten(old)
            else:
                old = dict(
                id=key,
                source_mode="rgb" if source_mode == "mono_rgb" else source_mode,
                engine="native",
                color_engine="native",
                base=recipe,
                drafts={"native": deepcopy(recipe)},
                saved={},
                history=[],
                initial_job=None,
                output_orientation=None,
                flatfield_input=None,
                source_shape=None,
                analysis_roi=None,
                measurement=dict(version=1, roi=None, density="negpy"),
                protected_rois=[],
                fixed_last_strength=0.5,
            )
            if not restore:
                self.inherit_flatfield(old)
            self.photos.append(old)
            self.edit_histories[key] = PhotoEditHistory(self._edit_snapshot(old))
        self.data["current"] = key
        self.data["checked"] = [key]
        self.save()
        return old

    def remove_photo(self, photo=None):
        """Unload one photo, retaining its settings and saved runs for reimport."""
        photo = photo or self.photo()
        return self.remove_photos([photo["id"]])[0]

    def remove_photos(self, identifiers):
        """Unload a selection in one save; original files and runs stay intact."""
        keys = set(identifiers)
        if keys - {entry["id"] for entry in self.photos}:
            raise ValueError("照片不在列表中")
        removed = [entry for entry in self.photos if entry["id"] in keys]
        if not removed:
            return []
        self.data.setdefault("removed", {}).update(
            (photo["id"], deepcopy(photo)) for photo in removed
        )
        self.photos[:] = [entry for entry in self.photos if entry["id"] not in keys]
        self.data["checked"] = [
            entry for entry in (self.data.get("checked") or []) if entry not in keys
        ]
        self.data["import_queue"] = [
            entry for entry in self.data.get("import_queue", []) if entry not in keys
        ]
        for key in keys:
            self.preview_metering.pop(key, None)
            self.edit_histories.pop(key, None)
        if self.data["current"] in keys:
            self.data["current"] = self.photos[0]["id"] if self.photos else None
        self.save()
        return removed

    @staticmethod
    def _color_engine(photo):
        engine = photo.get("color_engine", photo["engine"])
        if photo["source_mode"] == "bw":
            engine = "native"
        if engine not in ENGINES:
            raise ValueError("未知照片色彩引擎")
        photo["color_engine"] = engine
        return engine

    def color_engine(self, photo=None):
        return self._color_engine(photo or self.photo())

    def _activate_engine(self, photo, engine):
        if engine not in ENGINES or (
            photo["source_mode"] == "bw" and engine != "native"
        ):
            raise ValueError("黑白单张仅使用 NegPy 原生路线")
        orientation = self._shared_orientation(photo)
        flatfield = self._shared_flatfield(photo)
        measurement = self._shared_measurement(photo)
        if engine not in photo["drafts"]:
            source_engine = photo["engine"]
            source_draft = photo["drafts"][source_engine]
            recipe = self._with_crop(deepcopy(photo["base"]), source_draft)
            recipe = with_straighten(recipe, engine, straighten_values(source_draft, source_engine))
            if source_draft.get("lens_correction"):
                recipe["lens_correction"] = True
            recipe["flatfield"] = flatfield
            if flatfield and "flatfield_strength" in source_draft:
                recipe["flatfield_strength"] = source_draft["flatfield_strength"]
            if engine == "nikon":
                recipe["nikon_research_controls"] = nikon_controls()
            elif engine == "flex":
                document = flex_controls()
                if document is not None:
                    recipe["flex_research_controls"] = document
            photo["drafts"][engine] = self._with_measurement(
                self._with_orientation(recipe, orientation), engine, measurement
            )
        photo["engine"] = engine

    def select_engine(self, engine, *, remember_color=True):
        photo = self.photo()
        history = self._edit_history(photo)
        self._activate_engine(photo, engine)
        if remember_color:
            photo["color_engine"] = engine
            history.record(self._edit_snapshot(photo), "切换转换引擎")
        self.save()

    def enter_stage(self, index):
        """The combined frame/meter step retains the chosen color route."""
        index = canonical_stage(index)
        previous = canonical_stage(self.data.get("stage", 0))
        current = self.photo() if self.data["current"] else None
        before = current["engine"] if current else None
        if index == 2 and current is not None:
            self._color_engine(current)
            self._activate_engine(current, "native")
        elif previous == 2:
            # Several photos may have been inspected in the metering stage.
            # Restore every remembered route before batch review or export.
            for photo in self.photos:
                self._activate_engine(photo, self._color_engine(photo))
        elif current is not None:
            self._activate_engine(current, self._color_engine(current))
        self.data["stage"] = index
        self.save()
        return bool(current and current["engine"] != before)

    def recipe(self, photo=None, engine=None):
        photo = photo or self.photo()
        orientation = self._shared_orientation(photo)
        self._shared_measurement(photo)
        self._shared_flatfield(photo)
        return self._with_orientation(photo["drafts"][engine or photo["engine"]], orientation)

    def set_recipe(self, recipe, *, label="调整参数", record_edit=True):
        photo = self.photo()
        history = self._edit_history(photo)
        checked = checked_recipe(recipe)
        if checked["raw"] != photo["base"]["raw"]:
            raise ValueError("源文件变化需重新核对导入，不能覆盖已有照片")
        photo["drafts"][photo["engine"]] = checked
        self._set_shared_crop(photo, checked)
        self._set_shared_straighten(photo, straighten_values(checked, photo["engine"]))
        photo["flatfield_input"] = checked.get("flatfield")
        self._shared_flatfield(photo, strength=checked.get("flatfield_strength", 1.0))
        photo["measurement"] = self._measurement_from_recipe(
            checked, photo["engine"], photo.get("measurement")
        )
        self._shared_measurement(photo)
        photo["output_orientation"] = checked_output_orientation(checked.get("output_orientation"))
        self._shared_orientation(photo)
        if record_edit:
            history.record(self._edit_snapshot(photo), label)
        else:
            history.absorb(self._edit_snapshot(photo))
        self.save()
        return checked

    def reset_photo_adjustments(self, photo):
        """Restore import defaults as one edit and keep every saved render."""
        if not photo["initial_job"]:
            raise ValueError("请先完成照片导入")
        imported = self.workspace.job(photo["initial_job"])["recipe"]
        recipe = {**photo["base"], "roi": deepcopy(photo["analysis_roi"])}
        for field in ("raw_import", "flatfield", "flatfield_strength", "registration_check", "lens_correction"):
            if field in imported:
                recipe[field] = deepcopy(imported[field])
        recipe = checked_recipe(recipe)
        state = dict(
            drafts={"native": recipe}, color_engine="native", output_orientation=None,
            flatfield_input=recipe.get("flatfield"),
            measurement=self._measurement_from_recipe(recipe, "native"),
            protected_rois=[], fixed_last_strength=0.5,
        )
        repairs = len(self.workspace.data.get("repairs", ()))
        if repairs:
            state["repair_reset_index"] = repairs
        if state == self._edit_snapshot(photo):
            return False

        history = self._edit_history(photo)
        old = dict(photo)
        for key in self._EDIT_KEYS:
            photo.pop(key, None)
        photo.update(state, engine="native")
        photo.pop("geometry", None)
        try:
            self.save()
        except OSError:
            photo.clear()
            photo.update(old)
            raise
        self.preview_metering.pop(photo["id"], None)
        history.record(self._edit_snapshot(photo), "还原所有设置")
        return True

    def paste_adjustments(self, source, target_ids, groups):
        """Validate the entire selection before publishing one project update."""
        from .workflow_adjustment_clipboard import merged_adjustments

        photos_by_id = {photo["id"]: photo for photo in self.photos}
        updates = []
        for key in dict.fromkeys(target_ids):
            photo = photos_by_id.get(key)
            if photo is None or key == source["photo_id"] or not photo["initial_job"]:
                raise ValueError("只能粘贴到已导入的其他照片")
            candidate = deepcopy(photo)
            engine = self.color_engine(candidate)
            before = self.recipe(candidate, engine)
            before_edit = self._edit_snapshot(candidate)
            after = checked_recipe(merged_adjustments(source, candidate, engine, before, groups))
            candidate["drafts"][engine] = after
            if "engine" in groups and engine == "nikon":
                fixed = after.get("nikon_research_controls", {}).get("fixed_strength", 0)
                if fixed:
                    candidate["fixed_last_strength"] = fixed
            if "crop" in groups:
                self._set_shared_crop(candidate, after)
            if "geometry" in groups:
                self._set_shared_straighten(candidate, straighten_values(after, engine))
            if "orientation" in groups:
                candidate["output_orientation"] = checked_output_orientation(after.get("output_orientation"))
                self._shared_orientation(candidate)
            if self._edit_snapshot(candidate) == before_edit:
                continue
            updates.append((photo, deepcopy(photo), candidate))

        if not updates:
            return []
        histories = [self._edit_history(photo) for photo, _, _ in updates]
        for photo, _, candidate in updates:
            photo.clear()
            photo.update(candidate)
        try:
            self.save()
        except OSError:
            for photo, prior, _ in updates:
                photo.update(prior)
            raise
        for history, (photo, _, _) in zip(histories, updates):
            history.record(self._edit_snapshot(photo), "粘贴调整")
        return [photo["id"] for photo, _, _ in updates]

    def enqueue(self):
        photo = self.photo()
        self.prepare_highlight_reference()
        recipe = self.recipe()
        job = self.workspace.add(recipe)
        self.data["jobs"][job["id"]] = dict(photo=photo["id"], engine=photo["engine"])
        photo["history"].append(job["id"])
        self.save()
        return job

    def prepare_highlight_reference(self):
        photo = self.photo()
        recipe, automatic = self._highlight_reference_recipe(photo)
        if automatic is None:
            return recipe
        if automatic:
            photo["highlight_reference_origin"] = "automatic"
        return self.set_recipe(recipe, record_edit=False)

    def _highlight_reference_recipe(self, photo):
        """Prepare a detached recipe for a photo without changing its draft."""
        recipe = self.recipe(photo)
        if (
            photo["engine"] != "native"
            or not recipe.get("highlight_controls")
            or not recipe.get("native_spatial_controls")
        ):
            return recipe, None
        high = recipe["highlight_controls"]
        origin = photo.get("highlight_reference_origin")
        automatic = origin == "automatic" or (
            origin is None and (high.get("automatic_reference") or high["reference_roi"] == recipe["roi"])
        )
        avoidance = recipe["native_spatial_controls"].get("automatic_avoidance", {})
        if automatic and photo["source_mode"] == "rgb" and avoidance.get("action") == "exclude":
            # The worker can choose only after the current exclusions are frozen.
            high["automatic_reference"] = True
            high["reference_roi"] = align_highlight_reference(recipe, recipe["roi"])
            return recipe, True
        high.pop("automatic_reference", None)
        extras = []
        job = self.saved_job(photo)
        if job and avoidance.get("action") == "exclude":
            manifest = json.loads(
                (self.workspace.root / "runs" / job["id"] / "manifest.json").read_text()
            )
            report = (
                manifest.get("native_spatial_processing", {}).get("automatic_avoidance")
                or {}
            )
            before = dict(report.get("document", {}))
            if before:
                before["action"] = "exclude"
            if (
                manifest["roi"] == recipe["roi"]
                and before == avoidance
                and report.get("fallback") is None
            ):
                extras = report.get("candidate_exclusions", [])
        if automatic:
            recipe["highlight_controls"]["reference_roi"] = suggest_highlight_reference(
                recipe, photo["source_shape"], extras
            )
        validate_highlight_reference(
            recipe, photo["source_shape"], extras, require_metering_support=automatic
        )
        return recipe, automatic

    def completed(self, job_id):
        job = self.workspace.job(job_id)
        # Validate the full receipt before marking a UI version usable.
        sample = self.workspace.sample({**job, "state": "complete"})
        manifest = json.loads((sample.directory / "manifest.json").read_text())
        self.workspace.set_state(job_id, "complete")
        binding = self.data["jobs"][job_id]
        photo = self.photo(binding["photo"])
        photo["saved"][binding["engine"]] = job_id
        if photo["initial_job"] is None:
            photo.update(
                initial_job=job_id,
                source_shape=manifest["source_shape"],
                analysis_roi=manifest["roi"],
            )
            photo["base"]["roi"] = list(manifest["roi"])
            # The resolved auto ROI is not a user edit; synchronize the first draft.
            if photo["drafts"]["native"] == job["recipe"]:
                photo["drafts"]["native"]["roi"] = list(manifest["roi"])
            if photo["measurement"]["roi"] is None:
                photo["measurement"]["roi"] = list(manifest["roi"])
                self._shared_measurement(photo)
            # Initial source geometry is derived, not a reversible user edit.
            self.edit_histories[photo["id"]] = PhotoEditHistory(self._edit_snapshot(photo))
        self.save()
        return sample, manifest

    def saved_job(self, photo=None):
        photo = photo or self.photo()
        key = photo["saved"].get(photo["engine"])
        return self.workspace.job(key) if key else None

    def dirty(self, photo=None, *, saved=None):
        photo = photo or self.photo()
        job = saved if saved is not None else self.saved_job(photo)
        if not job:
            return True
        old = job["recipe"]
        if old.get("roi") is None and photo["analysis_roi"] is not None:
            old = {**old, "roi": photo["analysis_roi"]}
        current = self.recipe(photo)
        # Saved jobs are already checked. Legacy aliases only need metadata
        # normalization when their representation differs from the draft.
        if old != current and checked_recipe(old, verify_resources=False) != current:
            return True
        return self.highlight_reprocess_required(photo, saved=job)

    def highlight_reprocess_required(self, photo=None, *, saved=None):
        """Identify saved local-highlight results produced by an older algorithm."""
        photo = photo or self.photo()
        if not photo or photo["engine"] != "native":
            return False
        job = saved if saved is not None else self.saved_job(photo)
        if not job:
            return False
        highlight = job["recipe"].get("highlight_controls") or {}
        if not highlight.get("effect_area") and not highlight.get("protection"):
            return False
        path = self.workspace.root / "runs" / job["id"] / "manifest.json"
        try:
            stat = path.stat()
            identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            cached = self._highlight_policy_cache.get(job["id"])
            if cached and cached[0] == identity:
                return cached[1]
            manifest = json.loads(path.read_text())
            spatial = manifest.get("native_spatial_processing") or {}
            direct = manifest.get("highlight_processing") or {}
            processing = spatial.get("high_processing") or direct.get("highlight_processing") or {}
            stale = processing.get("local_selection_policy") != "highlight-luminance-soft-v2"
            self._highlight_policy_cache[job["id"]] = (identity, stale)
            return stale
        except (OSError, ValueError, TypeError, AttributeError):
            return True

    def source_job(self, photo=None):
        """Find a completed source with the current decode/gain identity."""
        photo = photo or self.photo()
        recipe = self.recipe(photo)
        identity = source_recipe_identity(recipe)
        identifiers = [photo["saved"].get(photo["engine"]),
                       *reversed(photo["history"]), photo.get("initial_job")]
        for identifier in dict.fromkeys(identifiers):
            if not identifier:
                continue
            try:
                job = self.workspace.job(identifier)
            except StopIteration:
                continue
            if job["state"] == "complete" and source_recipe_identity(job["recipe"]) == identity:
                return job
        return None

    def decode_anchor(self, photo=None, recipe=None):
        """Completed-run decode identity for the renderer's decoded-RAW cache.

        Render requests hold no manifest yet, so their cache may only restore
        against a prior completed run with the same decode/gain identity
        (strength included: it changes the decoded pixels). Photos without
        such a run return None and render exactly as before.
        """
        photo = photo or self.photo()
        recipe = recipe or self.recipe(photo)
        identity = source_recipe_identity(recipe)
        identifiers = [photo["saved"].get(photo["engine"]),
                      *reversed(photo["history"]), photo.get("initial_job")]
        for identifier in dict.fromkeys(i for i in identifiers if i):
            try:
                job = self.workspace.job(identifier)
                if job["state"] != "complete" or source_recipe_identity(job["recipe"]) != identity:
                    continue
                manifest = json.loads(
                    (self.workspace.root / "runs" / job["id"] / "manifest.json")
                    .read_text()
                )
                shape = manifest.get("source_shape")
                anchor = {
                    "source_shape": shape,
                    "raw_import_receipt": manifest.get("raw_import_receipt"),
                }
                if recipe["flatfield"]:
                    capture = (
                        (manifest.get("calibration") or {}).get("provenance") or {}
                    ).get("capture") or {}
                    anchor["flatfield_reference_shape"] = capture.get("sensor_shape")
            except (OSError, ValueError, KeyError, TypeError, StopIteration):
                continue
            return anchor
        return None

    def preview_base(self):
        photo = self.photo()
        job = self.source_job(photo)
        if not job:
            raise ValueError("请先核对导入并重新处理当前平场／源配置")
        return self.workspace.root / "runs" / job["id"] / "manifest.json"

    def selected_photo_ids(self):
        selected = set(self.data.get("checked") or ())
        return [photo["id"] for photo in self.photos if photo["id"] in selected]

    def select_photos(self, identifiers):
        selected = set(identifiers)
        if selected - {photo["id"] for photo in self.photos}:
            raise ValueError("所选照片已不在当前项目")
        self.data["checked"] = [photo["id"] for photo in self.photos if photo["id"] in selected]
        self.save()

    def export_items(self, destination):
        items = []
        ids = self.selected_photo_ids()
        if not ids:
            raise ValueError("请先在照片列表选择要导出的照片")
        for key in ids:
            photo = self.photo(key)
            mode = RESULT_MODES[photo["engine"]]
            if self.dirty(photo):
                # The export worker renders this detached snapshot at full
                # size. Subsequent edits cannot alter it or create queued jobs.
                recipe, _ = self._highlight_reference_recipe(photo)
                payload = dict(render={
                    **freeze_render_request(recipe), "only_mode": mode,
                    "decode_anchor": self.decode_anchor(photo, recipe),
                    "repair_plan": self.repair_context(photo, recipe),
                })
                revision = uuid.uuid4().hex[:8]
            else:
                job = self.saved_job(photo)
                sample = self.workspace.sample(job)
                record = self.confirmed_record(sample.record(mode, "baseline"), photo)
                payload = dict(record=deepcopy(record))
                revision = job["id"][:8]
            name = f"{Path(photo['base']['raw'][0]).stem}_{photo['engine']}_{revision}.tif"
            items.append(
                dict(
                    id=photo["id"],
                    mode=mode,
                    destination=str(Path(destination) / name),
                    **payload,
                )
            )
        return items

    def metering_receipt(self):
        photo = self.photo()
        transient = self.preview_metering.get(photo["id"])
        if transient and transient["recipe"] == self.recipe(photo):
            return deepcopy(transient["receipt"])
        key = photo["saved"].get("native")
        if not key:
            return {}
        manifest = json.loads(
            (self.workspace.root / "runs" / key / "manifest.json").read_text()
        )
        spatial = manifest["recipe"].get("native_spatial_controls", {})
        from .metering_hover_v45 import metering_zone_records

        processing = manifest.get("native_spatial_processing", {})
        reference_roi = processing.get("highlight_reference_roi")
        zones = metering_zone_records(processing.get("metering_window") or {})
        return dict(
            roi=manifest["roi"],
            metering=spatial.get("metering"),
            automatic=spatial.get("automatic_avoidance"),
            report=processing.get("automatic_avoidance") or {},
            **({"zones": zones} if zones else {}),
            **({"highlight_reference_roi": reference_roi}
               if reference_roi and manifest["recipe"] == self.recipe(photo) else {}),
        )

    def remember_preview_metering(self, recipe, receipt):
        """Keep an accepted draft analysis in memory, without a saved version."""
        if self.photo()["engine"] != "native" or recipe != self.recipe():
            return
        spatial = recipe.get("native_spatial_controls", {})
        if not isinstance(receipt, dict) or (
            receipt.get("roi") != recipe.get("roi")
            or receipt.get("metering") != spatial.get("metering")
            or receipt.get("automatic") != spatial.get("automatic_avoidance")
            or not isinstance(receipt.get("report", {}), dict)
        ):
            raise ValueError("草稿测量回执与当前选区或异常策略不一致")
        self.preview_metering[self.photo()["id"]] = dict(
            recipe=deepcopy(recipe), receipt=deepcopy(receipt)
        )

    def confirm_candidates(self):
        if self.photo()["engine"] != "native":
            raise ValueError("当前引擎不使用原生分区避让")
        receipt = self.metering_receipt()
        recipe = self.recipe()
        spatial = recipe.get("native_spatial_controls", {})
        if (
            recipe["roi"] != receipt.get("roi")
            or spatial.get("metering") != receipt.get("metering")
            or spatial.get("automatic_avoidance") != receipt.get("automatic")
        ):
            raise ValueError("密度设置已改变，请先应用测量后再确认候选")
        boxes = receipt.get("report", {}).get("candidate_exclusions", [])
        if not boxes:
            raise ValueError("本次没有可确认的候选；这不代表底片没有异常")
        excluded = spatial["metering"]["excluded_rois"] + boxes
        if len(excluded) > 64:
            raise ValueError("候选片段过多，请逐处绘制排除区域")
        spatial["metering"]["excluded_rois"] = excluded
        return self.set_recipe(recipe)

    def repair_paths(self, photo=None):
        if photo is None and self.data["current"]:
            photo = self.photo()
        start = (photo or {}).get("repair_reset_index", 0)
        return self.workspace.data.get("repairs", [])[start:]

    @staticmethod
    def _repair_file_identity(path):
        stat = Path(path).stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def _repair_json(self, path):
        """Cache small documents, never RAWs or repaired pixel buffers."""
        path = Path(path)
        before = self._repair_file_identity(path)
        cached = self._repair_documents.get(str(path))
        if cached and cached[0] == before:
            return cached[1], before
        data = json.loads(path.read_text())
        if before != self._repair_file_identity(path):
            raise ValueError("除尘来源清单在读取期间改变，请重试")
        if len(self._repair_documents) >= 512:
            self._repair_documents.clear()
        self._repair_documents[str(path)] = (before, data)
        return data, before

    def _repair_selection(self, photo=None, recipe=None):
        """Resolve old reviews once; colour edits only recheck small bindings."""
        from .repair_plan import geometry_key, make_plan, validate_frame

        photo = photo or self.photo()
        paths = tuple(self.repair_paths(photo))
        if not paths:
            return None, None
        recipe = recipe or self.recipe(photo)
        mode = RESULT_MODES[photo["engine"]]
        identifiers = tuple(dict.fromkeys([
            *reversed(photo["history"]), *photo["saved"].values(), photo.get("initial_job")]))
        identities = []
        for path in paths:
            try:
                identities.append((path, self._repair_file_identity(path)))
            except OSError:
                identities.append((path, None))
        signature = (identifiers, tuple(identities))
        cached = self._repair_candidates.get(photo["id"])
        if cached is None or cached["signature"] != signature:
            candidates = []
            for receipt_path in reversed(paths):
                try:
                    receipt, _ = self._repair_json(receipt_path)
                    if receipt.get("state") == "confirmed":
                        candidates.append((receipt_path, receipt))
                except (OSError, ValueError, TypeError):
                    continue
            manifests = []
            if candidates:
                for identifier in identifiers:
                    if not identifier:
                        continue
                    path = self.workspace.root / "runs" / identifier / "manifest.json"
                    try:
                        data, identity = self._repair_json(path)
                        if data.get("state") == "complete":
                            manifests.append((path, data, identity))
                    except (OSError, ValueError, TypeError):
                        continue
            cached = dict(signature=signature, plan=None, reason=None)
            for receipt_path, receipt in candidates:
                source = receipt.get("source", {})
                anchor = next(((path, data, identity, old_mode)
                    for path, data, identity in manifests
                    for old_mode, record in data.get("results", {}).items()
                    if record.get("sha256") == source.get("sha256")
                    and Path(record.get("path", "")).resolve() == Path(source.get("path", "")).resolve()), None)
                if anchor is None:
                    continue
                path, parent, identity, old_mode = anchor
                try:
                    cached.update(plan=make_plan(receipt_path, path, old_mode),
                                  parent_path=path, parent_identity=identity)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    cached["reason"] = str(error)
                break
            self._repair_candidates[photo["id"]] = cached
        plan = cached["plan"]
        if plan is None:
            return None, cached["reason"]
        try:
            if self._repair_file_identity(cached["parent_path"]) != cached["parent_identity"]:
                raise ValueError("除尘来源清单已改变，请重新进行除尘审核")
            current_job = self.source_job(photo)
            if current_job is None:
                return None, "来源配置已改变，请重新进行除尘审核"
            current_path = self.workspace.root / "runs" / current_job["id"] / "manifest.json"
            current, current_identity = self._repair_json(current_path)
            key = (signature, cached["parent_identity"], str(current_path), current_identity,
                   mode, geometry_key(recipe, current, mode))
            validated = self._repair_frames.get(photo["id"])
            if validated is None or validated[0] != key:
                try:
                    validate_frame(plan, recipe, current, mode)
                except ValueError as error:
                    self._repair_frames[photo["id"]] = (key, str(error))
                    raise
                self._repair_frames[photo["id"]] = (key, None)
            elif validated[1] is not None:
                raise ValueError(validated[1])
            return deepcopy(plan), None
        except (OSError, ValueError, KeyError, TypeError) as error:
            return None, str(error)

    def repair_context(self, photo=None, recipe=None):
        return self._repair_selection(photo, recipe)[0]

    def repair_status(self, photo=None, recipe=None):
        plan, reason = self._repair_selection(photo, recipe)
        return dict(state="active" if plan else "incompatible" if reason else "none",
                    message=("已保存隐藏除尘数据；调色、裁切、拉直和旋转后自动沿用。" if plan else reason or ""))

    def confirmed_record(self, source, photo=None):
        """Only a confirmed review of these exact source bytes replaces output."""
        paths = self.repair_paths(photo)
        if not paths:
            return source
        for path in reversed(paths):
            receipt, _ = self._repair_json(path)
            if receipt.get("source", {}).get("sha256") != source["sha256"]:
                continue
            if (receipt.get("source", {}).get("path")
                    and Path(receipt["source"]["path"]).resolve() != Path(source["path"]).resolve()):
                continue
            if receipt.get("revision") != source["sha256"]:
                raise ValueError("除尘审核来源与保存版本不一致")
            if receipt.get("state") == "confirmed":
                if not receipt.get("preview", {}).get("sha256"):
                    raise ValueError("已确认审核缺少可验证输出")
                return deepcopy(receipt["preview"])
        # New renders retain their original colour baseline. The separately
        # authenticated repair result is selected only for display/export.
        from .repair_plan import validate_frame

        photo = photo or self.photo()
        mode = RESULT_MODES[photo["engine"]]
        for identifier in dict.fromkeys([*photo["saved"].values(), *reversed(photo["history"])]):
            path = self.workspace.root / "runs" / identifier / "manifest.json"
            try:
                manifest, _ = self._repair_json(path)
            except (OSError, ValueError):
                continue
            original = manifest.get("results", {}).get(mode, {})
            if (original.get("sha256") != source["sha256"]
                    or Path(original.get("path", "")).resolve() != Path(source["path"]).resolve()):
                continue
            repaired = manifest.get("repair_results", {}).get(mode)
            if repaired:
                if repaired.get("source_sha256") != source["sha256"]:
                    raise ValueError("派生除尘版本与计算基准不一致")
                validate_frame(repaired["repair_plan"], manifest["recipe"], manifest, mode)
                active = self.repair_context(photo)
                if active and active["receipt_sha256"] == repaired["repair_plan"]["receipt_sha256"]:
                    return deepcopy(repaired)
        return source

    def export_request(self, destination):
        return dict(
            action="export_batch",
            directory=str(self.workspace.root / "exports" / uuid.uuid4().hex),
            items=self.export_items(destination),
        )

    def export_previous_highlight_request(self, destination):
        """Export the current photo's explicitly requested saved legacy result."""
        photo = self.photo()
        job = self.saved_job(photo)
        if not job or not self.highlight_reprocess_required(photo, saved=job):
            raise ValueError("当前照片没有需要更新的旧版高光成片")
        mode = RESULT_MODES[photo["engine"]]
        sample = self.workspace.sample(job)
        record = self.confirmed_record(sample.record(mode, "baseline"), photo)
        name = f"{Path(photo['base']['raw'][0]).stem}_{photo['engine']}_{job['id'][:8]}.tif"
        return dict(
            action="export_batch",
            directory=str(self.workspace.root / "exports" / uuid.uuid4().hex),
            items=[dict(id=photo["id"], mode=mode,
                        destination=str(Path(destination) / name), record=deepcopy(record))],
        )
