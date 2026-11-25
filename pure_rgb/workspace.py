"""Versioned local workspaces; recipes and completed render revisions are immutable."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import uuid

from .native import sha256
from .preview_data import Sample
from .professional_controls import (
    NATIVE_SPECS,
    checked_controls,
    checked_outputs,
    checked_engines,
    native_papers,
)
from .positive_post import checked_post
from .engine_resources import checked_resource
from .film_profiles import checked_film
from .native_float import checked_native_float
from .output_orientation import checked_document as checked_output_orientation


def source_recipe_identity(recipe):
    """The fields that affect decoded source pixels, before any color engine."""
    fields = ("source_mode", "raw", "raw_import", "flatfield", "registration_check", "lens_correction")
    return {
        **{key: recipe.get(key) for key in fields},
        "flatfield_strength": float(recipe.get("flatfield_strength", 1.0)) if recipe.get("flatfield") else 1.0,
        **({"chromatic_aberration": True} if recipe.get("chromatic_aberration") else {}),
        **({"lens_profile": recipe["lens_profile"]} if recipe.get("lens_profile") else {}),
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def checked_native_float_controls(document):
    """Sparse Native-only point controls; an empty object is an old-route alias."""
    if not isinstance(document, dict):
        raise ValueError("Native 浮点参数必须为版本化对象，或空对象")
    if not document:
        return None
    try:
        value = checked_native_float(document)
    except OverflowError as error:
        raise ValueError("Native 浮点参数必须为有限且范围内的 JSON 数值") from error
    return value if value["values"] else None


def checked_highlight_controls(document):
    from .highlight_workflow import checked_highlight_workflow

    if not isinstance(document, dict):
        raise ValueError("高光参考参数必须为版本化对象，或空对象")
    if not document:
        return None
    value = checked_highlight_workflow(document)
    return value if value["controls"]["strength"] > 0 else None


def checked_cool_headroom_controls(document):
    from .cool_headroom_workflow import checked_cool_headroom_controls as checked

    return checked(document)


def checked_flex_research_controls(document, *, verify_resources=True):
    from .flex_research_workflow import checked_flex_research_controls as checked

    if type(verify_resources) is not bool:
        raise ValueError("资源验证开关必须为明确布尔值")
    return checked(document, verify_resources=verify_resources)


def checked_nikon_research_controls(document):
    from .nikon_research_workflow import checked_nikon_research_controls as checked

    return checked(document)


def checked_recipe(recipe, *, verify_resources=True):
    """Validate recipe JSON; only historical metadata readers omit file access.

    The keyword is an internal call policy, never a recipe field. A historical
    structurally valid recipe does not establish current resource availability;
    new jobs, live sessions and renderers retain the default strict check.
    """
    if type(verify_resources) is not bool:
        raise ValueError("资源验证开关必须为明确布尔值")
    if isinstance(recipe, dict) and "source_mode" in recipe:
        from .bw_recipe import BW_SOURCE_MODE, checked_bw_recipe

        if recipe["source_mode"] != BW_SOURCE_MODE:
            raise ValueError("未知显式输入模式")
        return checked_bw_recipe(recipe, verify_resources=verify_resources)
    result = deepcopy(recipe)
    if set(result) - {
        "raw",
        "roi",
        "density",
        "density_metering",
        "measurement_method",
        "paper",
        "exposure_ev",
        "crop",
        "output_roi",
        "output_orientation",
        "flatfield",
        "flatfield_strength",
        "cool_tone",
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
        "flex_profile",
        "native_film",
        "common_positive_controls",
        "experimental_film_controls",
        "raw_import",
        "lens_correction",
        "chromatic_aberration",
        "lens_profile",
        "registration_check",
    }:
        raise ValueError("配方包含未支持字段")
    if "flatfield_strength" in result:
        strength = result["flatfield_strength"]
        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not math.isfinite(strength)
            or not 0 <= strength <= 2
        ):
            raise ValueError("平场强度需为 0..2 的数")
        result["flatfield_strength"] = float(strength)
    if "common_positive_controls" in result:
        from .common_positive import checked_common_positive_controls

        result["common_positive_controls"] = checked_common_positive_controls(
            result["common_positive_controls"]
        )
        if result["common_positive_controls"] is None:
            result.pop("common_positive_controls")
    if "experimental_film_controls" in result:
        from .experimental_film import checked_experimental_film_controls

        result["experimental_film_controls"] = checked_experimental_film_controls(
            result["experimental_film_controls"]
        )
        if result["experimental_film_controls"] is None:
            result.pop("experimental_film_controls")
    raw = result.get("raw", [])
    from .raw_import import (
        IMPORT_POLICIES,
        is_standard_import,
        STANDARD_DNG_IMPORT,
        STANDARD_MONO_IMPORT,
        checked_registration_check,
    )

    policy = result.get("raw_import", "legacy-v1")
    if policy not in IMPORT_POLICIES:
        raise ValueError("未知 RAW 导入策略")
    lens = result.get("lens_correction", False)
    if type(lens) is not bool or (lens and not is_standard_import(policy)):
        raise ValueError("RAW 内嵌镜头矫正需要标准三色 RAW 配方")
    if lens and policy == STANDARD_MONO_IMPORT:
        raise ValueError("单色 RGB 暂不支持 RAW 内嵌镜头矫正")
    if not lens:
        result.pop("lens_correction", None)
    chromatic = result.get("chromatic_aberration", False)
    if (type(chromatic) is not bool or
            (chromatic and policy not in ("standard-crop-v1", STANDARD_DNG_IMPORT))):
        raise ValueError("自动镜头色差矫正需要拜耳三色 RAW 配方")
    if not chromatic:
        result.pop("chromatic_aberration", None)
    profile = result.get("lens_profile")
    if profile is not None:
        if (not isinstance(profile, dict) or set(profile) != {"path", "sha256"}
                or not isinstance(profile["path"], str) or not profile["path"]
                or not isinstance(profile["sha256"], str) or len(profile["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in profile["sha256"])
                or policy not in ("standard-crop-v1", STANDARD_DNG_IMPORT)):
            raise ValueError("镜头配置需要有效的本机 LCP 文件及校验记录")
        result["lens_profile"] = {**profile, "path": str(Path(profile["path"]).expanduser().resolve())}
        if verify_resources and (chromatic or lens):
            from .adobe_lens_profiles import read_lcp

            read_lcp(result["lens_profile"])
    registration_check = checked_registration_check(
        policy, result.get("registration_check")
    )
    if registration_check is None:
        result.pop("registration_check", None)
    suffix = ".dng" if policy == STANDARD_DNG_IMPORT else ".rw2"
    if (
        len(raw) != 3
        or len(set(raw)) != 3
        or (policy != STANDARD_MONO_IMPORT and any(Path(p).suffix.lower() != suffix for p in raw))
    ):
        raise ValueError("配方需按 R/G/B 顺序包含三张不同 RW2")
    result["raw"] = [str(Path(p).resolve()) for p in raw]
    if len(set(result["raw"])) != 3:
        raise ValueError("三个通道不可指向同一个文件")
    roi = result.get("roi")
    if roi is not None and (
        len(roi) != 4
        or any(type(v) is not int for v in roi)
        or not 0 <= roi[0] < roi[2]
        or not 0 <= roi[1] < roi[3]
    ):
        raise ValueError("成像区域必须为半开区间 x0 y0 x1 y1，或自动识别")
    from .measurement_method import checked_method

    checked_method(result)
    result.setdefault("density", "negpy")
    result.setdefault("paper", "neutral")
    result.setdefault("exposure_ev", 0.0)
    result.setdefault("crop", True)
    result.setdefault("flatfield", None)
    if policy == STANDARD_DNG_IMPORT and result["flatfield"] is not None:
        raise ValueError("CFA DNG 导入暂不支持平场")
    if (policy == STANDARD_MONO_IMPORT and result["flatfield"] is not None
            and Path(result["flatfield"]).suffix.lower() != ".npz"):
        raise ValueError("单色 RGB 平场需要带传感器画幅记录的增益 NPZ")
    # Missing in an existing frozen recipe means the original conservative output.
    result.setdefault("cool_tone", "conservative")
    if result["cool_tone"] not in ("conservative", "soft-shoulder"):
        raise ValueError("未知 cool 呈现；191→255 截断曲线不作为软件选项")
    if result["density"] not in ("negpy", "percentile") or result["paper"] not in dict(
        native_papers()
    ):
        raise ValueError("未知密度或相纸模型")
    ev = result["exposure_ev"]
    if type(ev) not in (int, float) or not math.isfinite(ev) or not -4 <= ev <= 4:
        raise ValueError("曝光补偿范围为 -4 至 +4 EV")
    if type(result["crop"]) is not bool:
        raise ValueError("裁切参数必须为布尔值")
    if "output_roi" in result:
        output = result["output_roi"]
        if output is None:
            result["crop"] = False
            result.pop("output_roi")
        else:
            if (
                len(output) != 4
                or any(type(v) is not int for v in output)
                or not 0 <= output[0] < output[2]
                or not 0 <= output[1] < output[3]
            ):
                raise ValueError("输出裁切区域必须为独立的 x0 y0 x1 y1")
            result["crop"] = True
            if output == result.get("roi"):
                result.pop("output_roi")
    if "output_orientation" in result:
        value = checked_output_orientation(result["output_orientation"])
        if value is None:
            result.pop("output_orientation")
        else:
            result["output_orientation"] = value
    if "native_controls" in result:
        value = checked_controls(result["native_controls"], NATIVE_SPECS)
        if value["values"]:
            result["native_controls"] = value
        else:
            result.pop("native_controls")
    if "native_float_controls" in result:
        value = checked_native_float_controls(result["native_float_controls"])
        if value is None:
            result.pop("native_float_controls")
        else:
            result["native_float_controls"] = value
    for key, validator in (
        ("highlight_controls", checked_highlight_controls),
        ("cool_headroom_controls", checked_cool_headroom_controls),
        ("flex_research_controls", checked_flex_research_controls),
        ("nikon_research_controls", checked_nikon_research_controls),
    ):
        if key in result:
            value = (
                validator(result[key], verify_resources=verify_resources)
                if key == "flex_research_controls"
                else validator(result[key])
            )
            if value is None:
                result.pop(key)
            else:
                result[key] = value
    if "output_controls" in result:
        value = checked_outputs(result["output_controls"])
        if value:
            result["output_controls"] = value
        else:
            result.pop("output_controls")
    if "engine_controls" in result:
        value = checked_engines(result["engine_controls"])
        if value:
            result["engine_controls"] = value
        else:
            result.pop("engine_controls")
    if result.get("cool_headroom_controls") and result.get("engine_controls", {}).get(
        "cool-shared-rgb", {}
    ).get("values"):
        raise ValueError(
            "Cool 保留余量当前不与非默认引擎曲线控件组合；请先清除引擎曲线调整"
        )
    if result.get("nikon_research_controls") and (
        result["cool_tone"] != "conservative"
        or result.get("cool_headroom_controls")
        or result.get("engine_controls", {}).get("cool-shared-rgb", {}).get("values")
    ):
        raise ValueError(
            "Nikon 仅 CMS 研究链要求 conservative 呈现，且不能叠加 Cool 保留余量或旧引擎曲线"
        )
    if "post_controls" in result:
        value = checked_post(result["post_controls"])
        if value:
            result["post_controls"] = value
        else:
            result.pop("post_controls")
    if result.get("flex_profile"):
        result["flex_profile"] = checked_resource(result["flex_profile"])
    else:
        result.pop("flex_profile", None)
    if result.get("flex_research_controls") and (
        result.get("flex_profile")
        or result.get("engine_controls", {}).get("flex-endpoint", {}).get("values")
    ):
        raise ValueError(
            "Flex 研究链使用独立冻结曲线配方，不能叠加旧 Flex 配置或端点拟合控件"
        )
    if result.get("native_film") is not None:
        result["native_film"] = checked_film(result["native_film"])
    else:
        result.pop("native_film", None)
    if "density_metering" in result:
        from .shared_density_metering_v43 import checked_density_metering

        result["density_metering"] = checked_density_metering(result["density_metering"])
        if result["density"] != "negpy":
            raise ValueError("共享密度测量要求NegPy前置密度路线")
        if any(result.get(key) for key in (
            "native_spatial_controls", "native_controls", "native_float_controls",
            "highlight_controls", "native_film",
        )):
            raise ValueError("共享密度测量仅传递测量支持，不能混入Native创意处理")
    if "native_spatial_controls" in result:
        from .spatial_workflow_v31 import check_spatial_recipe

        check_spatial_recipe(result)
    return result


def output_rectangle(recipe, analysis_roi, shape):
    height, width = shape[:2]
    rectangle = (
        recipe.get("output_roi", analysis_roi)
        if recipe["crop"]
        else [0, 0, width, height]
    )
    x0, y0, x1, y1 = rectangle
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("输出裁切区域超出原始画面")
    return list(rectangle)


def freeze_render_request(recipe):
    """Bind a detached render recipe to source hashes without queuing a job."""
    from .bw_recipe import is_bw_recipe

    recipe = deepcopy(recipe)
    if not is_bw_recipe(recipe):
        recipe.setdefault("cool_tone", "soft-shoulder")
    recipe = checked_recipe(recipe)
    roles = ("BW",) if is_bw_recipe(recipe) else "RGB"
    inputs = [
        {"role": role, "path": path, "sha256": sha256(path)}
        for role, path in zip(roles, recipe["raw"])
    ]
    calibration = None
    if recipe["flatfield"]:
        path = str(Path(recipe["flatfield"]).resolve(strict=True))
        recipe["flatfield"] = path
        calibration = {"path": path, "sha256": sha256(path)}
    return dict(recipe=recipe, inputs=inputs, calibration=calibration, repairs=[])


class Workspace:
    def __init__(self, directory, create=False):
        self.root = Path(directory).resolve()
        self.path = self.root / "workspace.json"
        if create:
            self.root.mkdir(parents=True, exist_ok=False)
            self.data = {
                "schema": "pure-rgb-workspace",
                "version": 1,
                "jobs": [],
                "created": datetime.now(timezone.utc).isoformat(),
            }
            self.save()
        else:
            self.data = json.loads(self.path.read_text())
            if (
                self.data.get("schema") != "pure-rgb-workspace"
                or self.data.get("version") != 1
            ):
                raise ValueError("不支持的工作区版本")
            for job in self.data["jobs"]:
                checked_recipe(job["recipe"], verify_resources=False)
                if job["state"] in ("running", "cancelling"):
                    job.update(
                        state="interrupted",
                        error="上次处理被中断；可创建新任务重试，旧产物保留",
                    )
            self.save()

    def save(self):
        write_json(self.path, self.data)

    def add(self, recipe):
        job = {
            "id": uuid.uuid4().hex[:16],
            "state": "queued",
            **freeze_render_request(recipe),
        }
        self.data["jobs"].append(job)
        self.save()
        return job

    def job(self, identifier):
        return next(j for j in self.data["jobs"] if j["id"] == identifier)

    def set_state(self, identifier, state, **details):
        job = self.job(identifier)
        if job["state"] == "complete" and state != "complete":
            raise ValueError("已完成任务不可重解释；请创建新的配方版本")
        if state in ("running", "complete"):
            job.pop("error", None)
        job.update(state=state, **details)
        self.save()

    def sample(self, job):
        if job["state"] != "complete":
            raise ValueError("任务尚未完成")
        directory = self.root / "runs" / job["id"]
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest["state"] != "complete"
            or manifest["inputs"] != job["inputs"]
            or checked_recipe(manifest["recipe"], verify_resources=False)
            != checked_recipe(job["recipe"], verify_resources=False)
        ):
            raise ValueError("任务清单与冻结配方不一致")
        return Sample(
            job["id"],
            directory,
            tuple(manifest["inputs"]),
            tuple(manifest["roi"]),
            {f"baseline__{k}": v for k, v in manifest["results"].items()},
        )
