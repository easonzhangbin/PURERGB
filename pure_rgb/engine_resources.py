"""Hash-bound engine resource contracts; inspect all fields before execution."""

import hashlib
from pathlib import Path
import plistlib
from datetime import datetime

import numpy as np


def _describe(value):
    if isinstance(value, bytes):
        return {
            "type": "binary",
            "length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("配置包含非有限数值")
    if isinstance(value, dict):
        return {k: _describe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_describe(v) for v in value]
    return value


def checked_resource(document):
    if not isinstance(document, dict) or set(document) != {"path", "sha256"}:
        raise ValueError("引擎资源需 path 和 sha256")
    if not isinstance(document["path"], str) or not document["path"]:
        raise ValueError("引擎资源路径不能为空")
    digest = document["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("无效引擎资源 SHA256")
    return {"path": str(Path(document["path"]).resolve()), "sha256": digest}


def inventory_flex_profile(path):
    """Read genuine factory fields without inventing executable meanings.

    In particular real EndPoints=[1,1] must not be treated as an invalid
    black/white pair or silently changed to [0,255]. The strict legacy adapter
    inspector below remains separate until the factory field routing is proven.
    """
    path = Path(path).resolve(strict=True)
    payload = path.read_bytes()
    if len(payload) > 20_000_000:
        raise ValueError("胶片配置过大")
    data = plistlib.loads(payload)
    setting = data.get("ImageSetting", {}) if isinstance(data, dict) else {}
    correction = setting.get("ImageCorrection")
    if not isinstance(correction, dict):
        raise ValueError("缺少 ImageSetting.ImageCorrection")
    from .flex_curves import point_curve_lut

    curves = []
    for index, item in enumerate(correction.get("Gradations", [])):
        points = [
            [p.get("X"), p.get("Y"), p.get("DY", 1)] for p in item.get("Points", [])
        ]
        record = {"index": index, "points": points}
        try:
            table = point_curve_lut(points, bits=14)
            record.update(
                scalar_kernel_executable=True,
                table_sha256=hashlib.sha256(table.tobytes()).hexdigest(),
                channel_routing_verified=False,
            )
        except ValueError as error:
            record.update(scalar_kernel_executable=False, error=str(error))
        curves.append(record)
    return {
        "schema": "pure-rgb-factory-profile-inventory",
        "version": 1,
        "resource": {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()},
        "fields": _describe(correction),
        "curves": curves,
        "original_endpoints": _describe(correction.get("EndPoints")),
        "original_endpoints_semantics": "Two extension policies: 0=hold endpoint, 1=extend linearly to domain limits, 2=hard black/white; not input black/white code positions",
        "original_curve_graph": "Per-channel aggregate includes shared Gradations[0], adds [1+c] and subtracts [4+c] at a saved reference; intervening gamma/negative/contrast/gray stages must also be reconstructed",
        "complete_factory_route_executable": False,
        "scope": "Original fields retained verbatim; independent point-curve evaluation is not full factory routing, histogram, device LUT, or CMS validation",
    }


def inspect_flex_profile(path, expected_sha256=None):
    path = Path(path).resolve(strict=True)
    payload = path.read_bytes()
    if len(payload) > 20_000_000:
        raise ValueError("胶片配置过大")
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("FlexColor 胶片配置已变化，拒绝复用旧配方")
    data = plistlib.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("无效胶片配置 plist")
    setting = data.get("ImageSetting")
    if not isinstance(setting, dict) or not isinstance(
        setting.get("ImageCorrection"), dict
    ):
        raise ValueError("缺少 ImageSetting.ImageCorrection")
    ic = setting["ImageCorrection"]
    curves = ic.get("Gradations", [])
    if not isinstance(curves, list) or len(curves) not in (1, 3, 7):
        raise ValueError("Gradations 需 1、3 或 7 条曲线")
    parsed = []
    nondefault_dy = []
    for index, curve in enumerate(curves):
        points = curve.get("Points") if isinstance(curve, dict) else None
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError("曲线至少需要两个点")
        p = np.asarray(
            [
                [point.get("X", 0), point.get("Y", 0), point.get("DY", 1)]
                for point in points
            ],
            dtype=float,
        )
        if (
            not np.isfinite(p).all()
            or np.any(np.diff(p[:, 0]) <= 0)
            or np.any(p[:, :2] < 0)
            or np.any(p[:, :2] > 255)
        ):
            raise ValueError("曲线坐标需有限、X 递增、X/Y 在 0..255")
        if np.any(p[:, 2] != 1):
            nondefault_dy.append(index)
        parsed.append(p.tolist())
    gamma = float(ic.get("Gamma", 2))
    endpoints = np.asarray(ic.get("EndPoints", (0, 255)), dtype=float)
    if not np.isfinite(gamma) or not 0.5 <= gamma <= 4:
        raise ValueError("当前适配器支持负片 Gamma 0.5..4")
    if (
        endpoints.shape != (2,)
        or not np.isfinite(endpoints).all()
        or not 0 <= endpoints[0] < endpoints[1] <= 255
    ):
        raise ValueError(
            "当前研究适配器的端点需 0≤下端点<上端点≤255；"
            "原厂 EndPoints 是外延策略，不能直接当成黑白码值运行"
        )
    contrast, brightness = float(ic.get("Contrast", 0)), float(ic.get("Brightness", 0))
    if (
        not np.isfinite([contrast, brightness]).all()
        or not -80 <= contrast <= 100
        or not -64 <= brightness <= 64
    ):
        raise ValueError("胶片亮度或反差超出当前支持范围")
    unsupported = []
    for flag in ("ApplyCC", "ApplyHistogram", "ApplyDust", "ApplyUSM", "Convert"):
        if ic.get(flag):
            unsupported.append(flag)
    for field in ("Saturation", "Lightness", "AutoHighlight", "AutoShadow"):
        if ic.get(field, 0):
            unsupported.append(field)
    if ic.get("FilmType", 1) != 1:
        unsupported.append("non-color-negative FilmType")
    if ic.get("ApplyCurves", True):
        if nondefault_dy:
            unsupported.append("non-default DY interpolation")
        if any(any(abs(x - y) > 1e-9 for x, y, dy in p) for p in parsed[3:]):
            unsupported.append("non-identity extra Gradations[3:7]")
    known = {
        "Gamma",
        "Gradations",
        "EndPoints",
        "Contrast",
        "Brightness",
        "ApplyCurves",
        "ApplySliders",
        "FilmType",
        "Mode",
        "ApplyCC",
        "ApplyHistogram",
        "ApplyDust",
        "ApplyUSM",
        "Convert",
        "Saturation",
        "Lightness",
        "AutoHighlight",
        "AutoShadow",
        "ColorCorr",
        "ColorCor",
        "DotColor",
        "Gray",
        "Highlight",
        "Shadow",
        "USMAmount",
        "USMRadius",
        "USMDarkLimit",
        "USMNoiseLimit",
        "USMColFactor",
        "InputProfile",
        "RGBProfile",
        "CMYKProfile",
        "GrayProfile",
    }
    unknown = sorted(set(ic) - known)
    if unknown:
        unsupported.append("unknown fields: " + ", ".join(unknown))
    fields = {
        key: {
            "value": _describe(value),
            "execution": (
                "supported-reconstruction"
                if key
                in {
                    "Gamma",
                    "Gradations",
                    "EndPoints",
                    "Contrast",
                    "Brightness",
                    "ApplyCurves",
                    "ApplySliders",
                }
                else "metadata-or-unimplemented; not executed"
            ),
        }
        for key, value in ic.items()
    }
    report = {
        "schema": "pure-rgb-flex-profile-inspection",
        "version": 1,
        "resource": {"path": str(path), "sha256": digest},
        "fields": fields,
        "curves": parsed,
        "curve_count": len(parsed),
        "unsupported_active_stages": unsupported,
        "executable": not unsupported,
        "defaults": {
            "film_gamma": gamma,
            "brightness": brightness if ic.get("ApplySliders", True) else 0.0,
            "contrast": contrast if ic.get("ApplySliders", True) else 0.0,
            "black_code": float(endpoints[0]),
            "white_code": float(endpoints[1]),
        },
        "scope": "Public linear-interpolation reconstruction, not factory DY/ColorCor/device/CMS validation",
    }
    return ic, report


def load_flex_resource(document):
    resource = checked_resource(document)
    return inspect_flex_profile(resource["path"], resource["sha256"])


def require_flex_profile(document):
    ic, report = load_flex_resource(document)
    if not report["executable"]:
        raise ValueError(
            "配置启用尚无可验证执行实现的阶段："
            + "；".join(report["unsupported_active_stages"])
        )
    return ic, report
