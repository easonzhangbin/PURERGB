"""Read local Adobe LCP optical data without copying Adobe's profile library.

Only perspective models are supported. DCP camera colour profiles and vignette
models are deliberately outside this linear trichrome optics adapter.
"""

from functools import lru_cache
import hashlib
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def local_profile_roots():
    return [p for p in (
        Path.home() / "Library/Application Support/Adobe/CameraRaw/LensProfiles",
        Path("/Library/Application Support/Adobe/CameraRaw/LensProfiles"),
        Path("/Applications/Adobe Lightroom Classic/Adobe Lightroom Classic.app/Contents/Resources/LensProfiles"),
    ) if p.is_dir()]


def profile_binding(path):
    path = Path(path).expanduser().resolve(strict=True)
    if path.suffix.casefold() != ".lcp" or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("请选择不超过 8 MB 的 LCP 镜头配置")
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _attrs(element):
    return {k.rsplit("}", 1)[-1]: v for k, v in element.attrib.items()}


def _number(values, key, default=None):
    value = float(values[key]) if key in values else default
    if value is not None and not math.isfinite(value):
        raise ValueError("LCP 包含无效数值")
    return value


def _model(element, focal, crop):
    values = _attrs(element)
    description = next((c for c in element if c.tag.endswith("}Description")), None)
    if description is not None:
        values.update(_attrs(description))
    version = _number(values, "Version", 2)
    if version not in (1, 2):
        raise ValueError("此 LCP 的透视模型版本暂不支持")
    fx = _number(values, "FocalLengthX", focal * crop / 35)
    fy = _number(values, "FocalLengthY", fx)
    if not (0.01 <= fx <= 100 and 0.01 <= fy <= 100):
        raise ValueError("LCP 焦距标度无效")
    center = [_number(values, "ImageXCenter", 0.5), _number(values, "ImageYCenter", 0.5)]
    scale = _number(values, "ScaleFactor", 1.0)
    radial = [_number(values, f"RadialDistortParam{i}", 0.0) for i in (1, 2, 3)]
    tangential = [_number(values, f"TangentialDistortParam{i}", 0.0) for i in (1, 2)]
    # Telephoto LCPs legitimately have large high-order coefficients because
    # their normalized radii are tiny. Validate the resulting map as well.
    if any(abs(v) > 1e12 for v in radial + tangential) or not 0.25 < scale < 4:
        raise ValueError("LCP 模型系数超出支持范围")
    if any(not -0.25 <= v <= 1.25 for v in center):
        raise ValueError("LCP 光学中心超出支持范围")
    return dict(fx=fx, fy=fy, center=center, scale=scale, radial=radial, tangential=tangential)


@lru_cache(maxsize=24)
def _load(path, size, mtime, expected_sha):
    data = Path(path).read_bytes()
    if len(data) > 8 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("LCP 文件已改变，请重新选择配置")
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("LCP 不允许外部实体")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as error:
        raise ValueError("无法读取 LCP XML") from error
    samples = []
    metadata = None
    for element in root.iter():
        attrs = _attrs(element)
        if "FocalLength" not in attrs or "CameraRawProfile" not in attrs:
            continue
        if attrs["CameraRawProfile"].casefold() != "true":
            continue
        perspective = next((c for c in element if c.tag.endswith("}PerspectiveModel")), None)
        if perspective is None:
            continue
        focal, crop = _number(attrs, "FocalLength"), _number(attrs, "SensorFormatFactor", 1.0)
        if not 0 < focal <= 2000 or not 0.1 < crop <= 10:
            continue
        distortion = _model(perspective, focal, crop)
        children = {e.tag.rsplit("}", 1)[-1]: e for e in perspective.iter()}
        names = ("ChromaticRedGreenModel", "ChromaticGreenModel", "ChromaticBlueGreenModel")
        ca = [_model(children[n], focal, crop) for n in names] if all(n in children for n in names) else None
        samples.append(dict(focal=focal, focus=_number(attrs, "FocusDistance"),
                            aperture_value=_number(attrs, "ApertureValue"),
                            distortion=distortion, ca=ca))
        if metadata is None:
            metadata = {k: attrs.get(k, "") for k in (
                "Make", "Model", "UniqueCameraModel", "Lens", "LensPrettyName", "ProfileName",
                "ImageWidth", "ImageLength", "SensorFormatFactor")}
    if not samples:
        raise ValueError("此 LCP 没有受支持的 RAW 透视镜头模型（鱼眼暂不支持）")
    return dict(path=path, sha256=expected_sha, metadata=metadata, samples=samples,
                has_ca=any(s["ca"] for s in samples), has_distortion=True)


def read_lcp(binding):
    binding = profile_binding(binding) if isinstance(binding, (str, Path)) else binding
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        raise ValueError("LCP 配置引用无效")
    path = Path(binding["path"]).expanduser().resolve(strict=True)
    stat = path.stat()
    if path.suffix.casefold() != ".lcp" or stat.st_size > 8 * 1024 * 1024:
        raise ValueError("无效的 LCP 文件")
    return _load(str(path), stat.st_size, stat.st_mtime_ns, binding.get("sha256"))


def _normal(value):
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def capture_lens(paths):
    from negpy.infrastructure.loaders.helpers import read_exif_from_file

    def text(value):
        return value.decode(errors="replace").strip("\x00 ") if isinstance(value, bytes) else str(value or "")

    def ratio(value):
        if isinstance(value, tuple) and len(value) == 2 and value[1]:
            return value[0] / value[1]
        return float(value) if isinstance(value, (float, int)) else None

    captures = []
    for path in paths:
        exif = read_exif_from_file(str(path)) or {}
        primary, values = exif.get("0th", {}), exif.get("Exif", {})
        captures.append(dict(make=text(primary.get(271)), model=text(primary.get(272)),
                             lens=text(values.get(42036)), focal=ratio(values.get(37386)),
                             focal35=ratio(values.get(41989)), focus=ratio(values.get(37382))))
    if not captures or any(c != captures[0] for c in captures[1:]):
        raise ValueError("RGB 三张照片的镜头或拍摄参数不一致")
    return captures[0]


def find_matching_lcp(paths, *, roots=None):
    """Auto-select only an unambiguous exact lens/camera identity; no guesses."""
    capture = capture_lens(paths)
    lens, camera = _normal(capture["lens"]), _normal(capture["model"])
    if not lens:
        return None
    matches = []
    seen = set()
    for root in local_profile_roots() if roots is None else roots:
        for path in Path(root).rglob("*"):
            if path.suffix.casefold() != ".lcp" or path.resolve() in seen:
                continue
            seen.add(path.resolve())
            # Catalog filenames contain the lens name; avoid parsing the whole
            # proprietary library for every input. A manual choice remains possible.
            filename = _normal(path.stem)
            if lens not in filename:
                continue
            try:
                binding = profile_binding(path)
                profile = read_lcp(binding)
            except (ValueError, OSError):
                continue
            meta = profile["metadata"]
            if lens not in {_normal(meta["Lens"]), _normal(meta["LensPrettyName"])}:
                continue
            if meta["Model"] and _normal(meta["Model"]) != camera:
                continue
            if meta["Make"] and _normal(meta["Make"]) != _normal(capture["make"]):
                continue
            matches.append(binding)
    # Duplicate installations are harmless if the XML contents are identical.
    unique = {m["sha256"]: m for m in matches}
    return next(iter(unique.values())) if len(unique) == 1 else None


def _blend(a, b, fraction):
    return {key: ([(1 - fraction) * x + fraction * y for x, y in zip(a[key], b[key])]
                  if isinstance(a[key], list) else (1 - fraction) * a[key] + fraction * b[key])
            for key in a}


def select_models(profile, focal, *, focus=None, ca=False):
    key = "ca" if ca else "distortion"
    samples = [s for s in profile["samples"] if s[key] is not None]
    if not samples:
        return None
    focals = sorted({s["focal"] for s in samples})
    if focal is None:
        if len(focals) != 1:
            raise ValueError("镜头未记录焦距，无法匹配变焦 LCP")
        focal = focals[0]
    if focal < focals[0] * 0.95 or focal > focals[-1] * 1.05:
        raise ValueError("拍摄焦距超出所选 LCP 的校准范围")
    low = max((f for f in focals if f <= focal), default=focals[0])
    high = min((f for f in focals if f >= focal), default=focals[-1])

    def at(f):
        candidates = [s for s in samples if s["focal"] == f]
        # Focus is often absent in EXIF. Use the nearest-distance calibration
        # sample, report that choice, and never fabricate a macro distance.
        target = focus if focus and focus > 0 else min((s["focus"] for s in candidates if s["focus"]), default=1)
        sample = min(candidates, key=lambda s: abs(math.log(max(s["focus"] or target, 1e-6) / target)))
        return sample[key]

    a, b = at(low), at(high)
    fraction = 0 if low == high else (math.log(focal) - math.log(low)) / (math.log(high) - math.log(low))
    models = [_blend(x, y, fraction) for x, y in zip(a, b)] if ca else _blend(a, b, fraction)
    return dict(models=models, focal_bracket=[low, high], focal=focal, focus=focus,
                focus_policy="nearest-recorded-distance" if focus else "nearest-calibrated-distance")


def _coordinates(model, x, y, shape, *, base=None):
    h, w = shape[:2]
    basis = base or model
    fx, fy = basis["fx"] * max(h, w), basis["fy"] * max(h, w)
    nx, ny = (x - basis["center"][0] * w) / fx, (y - basis["center"][1] * h) / fy
    r2 = nx * nx + ny * ny
    k1, k2, k3 = model["radial"]
    p1, p2 = model["tangential"]
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    dx = nx * radial + 2 * p1 * nx * ny + p2 * (r2 + 2 * nx * nx)
    dy = ny * radial + p1 * (r2 + 2 * ny * ny) + 2 * p2 * nx * ny
    scale = model["scale"] if base else 1.0
    return (dx * scale * model["fx"] * max(h, w) + model["center"][0] * w,
            dy * scale * model["fy"] * max(h, w) + model["center"][1] * h)


def apply_lcp(source, binding, capture, plan, effective, orientation, *, ca=True, distortion=False,
              registration=None):
    """Full visible sensor coordinates, then registered crop and EXIF orientation.

    CA-only maps red/blue relative to green; G is copied bit-exactly. Unsupported
    edge reads keep their original samples, never introduce black/white borders.
    """
    from negpy.kernel.image.logic import apply_exif_orientation

    profile = read_lcp(binding)
    metadata = profile["metadata"]
    if metadata["Model"] and _normal(metadata["Model"]) != _normal(capture.get("model")):
        raise ValueError("所选 LCP 的相机型号与照片不匹配")
    profile_crop = float(metadata.get("SensorFormatFactor") or 1.0)
    if capture.get("focal") and capture.get("focal35"):
        if abs(capture["focal35"] / capture["focal"] / profile_crop - 1) > 0.08:
            raise ValueError("所选 LCP 的传感器覆盖范围与照片不匹配")
    selected_ca = select_models(profile, capture.get("focal"), focus=capture.get("focus"), ca=True) if ca else None
    selected_distortion = select_models(profile, capture.get("focal"), focus=capture.get("focus")) if distortion else None
    if not selected_ca and not selected_distortion:
        return source, dict(status="unchanged", method="local-adobe-lcp", reason="LCP 没有横向色差数据")
    inverse = {6: 8, 8: 6}.get(orientation, orientation)
    unrotated = np.ascontiguousarray(apply_exif_orientation(source, inverse))
    h, w = unrotated.shape[:2]
    sensor_shape = plan.source_shape[:2]
    if metadata.get("ImageWidth") and metadata.get("ImageLength"):
        profile_aspect = float(metadata["ImageWidth"]) / float(metadata["ImageLength"])
        if abs(sensor_shape[1] / sensor_shape[0] / profile_aspect - 1) > 0.03:
            raise ValueError("所选 LCP 的传感器比例与照片不匹配")
    left, top, _, _ = effective.visible_roi
    x = np.arange(w, dtype=np.float32)[None, :] + left
    result = unrotated.copy()
    for channel in range(3):
        if channel == 1 and not selected_distortion:
            continue
        plane = np.ascontiguousarray(unrotated[..., channel])
        for start in range(0, h, 256):
            stop = min(start + 256, h)
            y = np.arange(start, stop, dtype=np.float32)[:, None] + top
            mx, my = np.broadcast_arrays(x, y)
            if selected_distortion:
                mx, my = _coordinates(selected_distortion["models"], mx, my, sensor_shape)
            if selected_ca and channel != 1:
                models = selected_ca["models"]
                shift = ((registration or {}).get("channels", {}).get(("red", "green", "blue")[channel], {})
                         .get("shift_xy", [0.0, 0.0]))
                mx, my = _coordinates(models[channel], mx + shift[0], my + shift[1], sensor_shape, base=models[1])
                mx, my = mx - shift[0], my - shift[1]
            mx = np.ascontiguousarray(mx - left, dtype=np.float32)
            my = np.ascontiguousarray(my - top, dtype=np.float32)
            warped = cv2.remap(plane, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            valid = (mx >= 2) & (mx <= w - 3) & (my >= 2) & (my <= h - 3)
            if not np.isfinite(mx).all() or not np.isfinite(my).all():
                raise ValueError("LCP 产生无效采样坐标")
            if np.max(np.abs(mx - (x - left))) > max(sensor_shape) * 0.5 or np.max(np.abs(my - (y - top))) > max(sensor_shape) * 0.5:
                raise ValueError("LCP 采样位移超出此画幅的支持范围")
            result[start:stop, :, channel] = np.where(valid, warped, plane[start:stop])
    receipt = dict(version=1, status="applied", method="local-adobe-lcp", profile=binding,
                   profile_name=profile["metadata"]["ProfileName"], lateral_ca=bool(selected_ca),
                   distortion=bool(selected_distortion), green_unchanged=not bool(selected_distortion),
                   canvas_unchanged=True, ca_model=selected_ca, distortion_model=selected_distortion,
                   sensor_xyxy=list(effective.visible_roi), policy="registered-linear-after-flatfield")
    return np.ascontiguousarray(apply_exif_orientation(result, orientation)), receipt
