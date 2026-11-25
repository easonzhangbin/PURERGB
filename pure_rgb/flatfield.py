"""Single linear RAW → native sensor-layout gain, shared by import/render/preview."""

from dataclasses import replace
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from .native import sha256

RAW_EXTENSIONS = (
    ".rw2",
    ".dng",
    ".nef",
    ".nrw",
    ".arw",
    ".cr2",
    ".cr3",
    ".crw",
    ".raf",
    ".orf",
    ".pef",
    ".srw",
    ".raw",
    ".rwl",
    ".3fr",
    ".iiq",
    ".x3f",
)
FILE_FILTER = (
    "平场 RAW 或增益配置 ("
    + " ".join(f"*{s} *{s.upper()}" for s in RAW_EXTENSIONS)
    + " *.npz);;增益 NPZ (*.npz)"
)


def capture_info(path):
    from negpy.infrastructure.loaders.helpers import read_exif_from_file
    import rawpy

    exif = read_exif_from_file(str(path)) or {}

    def text(value):
        return (
            value.decode(errors="replace").strip("\x00 ")
            if isinstance(value, bytes)
            else str(value or "").strip()
        )

    def ratio(value):
        try:
            return (
                float(value[0] / value[1])
                if isinstance(value, (tuple, list))
                else float(value)
            )
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    with rawpy.imread(str(path)) as raw:
        sizes = raw.sizes
        shape = [int(sizes.height), int(sizes.width), 3]
        geometry = {
            "raw_shape": [int(sizes.raw_height), int(sizes.raw_width)],
            "visible_origin_raw": [int(sizes.left_margin), int(sizes.top_margin)],
        }
    return {
        "sensor_shape": shape,
        "raw_geometry": geometry,
        "make": text(exif.get("0th", {}).get(271)),
        "model": text(exif.get("0th", {}).get(272)),
        "aperture": ratio(exif.get("Exif", {}).get(33437)),
        "focal_length": ratio(exif.get("Exif", {}).get(37386)),
        "focal_length_35mm": ratio(exif.get("Exif", {}).get(41989)),
        "digital_zoom": ratio(exif.get("Exif", {}).get(41988)),
    }


def _sensor_shape(capture):
    value = capture.get("sensor_shape")
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(v) is not int or v < 1 for v in value) or value[2] != 3):
        raise ValueError("平场传感器画幅信息无效")
    return tuple(value)


def _visible_sensor_rectangle(capture):
    """A LibRaw visible rectangle in its *declared* RAW pixel canvas.

    Those coordinates may only be compared when both RAW canvases match. A
    smaller RAW canvas is not evidence that an APS-C frame covers a full sensor.
    """
    geometry = capture.get("raw_geometry")
    if geometry is None:
        return None
    if not isinstance(geometry, dict):
        raise ValueError("平场 RAW 画布元数据无效")
    raw_shape = geometry.get("raw_shape")
    origin = geometry.get("visible_origin_raw")
    if (not isinstance(raw_shape, (tuple, list)) or len(raw_shape) != 2
            or not isinstance(origin, (tuple, list)) or len(origin) != 2
            or any(type(v) is not int or v < 1 for v in raw_shape)
            or any(type(v) is not int or v < 0 for v in origin)):
        raise ValueError("平场 RAW 画布元数据不完整")
    height, width, _ = _sensor_shape(capture)
    left, top = origin
    if left + width > raw_shape[1] or top + height > raw_shape[0]:
        raise ValueError("平场传感器可见区域超出 RAW 画布")
    return tuple(raw_shape), (left, top, left + width, top + height)


def _crop_hint(capture):
    # EXIF's equivalent focal length is rounded; it can disprove equal coverage
    # but cannot establish a precise crop origin. Never infer a centered crop
    # from this ratio or from a smaller image alone.
    focal, equivalent = capture.get("focal_length"), capture.get("focal_length_35mm")
    values = []
    for value in (focal, equivalent):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            break
        values.append(value)
    factor = values[1] / values[0] if len(values) == 2 else None
    zoom = capture.get("digital_zoom")
    zoom = (max(1., float(zoom)) if isinstance(zoom, (int, float))
            and not isinstance(zoom, bool) and math.isfinite(zoom) and zoom >= 0 else None)
    return factor, zoom


def sensor_mapping(reference, target):
    """Bind a gain to target sensor coordinates without guessing crop positions.

    Same-field resolutions use normalized full-frame coordinates. Explicit
    LibRaw rectangles in an identical RAW canvas also support off-center crops.
    Aspect changes and conflicting EXIF crop hints require explicit geometry.
    """
    source_shape, target_shape = _sensor_shape(reference), _sensor_shape(target)
    source_geometry = _visible_sensor_rectangle(reference)
    target_geometry = _visible_sensor_rectangle(target)
    full = [0., 0., 1., 1.]
    region, method = full, "same-sensor-geometry"
    if source_geometry and target_geometry and source_geometry[0] == target_geometry[0]:
        sx0, sy0, sx1, sy1 = source_geometry[1]
        tx0, ty0, tx1, ty1 = target_geometry[1]
        if not sx0 <= tx0 < tx1 <= sx1 or not sy0 <= ty0 < ty1 <= sy1:
            raise ValueError("平场参考未覆盖照片的完整传感器区域，不能外推平场")
        region = [(tx0 - sx0) / (sx1 - sx0), (ty0 - sy0) / (sy1 - sy0),
                  (tx1 - sx0) / (sx1 - sx0), (ty1 - sy0) / (sy1 - sy0)]
        if region != full:
            method = "explicit-raw-canvas-crop"
    elif source_shape != target_shape:
        for a, b in zip(_crop_hint(reference), _crop_hint(target)):
            if ((a is not None and b is not None and abs(a / b - 1.) > .05)
                    or (a is None and b is not None and b > 1.1)
                    or (b is None and a is not None and a > 1.1)):
                raise ValueError("平场传感器画幅的 EXIF 裁切模式不同，缺少准确区域坐标，不能直接拉伸")
        source_aspect = source_shape[1] / source_shape[0]
        target_aspect = target_shape[1] / target_shape[0]
        # Permit decoder-border rounding (a few pixels), not a different aspect
        # crop. EXIF orientation is applied later and must not swap this canvas.
        if abs(target_aspect / source_aspect - 1.) > .005:
            raise ValueError("平场传感器画幅比例不同且缺少准确裁切坐标，不能直接拉伸")
        method = "normalized-full-sensor-resize"
    return {
        "version": 1,
        "method": method,
        "reference_sensor_shape": list(source_shape),
        "target_sensor_shape": list(target_shape),
        "reference_roi_normalized_xyxy": region,
    }


def compatibility(provenance, targets):
    reference = provenance.get("capture", {}) if provenance else {}
    targets = list(dict.fromkeys(targets))
    warnings = []
    if not reference:
        return {
            "checked": False,
            "warnings": ["旧增益配置没有相机/画幅信息，需自行核对拍摄条件。"],
        }
    mappings = []
    for path in targets:
        target = capture_info(path)
        def identifier(value):
            return " ".join(str(value or "").strip("\x00 ").casefold().split())

        camera_verified = all(identifier(capture.get(field))
                              for capture in (reference, target) for field in ("make", "model"))
        for field in ("make", "model"):
            if (
                identifier(reference.get(field))
                and identifier(target.get(field))
                and identifier(reference[field]) != identifier(target[field])
            ):
                raise ValueError("平场 RAW 与照片的相机型号不一致")
        mapping = sensor_mapping(reference, target)
        if mapping["reference_sensor_shape"] != mapping["target_sensor_shape"] and not camera_verified:
            raise ValueError("跨分辨率平场需确认相同相机型号；当前相机或传感器画幅元数据不完整")
        mappings.append({"path": str(path), **mapping, "camera_verified": camera_verified})
        for field, title in (("aperture", "光圈"), ("focal_length", "焦距")):
            a, b = reference.get(field), target.get(field)
            if a is not None and b is not None and abs(a - b) > 0.01:
                warning = f"{title}不同：平场 {a:g}，照片 {b:g}；适合对照，不能视为完全匹配的校正。"
                if warning not in warnings:
                    warnings.append(warning)
    return {"checked": bool(targets), "warnings": warnings, "sensor_mappings": mappings}


def gain_from_linear(reference):
    from negpy.features.flatfield.logic import (
        compute_gain,
        check_reference,
        check_copy,
        evenness,
    )

    reference = np.asarray(reference, dtype=np.float32)
    if (
        reference.ndim != 3
        or reference.shape[2] != 3
        or not np.isfinite(reference).all()
        or min(reference.shape[:2]) < 16
        or np.any(reference < 0)
    ):
        raise ValueError("需要有效的三通道线性平场图像")
    levels = np.quantile(reference[::8, ::8], 0.5, axis=(0, 1))
    if levels.min() < 0.005 or levels.min() / max(float(levels.max()), 1e-6) < 0.03:
        raise ValueError(
            "平场过暗或某通道信号不足；单张参考请使用均匀白光，不要只用一色照明"
        )
    gain = compute_gain(reference)
    check = check_reference(reference, gain)
    if check.clipped:
        raise ValueError("平场参考存在过曝通道，请降低曝光重新拍摄")
    if check.spread > 0.12 or check.color > 0.12:
        raise ValueError("参考校正后仍明显不均匀，请检查是否含底片、遮挡或明显纹理")
    return (
        gain,
        check_copy(reference),
        {
            "before": evenness(reference)._asdict(),
            "after": check._asdict(),
            "channel_medians": levels.tolist(),
            "gain_range": [float(gain.min()), float(gain.max())],
        },
    )


def read_gain(path):
    with np.load(path, allow_pickle=False) as profile:
        gain = np.asarray(profile["gain"], dtype=np.float32)
        if (
            str(profile.get("layout", "")) != "sensor"
            or gain.ndim != 3
            or gain.shape[2] != 3
            or not gain.size
            or not np.isfinite(gain).all()
            or np.any(gain <= 0)
        ):
            raise ValueError("需要有效的 sensor 布局三通道平场增益 NPZ")
        provenance = (
            json.loads(str(profile["pure_rgb_provenance"]))
            if "pure_rgb_provenance" in profile
            else None
        )
    return gain, provenance


def prepare_calibration(calibration, directory, targets=()):
    path = Path(calibration["path"]).resolve(strict=True)
    if sha256(path) != calibration["sha256"]:
        raise ValueError("平场源文件自选择/入队后发生变化")
    directory = Path(directory)
    if path.suffix.lower() == ".npz":
        gain_path, digest = path, calibration["sha256"]
    elif path.suffix.lower() in RAW_EXTENSIONS:
        if calibration.get("gain_path"):
            gain_path, digest = (
                Path(calibration["gain_path"]),
                calibration["gain_sha256"],
            )
            if sha256(gain_path) != digest:
                raise ValueError("已冻结的平场增益文件发生变化")
        else:
            from negpy.services.rendering.image_processor import ImageProcessor

            capture = capture_info(path)
            pixels, metadata = ImageProcessor(use_gpu=False)._decode_sensor_rgb(
                str(path), True, wb_override=[1, 1, 1, 1], highlight_mode=0
            )
            if pixels.dtype != np.uint16:
                raise ValueError("RAW 平场解码未返回原生线性 RGB16")
            gain, reference, quality = gain_from_linear(
                pixels.astype(np.float32) / 65535
            )
            capture["sensor_shape"] = list(pixels.shape)
            provenance = {
                "version": 1,
                "algorithm": "NegPy-0.68-compute_gain-sensor-v1",
                "source": {"path": str(path), "sha256": calibration["sha256"]},
                "capture": capture,
                "orientation": int(metadata.get("orientation", 1)),
                "quality": quality,
                "decode": "Full sensor RGB; linear gamma=1; neutral WB; fixed white level; user_flip=0; highlight reconstruction=0",
                "scope": "Single broadband reference; existing post-triplet-merge gain path, not per-LED calibration",
            }
            compatibility(provenance, targets)
            if sha256(path) != calibration["sha256"]:
                raise ValueError("平场 RAW 解码期间发生变化")
            directory.mkdir(parents=True, exist_ok=True)
            gain_path = directory / "flatfield-gain.npz"
            with gain_path.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    gain=gain,
                    layout="sensor",
                    reference=reference.astype(np.float16),
                    pure_rgb_provenance=json.dumps(
                        provenance, ensure_ascii=False, allow_nan=False
                    ),
                )
            digest = sha256(gain_path)
    else:
        raise ValueError("请选择单张相机 RAW 或平场增益 NPZ")
    gain, provenance = read_gain(gain_path)
    result = {
        "path": str(path),
        "sha256": calibration["sha256"],
        "gain_path": str(gain_path),
        "gain_sha256": digest,
        "kind": "single-raw-flatfield" if provenance else "existing-native-gain",
        "provenance": provenance,
        "compatibility": compatibility(provenance, targets),
    }
    return gain, result


def attach_calibration(config, calibration, directory, targets=(), strength=1.0):
    from negpy.features.flatfield.logic import set_gain_provider, flatfield_token
    from negpy.services.assets.flatfield import FlatFieldProfiles

    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not 0 <= strength <= 2
    ):
        raise ValueError("平场强度需为 0..2 的数")
    gain, report = prepare_calibration(calibration, directory, targets)
    if strength != 1.0:
        # Blend from the uncorrected source to NegPy's gain up to 100%.
        # Above 100%, amplify its exposure correction in stops. A 200%
        # correction squares both the brightening and attenuating gains.
        gain = (1.0 + strength * (gain - 1.0) if strength < 1.0
                else np.power(gain, strength))
        if not np.isfinite(gain).all() or np.any(gain <= 0):
            raise ValueError("此参考在指定强度下无法生成有限的正增益")
    mappings = report.get("compatibility", {}).get("sensor_mappings", [])
    if mappings:
        from .raw_flatfield import map_gain_to_sensor

        mapping = {k: v for k, v in mappings[0].items() if k not in ("path", "camera_verified")}
        if any({k: v for k, v in item.items() if k not in ("path", "camera_verified")} != mapping
               for item in mappings[1:]):
            raise ValueError("同一 RGB 组的传感器区域不同，无法共用平场")
        gain = map_gain_to_sensor(gain, mapping)
        original = report.get("provenance")
        if original and (mapping["reference_sensor_shape"] != mapping["target_sensor_shape"]
                         or mapping["reference_roi_normalized_xyxy"] != [0., 0., 1., 1.]):
            applied = deepcopy(original)
            applied["capture"]["sensor_shape"] = mapping["target_sensor_shape"]
            # The original frozen reference remains authoritative. This copy
            # binds the in-process gain to its target for existing render/cache
            # consumers; it must not masquerade as the original capture geometry.
            applied["capture"].pop("raw_geometry", None)
            applied["application_mapping"] = mapping
            report = {**report, "reference_provenance": original, "provenance": applied}
        report = {**report, "sensor_mapping": mapping}
    report = {**report, "strength": float(strength),
              "strength_curve": "linear-blend" if strength <= 1.0 else "exposure-stops-v1"}
    set_gain_provider(FlatFieldProfiles.load_gain)
    identifier = FlatFieldProfiles.import_gain(
        gain, name="PURE RGB flat field", source=calibration["path"]
    )
    config = replace(
        config, flatfield=replace(config.flatfield, apply=True, profile_id=identifier)
    )
    token = flatfield_token(config.flatfield)
    if not token:
        raise ValueError("平场增益未能进入原生解码器")
    return config, {**report, "applied_token": token}
