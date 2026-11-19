"""Single linear RAW → native sensor-layout gain, shared by import/render/preview."""

from dataclasses import replace
import json
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
        shape = [int(raw.sizes.height), int(raw.sizes.width), 3]
    return {
        "sensor_shape": shape,
        "make": text(exif.get("0th", {}).get(271)),
        "model": text(exif.get("0th", {}).get(272)),
        "aperture": ratio(exif.get("Exif", {}).get(33437)),
        "focal_length": ratio(exif.get("Exif", {}).get(37386)),
    }


def compatibility(provenance, targets):
    reference = provenance.get("capture", {}) if provenance else {}
    warnings = []
    if not reference:
        return {
            "checked": False,
            "warnings": ["旧增益配置没有相机/画幅信息，需自行核对拍摄条件。"],
        }
    for path in dict.fromkeys(targets):
        target = capture_info(path)
        if target["sensor_shape"] != reference["sensor_shape"]:
            raise ValueError("平场 RAW 与照片的传感器画幅不一致")
        for field in ("make", "model"):
            if (
                reference.get(field)
                and target.get(field)
                and reference[field].casefold() != target[field].casefold()
            ):
                raise ValueError("平场 RAW 与照片的相机型号不一致")
        for field, title in (("aperture", "光圈"), ("focal_length", "焦距")):
            a, b = reference.get(field), target.get(field)
            if a is not None and b is not None and abs(a - b) > 0.01:
                warning = f"{title}不同：平场 {a:g}，照片 {b:g}；适合对照，不能视为完全匹配的校正。"
                if warning not in warnings:
                    warnings.append(warning)
    return {"checked": bool(targets), "warnings": warnings}


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
