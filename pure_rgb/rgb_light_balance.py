"""Optional film-base exposure advice; never changes an image or its recipe.

The meter reads black-subtracted CFA photosites, normalized against the same
camera metadata for every frame. An optional validated flat-field gain is
sampled at each photosite before statistics, in unrotated sensor coordinates.
There is no white balance, demosaic, gamma, per-frame normalization or inversion.
The preview is a separate, fixed-scale sensor-RGB decode. Recommendations are
effective exposure ratios, not calibrated light-driver/knob percentages.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Mapping, Sequence

import numpy as np

CHANNELS = ("R", "G", "B")
SAFE_TARGET = 0.90
MIN_SAMPLES = 64
MIN_SIGNAL = 0.01
MAX_SATURATION_FRACTION = 0.002
MAX_UNIFORMITY_RANGE = 0.15
DEFAULT_REGION = (0.4, 0.4, 0.6, 0.6)
FULL_REGION = (0.0, 0.0, 1.0, 1.0)


def checked_region(region):
    """Fractions in the standard-cropped, unrotated RAW sensor canvas."""
    if not isinstance(region, (tuple, list)) or len(region) != 4:
        raise ValueError("请选择有效片基范围。")
    if any(isinstance(v, bool) or not isinstance(v, (int, float, np.number)) for v in region):
        raise ValueError("片基范围须为有限数值。")
    x0, y0, x1, y1 = (float(v) for v in region)
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or not (
        0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    ):
        raise ValueError("片基范围超出 RAW 画面。")
    return x0, y0, x1, y1


def region_pixels(region, width, height):
    x0, y0, x1, y1 = checked_region(region)
    left, top = int(math.floor(x0 * width)), int(math.floor(y0 * height))
    right = min(width, max(left + 1, int(math.ceil(x1 * width))))
    bottom = min(height, max(top + 1, int(math.ceil(y1 * height))))
    return left, top, right, bottom


def ordered_paths(paths):
    if isinstance(paths, Mapping):
        values = [paths.get(c, "") for c in CHANNELS]
    elif isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)):
        values = list(paths)
    else:
        values = []
    if len(values) != 3 or any(not str(p).strip() for p in values):
        raise ValueError("请选择红、绿、蓝各一张片基 RAW。")
    result = tuple(str(Path(p).expanduser().resolve()) for p in values)
    if len(set(result)) != 3:
        raise ValueError("红、绿、蓝须使用三张不同的曝光 RAW。")
    return result


def imbalance_grade(percent):
    if not math.isfinite(percent) or percent < 0:
        raise ValueError("Invalid imbalance")
    if percent <= 5:
        return "最佳"
    if percent <= 10:
        return "尚可"
    if percent <= 15:
        return "需调整"
    return "较差"


@dataclass(frozen=True)
class CaptureSettings:
    camera: tuple[str, str]
    iso: float | None
    shutter: float | None
    aperture: float | None
    shape: tuple[int, int]


@dataclass(frozen=True)
class ChannelReading:
    channel: str
    path: str
    settings: CaptureSettings
    sample_count: int
    signal: float | None
    median: float | None
    saturation_fraction: float
    uniformity_range: float | None
    issues: tuple[str, ...] = ()
    preview: np.ndarray | None = field(default=None, compare=False, repr=False)
    flatfield: dict | None = None
    raw_signal: float | None = None
    raw_limit_signal: float | None = None
    raw_saturation_fraction: float | None = None
    raw_channel_limits: tuple[dict, ...] = ()
    adjustment_issue: str | None = None


@dataclass(frozen=True)
class BalanceAnalysis:
    region: tuple[float, float, float, float]
    readings: tuple[ChannelReading, ...]
    issues: tuple[str, ...]
    imbalance_percent: float | None
    adjustments_percent: tuple[float | None, ...]
    target: float = SAFE_TARGET
    flatfield: dict | None = None

    @property
    def usable(self):
        return not self.issues and all(not item.issues for item in self.readings)

    @property
    def grade(self):
        return imbalance_grade(self.imbalance_percent) if self.usable else "参考待核对"

    def to_dict(self):
        readings = []
        for item in self.readings:
            readings.append({key: value for key, value in asdict(item).items() if key != "preview"})
        return {
            "schema": "rgb-film-base-exposure-v1", "region_xyxy_fraction": self.region,
            "coordinate_space": "standard-cropped RAW sensor, before EXIF rotation",
            "target_fraction": self.target, "usable": self.usable, "grade": self.grade,
            "imbalance_percent": self.imbalance_percent,
            "effective_exposure_adjustments_percent": self.adjustments_percent,
            "readings": readings, "issues": self.issues,
            "flatfield": self.flatfield,
            "measurement_basis": "平场后" if self.flatfield and self.flatfield.get("applied") else "未套用平场",
            "physical_light_power_calibrated": False,
        }

    def report_text(self):
        lines = ["PURE RGB · RGB 拍摄灯光配平", "参考：同卷已冲洗、无画面的片基；三张使用相同 ISO、快门和光圈。"]
        lines.append("范围：" + ("全幅" if self.region == FULL_REGION else "局部片基 " + ", ".join(f"{v:.4f}" for v in self.region)))
        corrected = bool(self.flatfield and self.flatfield.get("applied"))
        lines.append(f"量化目标：{'平场后采样信号' if corrected else '有效 RAW 范围'} {100 * self.target:.0f}%")
        if self.flatfield and self.flatfield.get("applied"):
            lines.append(f"量化基准：平场后；{self.flatfield['name']}；强度 {100 * self.flatfield['strength']:g}%")
            lines.append(f"平场：{self.flatfield['path']}；SHA-256：{self.flatfield['sha256']}")
            lines.append("通道占用与均匀性按逐位置平场后的信号计算；RAW 上限检查仍使用未经平场的原始感光点。")
        else:
            lines.append("量化基准：未套用平场")
        lines.append("当前不平衡：" + (f"{self.imbalance_percent:.2f}% · {self.grade}" if self.usable else "参考待核对"))
        for reading, adjustment in zip(self.readings, self.adjustments_percent):
            basis = "平场后信号" if reading.flatfield and reading.flatfield.get("applied") else "RAW 占用"
            level = "无法测量" if reading.signal is None else f"{basis} {100 * reading.signal:.2f}%"
            advice = "暂不估算" if adjustment is None else f"{adjustment:+.2f}%"
            lines.extend((f"{reading.channel}：{level}；建议有效曝光 {advice}", f"  文件：{reading.path}"))
            s = reading.settings
            lines.append(f"  相机：{' '.join(s.camera)}；ISO {s.iso}；快门 {s.shutter} s；光圈 f/{s.aperture}")
            lines.append(f"  样本 {reading.sample_count}；接近上限 {100 * reading.saturation_fraction:.3f}%")
            if reading.raw_limit_signal is not None:
                lines.append(f"  原始 RAW 全感光通道最高占用：{100 * reading.raw_limit_signal:.2f}%")
            lines.extend(f"  提示：{issue}" for issue in reading.issues)
            if reading.adjustment_issue:
                lines.append(f"  建议限制：{reading.adjustment_issue}")
        lines.extend(f"提示：{issue}" for issue in self.issues)
        lines.extend((
            "分级：≤5% 最佳；>5–10% 尚可；>10–15% 需调整；>15% 较差。",
            "这些等级是本软件的配平提示，不是论文规定的精度标准。",
            "不平衡比较三路与其中位值的最大偏差；建议增减按各路到 90% 目标的曝光比计算。",
            "建议数值是有效曝光估算，不等于灯具旋钮或物理光功率百分比；调整灯光后请复拍确认。",
            "均匀性检查不能证明所选区域确为片基。此分析不改变照片，也不影响其他操作。",
            "依据：RAW 信号取决于照明、胶片透射与传感器响应；胶片理论光谱曲线不能代替本次 RAW 实测。",
            "Kodak VISION3 500T 技术资料：https://www.kodak.com/content/pdfs/motion/KODAK-VISION3-500T-5219-7219-technical-information.pdf",
            "Flueckiger 等，2018，Film Material–Scanner Interaction：https://scan2screen.com/wp-content/uploads/2026/06/flueckigeretal_investigationfilmmaterialscannerinteraction_2018_v_1-1c.pdf",
        ))
        return "\n".join(lines) + "\n"


def _positive(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (ValueError, TypeError):
        return None


def settings_issues(readings):
    settings = [item.settings for item in readings]
    issues = []
    if any(not all(s.camera) for s in settings):
        issues.append("相机型号信息不完整，请核对三张是否来自同一相机。")
    elif any(s.camera != settings[0].camera for s in settings[1:]):
        issues.append("三张 RAW 的相机型号不同，请重新选择同机位参考。")
    if any(s.shape != settings[0].shape for s in settings[1:]):
        issues.append("三张 RAW 的画幅不同，无法共用同一片基范围。")
    for key, label in (("iso", "ISO"), ("shutter", "快门"), ("aperture", "光圈")):
        values = [getattr(s, key) for s in settings]
        if any(value is None for value in values):
            issues.append(f"{label}信息不完整，请用相同拍摄设置重新拍摄参考。")
        elif any(not math.isclose(values[0], value, rel_tol=1e-4) for value in values[1:]):
            issues.append(f"三张 RAW 的{label}不同，请使用相同拍摄设置。")
    return tuple(issues)


def measure_cfa(mosaic, colors, black, white, positions, region=FULL_REGION, *, gain=None):
    """Measure actual photosites, including both green sites; no interpolation."""
    mosaic, colors = np.asarray(mosaic), np.asarray(colors)
    if mosaic.ndim != 2 or colors.shape != mosaic.shape:
        raise ValueError("需要包含原始感光点数据的相机 RAW。")
    height, width = mosaic.shape
    x0, y0, x1, y1 = region_pixels(region, width, height)
    patch, pattern = mosaic[y0:y1, x0:x1], colors[y0:y1, x0:x1]
    if gain is not None:
        gain = np.asarray(gain)
        if (gain.shape != patch.shape or gain.dtype.kind != "f"
                or not np.isfinite(gain).all() or np.any(gain <= 0)):
            raise ValueError("片基平场增益必须对应选区的每个传感器位置，且为有限正值。")
    parts = []
    raw_parts = []
    normalized = np.full(patch.shape, np.nan, dtype=np.float32)
    for pos in positions:
        if not 0 <= pos < len(black) or not 0 <= pos < len(white):
            raise ValueError("RAW 的黑白电平元数据不完整。")
        lo, hi = float(black[pos]), float(white[pos])
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            raise ValueError("RAW 的黑白电平元数据无效。")
        mask = pattern == pos
        if mask.any():
            values = (patch[mask].astype(np.float32) - lo) / (hi - lo)
            raw_parts.append(values)
            if gain is not None:
                with np.errstate(over="ignore", invalid="ignore"):
                    values = values * gain[mask]
            parts.append(values)
            normalized[mask] = values
    samples = np.concatenate(parts) if parts else np.empty(0, np.float32)
    raw_samples = np.concatenate(raw_parts) if raw_parts else np.empty(0, np.float32)
    raw_limits = []
    for position in np.unique(pattern):
        index = int(position)
        if (not 0 <= index < min(len(black), len(white))
                or not math.isfinite(float(black[index])) or not math.isfinite(float(white[index]))
                or white[index] <= black[index]):
            raise ValueError("RAW 的全感光通道黑白电平元数据不完整或无效。")
        values = ((patch[pattern == position].astype(np.float32) - float(black[index]))
                  / (float(white[index]) - float(black[index])))
        if not np.isfinite(values).all():
            raise ValueError("RAW 片基样本含有无效数值。")
        raw_limits.append(dict(position=index, p999=float(np.percentile(values, 99.9)),
                               saturation_fraction=float(np.mean(values >= .998))))
    raw_metadata = dict(raw_signal=float(np.percentile(raw_samples, 99.9)) if raw_samples.size else None,
                        raw_limit_signal=max((item["p999"] for item in raw_limits), default=None),
                        raw_saturation_fraction=max((item["saturation_fraction"] for item in raw_limits), default=0.),
                        raw_channel_limits=tuple(raw_limits))
    issues = []
    if samples.size < MIN_SAMPLES:
        return dict(sample_count=int(samples.size), signal=None, median=None, saturation_fraction=0.0,
                    uniformity_range=None, issues=("片基样本过少，请稍微放大片基框。",), **raw_metadata)
    if not np.isfinite(samples).all() or not np.isfinite(raw_samples).all():
        raise ValueError("RAW 片基样本含有无效数值。")
    signal = float(np.percentile(samples, 99.9))
    median = float(np.median(samples))
    raw_signal = float(np.percentile(raw_samples, 99.9))
    saturation = float(np.mean(raw_samples >= 0.998))
    if min(signal, raw_signal) < MIN_SIGNAL or median <= 0:
        issues.append("片基信号过弱，暂不估算；请增加曝光后复拍。")
    if saturation > MAX_SATURATION_FRACTION or raw_signal >= 0.998:
        issues.append("片基接近或达到 RAW 上限，暂不估算；请降低曝光后复拍。")
    elif gain is not None and (raw_metadata["raw_limit_signal"] >= .998
                              or raw_metadata["raw_saturation_fraction"] > MAX_SATURATION_FRACTION):
        issues.append("片基的其他 CFA 感光通道接近或达到原始 RAW 上限，暂不估算；请降低曝光后复拍。")
    tiles = []
    for rows in np.array_split(normalized, 3, axis=0):
        for tile in np.array_split(rows, 3, axis=1):
            valid = tile[np.isfinite(tile)]
            if valid.size >= 4:
                tiles.append(float(np.median(valid)))
    # A tiny bright patch can evade tile medians yet dominate the ETTR
    # quantile. It must not turn a mostly non-base selection into a reference.
    uniformity = max(max(tiles) - min(tiles), signal - median) / max(median, 1e-8) if tiles else None
    if uniformity is None or uniformity > MAX_UNIFORMITY_RANGE:
        issues.append("片基范围不够均匀，请排除画面、齿孔、文字、阴影或灰尘后重测。")
    return dict(sample_count=int(samples.size), signal=signal, median=median,
                saturation_fraction=saturation, uniformity_range=uniformity, issues=tuple(issues), **raw_metadata)


@dataclass(frozen=True)
class _PreparedFlatfield:
    gain: np.ndarray = field(compare=False, repr=False)
    report: dict
    mappings: dict


def _prepare_flatfield(config, paths):
    """Read a frozen reference without writing to its folder or a workspace."""
    if config is None:
        return None
    if not isinstance(config, Mapping) or type(config.get("enabled", True)) is not bool:
        raise ValueError("平场设置无效，请重新选择当前平场。")
    if not config.get("enabled", True):
        return None
    strength = config.get("strength", 1.0)
    if (isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not math.isfinite(strength) or not 0 <= strength <= 2):
        raise ValueError("平场强度需为 0..2 的数。")
    if strength == 0:
        return None
    from .flatfield import prepare_calibration
    from .native import sha256

    path = Path(config.get("path") or "").expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("请选择有效平场文件。")
    digest = sha256(path)
    if config.get("sha256") and config["sha256"] != digest:
        raise ValueError("平场文件已改变，请重新选择。")
    with TemporaryDirectory(prefix="pure-rgb-balance-flatfield-") as directory:
        gain, calibration = prepare_calibration(dict(path=str(path), sha256=digest), directory, paths)
    mappings = calibration.get("compatibility", {}).get("sensor_mappings", [])
    if len(mappings) != len(paths) or any(not item.get("camera_verified") for item in mappings):
        raise ValueError("平场缺少可核对的相机型号或传感器画幅信息，不能用于片基量化。")
    if strength != 1.0:
        with np.errstate(over="ignore", invalid="ignore"):
            gain = (1.0 + strength * (gain - 1.0) if strength < 1.0
                    else np.power(gain, strength))
    if not np.isfinite(gain).all() or np.any(gain <= 0):
        raise ValueError("此平场在指定强度下无法生成有限的正增益。")
    gain.flags.writeable = False
    report = dict(applied=True, name=str(config.get("name") or path.name), path=str(path),
                  sha256=digest, gain_sha256=calibration["gain_sha256"], strength=float(strength),
                  strength_curve="linear-blend" if strength <= 1.0 else "exposure-stops-v1",
                  measurement_basis="平场后", saturation_basis="original-uncorrected-RAW",
                  coordinate_space="unrotated full sensor-visible RAW, then standard-crop ROI",
                  sensor_mappings=mappings,
                  warnings=calibration.get("compatibility", {}).get("warnings", []))
    return _PreparedFlatfield(gain, report, {item["path"]: item for item in mappings})


def _sensor_gain_patch(prepared, path, channel, plan, region):
    """Sample one RGB gain plane at the actual unrotated CFA pixel centers.

    This is equivalent to full-sensor gain interpolation before selecting the
    standard crop and measurement ROI. No EXIF transform or average gain enters
    the calculation, and no full-resolution RGB gain buffer is needed.
    """
    mapping = prepared.mappings.get(str(Path(path).expanduser().resolve()))
    if mapping is None or tuple(mapping["target_sensor_shape"]) != plan.source_shape:
        raise ValueError("片基 RAW 与已验证的平场传感器画幅不一致。")
    left, top, right, bottom = region_pixels(region, plan.crop_shape[1], plan.crop_shape[0])
    sx, sy, _, _ = plan.visible_roi
    x0, y0, x1, y1 = mapping["reference_roi_normalized_xyxy"]
    gain = prepared.gain[..., CHANNELS.index(channel)]
    gh, gw = gain.shape
    height, width, _ = plan.source_shape
    x = np.clip((x0 + (np.arange(left, right, dtype=np.float64) + sx + .5)
                 / width * (x1 - x0)) * gw - .5, 0, gw - 1)
    ix = np.floor(x).astype(np.intp)
    jx = np.minimum(ix + 1, gw - 1)
    wx = (x - ix).astype(np.float32)[None, :]
    result = np.empty((bottom - top, right - left), np.float32)
    rows = max(1, 262144 // result.shape[1])
    for start in range(0, result.shape[0], rows):
        end = min(start + rows, result.shape[0])
        y = np.clip((y0 + (np.arange(start, end, dtype=np.float64) + sy + top + .5)
                     / height * (y1 - y0)) * gh - .5, 0, gh - 1)
        iy = np.floor(y).astype(np.intp)
        jy = np.minimum(iy + 1, gh - 1)
        wy = (y - iy).astype(np.float32)[:, None]
        upper = gain[iy[:, None], ix[None, :]] * (1 - wx) + gain[iy[:, None], jx[None, :]] * wx
        lower = gain[jy[:, None], ix[None, :]] * (1 - wx) + gain[jy[:, None], jx[None, :]] * wx
        result[start:end] = upper * (1 - wy) + lower * wy
    return result


def read_capture(path, channel, region, *, with_preview=True, flatfield=None):
    """Call on a worker. The rawpy handle and full arrays never leave it."""
    import rawpy
    from negpy.infrastructure.loaders.helpers import read_exif_from_file
    from .raw_validity import plan_standard_crop

    source = Path(path)
    prepared = (flatfield if isinstance(flatfield, _PreparedFlatfield)
                else _prepare_flatfield(flatfield, [str(source.expanduser().resolve())]))
    before = source.stat()
    with rawpy.imread(str(source)) as raw:
        mosaic = raw.raw_image_visible
        colors = raw.raw_colors_visible
        if mosaic.ndim != 2 or colors is None:
            raise ValueError("当前参考分析支持带 CFA 阵列的相机 RAW。")
        plan = plan_standard_crop(raw.sizes._asdict(), (*mosaic.shape, 3), user_flip=0, half_size=False)
        x0, y0, x1, y1 = plan.visible_roi
        desc = raw.color_desc.decode("ascii", errors="ignore").upper()
        positions = tuple(index for index, letter in enumerate(desc) if letter == channel)
        if not positions:
            raise ValueError(f"这张 RAW 中没有可测量的 {channel} 感光通道。")
        black = tuple(raw.black_level_per_channel)
        calibrated = raw.camera_white_level_per_channel
        ceiling = float(raw.white_level)
        white = tuple(min(float(v), ceiling) if _positive(v) else ceiling for v in calibrated) if calibrated else (ceiling,) * len(black)
        gain = _sensor_gain_patch(prepared, source, channel, plan, region) if prepared is not None else None
        result = measure_cfa(mosaic[y0:y1, x0:x1], colors[y0:y1, x0:x1], black, white, positions, region, gain=gain)
        exif = (read_exif_from_file(str(source)) or {}).get("0th", {})

        def text(value):
            return value.decode("utf-8", errors="replace").strip("\0 ") if isinstance(value, bytes) else str(value or "").strip()

        info = raw.other
        settings = CaptureSettings(
            (text(exif.get(271)), text(exif.get(272))), _positive(info.iso_speed),
            _positive(info.shutter_speed), _positive(info.aperture), (y1 - y0, x1 - x0),
        )
        preview = None
        if with_preview:
            # A fixed display transform for locating the film base only. Neither
            # this demosaic nor its display gamma enters the numerical meter.
            from negpy.infrastructure.capture.raw_demosaic import _user_sat
            from negpy.infrastructure.loaders.helpers import is_xtrans
            from PIL import Image

            half = not is_xtrans(raw)
            rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, adjust_maximum_thr=0.0,
                                  user_sat=_user_sat(raw), use_camera_wb=False, user_wb=[1, 1, 1, 1],
                                  output_bps=16, output_color=rawpy.ColorSpace.raw,
                                  half_size=half, user_flip=0, highlight_mode=0)
            factor = 2 if half else 1
            band = rgb[y0 // factor:y1 // factor, x0 // factor:x1 // factor, CHANNELS.index(channel)]
            # Fixed 16-bit to display transform, shared by all three channels.
            image = Image.fromarray(np.rint(np.clip(band.astype(np.float32) / 65535, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8))
            image.thumbnail((760, 760), Image.Resampling.BILINEAR)
            preview = np.array(image, dtype=np.uint8, copy=True)
            preview.flags.writeable = False
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("参考 RAW 在读取时发生变化，请重新选择。")
    return ChannelReading(channel, str(source), settings, preview=preview,
                          flatfield=prepared.report if prepared is not None else None, **result)


def combine_readings(readings, region=FULL_REGION):
    readings = tuple(readings)
    if len(readings) != 3 or tuple(item.channel for item in readings) != CHANNELS:
        raise ValueError("需要按 R、G、B 顺序的三路参考。")
    region = checked_region(region)
    issues = settings_issues(readings)
    valid = not issues and not any(item.issues for item in readings)
    signals = np.asarray([item.signal if item.signal is not None else np.nan for item in readings], dtype=float)
    valid = valid and bool(np.isfinite(signals).all() and np.all(signals > 0))
    if valid:
        middle = float(np.median(signals))
        imbalance = round(float(100 * np.max(np.abs(signals / middle - 1))), 12)
        adjustments = []
        guarded_readings = []
        for item, value in zip(readings, signals):
            ratio = SAFE_TARGET / value
            issue = None
            if (item.flatfield and item.flatfield.get("applied") and item.raw_limit_signal is not None
                    and item.raw_limit_signal * ratio >= .998):
                issue = ("按平场后 90% 目标增加曝光会使原始 RAW 的感光通道削顶，"
                         "暂不给出该补偿；请核对平场与拍摄曝光后复测。")
            adjustments.append(None if issue else float(100 * (ratio - 1)))
            guarded_readings.append(replace(item, adjustment_issue=issue))
        readings, adjustments = tuple(guarded_readings), tuple(adjustments)
    else:
        imbalance, adjustments = None, (None, None, None)
    return BalanceAnalysis(region, readings, issues, imbalance, adjustments)


def analyze_triplet(paths, region=FULL_REGION, *, with_previews=True, progress: Callable | None = None,
                    flatfield=None):
    """Read-only standalone analysis. Never import into a workspace or render."""
    paths, region = ordered_paths(paths), checked_region(region)
    prepared = _prepare_flatfield(flatfield, paths)
    readings = []
    for index, (channel, path) in enumerate(zip(CHANNELS, paths)):
        if progress is not None:
            progress(f"正在读取 {channel} 片基 RAW（{index + 1}/3）…")
        options = {"with_preview": with_previews}
        if prepared is not None:
            options["flatfield"] = prepared
        readings.append(read_capture(path, channel, region, **options))
    if prepared is not None:
        from .native import sha256

        if sha256(prepared.report["path"]) != prepared.report["sha256"]:
            raise ValueError("平场文件在片基量化期间发生变化，请重新选择。")
    return replace(combine_readings(readings, region), flatfield=prepared.report if prepared is not None else None)
