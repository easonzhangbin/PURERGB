"""Optional film-base exposure advice; never changes an image or its recipe.

The meter reads black-subtracted CFA photosites, normalized against the same
camera metadata for every frame. An optional validated flat-field gain is
sampled at each photosite before statistics, in unrotated sensor coordinates.
There is no white balance, demosaic, gamma, per-frame normalization or inversion.
Relative balance uses the film-base median. RAW headroom uses original
photosites and is measured separately in the retained area and full frame.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Mapping, Sequence

import numpy as np

CHANNELS = ("R", "G", "B")
MIN_SAMPLES = 64
MIN_SIGNAL = 0.01
WEAK_REFERENCE_ISSUE = "片基信号过弱，暂不估算。"
RAW_NEAR_LIMIT = 0.998
SHUTTER_TRIAL_PEAK_LIMIT = 0.95
MAX_SATURATION_FRACTION = 0.002
MAX_SPATIAL_RANGE = 0.15
BRIGHT_TAIL_WARNING = 0.15
DEFAULT_REGION = (0.4, 0.4, 0.6, 0.6)
FULL_REGION = (0.0, 0.0, 1.0, 1.0)
_SHUTTER_STEPS = (
    *(1 / denominator for denominator in (
        8000, 6400, 5000, 4000, 3200, 2500, 2000, 1600, 1250, 1000, 800, 640, 500, 400, 320, 250,
        200, 160, 125, 100, 80, 60, 50, 40, 30, 25, 20, 15, 13, 10, 8, 6, 5, 4,
    )),
    .3, .4, .5, .6, .8, 1., 1.3, 1.6, 2., 2.5, 3.2, 4., 5., 6., 8., 10., 13., 15., 20., 25., 30.,
)


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
        return "建议复测"
    if percent <= 15:
        return "需调整"
    return "明显失衡"


@dataclass(frozen=True)
class CaptureSettings:
    camera: tuple[str, str]
    iso: float | None
    shutter: float | None
    aperture: float | None
    shape: tuple[int, int]


@dataclass(frozen=True)
class RawExposure:
    limit_signal: float | None
    sample_count: int
    near_count: int
    clipped_count: int
    risk: bool
    channel_limits: tuple[dict, ...]
    peak_signal: float | None = None


@dataclass(frozen=True)
class MeterAdvice:
    title: str
    detail: str
    severity: str


_CAPTURE_CONFIRMATION_ADVICE = MeterAdvice(
    "请确认保留画幅",
    "蓝框读数仅作预览。圈住最终保留的整个画面后，点击「确认保留画幅」。",
    "neutral",
)


@dataclass(frozen=True)
class ShutterAdvice:
    current_seconds: float
    suggested_seconds: float
    ev: float
    action: str
    title: str
    detail: str
    predicted_peak_signal: float | None = None


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
    spatial_range: float | None = None
    bright_tail: float | None = None
    warnings: tuple[str, ...] = ()
    capture: RawExposure | None = None
    full_capture: RawExposure | None = None
    raw_median: float | None = None


@dataclass(frozen=True)
class BalanceAnalysis:
    region: tuple[float, float, float, float]
    readings: tuple[ChannelReading, ...]
    issues: tuple[str, ...]
    imbalance_percent: float | None
    adjustments_percent: tuple[float | None, ...]
    flatfield: dict | None = None
    capture_region: tuple[float, float, float, float] = FULL_REGION
    reference_confirmed: bool = False
    deviations_percent: tuple[float, ...] | None = None
    capture_confirmed: bool = True

    @property
    def usable(self):
        return (self.reference_confirmed and self.deviations_percent is not None
                and not self.issues and all(not item.issues for item in self.readings))

    @property
    def grade(self):
        if self.deviations_percent is None:
            return "参考待核对"
        if not self.reference_confirmed:
            return "待确认片基"
        return imbalance_grade(self.imbalance_percent)

    @property
    def status(self):
        if not self.capture_confirmed:
            return _CAPTURE_CONFIRMATION_ADVICE.title
        if self.exposure_risk:
            scope = "全幅" if self.capture_region == FULL_REGION else "保留画幅"
            return f"{scope} RAW 接近上限 · {'、'.join(self._capture_risk_channels)}"
        if self.imbalance_percent is None:
            return "片基参考待核对"
        if not self.usable:
            proximity = "三路接近 · " if self.imbalance_percent <= 5 else ""
            return f"{proximity}片基待确认 · 试算最大偏差 {self.imbalance_percent:.2f}%"
        if self.imbalance_percent <= 5:
            return "已配平 · 保持灯光"
        return self.grade

    @property
    def _capture_risk_channels(self):
        return tuple(item.channel for item in self.readings if item.capture is not None and item.capture.risk)

    @property
    def exposure_risk(self):
        return bool(self._capture_risk_channels)

    @property
    def full_exposure_risk(self):
        return any(item.full_capture is not None and item.full_capture.risk for item in self.readings)

    @property
    def balance_advice(self):
        def advice(title, detail, severity):
            if self.flatfield and self.flatfield.get("warnings"):
                detail += "平场参数不完全匹配。"
            return MeterAdvice(title, detail, severity)

        if self.deviations_percent is None or self.issues or any(item.issues for item in self.readings):
            reasons = [_short_reason(issue) for issue in self.issues]
            channel_reasons = {}
            for item in self.readings:
                for issue in item.issues:
                    channel_reasons.setdefault(_short_reason(issue), []).append(item.channel)
            reasons.extend(f"{'、'.join(channels)} 路{reason}" for reason, channels in channel_reasons.items())
            detail = "。".join(reasons) + "。" if reasons else "未取得三路有效的片基中位值。"
            return advice("无法判定配平", detail + "核对片基量区和拍摄设置后复测。", "warning")
        difference = f"最大偏差 {self.imbalance_percent:.2f}%。"
        if not self.reference_confirmed:
            title = "试算在范围 · 待确认" if self.imbalance_percent <= 5 else "试算超出范围 · 待确认"
            return advice(title, difference + "请确认量区属于同一块无画面片基。", "neutral")
        if self.imbalance_percent <= 5:
            return advice("配平合格", difference + "三路均在 ±5% 内。", "good")
        if self.imbalance_percent <= 10:
            return advice("超出最佳范围 · 建议复测", difference + "按相同设置复拍，确认偏差是否稳定。", "warning")
        title = "明显失衡 · 需调整" if self.imbalance_percent > 15 else "超出范围 · 需调整"
        detail = self._relative_adjustment_detail()
        return advice(title, difference + detail, "warning")

    @property
    def _matched_capture_shutter(self):
        if len(self.readings) != len(CHANNELS) or self.issues or settings_issues(self.readings):
            return None
        if not all(_has_retained_exposure(item.capture) and _has_capture_peak(item.capture) for item in self.readings):
            return None
        if any(item.full_capture is not None and len(item.capture.channel_limits) != len(item.full_capture.channel_limits)
               for item in self.readings):
            return None
        return _positive(self.readings[0].settings.shutter)

    @property
    def shutter_advice(self):
        if not self.capture_confirmed:
            return None
        current = self._matched_capture_shutter
        if current is None:
            return None
        if self.exposure_risk:
            shorter = _shutter_step(current, -1)
            if not 0 < shorter < current:
                return None
            title = _shutter_change_label(current, shorter, -1)
            return ShutterAdvice(current, shorter, -1 / 3, "decrease", title,
                                 f"三路快门统一 {title}试拍，复拍后检查削顶。")

        exposures = tuple(item.capture for item in self.readings)
        if any(item.near_count for item in exposures):
            return None
        peak = max(item.peak_signal for item in exposures)
        for steps in (3, 2, 1):
            slower = _shutter_step(current, steps)
            scale = slower / current
            if not 1 < scale <= 2.000001 or peak * scale > SHUTTER_TRIAL_PEAK_LIMIT:
                continue
            predicted = peak * scale
            title = _shutter_change_label(current, slower, steps)
            return ShutterAdvice(current, slower, steps / 3, "increase", title,
                                 f"三路快门统一 {title}。预计 RAW 峰值 {predicted:.0%}。复拍核对。",
                                 predicted_peak_signal=predicted)
        return None

    @property
    def exposure_advice(self):
        if not self.capture_confirmed:
            return _CAPTURE_CONFIRMATION_ADVICE
        shutter = self.shutter_advice
        if self.exposure_risk:
            channels = "、".join(self._capture_risk_channels)
            if self.capture_region == FULL_REGION:
                detail = f"{channels} 路有削顶风险。只保留画面：先圈定再判断，目前不建议增加。"
                if shutter is not None:
                    detail = f"{channels} 路有削顶风险。只保留画面请先圈定。保留全幅时，{shutter.detail}"
                return MeterAdvice("保留全幅：降低曝光",
                                   detail, "danger")
            if shutter is not None:
                return MeterAdvice("降低曝光", f"{channels} 路有削顶风险。{shutter.detail}", "danger")
            return MeterAdvice(f"降低 {channels} 路曝光",
                               "所选保留画幅有削顶风险。降低后复拍，重新检查 RAW 余量。", "danger")

        if not all(_has_retained_exposure(item.capture) for item in self.readings):
            return MeterAdvice("暂缓判断曝光", "保留画幅的 RAW 余量数据不足。请扩大检查范围或重新分析。", "neutral")

        if shutter is not None:
            return MeterAdvice("可试加曝光（可选）", shutter.detail, "neutral")

        current = self._matched_capture_shutter
        measured = current is not None
        clipped = "、".join(item.channel for item in self.readings if item.capture.clipped_count)
        near = "、".join(item.channel for item in self.readings if item.capture.near_count)
        if clipped:
            detail = f"{clipped} 路有零星到顶。当前没有可建议的安全加曝光档位。"
        elif near:
            detail = f"{near} 路有零星近上限点。当前没有可建议的安全加曝光档位。"
        else:
            detail = "检查范围内未见明显削顶。"
            if measured:
                detail += "最高信号限制了可试加的快门档位。"
        if measured:
            severity = "neutral" if clipped or near else "good"
            return MeterAdvice("保持当前曝光", detail + f"三路保持 {_shutter_label(current)}。", severity)
        if self.deviations_percent is not None and not self.reference_confirmed:
            return MeterAdvice("先保持曝光", detail + "片基待确认，暂缓加曝光。", "neutral")
        if not self.usable:
            return MeterAdvice("当前无明显削顶风险", detail + "片基待核对，暂缓判断是否需要增加曝光。", "neutral")
        return MeterAdvice("保持当前曝光", detail, "good")

    def _relative_adjustment_detail(self):
        if not self.capture_confirmed:
            return "确认保留画幅后，再估算有效曝光调整。"
        directions = ["{} {} {:.2f}%".format(item.channel, "增加" if value > 0 else "降低", abs(value))
                      for item, value in zip(self.readings, self.adjustments_percent)
                      if value is not None and round(abs(value), 2) > 0]
        if not directions:
            return "先按各路提示处理 RAW 余量限制，再复拍确认三路偏差。"
        message = f"有效曝光试调估算 {'、'.join(directions)}。灯具小幅试调后，用同一片基复拍核对。"
        blocked = "、".join(item.channel for item in self.readings if item.adjustment_issue)
        if blocked:
            message += f"{blocked} 暂缓调整，先查看 RAW 余量提示。"
        return message

    @property
    def guidance(self):
        if not self.capture_confirmed:
            return _CAPTURE_CONFIRMATION_ADVICE.detail
        if self.exposure_risk:
            if self.capture_region == FULL_REGION:
                message = "若高值只出现在齿孔或片外，先圈定实际保留画幅后重测。若画面内仍接近上限，再降低拍摄曝光。"
            else:
                message = "保留画幅内已有余量风险。先降低拍摄曝光，再复测。"
            if self.deviations_percent is None:
                message += "片基量区或拍摄设置也需核对。"
            elif not self.reference_confirmed:
                message += "片基仍待确认。"
            else:
                message += "片基参考已通过核对。"
        elif self.deviations_percent is None:
            message = "查看各路提示，核对片基范围与拍摄设置后复测。"
        elif not self.reference_confirmed:
            message = "请确认三路量区都位于同一块已冲洗、无画面的片基，再决定是否调整。"
        elif self.imbalance_percent <= 5:
            message = "可用同一拍摄设置继续翻拍。"
        elif self.imbalance_percent <= 10:
            message = "先保持灯光，按相同拍摄设置再拍一组，确认偏差是否重复出现。"
        else:
            message = self._relative_adjustment_detail()
        if any(item.warnings for item in self.readings):
            message += "量区有亮尾提示，可展开数值与报告查看。"
        return message

    def to_dict(self):
        shutter = self.shutter_advice
        readings = []
        for item in self.readings:
            values = asdict(replace(item, preview=None))
            del values["preview"]
            readings.append(values)
        return {
            "schema": "rgb-film-base-exposure-v2", "region_xyxy_fraction": self.region,
            "capture_region_xyxy_fraction": self.capture_region,
            "coordinate_space": "standard-cropped RAW sensor, before EXIF rotation",
            "reference_confirmed": self.reference_confirmed,
            "capture_confirmed": self.capture_confirmed,
            "usable": self.usable, "grade": self.grade, "status": self.status, "guidance": self.guidance,
            "balance_advice": asdict(self.balance_advice), "exposure_advice": asdict(self.exposure_advice),
            "shutter_advice": asdict(shutter) if shutter is not None else None,
            "deviations_percent": self.deviations_percent, "imbalance_percent": self.imbalance_percent,
            "exposure_risk": self.exposure_risk, "full_exposure_risk": self.full_exposure_risk,
            "effective_exposure_adjustments_percent": self.adjustments_percent,
            "readings": readings, "issues": self.issues,
            "flatfield": self.flatfield,
            "measurement_basis": "平场后" if self.flatfield and self.flatfield.get("applied") else "未套用平场",
            "physical_light_power_calibrated": False,
        }

    def report_text(self):
        lines = ["PURE RGB · RGB 拍摄灯光配平", "建议使用同卷已冲洗、无画面的片基。三张使用相同 ISO、快门和光圈。"]
        lines.append(f"片基量区：{_region_label(self.region)} · {'已确认' if self.reference_confirmed else '待确认'}")
        lines.append(f"RAW 保留画幅：{_region_label(self.capture_region)} · {'已确认' if self.capture_confirmed else '待确认'}")
        corrected = bool(self.flatfield and self.flatfield.get("applied"))
        if corrected:
            lines.append(f"量化基准：平场后 · {self.flatfield['name']} · 强度 {self.flatfield['strength']:.0%}")
            lines.append(f"平场：{self.flatfield['path']} · SHA-256 {self.flatfield['sha256']}")
            lines.extend(f"平场提示：{warning}" for warning in self.flatfield.get("warnings", ()))
        else:
            lines.append("量化基准：未套用平场")
        for label, advice in (("曝光建议", self.exposure_advice), ("配平结论", self.balance_advice)):
            lines.extend((f"{label}：{advice.title}", advice.detail))
        deviations = self.deviations_percent or (None,) * len(self.readings)
        for reading, deviation, adjustment in zip(self.readings, deviations, self.adjustments_percent):
            lines.append(f"{reading.channel} · 片基中位值 {_percentage(reading.median)} · 片基 P99.9 {_percentage(reading.signal)}"
                         f" · 相对偏差 {_signed_percent(deviation)} · 有效曝光估算 {_signed_percent(adjustment)}")
            lines.append(f"  文件：{reading.path}")
            s = reading.settings
            lines.append(f"  相机：{' '.join(s.camera)} · ISO {s.iso} · 快门 {s.shutter} s · 光圈 f/{s.aperture}")
            lines.append(f"  样本 {reading.sample_count:,} · 九宫格空间差异 {_percentage(reading.spatial_range)}"
                         f" · 亮尾 {_percentage(reading.bright_tail)}")
            if reading.raw_limit_signal is not None:
                lines.append(f"  片基量区原始 RAW 中位值 {_percentage(reading.raw_median)}"
                             f" · 最高 CFA P99.9 {reading.raw_limit_signal:.2%}")
            for label, exposure in (("保留画幅", reading.capture), ("全幅", reading.full_capture)):
                if exposure is not None:
                    state = "余量不足" if exposure.risk else ("有零星近上限点" if exposure.near_count else "未见明显削顶风险")
                    lines.append(f"  {label} RAW · 最高 CFA P99.9 {_percentage(exposure.limit_signal)}"
                                 f" · 峰值 {_percentage(exposure.peak_signal)} · "
                                 f"达到上限 {exposure.clipped_count:,} 点 · 接近或达到上限 {exposure.near_count:,} / "
                                 f"{exposure.sample_count:,} 点 · {state}")
                    for plane in exposure.channel_limits:
                        lines.append(f"    CFA {plane['position']} · P99.9 {plane['p999']:.2%} · "
                                     f"达到上限 {plane['clipped_count']:,} 点 · 接近或达到上限 {plane['near_count']:,} / "
                                     f"{plane['sample_count']:,} 点")
            lines.extend(f"  提示：{issue}" for issue in reading.issues)
            lines.extend(f"  观察：{warning}" for warning in reading.warnings)
            if reading.adjustment_issue:
                lines.append(f"  建议限制：{reading.adjustment_issue}")
        lines.extend(f"提示：{issue}" for issue in self.issues)
        lines.extend((
            "相对偏差比较三路片基中位信号。有效曝光估算用三路中位信号的中间值作参考，灯具刻度须实测。",
            "分级为软件操作提示。≤5% 保持，超过 5% 至 10% 复测，超过 10% 至 15% 需调整，超过 15% 表示明显失衡。",
            "RAW 余量按未平场的感光点计算。达到上限以黑白电平元数据为准，接近上限起点为 99.8%。",
            "余量不足指任一 CFA 类别的 P99.9 达到 99.8%，或其接近上限的感光点超过 0.2%。",
            "100% 为原始 RAW 白电平上限。快门试拍假定灯光、ISO、光圈和保留画幅不变，复拍实测才可验证。",
            "PURE RGB 参考柯达 D-min 片基参照与扩展动态范围扫描建议设计。三色配平容差和快门试拍规则由本工具设定。",
            "Kodak H-61A 区分未曝光负片的 D-min 和用于印片控制的 LAD 标准片。",
            "Kodak 扩展动态范围扫描建议关注记录完整的密度范围。本报告将采集上限与片基相对配平单独评估。",
            "Kodak H-61A · https://www.kodak.com/content/products-brochures/Film/LAD-for-KODAK-Color-Negative-Film-H-61a.pdf",
            "Kodak 扩展动态范围扫描建议 · https://www.kodak.com/content/products-brochures/Film/Scanning-Recommendations-for-Extended-Dynamic-Range-Camera-Films-EN.pdf",
            "Flueckiger 等，2018 · https://diastor.ch/wp-content/uploads/2018/03/flueckigeretal_investigationfilmmaterialscannerinteraction_2018_v_1-1c.pdf",
        ))
        return "\n".join(lines) + "\n"


def _region_label(region):
    return "全幅" if region == FULL_REGION else ", ".join(f"{value:.4f}" for value in region)


def _percentage(value):
    return "—" if value is None else f"{value:.2%}"


def _signed_percent(value):
    return "—" if value is None else f"{value:+.2f}%"


def _short_reason(issue):
    return issue.split("。", 1)[0].split("，", 1)[0].split("；", 1)[0]


def _has_retained_exposure(exposure):
    return (exposure is not None and exposure.limit_signal is not None and math.isfinite(exposure.limit_signal)
            and exposure.sample_count >= MIN_SAMPLES
            and 0 <= exposure.clipped_count <= exposure.near_count <= exposure.sample_count)


def _has_capture_peak(exposure):
    peak = _positive(exposure.peak_signal)
    return peak is not None and peak >= exposure.limit_signal


def _shutter_step(seconds, steps):
    for index, shutter in enumerate(_SHUTTER_STEPS):
        if math.isclose(shutter, seconds, rel_tol=0.005) and 0 <= index + steps < len(_SHUTTER_STEPS):
            return _SHUTTER_STEPS[index + steps]
    return seconds * 2 ** (steps / 3)


def _shutter_label(seconds):
    if seconds < 0.3 and math.isfinite(1 / seconds):
        denominator = round(1 / seconds)
        if math.isclose(seconds, 1 / denominator, rel_tol=0.005):
            return f"1/{denominator} 秒"
    return f"{seconds:.5g} 秒"


def _shutter_change_label(current, suggested, steps):
    stop_count = "1" if abs(steps) == 3 else f"{abs(steps)}/3"
    sign = "+" if steps > 0 else "−"
    return f"{_shutter_label(current)} → {_shutter_label(suggested)}（约 {sign}{stop_count} EV）"


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


def _sample_patches(mosaic, colors, region):
    mosaic, colors = np.asarray(mosaic), np.asarray(colors)
    if mosaic.ndim != 2 or colors.shape != mosaic.shape or not mosaic.size:
        raise ValueError("需要包含原始感光点数据的相机 RAW。")
    height, width = mosaic.shape
    x0, y0, x1, y1 = region_pixels(region, width, height)
    return mosaic[y0:y1, x0:x1], colors[y0:y1, x0:x1]


def _linear_cfa_planes(patch, pattern, black, white):
    for position in np.unique(pattern):
        index = int(position)
        if index != position or not 0 <= index < min(len(black), len(white)):
            raise ValueError("RAW 的感光通道黑白电平元数据不完整。")
        lo, hi = float(black[index]), float(white[index])
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            raise ValueError("RAW 的感光通道黑白电平元数据无效。")
        mask = pattern == position
        values = patch[mask].astype(np.float32, copy=False)
        values -= lo
        values /= hi - lo
        if not np.isfinite(values).all():
            raise ValueError("RAW 样本含有无效数值。")
        yield index, mask, values


def _cfa_limit(position, values, *, overwrite=False):
    count = int(values.size)
    near = int(np.count_nonzero(values >= RAW_NEAR_LIMIT))
    clipped = int(np.count_nonzero(values >= 1.0))
    peak = float(np.max(values))
    p999 = float(np.percentile(values, 99.9, overwrite_input=overwrite))
    near_fraction = near / count
    return dict(position=position, p999=p999, peak=peak, sample_count=count,
                saturation_fraction=near_fraction, near_count=near, clipped_count=clipped,
                risk=p999 >= RAW_NEAR_LIMIT or near_fraction > MAX_SATURATION_FRACTION)


def _summarize_raw(limits):
    return RawExposure(
        limit_signal=max((item["p999"] for item in limits), default=None),
        sample_count=sum(item["sample_count"] for item in limits),
        near_count=sum(item["near_count"] for item in limits),
        clipped_count=sum(item["clipped_count"] for item in limits),
        risk=any(item["risk"] for item in limits),
        channel_limits=tuple(limits),
        peak_signal=max((item["peak"] for item in limits), default=None),
    )


def _measure_raw_exposure(mosaic, colors, black, white, region):
    patch, pattern = _sample_patches(mosaic, colors, region)
    limits = [_cfa_limit(index, values, overwrite=True)
              for index, _, values in _linear_cfa_planes(patch, pattern, black, white)]
    return _summarize_raw(limits)


def measure_cfa(mosaic, colors, black, white, positions, region=FULL_REGION, *, gain=None):
    """Measure actual photosites, including both green sites, without interpolation."""
    patch, pattern = _sample_patches(mosaic, colors, region)
    if gain is not None:
        gain = np.asarray(gain)
        if (gain.shape != patch.shape or gain.dtype.kind != "f"
                or not np.isfinite(gain).all() or np.any(gain <= 0)):
            raise ValueError("片基平场增益必须对应选区的每个传感器位置，且为有限正值。")
    selected = set(positions)
    if any(not 0 <= pos < min(len(black), len(white)) for pos in selected):
        raise ValueError("RAW 的黑白电平元数据不完整。")
    parts = []
    raw_parts = []
    raw_limits = []
    normalized = np.full(patch.shape, np.nan, dtype=np.float32)
    for index, mask, values in _linear_cfa_planes(patch, pattern, black, white):
        raw_limits.append(_cfa_limit(index, values, overwrite=index not in selected))
        if index not in selected:
            continue
        raw_parts.append(values)
        if gain is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                values = values * gain[mask]
            parts.append(values)
        normalized[mask] = values
    raw_samples = np.concatenate(raw_parts) if raw_parts else np.empty(0, np.float32)
    samples = np.concatenate(parts) if parts else raw_samples
    raw_median, raw_signal = np.percentile(raw_samples, (50, 99.9)) if raw_samples.size else (None, None)
    raw_metadata = dict(raw_median=float(raw_median) if raw_median is not None else None,
                        raw_signal=float(raw_signal) if raw_signal is not None else None,
                        raw_limit_signal=max((item["p999"] for item in raw_limits), default=None),
                        raw_saturation_fraction=max((item["saturation_fraction"] for item in raw_limits), default=0.),
                        raw_channel_limits=tuple(raw_limits))
    issues = []
    if samples.size < MIN_SAMPLES:
        return dict(sample_count=int(samples.size), signal=None, median=None, saturation_fraction=0.0,
                    uniformity_range=None, issues=("片基样本过少，请稍微放大片基框。",), **raw_metadata)
    if gain is not None and not np.isfinite(samples).all():
        raise ValueError("RAW 片基样本含有无效数值。")
    median, signal = map(float, np.percentile(samples, (50, 99.9)) if gain is not None else (raw_median, raw_signal))
    saturation = float(np.mean(raw_samples >= RAW_NEAR_LIMIT))
    if min(median, raw_median) < MIN_SIGNAL:
        issues.append(WEAK_REFERENCE_ISSUE)
    if any(item["risk"] and item["position"] in selected for item in raw_limits):
        issues.append("片基量区接近或达到原始 RAW 上限。")
    elif any(item["risk"] for item in raw_limits):
        issues.append("片基量区的其他 CFA 感光通道接近或达到原始 RAW 上限。")
    tiles = []
    for rows in np.array_split(normalized, 3, axis=0):
        for tile in np.array_split(rows, 3, axis=1):
            valid = tile[np.isfinite(tile)]
            if valid.size >= 4:
                tiles.append(float(np.median(valid)))
    spatial_range = (max(tiles) - min(tiles)) / median if tiles and median > 0 else None
    bright_tail = (signal - median) / median if median > 0 else None
    warnings = []
    if spatial_range is None or spatial_range > MAX_SPATIAL_RANGE:
        issues.append("片基量区的空间差异过大。请检查画面、齿孔、文字和阴影。")
    if bright_tail is not None and bright_tail > BRIGHT_TAIL_WARNING:
        warnings.append("亮尾偏高。请查看量区是否含灰尘、亮斑或透光边缘。")
    uniformity = max(spatial_range, bright_tail) if spatial_range is not None else None
    return dict(sample_count=int(samples.size), signal=signal, median=median,
                saturation_fraction=saturation, uniformity_range=uniformity, issues=tuple(issues),
                spatial_range=spatial_range, bright_tail=bright_tail, warnings=tuple(warnings), **raw_metadata)


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


def read_capture(path, channel, region, *, with_preview=True, flatfield=None, capture_region=FULL_REGION):
    """Call on a worker. The rawpy handle and full arrays never leave it."""
    import rawpy
    from negpy.infrastructure.loaders.helpers import read_exif_from_file
    from .raw_validity import plan_standard_crop

    region, capture_region = checked_region(region), checked_region(capture_region)
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
        cropped = mosaic[y0:y1, x0:x1]
        pattern = colors[y0:y1, x0:x1]
        result = measure_cfa(cropped, pattern, black, white, positions, region, gain=gain)
        reference_exposure = _summarize_raw(result["raw_channel_limits"])
        full_exposure = (reference_exposure if region == FULL_REGION else
                         _measure_raw_exposure(cropped, pattern, black, white, FULL_REGION))
        capture_exposure = full_exposure
        if capture_region != FULL_REGION:
            capture_exposure = (reference_exposure if capture_region == region else
                                _measure_raw_exposure(cropped, pattern, black, white, capture_region))
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
                          flatfield=prepared.report if prepared is not None else None,
                          capture=capture_exposure, full_capture=full_exposure, **result)


def combine_readings(readings, region=FULL_REGION, *, capture_region=FULL_REGION, reference_confirmed=False,
                     capture_confirmed=True):
    readings = tuple(readings)
    if len(readings) != 3 or tuple(item.channel for item in readings) != CHANNELS:
        raise ValueError("需要按 R、G、B 顺序的三路参考。")
    region, capture_region = checked_region(region), checked_region(capture_region)
    if type(reference_confirmed) is not bool:
        raise ValueError("片基确认状态无效。")
    if type(capture_confirmed) is not bool:
        raise ValueError("保留画幅确认状态无效。")
    issues = settings_issues(readings)
    valid = not issues and not any(item.issues for item in readings)
    medians = np.asarray([item.median if item.median is not None else np.nan for item in readings], dtype=float)
    valid = valid and bool(np.isfinite(medians).all() and np.all(medians > 0))
    if not valid:
        if not capture_confirmed:
            readings = tuple(replace(item, adjustment_issue=None) for item in readings)
        return BalanceAnalysis(region, readings, issues, None, (None,) * len(readings),
                               capture_region=capture_region, reference_confirmed=reference_confirmed,
                               capture_confirmed=capture_confirmed)

    middle = float(np.median(medians))
    deviations = tuple(round(float(100 * (value / middle - 1)), 12) for value in medians)
    imbalance = max(map(abs, deviations))
    retained_risk = any(item.capture is not None and item.capture.risk for item in readings)
    ratios = middle / medians
    adjustments = []
    guarded_readings = []
    for item, ratio in zip(readings, ratios):
        adjustment = None
        issue = None
        if capture_confirmed and reference_confirmed and imbalance > 10:
            if retained_risk:
                issue = "保留画幅已有 RAW 削顶风险。请先降低拍摄曝光，再复测配平。"
            elif ratio > 1 and (item.capture is None or item.capture.limit_signal is None):
                issue = "缺少保留画幅的 RAW 上限数据，暂不估算增加量。"
            elif ratio > 1 and item.capture.limit_signal * ratio >= RAW_NEAR_LIMIT:
                issue = "增加到共同参考会使保留画幅原始 RAW 接近或达到削顶。请先降低整体曝光再复测。"
            else:
                adjustment = float(100 * (ratio - 1))
        adjustments.append(adjustment)
        guarded_readings.append(replace(item, adjustment_issue=issue))
    return BalanceAnalysis(region, tuple(guarded_readings), issues, imbalance, tuple(adjustments),
                           capture_region=capture_region, reference_confirmed=reference_confirmed,
                           deviations_percent=deviations, capture_confirmed=capture_confirmed)


def analyze_triplet(paths, region=FULL_REGION, *, with_previews=True, progress: Callable | None = None,
                    flatfield=None, capture_region=FULL_REGION, reference_confirmed=False, capture_confirmed=True):
    """Read-only standalone analysis. Never import into a workspace or render."""
    paths = ordered_paths(paths)
    region, capture_region = checked_region(region), checked_region(capture_region)
    if type(reference_confirmed) is not bool:
        raise ValueError("片基确认状态无效。")
    if type(capture_confirmed) is not bool:
        raise ValueError("保留画幅确认状态无效。")
    prepared = _prepare_flatfield(flatfield, paths)
    readings = []
    for index, (channel, path) in enumerate(zip(CHANNELS, paths)):
        if progress is not None:
            progress(f"正在读取 {channel} 片基 RAW（{index + 1}/3）…")
        options = {"with_preview": with_previews}
        if capture_region != FULL_REGION:
            options["capture_region"] = capture_region
        if prepared is not None:
            options["flatfield"] = prepared
        readings.append(read_capture(path, channel, region, **options))
    if prepared is not None:
        from .native import sha256

        if sha256(prepared.report["path"]) != prepared.report["sha256"]:
            raise ValueError("平场文件在片基量化期间发生变化，请重新选择。")
    result = combine_readings(readings, region, capture_region=capture_region,
                              reference_confirmed=reference_confirmed, capture_confirmed=capture_confirmed)
    return replace(result, flatfield=prepared.report if prepared is not None else None)
