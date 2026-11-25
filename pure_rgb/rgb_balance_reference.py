"""Portable, editable-text acquisition references for the RGB balance report."""

from dataclasses import dataclass
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import re
import unicodedata

from .rgb_light_balance import CHANNELS, CaptureSettings, _shutter_label


SCHEMA = "pure-rgb-balance-reference"
VERSION = 1
LAMP_ONLY_VERSION = 2
REFERENCE_BEGIN = "[PURE RGB BALANCE REFERENCE BEGIN]"
REFERENCE_END = "[PURE RGB BALANCE REFERENCE END]"
MAX_REPORT_BYTES = 256 * 1024
MAX_REFERENCE_BYTES = 16 * 1024
MAX_LAMP_VALUE_LENGTH = 48
_EMPTY = (None, None, None)
_CAPTURE_KEYS = ("camera", "iso", "shutter_seconds", "aperture", "shape", "source_name")
_CHANNEL_KEYS = {"channel", "lamp", *_CAPTURE_KEYS}
_LAMP_CHARACTERS = str.maketrans("０１２３４５６７８９．％＋－−", "0123456789.%+--")
_LAMP_NUMBER = re.compile(r"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))\s*(%)?")


@dataclass(frozen=True)
class BalanceReference:
    settings: tuple[CaptureSettings, CaptureSettings, CaptureSettings] | None
    lamp_values: tuple[str | None, str | None, str | None] = _EMPTY
    source_names: tuple[str | None, str | None, str | None] = _EMPTY

    def __post_init__(self):
        lamps = tuple(_optional_text(value, MAX_LAMP_VALUE_LENGTH, f"{channel} 路灯光值")
                      for channel, value in zip(CHANNELS, _triplet(self.lamp_values)))
        names = tuple(_source_name(value) for value in _triplet(self.source_names))
        settings = None
        if self.settings is None:
            if not any(lamps):
                raise ValueError("请至少填写一路灯具读数。")
            if any(names):
                raise ValueError("仅灯具读数的参考不能包含 RAW 文件名。")
        else:
            settings = tuple(_capture(value) for value in _triplet(self.settings))
        object.__setattr__(self, "settings", settings)
        object.__setattr__(self, "lamp_values", lamps)
        object.__setattr__(self, "source_names", names)

    @classmethod
    def from_lamps(cls, lamp_values):
        return cls(None, lamp_values)

    @classmethod
    def from_analysis(cls, analysis, lamp_values=_EMPTY):
        readings = tuple(analysis.readings)
        if tuple(reading.channel for reading in readings) != CHANNELS:
            raise ValueError("请先完成红、绿、蓝三路的 RAW 分析。")
        return cls(tuple(reading.settings for reading in readings), lamp_values,
                   tuple(Path(reading.path).name if reading.path else None for reading in readings))

    @classmethod
    def from_report(cls, text):
        if _utf8_size(text) > MAX_REPORT_BYTES:
            raise ValueError("报告太大，无法载入曝光参考。")
        lines = text.lstrip("\ufeff").splitlines()
        starts = [index for index, line in enumerate(lines) if line.strip() == REFERENCE_BEGIN]
        ends = [index for index, line in enumerate(lines) if line.strip() == REFERENCE_END]
        if not starts and not ends:
            raise ValueError("这份报告没有可载入的曝光参考。")
        if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
            raise ValueError("这份报告的曝光参考区块缺失、重复或不完整。")
        block = "\n".join(lines[starts[0] + 1:ends[0]])
        if _utf8_size(block) > MAX_REFERENCE_BYTES:
            raise ValueError("报告里的曝光参考数据过大。")
        try:
            payload = json.loads(block, object_pairs_hook=_unique_object)
        except (ValueError, RecursionError):
            raise ValueError("报告里的曝光参考格式无效。") from None
        if not isinstance(payload, dict) or set(payload) != {"schema", "version", "channels"}:
            raise ValueError("报告里的曝光参考字段无效。")
        version = payload["version"]
        if payload["schema"] != SCHEMA or type(version) is not int or version not in (VERSION, LAMP_ONLY_VERSION):
            raise ValueError("这份曝光参考使用了不支持的报告版本。")

        settings, lamps, sources = [], [], []
        for channel, row in zip(CHANNELS, _triplet(payload["channels"])):
            if not isinstance(row, dict) or set(row) != _CHANNEL_KEYS or row["channel"] != channel:
                raise ValueError("报告里的曝光参考缺少正确的 R、G、B 顺序或字段。")
            lamps.append(row["lamp"])
            if version == LAMP_ONLY_VERSION:
                if any(row[key] is not None for key in _CAPTURE_KEYS):
                    raise ValueError("仅灯具读数的报告不能包含实拍曝光或 RAW 文件信息。")
            else:
                settings.append(CaptureSettings(row["camera"], row["iso"], row["shutter_seconds"],
                                                row["aperture"], row["shape"]))
                sources.append(row["source_name"])
        if version == LAMP_ONLY_VERSION:
            return cls.from_lamps(tuple(lamps))
        return cls(tuple(settings), tuple(lamps), tuple(sources))

    @classmethod
    def read_report(cls, path):
        try:
            file = Path(path)
            if not file.is_file():
                raise OSError("not a regular file")
            with file.open("rb") as stream:
                content = stream.read(MAX_REPORT_BYTES + 1)
        except (OSError, TypeError, ValueError):
            raise ValueError("无法读取所选的曝光参考报告。") from None
        if len(content) > MAX_REPORT_BYTES:
            raise ValueError("报告太大，无法载入曝光参考。")
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError("请载入 UTF-8 文本格式的曝光参考报告。") from None
        return cls.from_report(text)

    def to_dict(self):
        channels = []
        for channel, setting, lamp, name in zip(CHANNELS, self.settings or _EMPTY, self.lamp_values, self.source_names):
            if setting is None:
                capture = dict.fromkeys(_CAPTURE_KEYS)
            else:
                capture = dict(camera=list(setting.camera), iso=setting.iso, shutter_seconds=setting.shutter,
                               aperture=setting.aperture, shape=list(setting.shape), source_name=name)
            channels.append(dict(channel=channel, lamp=lamp, **capture))
        version = LAMP_ONLY_VERSION if self.settings is None else VERSION
        return dict(schema=SCHEMA, version=version, channels=channels)

    def append_to_report(self, human_text, *, include_reference_data=True):
        if _utf8_size(human_text) > MAX_REPORT_BYTES:
            raise ValueError("报告太大，无法附加曝光参考。")
        if type(include_reference_data) is not bool:
            raise ValueError("曝光参考的导出选项无效。")
        if any(line.strip() in (REFERENCE_BEGIN, REFERENCE_END) for line in human_text.splitlines()):
            raise ValueError("这份报告已含曝光参考区块。")
        heading, _, body = human_text.strip("\r\n").partition("\n")
        parts = [part for part in (heading.rstrip("\r"), "下次拍摄参考\n" + self.summary_text(), body.rstrip()) if part]
        if include_reference_data:
            block = json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            if _utf8_size(block) > MAX_REFERENCE_BYTES:
                raise ValueError("曝光参考数据过大。")
            parts.append(f"{REFERENCE_BEGIN}\n{block}\n{REFERENCE_END}")
        text = "\n\n".join(parts) + "\n"
        if _utf8_size(text) > MAX_REPORT_BYTES:
            raise ValueError("报告太大，无法附加曝光参考。")
        return text

    def exposure_summary_text(self):
        if self.settings is None:
            return "未记录实拍曝光"
        first = self.settings[0]
        if all(_compare_settings(first, setting) == ((), True) for setting in self.settings[1:]):
            return _settings_text(first)
        return "\n".join(f"{channel} · {_settings_text(setting)}"
                         for channel, setting in zip(CHANNELS, self.settings))

    def summary_text(self):
        lines = [self.exposure_summary_text()]
        if any(value is not None for value in self.lamp_values):
            lamps = " · ".join(f"{channel} {value if value is not None else '—'}"
                               for channel, value in zip(CHANNELS, self.lamp_values))
            lines.insert(0, f"灯具读数  {lamps}")
        return "\n".join(lines)

    def lamp_adjustment_text(self, current_values):
        """Describe same-unit changes from current lamp controls to the saved readings."""
        changes = []
        for channel, saved, supplied in zip(CHANNELS, self.lamp_values, _triplet(current_values)):
            try:
                current = _optional_text(supplied, MAX_LAMP_VALUE_LENGTH, f"{channel} 路灯光值")
            except ValueError:
                changes.append(f"{channel} 读数无法比较")
                continue
            if saved is None and current is None:
                continue
            changes.append(_lamp_adjustment(channel, saved, current))
        return " · ".join(changes)

    def compare_exposure(self, other):
        if not isinstance(other, BalanceReference):
            raise ValueError("需要两份有效的曝光参考才能比较。")
        if self.settings is None or other.settings is None:
            return (), False
        differences = []
        complete = True
        for channel, current, saved in zip(CHANNELS, self.settings, other.settings):
            fields, both_complete = _compare_settings(current, saved)
            differences.extend(f"{channel} {field}" for field in fields)
            complete = complete and both_complete
        return tuple(differences), complete


def _triplet(values):
    if not isinstance(values, (tuple, list)) or len(values) != len(CHANNELS):
        raise ValueError("曝光参考需要按 R、G、B 顺序的三路数据。")
    return values


def _optional_text(value, max_length, name):
    if value is None:
        return None
    if (not isinstance(value, str) or len(value) > max_length
            or any(not char.isprintable() and unicodedata.category(char) != "Zs" for char in value)):
        raise ValueError(f"{name}应为至多 {max_length} 字的一行文字。")
    return value.strip() or None


def _lamp_adjustment(channel, saved, current):
    if saved is None:
        return f"{channel} 缺少已存读数"
    if current is None:
        return f"{channel} 缺少当前读数"

    target_value = _LAMP_NUMBER.fullmatch(saved.translate(_LAMP_CHARACTERS))
    current_value = _LAMP_NUMBER.fullmatch(current.translate(_LAMP_CHARACTERS))
    if target_value is None or current_value is None:
        return f"{channel} 读数无法比较"
    if target_value[2] != current_value[2]:
        return f"{channel} 读数单位不同"

    target, actual = Decimal(target_value[1]), Decimal(current_value[1])
    # Each input is bounded. The sum of their widths accommodates both place
    # values and a carry even when they use opposite signs and scales.
    with localcontext() as context:
        context.prec = 2 * MAX_LAMP_VALUE_LENGTH + 2
        difference = target - actual
    is_percent = target_value[2] is not None
    destination = _decimal_text(target) + ("%" if is_percent else "")
    if difference.is_zero():
        return f"{channel} 保持 {destination}"
    action = "增加" if difference > 0 else "减少"
    magnitude = _decimal_text(difference.copy_abs())
    units = " 个百分点" if is_percent else ""
    return f"{channel} {action} {magnitude}{units} → {destination}"


def _decimal_text(value):
    if value.is_zero():
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _source_name(value):
    name = _optional_text(value, 255, "RAW 文件名")
    if name is not None and ("/" in name or "\\" in name or name in (".", "..")):
        raise ValueError("曝光参考只接受 RAW 文件名，不接受路径。")
    return name


def _number(value, name):
    if value is None:
        return None
    if type(value) not in (int, float):
        raise ValueError(f"曝光参考中的{name}应为有效的正数。")
    try:
        number = float(value)
    except OverflowError:
        number = math.inf
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"曝光参考中的{name}应为有效的正数。")
    return number


def _capture(setting):
    if not isinstance(setting, CaptureSettings):
        raise ValueError("曝光参考缺少相机实拍参数。")
    if (not isinstance(setting.camera, (tuple, list)) or len(setting.camera) != 2
            or any(not isinstance(value, str) for value in setting.camera)):
        raise ValueError("曝光参考中的相机型号字段无效。")
    camera = tuple(_optional_text(value, 120, "相机型号") or "" for value in setting.camera)
    if (not isinstance(setting.shape, (tuple, list)) or len(setting.shape) != 2
            or any(type(value) is not int or not 0 < value <= 1_000_000 for value in setting.shape)):
        raise ValueError("曝光参考中的 RAW 画幅尺寸无效。")
    return CaptureSettings(camera, _number(setting.iso, "ISO"), _number(setting.shutter, "快门"),
                           _number(setting.aperture, "光圈"), tuple(setting.shape))


def _compare_settings(first, second):
    differences = []
    complete = bool(all(first.camera) and all(second.camera))
    if complete and tuple(value.casefold() for value in first.camera) != tuple(value.casefold() for value in second.camera):
        differences.append("机身")
    for name, label in (("iso", "ISO"), ("shutter", "快门"), ("aperture", "光圈")):
        a, b = getattr(first, name), getattr(second, name)
        if a is None or b is None:
            complete = False
        elif not math.isclose(a, b, rel_tol=1e-4):
            differences.append(label)
    return tuple(differences), complete


def _settings_text(setting):
    source = " ".join(value for value in setting.camera if value) or "相机未记录"
    iso = f"ISO {setting.iso:g}" if setting.iso is not None else "ISO 未记录"
    if setting.shutter is None:
        shutter = "快门未记录"
    elif 1e-6 <= setting.shutter <= 1e4:
        shutter = _shutter_label(setting.shutter)
    else:
        shutter = f"{setting.shutter:.5g} 秒"
    aperture = f"f/{setting.aperture:g}" if setting.aperture is not None else "光圈未记录"
    return " · ".join((source, iso, shutter, aperture))


def _utf8_size(text):
    if isinstance(text, str):
        try:
            return len(text.encode("utf-8"))
        except UnicodeEncodeError:
            pass
    raise ValueError("曝光参考报告需要有效的 UTF-8 文本。")


def _unique_object(pairs):
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ValueError("重复字段")
    return fields
