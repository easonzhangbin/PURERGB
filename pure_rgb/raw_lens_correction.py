"""Apply a matched embedded optical profile in registered sensor coordinates."""

from dataclasses import dataclass, replace

from negpy.features.lens.models import LensCorrections, LensMetadata


@dataclass(frozen=True)
class RegisteredLensWarp:
    warp: object
    sensor_shape: tuple
    visible_roi: tuple

    @property
    def has_distortion(self):
        return self.warp.has_distortion

    @property
    def has_ca(self):
        return self.warp.has_ca

    def remap(self, lens, shape, start, stop, channel, corrections):
        left, top, right, _ = self.visible_roi
        x, y = self.warp.remap(
            lens, self.sensor_shape, start + top, stop + top, channel, corrections
        )
        return x[:, left:right] - left, y[:, left:right] - top


def matching_lens(profiles):
    first = profiles[0]
    if not isinstance(first, LensMetadata) or not first.available:
        raise ValueError("RAW 未提供可用的内嵌镜头矫正参数")
    if any(profile != first for profile in profiles[1:]):
        raise ValueError("三张 RAW 的内嵌镜头矫正参数不一致")
    return first


def correct_registered_lens(source, profiles, plan, effective, orientation):
    from negpy.features.lens.logic import apply_lens, fill_scale

    lens = matching_lens(profiles)
    mapped = replace(lens, warps=tuple(
        RegisteredLensWarp(warp, plan.source_shape, effective.visible_roi)
        for warp in lens.warps
    ))
    options = LensCorrections(lens.distortion, lens.ca)
    x0, y0, x1, y1 = effective.visible_roi
    scale = fill_scale(mapped, (y1 - y0, x1 - x0), options)
    pixels = apply_lens(source, mapped, orientation, options)
    return pixels, dict(version=1, profile=lens.source, distortion=lens.distortion,
                        lateral_ca=lens.ca, fill_scale=scale,
                        sensor_xyxy=list(effective.visible_roi),
                        policy="registered-linear-after-flatfield")


def camera_lens_info(paths, correctable):
    from negpy.infrastructure.loaders.helpers import read_exif_from_file
    from negpy.infrastructure.loaders.lens_metadata import read_lens_metadata

    exif = read_exif_from_file(paths[len(paths) // 2]) or {}
    primary, capture = exif.get("0th", {}), exif.get("Exif", {})

    def text(value):
        return value.decode(errors="replace").strip("\x00 ") if isinstance(value, bytes) else str(value or "")

    camera = " ".join(filter(None, (text(primary.get(271)), text(primary.get(272)))))
    name = text(capture.get(42036)) or "未记录镜头型号"
    focal = capture.get(37386)
    if isinstance(focal, tuple) and len(focal) == 2 and focal[1]:
        name += f" {focal[0] / focal[1]:g} mm"
    description = f"{camera or '未读取机身型号'} · {name}"
    if not correctable:
        return description, False, "此输入方式可使用匹配平场；当前内嵌镜头矫正用于标准三色 RAW。"
    try:
        lens = matching_lens([read_lens_metadata(path) for path in paths])
    except ValueError as error:
        return description, False, str(error)
    options = "＋".join(label for active, label in ((lens.distortion, "畸变"), (lens.ca, "横向色差")) if active)
    return description, True, f"检测到匹配的 RAW 内嵌参数，可矫正{options}。暗角与照明不均使用平场。"
