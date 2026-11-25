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


def correct_registered_lens(source, profiles, plan, effective, orientation, *, corrections=None):
    from negpy.features.lens.logic import apply_lens, fill_scale

    lens = matching_lens(profiles)
    mapped = replace(lens, warps=tuple(
        RegisteredLensWarp(warp, plan.source_shape, effective.visible_roi)
        for warp in lens.warps
    ))
    options = corrections if corrections is not None else LensCorrections(lens.distortion, lens.ca)
    x0, y0, x1, y1 = effective.visible_roi
    scale = fill_scale(mapped, (y1 - y0, x1 - x0), options)
    pixels = apply_lens(source, mapped, orientation, options)
    return pixels, dict(version=1, profile=lens.source, distortion=bool(options.distortion and lens.distortion),
                        lateral_ca=bool(options.ca and lens.ca), fill_scale=scale,
                        sensor_xyxy=list(effective.visible_roi),
                        policy="registered-linear-after-flatfield")


def correct_additional_optics(source, profiles, paths, plan, effective, orientation, *,
                              chromatic_aberration=False, lens_profile=None,
                              distortion=False, embedded_receipt=None, registration=None,
                              baseline_context=None):
    """Resolve explicit local LCP, embedded CA, then cautious image estimation."""
    from .adobe_lens_profiles import apply_lcp, capture_lens

    if lens_profile:
        capture = capture_lens(paths)
        if distortion and chromatic_aberration and baseline_context is not None:
            # Authenticate the same common distortion-only image as a recipe
            # with CA bypassed. Its green pixels are the dust coordinate frame.
            base, base_receipt = apply_lcp(source, lens_profile, capture, plan, effective, orientation,
                                           ca=False, distortion=True, registration=registration)
            baseline_context.update(pixels=base, lens_receipt=base_receipt)
        pixels, receipt = apply_lcp(source, lens_profile, capture, plan, effective,
                                    orientation, ca=chromatic_aberration, distortion=distortion,
                                    registration=registration)
        if not chromatic_aberration or receipt.get("lateral_ca"):
            return pixels, receipt if distortion else None, receipt
        # An explicitly selected distortion-only LCP can still be followed by
        # conservative CA estimation; record both parts of the optical chain.
        source = pixels
        optical_receipt = receipt if distortion else None
    else:
        optical_receipt = None
    if not chromatic_aberration:
        return source, optical_receipt, dict(status="disabled", green_unchanged=True)
    if embedded_receipt and embedded_receipt.get("lateral_ca"):
        return source, None, dict(version=1, status="applied", method="embedded-raw",
                                 green_unchanged=True, canvas_unchanged=True,
                                 reason="内嵌镜头参数已包含横向色差，未重复应用")
    if not lens_profile:
        try:
            lens = matching_lens(profiles)
        except ValueError:
            lens = None
        if lens is not None and lens.ca:
            pixels, receipt = correct_registered_lens(source, profiles, plan, effective, orientation,
                                                       corrections=LensCorrections(False, True))
            return pixels, None, dict(version=1, status="applied", method="embedded-raw",
                                      green_unchanged=True, canvas_unchanged=True, optical_model=receipt)
    from .chromatic_aberration import correct_chromatic_aberration

    pixels, receipt = correct_chromatic_aberration(source)
    if optical_receipt:
        receipt["preceding_lcp"] = optical_receipt
        receipt["green_unchanged"] = False
    return pixels, optical_receipt, receipt


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
