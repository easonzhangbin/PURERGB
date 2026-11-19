"""Explicit FlexColor research ICC bridges and original Gray-CMS arithmetic.

LittleCMS is a portable ICC implementation, not proof of legacy ColorSync or
camera-to-scanner equivalence. No raw/density/scalar-curve array receives an ICC
implicitly. Profile bytes, hashes, input domain and assumptions are mandatory.
"""

from dataclasses import dataclass
import hashlib
import math
import struct
import time

import imagecodecs
import numpy as np

from .flex_chain import gray_no_cms_parameters


@dataclass(frozen=True)
class ICCBridgeResult:
    pixels: np.ndarray
    icc: bytes
    receipt: dict


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _pixel_sha(pixels):
    return _sha(memoryview(pixels).cast("B"))


def _profile(payload, expected, role):
    if not isinstance(payload, bytes) or not 132 <= len(payload) <= 20_000_000:
        raise ValueError(role + "需明确的 ICC bytes，不能是默认名称／路径标记")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or _sha(payload) != expected
    ):
        raise ValueError(role + " ICC SHA256 不匹配")
    size = struct.unpack_from(">I", payload)[0]
    if size != len(payload) or payload[36:40] != b"acsp":
        raise ValueError(role + " ICC header／长度无效")
    if payload[16:20] != b"RGB " or payload[20:24] not in (b"XYZ ", b"Lab "):
        raise ValueError(role + "需 RGB→XYZ/Lab ICC；CMYK、GRAY、DeviceLink 不在本接口")
    if payload[12:16] not in (b"scnr", b"mntr", b"prtr", b"spac"):
        raise ValueError(role + " ICC class 不在 RGB profile 研究范围")
    # Real Hasselblad RGB is tagged scnr; its declared output role is explicit,
    # not inferred from class alone. LittleCMS must still build the actual route.
    return {
        "sha256": expected,
        "bytes": size,
        "class": payload[12:16].decode("ascii"),
        "color_space": "RGB",
        "PCS": payload[20:24].decode("ascii").strip(),
        "role": role,
    }


def _codes(pixels, maximum):
    if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint16:
        raise ValueError("需明确的 uint16 RGB 编码，不可隐式解释浮点RAW／密度")
    if pixels.ndim not in (2, 3) or pixels.shape[-1] != 3 or not pixels.size:
        raise ValueError("需非空 N×3 或 H×W×3 RGB 数据")
    if not pixels.flags.c_contiguous:
        raise ValueError("需明确 C-contiguous 输入，避免大图隐式复制")
    if pixels.max() > maximum:
        raise ValueError(f"输入超过已声明的 0..{maximum} 编码域")
    return pixels


def _transform(
    pixels,
    source_icc,
    output_icc,
    *,
    source_sha256,
    output_sha256,
    input_domain,
    provenance,
    intent,
    chunk_pixels,
    input_shift=0,
):
    source = _profile(source_icc, source_sha256, "input-profile")
    target = _profile(output_icc, output_sha256, "output-rgb-profile")
    if type(intent) is not int or intent not in (0, 1, 2, 3):
        raise ValueError("需明确 ICC rendering intent 0／1／2／3")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("需记录输入域与 profile 关联的来源／研究假设")
    if type(chunk_pixels) is not int or not 1 <= chunk_pixels <= 4_194_304:
        raise ValueError("ICC 分块需 1..4194304 像素")
    started = time.monotonic()
    output = np.empty_like(pixels)
    flat, destination = pixels.reshape(-1, 3), output.reshape(-1, 3)
    encoded_hash = hashlib.sha256()
    for offset in range(0, len(flat), chunk_pixels):
        block = flat[offset : offset + chunk_pixels].reshape(1, -1, 3)
        if input_shift:
            block = np.left_shift(block, input_shift)
        encoded_hash.update(memoryview(block).cast("B"))
        values = imagecodecs.cms_transform(
            block,
            source_icc,
            output_icc,
            colorspace="RGB",
            outcolorspace="RGB",
            outdtype=np.uint16,
            intent=intent,
            flags=0x0100,
        )
        if values.dtype != np.uint16 or values.shape != block.shape:
            raise ValueError("ICC 引擎没有返回约定的 RGB16 数据")
        values = values.reshape(-1, 3)
        destination[offset : offset + len(values)] = values
    receipt = {
        "engine": imagecodecs.cms_version(),
        "intent": intent,
        "flags": {
            "value": 0x0100,
            "meaning": "NOOPTIMIZE; no implicit black-point compensation",
        },
        "input_domain": input_domain,
        "provenance": provenance,
        "source_profile": source,
        "output_profile": target,
        "input_pixels_sha256": encoded_hash.hexdigest(),
        "output_pixels_sha256": _pixel_sha(output),
        "shape": list(pixels.shape),
        "dtype": "uint16",
        "maximum_temporary_transform_pixels": min(len(flat), chunk_pixels),
        "elapsed_seconds": time.monotonic() - started,
        "original_colorsync_equivalence_proven": False,
        "camera_to_scanner_calibration_proven": False,
        "extra_gamma_or_oetf_applied": False,
    }
    return ICCBridgeResult(output, output_icc, receipt)


def rgb16_icc_transform(
    pixels,
    source_icc,
    output_icc,
    *,
    source_sha256,
    output_sha256,
    input_domain,
    provenance,
    intent=1,
    chunk_pixels=1_048_576,
):
    """Transform explicitly profile-encoded RGB16, retaining actual output ICC.

    ICC assignment must already be justified by the caller's source provenance.
    Unknown raw, density, scalar-curve and negative domains are rejected.
    """
    if input_domain != "icc-encoded-rgb16":
        raise ValueError("该接口只接受已声明为 icc-encoded-rgb16 的数据")
    _codes(pixels, 65535)
    return _transform(
        pixels,
        source_icc,
        output_icc,
        source_sha256=source_sha256,
        output_sha256=output_sha256,
        input_domain=input_domain,
        provenance=provenance,
        intent=intent,
        chunk_pixels=chunk_pixels,
    )


def scanner_rgb14_icc_bridge(
    pixels,
    source_icc,
    output_icc,
    *,
    source_sha256,
    output_sha256,
    input_domain,
    assumption,
    intent=1,
    chunk_pixels=1_048_576,
):
    """Explicit uncalibrated scanner-code hypothesis, not accepted production CMS.

    Original ColorSync RGBToRGB prepares CMM words with input<<2. We preserve
    RGB16 CMM output for export rather than throwing away its bottom two bits;
    the original internal return-code route would use output>>2.
    """
    if input_domain != "assumed-flextight-scanner-rgb14":
        raise ValueError(
            "仅允许显式 assumed-flextight-scanner-rgb14 研究分支；未知相机输入不能自动标定"
        )
    _codes(pixels, 16383)
    if not isinstance(assumption, str) or not assumption.strip():
        raise ValueError("需明确记录相机→扫描域尚未验证的假设")
    result = _transform(
        pixels,
        source_icc,
        output_icc,
        source_sha256=source_sha256,
        output_sha256=output_sha256,
        input_domain=input_domain,
        provenance=assumption,
        intent=intent,
        chunk_pixels=chunk_pixels,
        input_shift=2,
    )
    result.receipt.update(
        {
            "source_rgb14_sha256": _pixel_sha(pixels),
            "cmm_input_adapter": "unsigned 14-bit word << 2; maximum 16383 -> 65532",
            "output_adapter": "retain actual CMM RGB16; original internal return uses >>2",
            "scope": "InputProfile -> explicitly selected RGBProfile research bridge; not original ColorSync or calibrated camera capture",
        }
    )
    return result


def gray_cms_request(
    gray, *, film_type, use_colorsync, wants_embed, manager_present=True
):
    """Original CalcNeutralCurves gate and neutral sample in internal code units."""
    if (
        not isinstance(gray, (list, tuple))
        or len(gray) != 4
        or any(type(value) is not int or not 0 <= value <= 255 for value in gray)
    ):
        raise ValueError("Gray 需四个0..255整数控制值")
    if type(film_type) is not int or film_type not in (0, 1, 2):
        raise ValueError("FilmType需0／1／2")
    if any(
        type(value) is not bool
        for value in (use_colorsync, wants_embed, manager_present)
    ):
        raise ValueError("CMS开关需明确布尔值")
    mean = np.float32(sum(gray[1:]) / 3.0)
    active = (
        manager_present
        and (use_colorsync or wants_embed)
        and tuple(gray[1:]) != (128, 128, 128)
        and film_type == 0
    )
    neutral = math.trunc(float(np.float32(mean * np.float32(64))))
    return {
        "gray": list(gray),
        "mean_float32": float(mean),
        "cms_required": active,
        "neutral_rgb14": [neutral] * 3,
        "film_type": film_type,
        "use_colorsync": use_colorsync,
        "wants_embed": wants_embed,
        "manager_present": manager_present,
        "original_call": {
            "color_world_type": 0,
            "count": 1,
            "output_color_type": 2,
            "input_profile_role": "InputProfile",
            "target_profile_role": "DisplayProfile",
        },
        "scope": "original Gray CMS gate/sample; world=0 is InputProfile -> DisplayProfile, separate from RGB output route",
    }


def gray_parameters_with_cms_sample(
    gray,
    transformed_neutral_rgb14=None,
    *,
    film_type,
    use_colorsync,
    wants_embed,
    manager_present=True,
):
    """Finish original Gray arithmetic with caller-supplied real/declared CMS sample.

    An active CMS branch requires its transformed neutral sample. It never silently
    uses the no-CMS arithmetic mean as a substitute for a missing color service.
    """
    request = gray_cms_request(
        gray,
        film_type=film_type,
        use_colorsync=use_colorsync,
        wants_embed=wants_embed,
        manager_present=manager_present,
    )
    if not request["cms_required"]:
        return {
            "parameters": gray_no_cms_parameters(gray),
            "request": request,
            "cms_branch_selected": False,
            "cms_service_executed_here": False,
            "source": "original gate bypass; no-CMS branch",
        }
    if (
        not isinstance(transformed_neutral_rgb14, (list, tuple))
        or len(transformed_neutral_rgb14) != 3
        or any(
            type(v) is not int or not 0 <= v <= 16383 for v in transformed_neutral_rgb14
        )
    ):
        raise ValueError(
            "原CMS分支需要明确的三通道变换后0..16383中性样点，不能用平均灰替代"
        )
    base = request["neutral_rgb14"][0]
    deltas = [value - base for value in transformed_neutral_rgb14]
    mean_delta = math.trunc(sum(deltas) / 3)
    parameters = []
    for value, delta in zip(gray[1:], deltas):
        correction = math.trunc((delta - mean_delta) / 64)
        target = math.trunc(
            float(
                np.float32(np.float32(request["mean_float32"]) - np.float32(correction))
            )
        )
        signed_target = ((target + 32768) & 65535) - 32768
        parameters.append((float(value), float(signed_target)))
    return {
        "parameters": tuple(parameters),
        "request": request,
        "cms_branch_selected": True,
        "cms_service_executed_here": False,
        "transformed_neutral_rgb14": list(transformed_neutral_rgb14),
        "mean_channel_delta": mean_delta,
        "source": "original post-CMS integer/float32 arithmetic; CMS numeric service supplied separately",
    }


def gray_parameters_via_icc(
    gray,
    input_icc,
    display_icc,
    *,
    source_sha256,
    display_sha256,
    input_domain,
    assumption,
    film_type,
    use_colorsync,
    wants_embed,
    manager_present=True,
    intent=1,
):
    """Portable Gray world=0 model with explicit display ICC, not an output preset.

    The branch/packing/arithmetic follow original instructions. Numeric conversion
    is LittleCMS and is not proof of original ColorSync equivalence. Negative
    FilmType 1/2 correctly bypass this Gray CMS conversion.
    """
    request = gray_cms_request(
        gray,
        film_type=film_type,
        use_colorsync=use_colorsync,
        wants_embed=wants_embed,
        manager_present=manager_present,
    )
    if not request["cms_required"]:
        result = gray_parameters_with_cms_sample(
            gray,
            film_type=film_type,
            use_colorsync=use_colorsync,
            wants_embed=wants_embed,
            manager_present=manager_present,
        )
        result["portable_icc_receipt"] = None
        return result
    bridge = scanner_rgb14_icc_bridge(
        np.array([request["neutral_rgb14"]], dtype=np.uint16),
        input_icc,
        display_icc,
        source_sha256=source_sha256,
        output_sha256=display_sha256,
        input_domain=input_domain,
        assumption=assumption,
        intent=intent,
    )
    transformed = np.right_shift(bridge.pixels.reshape(-1, 3)[0], 2).tolist()
    result = gray_parameters_with_cms_sample(
        gray,
        transformed,
        film_type=film_type,
        use_colorsync=use_colorsync,
        wants_embed=wants_embed,
        manager_present=manager_present,
    )
    bridge.receipt["profile_route"] = (
        "InputProfile -> DisplayProfile (Gray world=0); not RGB output profile"
    )
    result["portable_icc_receipt"] = bridge.receipt
    return result
