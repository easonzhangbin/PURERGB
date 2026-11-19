"""Explicit macOS ColorSync RGB16 research boundary, separate from defaults.

This invokes the installed framework. It is not an oracle for the ColorSync
version/CMM/preferences used by the historical FlexColor application. Input
codes must already be associated with the explicitly supplied profile.
"""

import ctypes
import hashlib
import platform
import sys

import numpy as np

from .flex_color_management import _profile

DOMAIN = "profile-associated-rgb16"
FRAMEWORK = "/System/Library/Frameworks/ColorSync.framework/ColorSync"
FOUNDATION = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
INTENTS = ("Perceptual", "Relative", "Saturation", "Absolute")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _layout(image):
    return (image.shape, image.dtype.str, image.strides, bool(image.flags.c_contiguous))


def _api():
    if sys.platform != "darwin":
        raise RuntimeError("Modern ColorSync requires macOS; no silent CMS fallback")
    if sys.byteorder != "little":
        raise RuntimeError(
            "Modern ColorSync probe requires explicit little-endian host"
        )
    cf, cs = ctypes.CDLL(FOUNDATION), ctypes.CDLL(FRAMEWORK)
    pointer = ctypes.c_void_p
    signatures = {
        "CFDataCreate": (pointer, [pointer, pointer, ctypes.c_long]),
        "CFDictionaryCreate": (
            pointer,
            [pointer, pointer, pointer, ctypes.c_long, pointer, pointer],
        ),
        "CFArrayCreate": (pointer, [pointer, pointer, ctypes.c_long, pointer]),
        "CFDictionaryGetValue": (pointer, [pointer, pointer]),
        "CFGetTypeID": (ctypes.c_ulong, [pointer]),
        "CFStringGetTypeID": (ctypes.c_ulong, []),
        "CFStringGetCString": (
            ctypes.c_bool,
            [pointer, pointer, ctypes.c_long, ctypes.c_uint32],
        ),
        "CFRelease": (None, [pointer]),
    }
    for name, (result, args) in signatures.items():
        getattr(cf, name).restype, getattr(cf, name).argtypes = result, args
    cs.ColorSyncProfileCreate.restype = pointer
    cs.ColorSyncProfileCreate.argtypes = [pointer, ctypes.POINTER(pointer)]
    cs.ColorSyncTransformCreate.restype = pointer
    cs.ColorSyncTransformCreate.argtypes = [pointer, pointer]
    cs.ColorSyncTransformCopyProperty.restype = pointer
    cs.ColorSyncTransformCopyProperty.argtypes = [pointer, pointer, pointer]
    cs.ColorSyncTransformConvert.restype = ctypes.c_bool
    cs.ColorSyncTransformConvert.argtypes = [
        pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        pointer,
        ctypes.c_uint,
        ctypes.c_uint32,
        ctypes.c_size_t,
        pointer,
        ctypes.c_uint,
        ctypes.c_uint32,
        ctypes.c_size_t,
        pointer,
    ]
    return cf, cs


def convert_modern_colorsync_rgb16(
    source,
    source_icc,
    target_icc,
    *,
    source_sha256,
    target_sha256,
    input_domain,
    assumption,
    intent=1,
    block_pixels=262144,
):
    """Bounded RGB16 conversion with explicit intent, best quality and BPC off."""
    if (
        not isinstance(source, np.ndarray)
        or source.dtype != np.dtype("uint16")
        or source.ndim not in (2, 3)
        or source.shape[-1] != 3
        or not source.size
        or not source.flags.c_contiguous
        or source.flags.writeable
    ):
        raise ValueError("ColorSync source must be readonly native uint16 C-order RGB")
    if (
        input_domain != DOMAIN
        or not isinstance(assumption, str)
        or not assumption.strip()
    ):
        raise ValueError(
            "Explicit profile-associated code domain and provenance required"
        )
    if type(intent) is not int or intent not in range(4):
        raise ValueError("Explicit ICC intent0..3 required")
    if type(block_pixels) is not int or not 1 <= block_pixels <= 4194304:
        raise ValueError("Bounded integer block_pixels required")
    src_profile = _profile(source_icc, source_sha256, "declared-input-profile")
    dst_profile = _profile(target_icc, target_sha256, "declared-output-profile")
    source_layout, source_pixels_sha = (
        _layout(source),
        _sha(memoryview(source).cast("B")),
    )
    cf, cs = _api()
    owned = []

    def own(value, name):
        if not value:
            raise ValueError("Modern ColorSync could not create " + name)
        owned.append(value)
        return value

    def symbol(library, name):
        return ctypes.c_void_p.in_dll(library, name).value

    def callbacks(name):
        # Address of the actual exported callback table, not its first field.
        return ctypes.addressof(ctypes.c_byte.in_dll(cf, name))

    def profile(payload):
        raw = ctypes.create_string_buffer(payload)
        data = own(cf.CFDataCreate(None, raw, len(payload)), "profile CFData")
        error = ctypes.c_void_p()
        value = cs.ColorSyncProfileCreate(data, ctypes.byref(error))
        if error.value:
            cf.CFRelease(error.value)
        return own(value, "ICC profile")

    def dictionary(items):
        keys = (ctypes.c_void_p * len(items))(*[key for key, _ in items])
        values = (ctypes.c_void_p * len(items))(*[value for _, value in items])
        return own(
            cf.CFDictionaryCreate(
                None,
                keys,
                values,
                len(items),
                callbacks("kCFTypeDictionaryKeyCallBacks"),
                callbacks("kCFTypeDictionaryValueCallBacks"),
            ),
            "dictionary",
        )

    def text(value):
        if not value or cf.CFGetTypeID(value) != cf.CFStringGetTypeID():
            return None
        buffer = ctypes.create_string_buffer(4096)
        if not cf.CFStringGetCString(value, buffer, len(buffer), 0x08000100):
            return None
        return buffer.value.decode("utf-8")

    try:
        first, second = profile(source_icc), profile(target_icc)
        intent_symbol = symbol(cs, "kColorSyncRenderingIntent" + INTENTS[intent])
        common = [
            (symbol(cs, "kColorSyncRenderingIntent"), intent_symbol),
            (
                symbol(cs, "kColorSyncBlackPointCompensation"),
                symbol(cf, "kCFBooleanFalse"),
            ),
        ]
        src = dictionary(
            [
                *common,
                (symbol(cs, "kColorSyncProfile"), first),
                (
                    symbol(cs, "kColorSyncTransformTag"),
                    symbol(cs, "kColorSyncTransformDeviceToPCS"),
                ),
            ]
        )
        dst = dictionary(
            [
                *common,
                (symbol(cs, "kColorSyncProfile"), second),
                (
                    symbol(cs, "kColorSyncTransformTag"),
                    symbol(cs, "kColorSyncTransformPCSToDevice"),
                ),
            ]
        )
        members = (ctypes.c_void_p * 2)(src, dst)
        sequence = own(
            cf.CFArrayCreate(None, members, 2, callbacks("kCFTypeArrayCallBacks")),
            "profile sequence",
        )
        options = dictionary(
            [
                (
                    symbol(cs, "kColorSyncConvertQuality"),
                    symbol(cs, "kColorSyncBestQuality"),
                )
            ]
        )
        transform = own(cs.ColorSyncTransformCreate(sequence, options), "transform")
        creator = None
        info = cs.ColorSyncTransformCopyProperty(
            transform, symbol(cs, "kColorSyncTransformInfo"), None
        )
        if info:
            owned.append(info)
            creator = text(
                cf.CFDictionaryGetValue(info, symbol(cs, "kColorSyncTransformCreator"))
            )
        output = np.empty(source.shape, dtype=np.uint16)
        source_flat, dest_flat = source.reshape(-1, 3), output.reshape(-1, 3)
        conversions = 0
        for start in range(0, len(source_flat), block_pixels):
            block = source_flat[start : start + block_pixels]
            destination = dest_flat[start : start + len(block)]
            ok = cs.ColorSyncTransformConvert(
                transform,
                len(block),
                1,
                destination.ctypes.data,
                3,
                1 << 12,
                destination.nbytes,
                block.ctypes.data,
                3,
                1 << 12,
                block.nbytes,
                None,
            )
            if not ok:
                raise RuntimeError(
                    "Modern ColorSync RGB16 conversion failed; no fallback"
                )
            conversions += 1
        if (
            _layout(source) != source_layout
            or _sha(memoryview(source).cast("B")) != source_pixels_sha
        ):
            raise ValueError("ColorSync source layout or bytes changed")
        output.setflags(write=False)
        return output, {
            "schema": "pure-rgb-modern-colorsync-rgb16",
            "version": 1,
            "source_profile": src_profile,
            "target_profile": dst_profile,
            "input_domain": input_domain,
            "assumption": assumption,
            "shape": list(source.shape),
            "source_pixels_sha256": source_pixels_sha,
            "output_pixels_sha256": _sha(memoryview(output).cast("B")),
            "intent": intent,
            "BPC": False,
            "quality": "explicit-BestQuality",
            "source_tag": "DeviceToPCS",
            "target_tag": "PCSToDevice",
            "component_depth": "native uint16 RGB, explicit little-endian layout",
            "backend": FRAMEWORK,
            "framework_conversion_calls": conversions,
            "CMM_creator": creator,
            "macOS": platform.mac_ver()[0],
            "platform_release": platform.release(),
            "machine": platform.machine(),
            "current_system_framework_not_byte_bound": True,
            "legacy_FlexColor_ColorSync_equivalence_claimed": False,
            "camera_scanner_calibration_certified": False,
            "production_defaults_changed": False,
            "source_unchanged": True,
        }
    finally:
        for value in reversed(owned):
            cf.CFRelease(value)
