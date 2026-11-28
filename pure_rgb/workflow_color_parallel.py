"""Pixel-exact display ICC transforms in independent horizontal bands."""

from concurrent.futures import ThreadPoolExecutor, wait
import ctypes
from functools import lru_cache
import io
from threading import Lock
from weakref import finalize

import numpy as np
from PIL import Image, ImageCms


# LittleCMS packed formats. Qt RGB32 on little-endian hosts stores B, G, R, 255.
_RGB8 = (4 << 16) | (3 << 3) | 1
_BGRA8 = _RGB8 | (1 << 7) | (1 << 10) | (1 << 14)


@lru_cache(maxsize=1)
def _pillow_lcms():
    """Use the exact LCMS linked by Pillow, with no extra system dependency."""
    try:
        from PIL import _imagingcms

        library = ctypes.CDLL(_imagingcms.__file__)
        pointer, uint32 = ctypes.c_void_p, ctypes.c_uint32
        for name, result, arguments in (
            ("cmsOpenProfileFromMem", pointer, (ctypes.c_char_p, uint32)),
            ("cmsCloseProfile", ctypes.c_int, (pointer,)),
            ("cmsCreateTransform", pointer, (pointer, uint32, pointer, uint32, uint32, uint32)),
            ("cmsDeleteTransform", None, (pointer,)),
            ("cmsDoTransform", None, (pointer, pointer, pointer, uint32)),
        ):
            function = getattr(library, name)
            function.restype, function.argtypes = result, arguments
    except (ImportError, OSError, AttributeError):
        return None
    return library


class _PackedTransform:
    def __init__(self, library, handle):
        self.handle = handle
        self.lock = Lock()
        self.release = finalize(self, library.cmsDeleteTransform, handle)


@lru_cache(maxsize=24)
def _packed_transform(source_icc, target_icc, output_format, index):
    library = _pillow_lcms()
    if library is None:
        return None
    source = library.cmsOpenProfileFromMem(source_icc, len(source_icc))
    target = library.cmsOpenProfileFromMem(target_icc, len(target_icc))
    try:
        if not source or not target:
            return None
        handle = library.cmsCreateTransform(
            source, _RGB8, target, output_format,
            ImageCms.Intent.RELATIVE_COLORIMETRIC, ImageCms.Flags.BLACKPOINTCOMPENSATION,
        )
        return _PackedTransform(library, handle) if handle else None
    finally:
        if source:
            library.cmsCloseProfile(source)
        if target:
            library.cmsCloseProfile(target)


@lru_cache(maxsize=1)
def _packed_pool():
    return ThreadPoolExecutor(max_workers=8, thread_name_prefix="display-icc-packed")


def transform_packed_rgb(codes, output, source_icc, target_icc, *, workers):
    """Write RGB8 or BGRA8 into separate, contiguous caller-owned buffers.

    For BGRA the caller initializes the alpha byte to 255. Return false when
    the Pillow build does not expose the packed native transform.
    """
    library = _pillow_lcms()
    if library is None:
        return False
    if (codes.ndim != 3 or output.ndim != 3 or codes.shape[2] != 3
            or output.shape[2] not in (3, 4) or output.shape[:2] != codes.shape[:2]
            or not 0 < codes.shape[0] * codes.shape[1] < 2**32
            or codes.dtype != np.uint8 or output.dtype != np.uint8
            or not codes.flags.c_contiguous or not output.flags.c_contiguous
            or not output.flags.writeable or np.may_share_memory(codes, output)):
        raise ValueError("显示ICC需要独立且连续的RGB缓冲区")
    if type(workers) is not int or not 1 <= workers <= 256:
        raise ValueError("显示变换线程数必须为1至256")
    height, width, _ = codes.shape
    channels = output.shape[2]
    count = min(8, workers, height) if height * width >= 150_000 else 1
    output_format = _BGRA8 if channels == 4 else _RGB8
    transforms = [_packed_transform(source_icc, target_icc, output_format, index)
                  for index in range(count)]
    if any(transform is None for transform in transforms):
        return False
    source_address, output_address = codes.ctypes.data, output.ctypes.data

    def transform_band(index):
        first = index * height // count * width
        last = (index + 1) * height // count * width
        transform = transforms[index]
        with transform.lock:
            library.cmsDoTransform(transform.handle, source_address + first * 3,
                                   output_address + first * channels, last - first)

    if count == 1:
        transform_band(0)
    else:
        jobs = [_packed_pool().submit(transform_band, index) for index in range(count)]
        # Hold both arrays and the destination image until every native write ends.
        wait(jobs)
        for job in jobs:
            job.result()
    return True


@lru_cache(maxsize=24)
def _band_transform(source_icc, target_icc, index):
    # The band index retains separate LittleCMS handles so the normal
    # foreground conversion can run in parallel. The lock also protects the
    # handle if a second display conversion uses the same profiles at once.
    transform = ImageCms.buildTransform(
        ImageCms.ImageCmsProfile(io.BytesIO(source_icc)),
        ImageCms.ImageCmsProfile(io.BytesIO(target_icc)),
        "RGB",
        "RGB",
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
    )
    return transform, Lock()


def transform_rgb_image(
    image, source_icc, target_icc, *, workers=1, min_pixels=500_000
):
    """Transform RGB8 pixels; each worker owns its LittleCMS transform instance.

    LittleCMS transforms pixels independently. Splitting by whole rows keeps
    the same intent/black-point settings and never changes pixel coordinates.
    The original Pillow image and profile bytes are not modified.
    """
    if image.mode != "RGB" or min(image.size) <= 0:
        raise ValueError("显示变换需要非空RGB图像")
    if type(workers) is not int or not 1 <= workers <= 256:
        raise ValueError("显示变换线程数必须为1至256")
    if type(min_pixels) is not int or min_pixels < 0:
        raise ValueError("显示变换并行阈值无效")
    count = min(8, workers, image.height)
    if image.width * image.height < min_pixels:
        count = 1

    source_icc, target_icc = bytes(source_icc), bytes(target_icc)

    def transform(band, index):
        lcms, lock = _band_transform(source_icc, target_icc, index)
        with lock:
            return ImageCms.applyTransform(band, lcms)

    if count == 1:
        return transform(image, 0)
    ranges = [
        (index, index * image.height // count, (index + 1) * image.height // count)
        for index in range(count)
    ]

    def band_task(bounds):
        index, first, last = bounds
        return first, transform(image.crop((0, first, image.width, last)), index)

    result = Image.new("RGB", image.size)
    with ThreadPoolExecutor(
        max_workers=count, thread_name_prefix="display-icc"
    ) as pool:
        for first, band in pool.map(band_task, ranges):
            result.paste(band, (0, first))
    result.info["icc_profile"] = bytes(target_icc)
    return result
