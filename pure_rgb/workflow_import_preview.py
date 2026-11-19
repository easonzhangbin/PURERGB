"""Quick positives for checking sources and channel order before import."""

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
from PyQt6.QtGui import QColorSpace, QImage


def _small_rgb(path):
    import rawpy

    with rawpy.RawPy() as raw:
        raw.open_file(path)
        try:
            thumb = raw.extract_thumb()
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            raw.unpack()
            image = Image.fromarray(raw.postprocess(half_size=True))
        else:
            if thumb.format == rawpy.ThumbFormat.JPEG:
                with Image.open(BytesIO(thumb.data)) as jpeg:
                    image = ImageOps.exif_transpose(jpeg).convert("RGB")
            else:
                image = Image.fromarray(thumb.data).convert("RGB")
    image.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
    return image


def _positive_plane(transmission):
    height, width = transmission.shape
    center = transmission[height // 8:height - height // 8, width // 8:width - width // 8]
    low, high = np.percentile(center, [1, 99])
    if high - low < 8:
        low, high = 0, 255
    return np.rint(255 * np.clip((high - transmission.astype(np.float32)) / (high - low), 0, 1)).astype(np.uint8)


def import_preview(paths, display_icc=None):
    images = [_small_rgb(path) for path in paths]
    if len(images) == 1:
        plane = _positive_plane(np.asarray(images[0].convert("L")))
        pixels = np.repeat(plane[..., None], 3, axis=2)
    else:
        size = images[1].size
        pixels = np.stack([
            _positive_plane(np.asarray(ImageOps.fit(image, size, Image.Resampling.BILINEAR))[..., channel])
            for channel, image in enumerate(images)
        ], axis=-1)
    height, width = pixels.shape[:2]
    result = QImage(pixels.tobytes(), width, height, width * 3, QImage.Format.Format_RGB888).copy()
    result.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
    if display_icc:
        target = QColorSpace.fromIccProfile(display_icc)
        if target.isValid():
            result = result.convertedToColorSpace(target)
    return result
