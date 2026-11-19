"""Pixel-exact display ICC transforms in independent horizontal bands."""

from concurrent.futures import ThreadPoolExecutor
import io

from PIL import Image, ImageCms


def transform_rgb_image(
    image, source_icc, target_icc, *, workers=1, min_pixels=2_000_000
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

    def transform(band):
        return ImageCms.profileToProfile(
            band,
            ImageCms.ImageCmsProfile(io.BytesIO(source_icc)),
            ImageCms.ImageCmsProfile(io.BytesIO(target_icc)),
            outputMode="RGB",
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
        )

    if count == 1:
        return transform(image)
    ranges = [
        (index * image.height // count, (index + 1) * image.height // count)
        for index in range(count)
    ]

    def band_task(bounds):
        first, last = bounds
        return first, transform(image.crop((0, first, image.width, last)))

    result = Image.new("RGB", image.size)
    with ThreadPoolExecutor(
        max_workers=count, thread_name_prefix="display-icc"
    ) as pool:
        for first, band in pool.map(band_task, ranges):
            result.paste(band, (0, first))
    result.info["icc_profile"] = bytes(target_icc)
    return result
