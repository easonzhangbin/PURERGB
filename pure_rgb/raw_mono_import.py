"""Three user-assigned monochrome sensor exposures assembled as RGB."""

from copy import deepcopy
from pathlib import Path

import numpy as np

from .raw_import import STANDARD_MONO_IMPORT, StandardRawProcessor, _digest, file_identity
from .raw_validity import plan_standard_crop

MONO_NORMALIZATION_VERSION = "raw-visible-sensor-black-white-v2"


def _check_mono_calibration(path, raw, sizes):
    if (sizes["raw_height"], sizes["raw_width"]) != (5494, 7308):
        return
    if raw.black_level_per_channel != [0, 0, 0, 0] or raw.white_level != 64508:
        return
    with open(path, "rb") as stream:
        header = stream.read(16)
    if header[8:16] == b"IIIICwaR":
        raise ValueError(
            "P45 原始 IIQ 的黑白位无法可靠确定，请使用带黑白位标记的单色 DNG。"
        )


def inspect_mono_header(path):
    import rawpy
    from negpy.infrastructure.loaders.helpers import read_exif_from_file, read_orientation

    identity = file_identity(path)
    with rawpy.RawPy() as raw:
        raw.open_file(str(path))
        sizes = raw.sizes._asdict()
        _check_mono_calibration(path, raw, sizes)
    # Some camera RAW containers expose the full visible area without an inset.
    if not sizes["crop_width"] and not sizes["crop_height"]:
        sizes.update(crop_left_margin=sizes["left_margin"],
                     crop_top_margin=sizes["top_margin"],
                     crop_width=sizes["width"], crop_height=sizes["height"])
    plan = plan_standard_crop(
        sizes, (sizes["height"], sizes["width"], 3), user_flip=0, half_size=False
    )
    orientation = read_orientation(str(path))
    if type(orientation) is not int or not 1 <= orientation <= 8:
        raise ValueError("RAW EXIF orientation must be explicit 1..8")
    exif = read_exif_from_file(str(path)) or {}
    camera = [str(exif.get("0th", {}).get(key, "")) for key in (271, 272)]
    if file_identity(path) != identity:
        raise ValueError("RAW changed while its header was read")
    return {"plan": plan, "orientation": orientation, "camera": camera}


def normalized_mono_plane(raw, plan):
    """Black-subtract and normalize the visible sensor samples without demosaicing."""
    mosaic = raw.raw_image_visible
    if mosaic.ndim != 2 or mosaic.shape != plan.source_shape[:2]:
        raise ValueError("Monochrome RGB requires a two-dimensional sensor plane")
    black = np.asarray(raw.black_level_per_channel, dtype=np.float64)
    if black.shape != (4,) or not np.isfinite(black).all():
        raise ValueError("RAW black levels are unavailable")
    white = raw.camera_white_level_per_channel
    if white is None:
        white = [raw.white_level] * 4
    else:
        white = [value if value and value > 0 else raw.white_level for value in white]
    white = np.asarray(white, dtype=np.float64)
    if white.shape != (4,) or not np.isfinite(white).all() or np.any(white <= black):
        raise ValueError("RAW white levels are unavailable or invalid")
    x0, y0, x1, y1 = plan.visible_roi
    result = mosaic[y0:y1, x0:x1].astype(np.float32, copy=True)
    if np.all(black == black[0]) and np.all(white == white[0]):
        np.subtract(result, np.float32(black[0]), out=result)
        np.divide(result, np.float32(white[0] - black[0]), out=result)
    else:
        colors = raw.raw_colors_visible[y0:y1, x0:x1]
        if colors.shape != result.shape or np.any(colors > 3):
            raise ValueError("RAW per-site levels cannot be mapped to the sensor plane")
        for channel in range(4):
            mask = colors == channel
            np.subtract(result, np.float32(black[channel]), out=result, where=mask)
            np.divide(result, np.float32(white[channel] - black[channel]), out=result, where=mask)
    np.clip(result, 0.0, 1.0, out=result)
    pattern = raw.raw_pattern
    evidence = {
        "sensor_mode": "user-declared-monochrome",
        "reported_colors": int(raw.num_colors),
        "reported_pattern": pattern.tolist() if pattern is not None else None,
        "black_levels": black.tolist(),
        "white_levels": white.tolist(),
        "normalization": MONO_NORMALIZATION_VERSION,
    }
    return result, evidence


class StandardMonoProcessor(StandardRawProcessor):
    policy = STANDARD_MONO_IMPORT

    @staticmethod
    def _paths(file_path, params):
        from negpy.features.rgbscan.models import is_rgb_triplet
        from negpy.features.hdr.models import hdr_active
        from negpy.features.process.logic import effective_highlight_reconstruction, effective_linear_raw
        from negpy.services.rendering.lens import metadata_lens_corrections

        if not is_rgb_triplet(params.rgbscan) or hdr_active(params.hdr):
            raise ValueError("Monochrome RGB import requires three exposures without HDR")
        if params.stitch.stitch_enabled or metadata_lens_corrections(params):
            raise ValueError("Monochrome RGB import does not support stitch or lens remapping")
        if not effective_linear_raw(params.process) or effective_highlight_reconstruction(params.process) != 0:
            raise ValueError("Monochrome RGB import requires linear RAW without highlight reconstruction")
        paths = tuple(str(Path(p).resolve(strict=True)) for p in
                      (file_path, params.rgbscan.green_path, params.rgbscan.blue_path))
        if len(set(paths)) != 3:
            raise ValueError("Monochrome RGB import requires three distinct RAW files")
        return paths

    def _header(self, path):
        return inspect_mono_header(path)

    def _decode_band(self, path, band, plan, orientation, params):
        import rawpy
        from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE

        with rawpy.imread(path) as raw:
            plane, evidence = normalized_mono_plane(raw, plan)
        self._mono_evidence[path] = evidence
        return plane, {"orientation": orientation, "cam_xyz": None,
                       "color_space": WORKING_COLOR_SPACE}

    def _decode_oriented_f32(self, file_path, params, fast_decode=False, wb_override=None, unmix=True):
        self._mono_evidence = {}
        result = super()._decode_oriented_f32(file_path, params, fast_decode, wb_override, unmix)
        receipt = deepcopy(self.pure_raw_import_receipt)
        for band in receipt["bands"]:
            band["monochrome_sensor"] = self._mono_evidence[band["path"]]
        receipt["geometry_token"] = _digest({
            key: receipt[key] for key in ("bands", "registration", "effective_geometry")
        })
        receipt["decode"].update(sensor_mode="user-declared-monochrome",
                                 demosaic="none", neutral_WB=True,
                                 normalization=MONO_NORMALIZATION_VERSION)
        self.pure_raw_import_receipt = receipt
        self.pure_raw_import_token = _digest(receipt)
        return result
