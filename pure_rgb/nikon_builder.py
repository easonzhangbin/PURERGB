"""Bound scanner RGB16 -> three builders -> fixed LUT -> captured Nikon CML4.

This complete public downstream chain accepts an explicit LS-5000 scanner domain.
It cannot infer a per-acquisition builder or turn uncalibrated RW2 into that domain.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import tifffile

from .diagnostics import setup, span, MemorySampler
from .engine_resources import checked_resource
from .native import sha256
from .nikon_cms import (
    CMSResult,
    INPUT_DOMAIN as CMS_DOMAIN,
    _model,
    evaluate_device_codes,
)
from .workspace import write_json

SCANNER_DOMAIN = "nikon-ls5000-unadjusted-scanner-rgb16"


def pixel_sha256(pixels):
    return hashlib.sha256(
        np.asarray(pixels, dtype="<u2", order="C").tobytes()
    ).hexdigest()


def evaluate_bound_builders(pixels, document, *, threads=1):
    if type(threads) is not int or not 1 <= threads <= 16:
        raise ValueError("Nikon CMS 线程数必须是 1..16 整数")
    if (
        not isinstance(document, dict)
        or set(document)
        != {"schema", "version", "input_domain", "source_pixels_sha256", "channels"}
        or document["schema"] != "pure-rgb-nikon-builders"
        or type(document["version"]) is not int
        or document["version"] != 1
    ):
        raise ValueError("无效 Nikon builder 资源清单")
    if document["input_domain"] != SCANNER_DOMAIN:
        raise ValueError("需 LS-5000 未调整 scanner RGB16；不接受 RAW、密度或工作 RGB")
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.uint16
        or pixels.ndim != 3
        or pixels.shape[2] != 3
        or not pixels.size
    ):
        raise ValueError("Nikon scanner 输入需 H×W×3 uint16")
    source_hash = pixel_sha256(pixels)
    if source_hash != document["source_pixels_sha256"]:
        raise ValueError("builder 与当前帧像素绑定不匹配")
    channels = document["channels"]
    if not isinstance(channels, dict) or set(channels) != set("RGB"):
        raise ValueError("需完整 R/G/B builder 资源")
    resources = []
    builders = []
    for channel in "RGB":
        resource = checked_resource(channels[channel])
        payload = Path(resource["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != resource["sha256"]:
            raise ValueError(channel + " builder 文件已变化")
        array = np.load(io.BytesIO(payload), allow_pickle=False)
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.uint16
            or array.size != 65536
        ):
            raise ValueError(channel + " builder 必须包含 65536 项 uint16")
        builders.append(array.reshape(-1))
        resources.append({"channel": channel, **resource})
    model = _model(threads)
    from invert_c41 import apply_builder

    with span("nikon_builder_fixed", source_pixels_sha256=source_hash):
        device = apply_builder(pixels, builders, model.fixed_lut)
    with span("nikon_cml4", threads=threads):
        result = evaluate_device_codes(device, input_domain=CMS_DOMAIN, threads=threads)
    if pixel_sha256(pixels) != source_hash:
        raise ValueError("Nikon builder 改变了源像素")
    return CMSResult(
        result.pixels,
        result.icc,
        {
            **result.receipt,
            "input_domain": SCANNER_DOMAIN,
            "source_pixels_sha256": source_hash,
            "builder_resources": resources,
            "source_pixel_binding_verified": True,
            "acquisition_binding_verified_by_this_interface": False,
            "scope": "Hash-bound provided per-frame builders + pinned fixed LUT + CML4. Caller-declared LS-5000 domain; no camera calibration inferred.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--builders", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory")
    parser.add_argument("--threads", type=int, choices=range(1, 17), default=1)
    args = parser.parse_args()
    setup("nikon-builder")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = {"state": "running"}
    try:
        source_hash, builder_hash = sha256(args.input), sha256(args.builders)
        document = json.loads(args.builders.read_text())
        for resource in document.get("channels", {}).values():
            if (
                isinstance(resource, dict)
                and isinstance(resource.get("path"), str)
                and not Path(resource["path"]).is_absolute()
            ):
                resource["path"] = str(
                    (args.builders.resolve().parent / resource["path"]).resolve()
                )
        with tifffile.TiffFile(args.input) as tif:
            if len(tif.pages) != 1:
                raise ValueError("scanner TIFF 需单页")
            page = tif.pages[0]
            if page.tags.get(274) is not None and page.tags[274].value != 1:
                raise ValueError("需未重排的 scanner 像素（TIFF Orientation=1）")
            pixels = page.asarray()
        with MemorySampler() as resources:
            result = evaluate_bound_builders(pixels, document, threads=args.threads)
        if source_hash != sha256(args.input) or builder_hash != sha256(args.builders):
            raise ValueError("源或资源清单处理期间发生变化")
        target = directory / "nikon-cml4.tiff"
        tifffile.imwrite(
            target,
            result.pixels,
            photometric="rgb",
            metadata=None,
            compression="deflate",
            extratags=[
                (34675, 7, len(result.icc), result.icc, False),
                (274, 3, 1, 1, False),
            ],
        )
        from .study import read_rgb

        actual, icc = read_rgb(target)
        if not np.array_equal(actual, result.pixels) or icc != result.icc:
            raise ValueError("Nikon RGB16/ICC 写出核验失败")
        report.update(
            state="complete",
            input_sha256=source_hash,
            builders_sha256=builder_hash,
            output=str(target),
            output_sha256=sha256(target),
            receipt=result.receipt,
            performance=resources.report(),
        )
    except Exception as error:
        report.update(state="failed", error=str(error))
        raise
    finally:
        write_json(directory / "manifest.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
