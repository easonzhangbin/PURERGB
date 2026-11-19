"""Verified CML4 downstream interface, deliberately separate from the RW2 modes.

This consumes LS-5000 C-41 device codes AFTER an acquisition-bound builder and
fixed Stage-1 LUT. It does not generate builders or convert camera/working RGB
into scanner RGB. Its own Nikon ICC must accompany its output.
"""

from dataclasses import dataclass
from functools import lru_cache
import json
import sys

import numpy as np

from .engines import ROOT, check_reference_sources

INPUT_DOMAIN = "nikon-ls5000-c41-after-bound-builder-and-fixed-rgb16"


@dataclass(frozen=True)
class CMSResult:
    pixels: np.ndarray
    icc: bytes
    receipt: dict


@lru_cache(maxsize=1)
def _model(threads):
    check_reference_sources()
    source = str(ROOT / "vendor/cool-colors")
    if source not in sys.path:
        sys.path.insert(0, source)
    from nikon_cml4 import PortableNikonCML4

    return PortableNikonCML4(threads=threads)


def evaluate_device_codes(rgb16, *, input_domain, threads=1):
    if input_domain != INPUT_DOMAIN:
        raise ValueError(
            "CML4 仅接受指定 LS-5000 builder/fixed 后设备域；不能直接输入 RW2、密度或工作 RGB"
        )
    if (
        not isinstance(rgb16, np.ndarray)
        or rgb16.dtype != np.uint16
        or rgb16.ndim != 3
        or rgb16.shape[2] != 3
        or not rgb16.size
    ):
        raise ValueError("CML4 输入必须为非空 H×W×3 RGB16")
    # Validate before lru_cache: bool must not alias the integer 1 key.
    if type(threads) is not int or not 1 <= threads <= 16:
        raise ValueError("CML4 线程数必须为 1 至 16 的整数")
    model = _model(threads)
    from nikon_cml4 import rgb16_sha256

    before = rgb16_sha256(rgb16)
    pixels = model.evaluate(rgb16)
    if rgb16_sha256(rgb16) != before:
        raise ValueError("CML4 改变了输入")
    return CMSResult(
        pixels,
        model.profile,
        {
            "input_domain": INPUT_DOMAIN,
            "source_sha256": before,
            "output_sha256": rgb16_sha256(pixels),
            "model": model.bundle_receipt(),
            "rw2_adapter_validated": False,
            "acquisition_binding_verified_by_this_interface": False,
            "scope": "Downstream captured LS-5000 CML4 model only; domain declaration is not acquisition proof",
        },
    )


def main():
    source = np.array(
        [[[0, 0, 0], [65535, 65535, 65535], [32768, 32768, 32768]]], np.uint16
    )
    result = evaluate_device_codes(source, input_domain=INPUT_DOMAIN)
    print(
        json.dumps(
            {"receipt": result.receipt, "known_vector_output": result.pixels.tolist()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
