"""Read-only UI adapter for the verified original rgb2 range study."""

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np

from . import WORKSPACE_ROOT as ROOT
from .study import read_rgb

MODES = {
    "native": "negpy · 原生",
    "cool-shared-rgb": "Nikon CMS · 演算",
    "flex-endpoint": "FlexColor · 演算",
}
PROBES = {
    "baseline": "基准效果",
    "highlight-only": "高光侧范围实验",
    "shadow-only": "阴影侧范围实验",
    "both": "两侧范围实验",
}
DEFAULT_STUDY = ROOT / "outputs/primary-range-3f8584ef4c8c"


@dataclass(frozen=True)
class Sample:
    id: str
    directory: Path
    inputs: tuple
    roi: tuple
    results: dict

    def record(self, mode, probe):
        if mode not in MODES or probe not in PROBES:
            raise ValueError("Unknown mode or range probe")
        return self.results[f"{probe}__{mode}"]


def load_samples(study=DEFAULT_STUDY):
    index = json.loads((Path(study) / "index.json").read_text())
    if index["state"] != "complete":
        raise ValueError("样片批次未完成，不能载入预览")
    samples = []
    for case in index["cases"]:
        if case["state"] != "complete":
            raise ValueError("样片尚未完成")
        directory = Path(case["directory"])
        m = json.loads((directory / "manifest.json").read_text())
        if m["state"] != "complete":
            raise ValueError("样片清单未完成")
        for mode in MODES:
            for probe in PROBES:
                if f"{probe}__{mode}" not in m["results"]:
                    raise ValueError("样片缺少模式或范围结果")
        samples.append(
            Sample(
                case["id"], directory, tuple(m["inputs"]), tuple(m["roi"]), m["results"]
            )
        )
    if not samples:
        raise ValueError("没有可用样片")
    return samples


def verified_pixels(record):
    from .diagnostics import span

    path = Path(record["path"])
    with span("verify_tiff_sha256"):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != record["sha256"]:
        raise ValueError("样片 TIFF 已变化，哈希核验失败")
    with span("verify_tiff_decode"):
        pixels, icc = read_rgb(path)
    if pixels.dtype != np.uint16 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("样片必须是带 ICC 的 RGB16 TIFF")
    if not icc:
        raise ValueError("样片缺少 ICC")
    return pixels, icc


def _publish_exclusive(temporary, destination):
    """Atomically publish a complete same-directory file without replacement.

    Apple xnu bsd/sys/stdio.h defines RENAME_EXCL=0x00000004; its rename(2)
    contract returns EEXIST for an existing target and ENOTSUP when the volume
    cannot honor this flag. Do not fall back to an overwriting rename.
    Other platforms use atomic hard-link creation, or explicitly fail if the
    filesystem cannot provide that no-replace primitive.
    """
    if sys.platform == "darwin":
        library = ctypes.CDLL(None, use_errno=True)
        try:
            rename = library.renamex_np
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "Atomic exclusive rename is unavailable",
                str(destination),
            ) from error
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(os.fsencode(temporary), os.fsencode(destination), 0x00000004):
            code = ctypes.get_errno() or errno.EIO
            raise OSError(code, os.strerror(code), str(destination))
    else:
        os.link(temporary, destination)


def copy_verified_tiff(record, destination):
    """Verify, sync and atomically publish bytes; never expose a partial TIFF.

    A hard process kill may leave this call's hidden .part file, never a
    partially written final filename. No rendering or color conversion occurs.
    """
    source, destination = Path(record["path"]), Path(destination)
    verified_pixels(record)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(destination))
    temporary = destination.with_name(f".pure-rgb-export-{uuid.uuid4().hex}.part")
    digest = hashlib.sha256()
    created = False
    try:
        with source.open("rb") as src, temporary.open("xb") as dst:
            created = True
            while chunk := src.read(1024 * 1024):
                digest.update(chunk)
                dst.write(chunk)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError("导出期间源文件发生变化")
            dst.flush()
            os.fsync(dst.fileno())
        _publish_exclusive(temporary, destination)
    finally:
        if created:
            temporary.unlink(missing_ok=True)
    return str(destination)
