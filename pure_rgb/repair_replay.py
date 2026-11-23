"""Reapply a confirmed final repair mask to newly rendered positive pixels.

Full-frame mask coordinates are final output pixels, not RAW/source pixels.
``output_size`` is [width, height]. A regular preview samples [::step, ::step];
a viewport samples [y:y+height:step, x:x+width:step], including that exact phase.
Reduced/viewport healing is a screen approximation because its donor context
differs. ``replay_record`` alone produces the full-resolution final result.
The caller must validate the plan's RAW binding before calling here. Supported
output edits carry a mapping from the authenticated review to the current grid.
"""

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import uuid

import numpy as np
import tifffile

from .native import sha256
from .preview_data import verified_pixels
from .repair_bridge import DEFAULT_BACKEND, backend_identity
from .workspace import write_json


_LOCK = threading.RLock()
_VERIFIED = OrderedDict()
_MASK = None
_BACKEND = None


def _identity(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash_value(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _checked_digest(value):
    return (type(value) is str and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _verified_file(path, digest):
    """Bounded identity cache; changed file identities always rehash."""
    if not _checked_digest(digest):
        raise ValueError("除尘计划缺少有效文件哈希")
    path = Path(path).resolve(strict=True)
    before = _identity(path)
    key = str(path), before, digest
    with _LOCK:
        hit = key in _VERIFIED
        if hit:
            _VERIFIED.move_to_end(key)
    if not hit:
        if sha256(path) != digest or _identity(path) != before:
            raise ValueError("除尘计划关联文件已变化，哈希核验失败")
        with _LOCK:
            _VERIFIED[key] = True
            while len(_VERIFIED) > 32:
                _VERIFIED.popitem(last=False)
    return path, before


def _backend_files(root):
    names = {"runtime/node", "runtime/image-codec", "src/cli.cjs", "src/processor.cjs",
             "src/review.cjs", "src/tiff.cjs", "src/brush.cjs"}
    for directory in (root / "src", root / "vendor", root / "runtime/FibreNet.mlmodelc"):
        names.update(str(path.relative_to(root)) for path in directory.rglob("*") if path.is_file())
    return tuple((name, _identity(root / name)) for name in sorted(names))


def _verify_backend(expected):
    global _BACKEND
    if not isinstance(expected, dict) or not _checked_digest(expected.get("fingerprint")):
        raise ValueError("除尘计划的引擎身份无效")
    # Historical project data identifies the engine but cannot choose which
    # executable the desktop runs on a background preview.
    root = DEFAULT_BACKEND.resolve(strict=True)
    if Path(expected["root"]).resolve() != root:
        raise ValueError("此除尘审核来自其他引擎安装位置，请使用本机 PURE Neg 重新审核")
    identity = str(root), _backend_files(root)
    with _LOCK:
        cached = _BACKEND
    if cached is None or cached[0] != identity:
        actual = backend_identity(root)
        if _backend_files(root) != identity[1]:
            raise ValueError("PURE Neg 引擎在核验期间发生变化")
        with _LOCK:
            _BACKEND = identity, actual
    else:
        actual = cached[1]
    if (actual["fingerprint"] != expected["fingerprint"]
            or actual.get("files") != expected.get("files")):
        raise ValueError("PURE Neg 引擎自确认后发生变化，请重新审核")
    return root, identity


def _verified_plan(plan):
    global _MASK
    if (not isinstance(plan, dict) or type(plan.get("version")) is not int
            or plan["version"] != 1):
        raise ValueError("不支持的除尘重放计划")
    shape = plan.get("shape")
    if (type(shape) is not list or len(shape) != 3 or shape[2] != 3
            or any(type(value) is not int or value < 1 for value in shape)):
        raise ValueError("除尘计划缺少有效完整输出尺寸")
    receipt_path, receipt_identity = _verified_file(plan["receipt_path"], plan["receipt_sha256"])
    receipt = json.loads(receipt_path.read_text())
    preview, source = receipt.get("preview", {}), receipt.get("source", {})
    if (receipt.get("state") != "confirmed" or receipt.get("revision") != source.get("sha256")
            or preview.get("outside_mask_changed_pixels") != 0
            or preview.get("shape") != shape
            or preview.get("mask_sha256") != plan.get("mask_sha256")
            or Path(preview.get("mask_path", "")).resolve() != Path(plan["mask_path"]).resolve()
            or receipt.get("backend") != plan.get("backend")):
        raise ValueError("除尘计划与已确认审核不一致")
    source_path, source_identity = _verified_file(source["path"], source["sha256"])
    with tifffile.TiffFile(source_path) as document:
        page = document.pages[0]
        if list(page.shape) != shape or page.dtype != np.uint16:
            raise ValueError("原审核成片尺寸或RGB16格式与除尘计划不一致")
    mask_path, mask_identity = _verified_file(plan["mask_path"], plan["mask_sha256"])
    key = str(mask_path), mask_identity, plan["mask_sha256"], tuple(shape)
    with _LOCK:
        cached = _MASK
    if cached is None or cached[0] != key:
        mask = tifffile.imread(mask_path)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[..., 0]
        if mask.dtype != np.uint8 or list(mask.shape) != shape[:2]:
            raise ValueError("已确认除尘蒙版必须是原尺寸的8位灰度图")
        mask.flags.writeable = False
        if _identity(mask_path) != mask_identity:
            raise ValueError("除尘蒙版在读取期间发生变化")
        with _LOCK:
            _MASK = key, mask
    else:
        mask = cached[1]
    backend_root, backend_stamp = _verify_backend(plan["backend"])
    evidence = [(receipt_path, receipt_identity, plan["receipt_sha256"]),
                (source_path, source_identity, source["sha256"]),
                (mask_path, mask_identity, plan["mask_sha256"])]
    _unchanged(evidence)
    return mask, backend_root, backend_stamp, evidence


def _unchanged(evidence):
    for path, identity, *digests in evidence:
        if _identity(path) == identity:
            continue
        # macOS may attach metadata to a newly created TIFF when the helper
        # opens it. A ctime change is not a pixel change: authenticate bytes
        # again instead of discarding a valid large-image repair.
        if digests:
            for _ in range(2):
                current = _identity(path)
                if sha256(path) == digests[0] and _identity(path) == current:
                    break
            else:
                raise ValueError("除尘重放期间审核来源或蒙版内容发生变化")
        else:
            raise ValueError("除尘重放期间审核来源或蒙版发生变化")


def _pixel_hash(pixels):
    digest = hashlib.sha256()
    # Row chunks also accept strided previews without a full-frame copy.
    for start in range(0, pixels.shape[0], 64):
        digest.update(memoryview(np.ascontiguousarray(pixels[start:start + 64])).cast("B"))
    return digest.hexdigest()


def _sample_mask(mask, pixels, output_size, sample_step, viewport):
    from .viewport_preview_v43 import checked_viewport

    size = [mask.shape[1], mask.shape[0]]
    if output_size is not None and output_size != size:
        raise ValueError("除尘计划完整尺寸与当前输出不一致")
    if type(sample_step) is not int or sample_step < 1:
        raise ValueError("除尘蒙版采样步长必须为正整数")
    if viewport is None:
        x = y = 0
        width, height = size
    else:
        viewport = checked_viewport(viewport, size)
        if viewport["sample_step"] != sample_step:
            raise ValueError("除尘视口采样步长与当前预览不一致")
        x, y, width, height = (viewport[key] for key in ("x", "y", "width", "height"))
    sampled = mask[y:y + height:sample_step, x:x + width:sample_step]
    if sampled.shape != pixels.shape[:2]:
        raise ValueError("除尘蒙版采样网格与当前画面不一致")
    return sampled, dict(x=x, y=y, width=width, height=height, sample_step=sample_step,
                         output_size=size)


def _mapped_mask(plan, mask, mapping, pixels, output_size, sample_step, viewport):
    from .repair_geometry import frame_descriptor, sample_repair_mask

    if (not isinstance(mapping, dict) or set(mapping) != {"version", "source", "target"}
            or type(mapping.get("version")) is not int or mapping["version"] != 1):
        raise ValueError("除尘坐标映射记录无效")
    parent_path, identity = _verified_file(plan["parent_manifest"]["path"],
                                          plan["parent_manifest"]["sha256"])
    parent = json.loads(parent_path.read_text())
    original = frame_descriptor(parent["recipe"], parent, plan["binding"]["mode"])
    if original != mapping["source"]:
        raise ValueError("除尘坐标映射与已认证原审核不一致")
    sampled, grid = sample_repair_mask(mask, original, mapping["target"],
        output_size=output_size, sample_step=sample_step, viewport=viewport)
    if sampled.shape != pixels.shape[:2]:
        raise ValueError("除尘映射蒙版与当前预览采样网格不一致")
    evidence = (parent_path, identity, plan["parent_manifest"]["sha256"])
    _unchanged([evidence])
    return sampled, grid, evidence


def _write_tiff(path, pixels, icc):
    tifffile.imwrite(path, pixels, photometric="rgb", compression=None, metadata=None,
                     extratags=[(34675, 7, len(icc), icc, False), (274, 3, 1, 1, False)])


def _changed_pixels(actual, original, mask):
    """Verify every output pixel without whole-frame RGB boolean buffers."""
    count = 0
    for start in range(0, original.shape[0], 64):
        stop = start + 64
        changed = np.any(actual[start:stop] != original[start:stop], axis=2)
        if np.any(changed & (mask[start:stop] == 0)):
            raise ValueError("除尘重放在已确认蒙版外改变了像素")
        count += int(np.count_nonzero(changed))
    return count


def _run_helper(root, scratch, request, on_progress):
    path = scratch / "request.json"
    path.write_bytes(_canonical(request))
    argv = [str(root / "runtime/node"), str(Path(__file__).with_suffix(".cjs")), str(path)]
    process = None
    final = None
    with (scratch / "stderr.log").open("wb") as stderr:
        try:
            process = subprocess.Popen(argv, cwd=root, stdout=subprocess.PIPE, stderr=stderr,
                                       text=True, encoding="utf-8")
            for line in process.stdout:
                if len(line) > 65536:
                    raise ValueError("除尘重放返回数据过大")
                event = json.loads(line)
                if event.get("type") == "progress":
                    on_progress(event)
                elif event.get("type") == "error":
                    raise RuntimeError(event.get("message", "除尘重放失败"))
                elif event.get("type") == "result":
                    final = event
            code = process.wait()
            if code != 0 or final is None or final.get("detector_calls") != 0:
                raise RuntimeError(f"除尘重放未完成，退出码 {code}")
            return final
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process is not None and process.stdout is not None:
                process.stdout.close()


def replay_pixels(pixels, icc, plan, *, directory, output_size=None, sample_step=1,
                  viewport=None, on_progress=lambda event: None, frame_mapping=None,
                  mask_destination=None):
    """Return new RGB16 pixels plus replay evidence, leaving input unchanged.

    ``on_progress`` may raise to cancel; the helper is stopped and all temporary
    files are removed. Viewports use the final-output x/y phase exactly. Neither
    downsampled previews nor tiles claim full-frame donor equivalence.
    """
    if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint16
            or pixels.ndim != 3 or pixels.shape[2] != 3
            or not isinstance(icc, bytes) or not icc):
        raise ValueError("除尘重放需要带ICC的RGB16画面")
    plan = deepcopy(plan)
    mask, root, backend_stamp, evidence = _verified_plan(plan)
    if frame_mapping is None:
        sampled, grid = _sample_mask(mask, pixels, output_size, sample_step, viewport)
    else:
        sampled, grid, parent_evidence = _mapped_mask(
            plan, mask, frame_mapping, pixels, output_size, sample_step, viewport)
        evidence.append(parent_evidence)
    if mask_destination is not None and (sample_step != 1 or viewport is not None):
        raise ValueError("仅完整分辨率输出可保存除尘派生蒙版")
    before = _pixel_hash(pixels)
    count = int(np.count_nonzero(sampled))
    provenance = dict(version=1, plan_hash=_hash_value(plan), source_pixel_sha256=before,
                      icc_sha256=hashlib.sha256(icc).hexdigest(), grid=grid,
                      quality="full-resolution" if sample_step == 1 and viewport is None else "screen-approximation",
                      mask_pixels=count, outside_mask_changed_pixels=0, detector_calls=0)
    if frame_mapping is not None:
        provenance["frame_mapping"] = deepcopy(frame_mapping)

    def save_mask():
        if mask_destination is not None:
            destination = Path(mask_destination).resolve()
            if destination.exists():
                raise ValueError("除尘派生蒙版已存在，未覆盖")
            tifffile.imwrite(destination, sampled, photometric="minisblack",
                             compression="deflate", metadata=None)
            provenance["effective_mask"] = dict(path=str(destination), sha256=sha256(destination),
                                                 shape=list(sampled.shape))

    if not count:
        _unchanged(evidence)
        save_mask()
        return pixels, {**provenance, "changed_pixels": 0, "algorithm": "empty-mask"}
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    on_progress({"type": "progress", "fraction": 0, "message": "核验已确认除尘计划"})
    with tempfile.TemporaryDirectory(prefix=".repair-replay-", dir=directory) as temporary:
        scratch = Path(temporary)
        source_path, mask_path, result_path = (scratch / name for name in ("input.tiff", "mask.tiff", "output.tiff"))
        _write_tiff(source_path, pixels, icc)
        tifffile.imwrite(mask_path, sampled, photometric="minisblack", compression=None, metadata=None)
        temporary_evidence = [(path, _identity(path), sha256(path)) for path in (source_path, mask_path)]
        result = _run_helper(root, scratch, dict(backend=str(root), input=str(source_path),
                             mask=str(mask_path), output=str(result_path)), on_progress)
        actual, actual_icc = verified_pixels(dict(path=str(result_path), sha256=sha256(result_path)))
        if actual.shape != pixels.shape or actual_icc != icc:
            raise ValueError("除尘重放结果尺寸或ICC不一致")
        changed_count = _changed_pixels(actual, pixels, sampled)
        _unchanged(evidence + temporary_evidence)
        if _pixel_hash(pixels) != before:
            raise ValueError("除尘重放期间输入像素发生变化")
        if (str(root), _backend_files(root)) != backend_stamp:
            raise ValueError("PURE Neg 引擎在重放期间发生变化")
        provenance.update(changed_pixels=changed_count, algorithm=result["algorithm"],
                          healed=result.get("healed", {}))
        save_mask()
        return actual, provenance


def replay_record(source_record, plan, *, directory, on_progress=lambda event: None,
                  frame_mapping=None):
    """Write a new standalone full-resolution TIFF/receipt without replacement."""
    source_record = deepcopy(source_record)
    source_path = Path(source_record["path"]).resolve(strict=True)
    before = _identity(source_path)
    pixels, icc = verified_pixels(source_record)
    destination = Path(directory).resolve() / ("replayed-" + uuid.uuid4().hex)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        options = dict(directory=destination, on_progress=on_progress)
        if frame_mapping is not None:
            options.update(frame_mapping=frame_mapping, mask_destination=destination / "mask.tiff")
        actual, provenance = replay_pixels(pixels, icc, plan, **options)
        if _identity(source_path) != before or sha256(source_path) != source_record["sha256"]:
            raise ValueError("除尘重放期间新成片来源发生变化")
        output = destination / "cleaned.tiff"
        _write_tiff(output, actual, icc)
        effective = provenance.get("effective_mask", {})
        record = dict(path=str(output), sha256=sha256(output), shape=list(actual.shape),
                      icc_sha256=hashlib.sha256(icc).hexdigest(),
                      mask_path=effective.get("path", plan["mask_path"]),
                      mask_sha256=effective.get("sha256", plan["mask_sha256"]),
                      changed_pixels=provenance["changed_pixels"],
                      mask_pixels=provenance["mask_pixels"], outside_mask_changed_pixels=0)
        provenance.update(source_sha256=source_record["sha256"], source_sha=source_record["sha256"],
                          parent_receipt=plan["receipt_path"])
        receipt_path = destination / "receipt.json"
        record["repair_replay"] = provenance
        record["repair_plan"] = deepcopy(plan)
        record["source_sha256"] = source_record["sha256"]
        record["replay_receipt"] = str(receipt_path)
        write_json(receipt_path, dict(state="confirmed", version=1, source=source_record,
                   revision=source_record["sha256"], backend=deepcopy(plan["backend"]),
                   plan=deepcopy(plan), preview=deepcopy(record), provenance=provenance))
        return record
    except BaseException:
        shutil.rmtree(destination)
        raise
