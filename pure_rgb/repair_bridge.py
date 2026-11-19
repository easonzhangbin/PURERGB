"""Installed PURE Neg CLI transactions. Never edit its signed review records."""

import hashlib
import json
from contextlib import suppress
from pathlib import Path
import subprocess

import numpy as np
import tifffile

from .native import sha256
from .preview_data import verified_pixels
from .workspace import write_json

DEFAULT_BACKEND = Path("/Applications/PURE Neg.app/Contents/Resources")


def backend_identity(root=DEFAULT_BACKEND):
    root = Path(root).resolve(strict=True)
    names = [
        "runtime/node",
        "runtime/image-codec",
        "src/cli.cjs",
        "src/processor.cjs",
        "src/review.cjs",
        "src/tiff.cjs",
        "src/brush.cjs",
    ]
    assets = {name: sha256(root / name) for name in names}
    # Include engine and learned-model resources, not brand artwork or support images.
    for directory in (
        root / "src",
        root / "vendor",
        root / "runtime/FibreNet.mlmodelc",
    ):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                assets[str(path.relative_to(root))] = sha256(path)
    return {
        "root": str(root),
        "files": assets,
        "fingerprint": hashlib.sha256(
            json.dumps(assets, sort_keys=True).encode()
        ).hexdigest(),
    }


def run_cli(root, work_root, command, flags, on_progress=lambda event: None):
    root, work_root = Path(root).resolve(), Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    argv = [
        str(root / "runtime/node"),
        str(root / "src/cli.cjs"),
        command,
        "--work-dir",
        str(work_root),
    ]
    for name, value in flags.items():
        if name in ("input", "output", "review", "edits"):
            value = Path(value).resolve()
        argv.extend(["--" + name, str(value)])
    expected = {
        "prepare": "review",
        "inspect-review": "review",
        "preview": "preview",
        "confirm": "result",
        "redetect": "redetected",
    }[command]
    final = None
    with (work_root / "stderr.log").open("ab") as stderr:
        process = subprocess.Popen(
            argv,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            encoding="utf-8",
        )
        try:
            for line in process.stdout:
                if len(line) > 4_000_000:
                    raise ValueError("PURE Neg 返回的数据行过大")
                event = json.loads(line)
                if event.get("type") == "progress":
                    on_progress(event)
                elif event.get("type") == "error":
                    raise RuntimeError(event.get("message", "PURE Neg 处理失败"))
                elif event.get("type") == expected:
                    final = event
            code = process.wait()
            if code != 0 or final is None:
                raise RuntimeError(
                    f"PURE Neg 未完成 {command}，退出码 {code}；请查看 stderr.log"
                )
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    return final


def verify_repair(source, cleaned, mask):
    original, oi = verified_pixels(source)
    actual, ai = verified_pixels({"path": str(cleaned), "sha256": sha256(cleaned)})
    coverage = tifffile.imread(mask)
    if coverage.ndim == 3 and coverage.shape[2] == 1:
        coverage = coverage[..., 0]
    if (
        actual.shape != original.shape
        or ai != oi
        or coverage.shape != original.shape[:2]
        or coverage.dtype != np.uint8
    ):
        raise ValueError("除尘结果尺寸、RGB16、ICC 或蒙版不匹配")
    changed = np.any(original != actual, axis=2)
    outside = int(np.count_nonzero(changed & (coverage == 0)))
    if outside:
        raise ValueError(f"除尘蒙版外有 {outside} 像素变化，结果未通过")
    return {
        "path": str(cleaned),
        "sha256": sha256(cleaned),
        "icc_sha256": hashlib.sha256(ai).hexdigest(),
        "shape": list(actual.shape),
        "mask_path": str(mask),
        "mask_sha256": sha256(mask),
        "changed_pixels": int(changed.sum()),
        "mask_pixels": int(np.count_nonzero(coverage)),
        "outside_mask_changed_pixels": outside,
    }


def prepare(request, on_progress):
    root = Path(request["directory"]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    identity = backend_identity(request.get("backend", DEFAULT_BACKEND))
    source = request["source"]
    verified_pixels(source)
    receipt = {
        "state": "preparing",
        "source": source,
        "backend": identity,
        "work_root": str(root / "backend-work"),
        "revision": request["revision"],
    }
    write_json(root / "receipt.json", receipt)
    result = run_cli(
        identity["root"],
        receipt["work_root"],
        "prepare",
        {
            "input": source["path"],
            "output": root / "published",
            "polarity": request.get("polarity", "bright"),
            "detect": request.get("detect", "dust,scratches,fibers"),
            "sensitivity": request.get("sensitivity", 1),
        },
        on_progress,
    )
    review = result["review"]
    if review["depth"] != 16:
        raise ValueError("PURE Neg 审核源位深发生变化")
    validation = verify_repair(source, review["cleanedPath"], review["maskPath"])
    receipt.update(
        state="ready",
        review_path=result["reviewPath"],
        review=review,
        preview=validation,
        edits={"excludedRegions": [], "strokes": []},
    )
    write_json(root / "receipt.json", receipt)
    return receipt


REDETECT_THRESHOLD_FIELDS = (
    "dust", "fibers", "scratches", "patches", "curl", "softness"
)


def _redetect_thresholds(request):
    """Map the dialog's percent-style request onto PURE Neg 0.5..2 factors."""
    thresholds = {}
    for field in REDETECT_THRESHOLD_FIELDS:
        value = request.get(field, 1.0)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"重新检测 {field} 灵敏度必须是数值")
        value = float(value)
        if not 0.5 <= value <= 2.0:
            raise ValueError(f"重新检测 {field} 灵敏度必须在 0.5 到 2 之间")
        thresholds[field] = value
    keep_rebate = request.get("keep_rebate", False)
    if type(keep_rebate) is not bool:
        raise ValueError("保留片边选项必须是布尔值")
    return thresholds, keep_rebate


def redetect(request, on_progress):
    """Re-run detection with explicit thresholds against the same source.

    The CLI builds a brand-new signed review and remaps prior exclusions; the
    bridge re-verifies the new raster/mask against the immutable source before
    swapping the receipt's review pointer.
    """
    path = Path(request["receipt"]).resolve()
    receipt = json.loads(path.read_text())
    if receipt["state"] not in ("ready", "previewed"):
        raise ValueError("审核不可重新检测；请重新分析，不能改写签名记录或锁")
    if request["revision"] != receipt["revision"]:
        raise ValueError("当前配方与审核来源不一致，请重新除尘")
    current = backend_identity(receipt["backend"]["root"])
    if current["fingerprint"] != receipt["backend"]["fingerprint"]:
        raise ValueError("PURE Neg 引擎自分析后发生变化，请重新除尘")
    verified_pixels(receipt["source"])
    edits = request.get("edits", receipt["edits"])
    if set(edits) - {"excludedRegions", "strokes"}:
        raise ValueError("本桥接只支持灰尘排除与补修，不包含牛顿环")
    thresholds, keep_rebate = _redetect_thresholds(request)
    import uuid

    edits_path = path.parent / ("edits-" + uuid.uuid4().hex + ".json")
    edits_path.write_text(
        json.dumps(edits, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    )
    receipt.update(state="redetecting")
    write_json(path, receipt)
    try:
        result = run_cli(
            current["root"],
            receipt["work_root"],
            "redetect",
            {
                "review": receipt["review_path"],
                "edits": edits_path,
                "dust-sensitivity": thresholds["dust"],
                "fiber-sensitivity": thresholds["fibers"],
                "scratch-sensitivity": thresholds["scratches"],
                "patch-sensitivity": thresholds["patches"],
                "fiber-curl": thresholds["curl"],
                "fiber-softness": thresholds["softness"],
                "keep-rebate": "yes" if keep_rebate else "no",
            },
            on_progress,
        )
        review = result["review"]
        if review["depth"] != 16:
            raise ValueError("PURE Neg 重新检测源位深发生变化")
        validation = verify_repair(
            receipt["source"], review["cleanedPath"], review["maskPath"]
        )
    except BaseException as error:
        receipt.update(state="uncertain", error=str(error))
        write_json(path, receipt)
        raise
    remapped_edits = {
        "excludedRegions": result.get("excludedRegions", []),
        "strokes": edits["strokes"],
    }
    receipt.update(
        state="ready",
        review_path=result["reviewPath"],
        review=review,
        preview=validation,
        edits=remapped_edits,
        redetection={
            "thresholds": thresholds,
            "keep_rebate": keep_rebate,
            "unmatched_exclusions": result.get("unmatchedExclusions", 0),
            "edits_path": str(edits_path),
        },
    )
    write_json(path, receipt)
    return receipt


def revise(request, on_progress, confirm=False):
    path = Path(request["receipt"]).resolve()
    receipt = json.loads(path.read_text())
    if receipt["state"] not in ("ready", "previewed"):
        raise ValueError("审核不可继续；请重新分析，不能改写签名记录或锁")
    if request["revision"] != receipt["revision"]:
        raise ValueError("当前配方与审核来源不一致，请重新除尘")
    current = backend_identity(receipt["backend"]["root"])
    if current["fingerprint"] != receipt["backend"]["fingerprint"]:
        raise ValueError("PURE Neg 引擎自分析后发生变化，请重新分析")
    verified_pixels(receipt["source"])
    edits = request.get("edits", receipt["edits"])
    if set(edits) - {"excludedRegions", "strokes"}:
        raise ValueError("本桥接只支持灰尘排除与补修，不包含牛顿环")
    import uuid

    edits_path = path.parent / ("edits-" + uuid.uuid4().hex + ".json")
    encoded_edits = json.dumps(
        edits, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    if len(encoded_edits.encode("utf-8")) > 8_000_000:
        raise ValueError("审核选择文件超过 PURE Neg 接口限制")
    edits_path.write_text(encoded_edits)
    flags = {"review": receipt["review_path"], "edits": edits_path}
    replace_live = not confirm and bool(request.get("replace_live_preview"))
    previous_stage = receipt.get("live_preview_stage") if replace_live else None
    previous_edits = receipt.get("edits_path") if previous_stage else None
    if confirm:
        flags.update(
            output=request["output"],
            format="tiff",
            mask="yes",
            **{
                "export-settings": '{"space":"keep","depth":"keep","compression":"none"}'
            },
        )
    if confirm:
        receipt.update(state="confirming", edits=edits, edits_path=str(edits_path))
        write_json(path, receipt)
    try:
        result = run_cli(
            current["root"],
            receipt["work_root"],
            "confirm" if confirm else "preview",
            flags,
            on_progress,
        )
        cleaned = result["paths"]["cleaned"] if confirm else result["cleanedPath"]
        mask = result["paths"]["mask"] if confirm else result["maskPath"]
        validation = verify_repair(receipt["source"], cleaned, mask)
    except BaseException as error:
        if confirm:
            receipt.update(state="uncertain", error=str(error))
            write_json(path, receipt)
        raise
    receipt.update(
        state="confirmed" if confirm else "previewed",
        edits=edits,
        edits_path=str(edits_path),
        preview=validation,
    )
    if confirm:
        receipt["published"] = result
    if replace_live:
        receipt["live_preview_stage"] = str(Path(cleaned).parent)
    else:
        receipt.pop("live_preview_stage", None)
    write_json(path, receipt)
    if previous_stage:
        # Only reclaim files from the previous automatic preview after the
        # receipt points at the verified replacement. The signed review and
        # published outputs are outside this temporary stage.
        stage = Path(previous_stage).resolve()
        review_dir = Path(receipt["review_path"]).resolve().parent
        if stage.parent == review_dir and stage.name.startswith("preview-") and stage != Path(cleaned).resolve().parent:
            with suppress(OSError):
                for name in ("cleaned.tiff", "dust-mask.tiff", "report.json"):
                    (stage / name).unlink(missing_ok=True)
                stage.rmdir()
            if previous_edits:
                old_input = Path(previous_edits)
                if old_input.parent == path.parent and old_input.match("edits-*.json"):
                    with suppress(OSError):
                        old_input.unlink(missing_ok=True)
    return receipt
