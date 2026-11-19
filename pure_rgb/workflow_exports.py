"""Export frozen draft snapshots at full size; existing TIFFs remain byte copies."""

import json
from pathlib import Path
import subprocess
import sys
import os
import threading

from .preview_data import ROOT, copy_verified_tiff
from .workspace import write_json
from .diagnostics import event, failure, span


def _render_concurrency():
    """One render peaks near 12GB; unknown hardware stays serial."""
    from .workflow_performance import hardware_info

    memory = hardware_info().get("total_memory_bytes")
    if not memory:
        return 1
    return max(1, min(3, int(memory // (12 * 1024**3))))


def _export_item(item, index, total, root, report_progress):
    event(
        "export_item_start",
        sample=item["id"],
        mode=item["mode"],
        destination=item["destination"],
        draft=bool(item.get("render")),
    )
    if Path(item["destination"]).exists():
        raise FileExistsError("目标文件已存在，请选择新文件名")
    record = item.get("record")
    if item.get("render"):
        payload = {
            **item["render"],
            "action": "render",
            "directory": str(root / "renders" / str(index)),
        }
        file = root / f"request-{index}.json"
        write_json(file, payload)
        # Fresh interpreter per render preserves per-job native asset state.
        # It inherits this worker's process group for cancellation.
        with (root / f"render-{index}.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "pure_rgb.workflow_worker",
                    str(file),
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                env={**os.environ, "PURE_RGB_INHERIT_PROCESS_GROUP": "1"},
            )
            report_progress("spawn", process)
            error = ""
            try:
                for line in process.stdout:
                    try:
                        response = json.loads(line)
                    except ValueError:
                        continue
                    if response.get("type") == "error":
                        error = str(response.get("message", ""))
                    if response.get("type") == "progress":
                        report_progress(
                            "item_fraction",
                            None,
                            fraction=float(response.get("fraction", 0)),
                            message="按当前参数生成全尺寸 TIFF："
                            + response.get("message", ""),
                        )
                code = process.wait()
            finally:
                report_progress("exit", process)
        if code:
            raise RuntimeError(error or f"全尺寸处理失败（{code}）")
        manifest = json.loads(
            (Path(payload["directory"]) / "manifest.json").read_text()
        )
        if manifest["state"] != "complete":
            raise ValueError("全尺寸结果尚未完成")
        record = manifest["results"][item["mode"]]
    with span("export_verified_copy", sample=item["id"], mode=item["mode"]):
        target = copy_verified_tiff(record, item["destination"])
    return {
        "id": item["id"],
        "path": target,
        "mode": item["mode"],
        "rendered_draft": bool(item.get("render")),
    }


def export_batch(request, progress):
    root = Path(request["directory"]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {"state": "running", "saved": [], "errors": [], "items": request["items"]}
    write_json(root / "export.json", report)
    items = request["items"]
    total = len(items)
    lock = threading.Lock()
    done = [0]
    ceiling = [0.0]
    active = []

    def report_progress(kind, process, fraction=0.0, message=""):
        with lock:
            if kind == "spawn":
                active.append(process)
                return
            if kind == "exit":
                if process in active:
                    active.remove(process)
                return
            ceiling[0] = max(ceiling[0], (done[0] + fraction) / total)
            progress(
                {
                    "type": "progress",
                    "fraction": ceiling[0],
                    "message": message,
                }
            )

    def handle(index, item):
        with lock:
            ceiling[0] = max(ceiling[0], done[0] / total)
            progress(
                {
                    "type": "progress",
                    "fraction": ceiling[0],
                    "message": f"导出 {index + 1}/{total}：{item['id']}",
                }
            )
        try:
            saved = _export_item(item, index, total, root, report_progress)
            with lock:
                report["saved"].append(saved)
        except Exception as error:
            failure(
                "export_item_error", error, sample=item["id"], mode=item["mode"]
            )
            with lock:
                report["errors"].append({"id": item["id"], "message": str(error)})
        with lock:
            done[0] += 1
            write_json(root / "export.json", report)

    try:
        limit = _render_concurrency()
        if limit <= 1 or total <= 1:
            for index, item in enumerate(items):
                handle(index, item)
        else:
            gate = threading.BoundedSemaphore(limit)

            def guarded(index, item):
                with gate:
                    handle(index, item)

            threads = [
                threading.Thread(target=guarded, args=(index, item), daemon=True)
                for index, item in enumerate(items)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        report["state"] = "partial" if report["errors"] else "complete"
        event(
            "export_complete",
            saved=len(report["saved"]),
            errors=len(report["errors"]),
            directory=str(root),
        )
    finally:
        with lock:
            running = list(active)
        for process in running:
            try:
                process.terminate()
            except OSError:
                pass
        write_json(root / "export.json", report)
    return report
