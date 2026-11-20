"""Export frozen draft snapshots at full size; existing TIFFs remain byte copies."""

import json
from pathlib import Path
import subprocess
import sys
import os
import re
import threading

from .preview_data import ROOT, copy_verified_tiff
from .workspace import write_json
from .diagnostics import event, failure, span

_MEMORY_RECHECK_SECONDS = 5.0


def _available_memory():
    """Estimate reclaimable physical memory, falling back to serial if unknown."""
    try:
        if sys.platform == "darwin":
            output = subprocess.check_output(
                ["/usr/bin/vm_stat"], text=True, timeout=2
            )
            page_size = re.search(r"page size of (\d+) bytes", output)
            pages = {}
            for line in output.splitlines():
                match = re.fullmatch(r"Pages (free|inactive|speculative):\s*(\d+)\.", line)
                if match:
                    pages[match.group(1)] = int(match.group(2))
            if page_size and len(pages) == 3:
                return int(page_size.group(1)) * sum(pages.values())
        elif sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="ascii") as stream:
                for line in stream:
                    match = re.fullmatch(r"MemAvailable:\s*(\d+)\s+kB\s*", line)
                    if match:
                        return int(match.group(1)) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def _render_concurrency(resident_bytes=0):
    """Budget 12 GiB per render, accounting for already resident workers."""
    from .workflow_performance import hardware_info

    total = hardware_info().get("total_memory_bytes")
    available = _available_memory()
    if not total or available is None:
        return 1
    return max(1, min(3, int(min(total, available + resident_bytes) // (12 * 1024**3))))


def _active_render_resident_bytes(processes):
    """Conservatively credit private memory already held by active renderers."""
    from .workflow_performance import process_rss_bytes

    gib = 1024**3
    resident = 0
    for process in processes:
        pid = getattr(process, "pid", None)
        rss = process_rss_bytes(pid)
        if rss is not None:
            # RSS includes potentially shared or reclaimable pages. Keep a
            # 1 GiB allowance per worker and never credit beyond its budget.
            resident += min(12 * gib, max(0, rss - gib))
    return resident


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
            except BaseException:
                try:
                    process.terminate()
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                except OSError:
                    pass
                raise
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
    fractions = [0.0] * total
    fraction_sum = [0.0]
    active = []
    condition = threading.Condition()
    closing = [False]

    def terminate_running():
        with lock:
            running = list(active)
        for process in running:
            try:
                process.terminate()
            except OSError:
                pass

    def report_progress(index, kind, process, fraction=0.0, message=""):
        with lock:
            if kind == "spawn":
                active.append(process)
                if closing[0]:
                    try:
                        process.terminate()
                    except OSError:
                        pass
                return
            if kind == "exit":
                if process in active:
                    active.remove(process)
                return
            updated = max(fractions[index], min(0.99, max(0.0, fraction)))
            fraction_sum[0] += updated - fractions[index]
            fractions[index] = updated
            progress(
                {
                    "type": "progress",
                    "fraction": min(1.0, fraction_sum[0] / total),
                    "message": message,
                }
            )

    def handle(index, item):
        with lock:
            progress(
                {
                    "type": "progress",
                    "fraction": min(1.0, fraction_sum[0] / total),
                    "message": f"导出 {index + 1}/{total}：{item['id']}",
                }
            )
        try:
            def item_progress(kind, process, fraction=0.0, message=""):
                report_progress(index, kind, process, fraction=fraction, message=message)

            saved = _export_item(item, index, total, root, item_progress)
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
            fraction_sum[0] += 1.0 - fractions[index]
            fractions[index] = 1.0
            write_json(root / "export.json", report)
            progress({"type": "progress", "fraction": 1.0 if done[0] == total else min(1.0, fraction_sum[0] / total),
                      "message": f"已完成 {done[0]}/{total}"})

    try:
        if total <= 1:
            for index, item in enumerate(items):
                handle(index, item)
        else:
            scheduled = [0]
            next_index = [0]
            thread_errors = []

            def guarded():
                while True:
                    with condition:
                        while not closing[0] and next_index[0] < total:
                            with lock:
                                processes = list(active)
                            resident = _active_render_resident_bytes(processes)
                            if scheduled[0] < _render_concurrency(resident):
                                break
                            # Available memory can increase while an existing
                            # render runs, without any task-completion signal.
                            condition.wait(timeout=_MEMORY_RECHECK_SECONDS)
                        if closing[0] or next_index[0] >= total:
                            return
                        index = next_index[0]
                        next_index[0] += 1
                        scheduled[0] += 1
                    try:
                        handle(index, items[index])
                    except BaseException as error:
                        with condition:
                            thread_errors.append(error)
                            closing[0] = True
                            condition.notify_all()
                        terminate_running()
                        return
                    finally:
                        with condition:
                            scheduled[0] -= 1
                            condition.notify_all()

            threads = [
                threading.Thread(target=guarded, daemon=True)
                for _ in range(min(total, 3))
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if thread_errors:
                raise thread_errors[0]
        report["state"] = "partial" if report["errors"] else "complete"
        event(
            "export_complete",
            saved=len(report["saved"]),
            errors=len(report["errors"]),
            directory=str(root),
        )
    except BaseException:
        report["state"] = "failed"
        raise
    finally:
        with condition:
            closing[0] = True
            condition.notify_all()
        terminate_running()
        write_json(root / "export.json", report)
    return report
