"""Export frozen draft snapshots at full size; existing TIFFs remain byte copies."""

import json
import _thread
from pathlib import Path
import subprocess
import sys
import os
import re
import threading
import time

from .preview_data import ROOT, copy_verified_tiff
from .workspace import write_json
from .diagnostics import event, failure, span

_MEMORY_RECHECK_SECONDS = 5.0
_HEADER_TIMEOUT_SECONDS = 1.0
_GIB = 1024**3
_DEFAULT_RENDER_BUDGET = 12 * _GIB
_SIMPLE_RENDER_BUDGET = 8 * _GIB
_LARGE_RAW_PIXELS = 100_000_000
_SIMPLE_RAW_PIXELS = 25_000_000


def _render_budget(item):
    """Estimate an admission budget, failing closed when RAW evidence is missing.

    Only the measured, simple Native route with three checked RAW headers may
    use the smaller budget. Sensor dimensions are deliberately used instead
    of a potentially smaller crop or a stale completed-run source_shape.
    """
    render = item.get("render") or {}
    if not render:
        try:
            import tifffile
            from .raw_import import file_identity

            path = item["record"]["path"]
            identity = file_identity(path)
            with tifffile.TiffFile(path) as tif:
                shape = tif.pages[0].shape
                valid = len(shape) == 3 and shape[-1] == 3 and tif.pages[0].dtype.name == "uint16"
            if file_identity(path) != identity or not valid:
                raise ValueError("TIFF changed or geometry is unsupported")
            pixels = shape[0] * shape[1]
            if pixels > _LARGE_RAW_PIXELS:
                return _DEFAULT_RENDER_BUDGET, True, ((path, identity),)
            # The copy path decodes the TIFF to verify it, but does not run
            # the RAW renderer. Retain room for decompression and the RGB16 frame.
            return 4 * _GIB, False, ((path, identity),)
        except Exception:
            return _DEFAULT_RENDER_BUDGET, False, ()
    if not isinstance(render, dict):
        return _DEFAULT_RENDER_BUDGET, False, ()
    recipe = render.get("recipe") or {}
    if not isinstance(recipe, dict):
        return _DEFAULT_RENDER_BUDGET, False, ()
    raw = recipe.get("raw")
    if not isinstance(raw, list) or len(raw) != 3:
        return _DEFAULT_RENDER_BUDGET, False, ()
    try:
        from .raw_import import file_identity, inspect_raw_header

        identities = []
        shapes = []
        for path in raw:
            identity = file_identity(path)
            header = inspect_raw_header(path)
            if file_identity(path) != identity:
                raise ValueError("RAW changed after header inspection")
            shape = tuple(header["plan"].raw_shape)
            if len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
                raise ValueError("invalid RAW sensor geometry")
            identities.append((path, identity))
            shapes.append(shape)
    except Exception:
        # Unknown dimensions retain the existing conservative budget.
        return _DEFAULT_RENDER_BUDGET, False, ()
    if any(height * width > _LARGE_RAW_PIXELS for height, width in shapes):
        return _DEFAULT_RENDER_BUDGET, True, tuple(identities)
    allowed = {"raw", "roi", "density", "paper", "exposure_ev", "crop", "flatfield", "cool_tone", "raw_import"}
    simple = (
        item.get("mode") == "native"
        and render.get("only_mode", "native") == "native"
        and not render.get("repairs")
        and not render.get("calibration")
        and set(recipe) <= allowed
        and recipe.get("raw_import") == "standard-crop-v1"
        and recipe.get("density", "negpy") == "negpy"
        and recipe.get("paper", "neutral") == "neutral"
        and recipe.get("exposure_ev", 0) == 0
        and recipe.get("crop", False) is False
        and not recipe.get("flatfield")
        and len(set(shapes)) == 1
        and all(height * width <= _SIMPLE_RAW_PIXELS for height, width in shapes)
    )
    return (_SIMPLE_RENDER_BUDGET if simple else _DEFAULT_RENDER_BUDGET), False, tuple(identities)


def _budget_still_valid(estimate):
    """Reject a reduced estimate if any source changed after inspection."""
    budget, serial, identities = estimate
    if not identities:
        return _DEFAULT_RENDER_BUDGET, serial
    try:
        from .raw_import import file_identity
        if any(file_identity(path) != identity for path, identity in identities):
            return _DEFAULT_RENDER_BUDGET, True
    except (OSError, ValueError, TypeError):
        return _DEFAULT_RENDER_BUDGET, True
    return budget, serial


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


def _has_admission_capacity(reservations, processes, candidate):
    """Reserve every active task's remaining budget before starting another."""
    budget, serial = candidate
    if not reservations:
        # Always allow one job: failing to probe memory must not deadlock export.
        return True
    if serial or any(other_serial for _, other_serial in reservations.values()):
        return False
    resident = _active_render_resident_bytes(processes)
    if budget == _DEFAULT_RENDER_BUDGET and all(
        other == _DEFAULT_RENDER_BUDGET for other, _ in reservations.values()
    ):
        # Preserve the long-standing unknown/complex-route policy.
        return len(reservations) < _render_concurrency(resident)
    from .workflow_performance import hardware_info

    total = hardware_info().get("total_memory_bytes")
    available = _available_memory()
    if not total or available is None:
        return False
    reserved = sum(other for other, _ in reservations.values())
    return reserved + budget <= min(total, available + resident)


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
    # Header probing happens only when the next item reaches admission.  A
    # daemon probe avoids blocking cancellation on a stalled network volume.
    estimates = [None] * total
    deadlines = [None] * total
    validations = [None] * total
    validation_deadlines = [None] * total
    pending = object()
    lock = threading.Lock()
    done = [0]
    fractions = [0.0] * total
    fraction_sum = [0.0]
    active = []
    condition = threading.Condition()
    closing = [False]
    probe_slot = threading.BoundedSemaphore(1)

    def start_probe(callback, complete):
        # A stalled filesystem call cannot be cancelled. Keep at most one
        # outstanding probe and use the serial fallback while it is stalled.
        if not probe_slot.acquire(blocking=False):
            return False

        def run():
            try:
                result = callback()
            except Exception:
                result = None
            with condition:
                probe_slot.release()
                complete(result)
                condition.notify_all()

        try:
            _thread.start_new_thread(run, ())
        except RuntimeError:
            probe_slot.release()
            return False
        return True

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
            reservations = {}
            next_index = [0]
            thread_errors = []

            def guarded():
                while True:
                    with condition:
                        while not closing[0] and next_index[0] < total:
                            index = next_index[0]
                            if estimates[index] is None:
                                estimates[index] = pending
                                deadlines[index] = time.monotonic() + _HEADER_TIMEOUT_SECONDS

                                def inspect_item(target=index):
                                    return _render_budget(items[target])

                                def inspected(value, target=index):
                                    if estimates[target] is pending:
                                        estimates[target] = value or (_DEFAULT_RENDER_BUDGET, True, ())

                                if not start_probe(inspect_item, inspected):
                                    estimates[index] = (_DEFAULT_RENDER_BUDGET, True, ())
                            if estimates[index] is pending:
                                remaining = deadlines[index] - time.monotonic()
                                if remaining > 0:
                                    condition.wait(timeout=remaining)
                                    continue
                                # A slow or unavailable volume is unknown. Run
                                # it alone; never infer small from a timeout.
                                estimates[index] = (_DEFAULT_RENDER_BUDGET, True, ())
                            with lock:
                                processes = list(active)
                            tentative = estimates[index][:2]
                            if _has_admission_capacity(reservations, processes, tentative):
                                if validations[index] is None:
                                    token = object()
                                    validations[index] = token
                                    validation_deadlines[index] = time.monotonic() + _HEADER_TIMEOUT_SECONDS

                                    def validate_item(target=index):
                                        return _budget_still_valid(estimates[target])

                                    def validated(result, target=index, generation=token):
                                        if validations[target] is generation:
                                            validations[target] = result or (_DEFAULT_RENDER_BUDGET, True)

                                    if not start_probe(validate_item, validated):
                                        validations[index] = (_DEFAULT_RENDER_BUDGET, True)
                                if type(validations[index]) is not tuple:
                                    remaining = validation_deadlines[index] - time.monotonic()
                                    if remaining > 0:
                                        condition.wait(timeout=remaining)
                                        continue
                                    # A stalled final stat cannot block the queue;
                                    # reserve the unknown input exclusively.
                                    validations[index] = (_DEFAULT_RENDER_BUDGET, True)
                                candidate = validations[index]
                                if _has_admission_capacity(reservations, processes, candidate):
                                    break
                            # Available memory can increase while an existing
                            # render runs, without any task-completion signal.
                            validations[index] = None
                            condition.wait(timeout=_MEMORY_RECHECK_SECONDS)
                        if closing[0] or next_index[0] >= total:
                            return
                        index = next_index[0]
                        next_index[0] += 1
                        scheduled[0] += 1
                        reservations[index] = candidate
                        event(
                            "export_admission", sample=items[index].get("id"),
                            reserved_bytes=candidate[0], exclusive=candidate[1],
                            scheduled=scheduled[0],
                        )
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
                            reservations.pop(index, None)
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
