"""Focused acceptance of draft recovery, presets, export selection and cancellation."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import uuid

from PyQt6.QtCore import QPoint, QPointF, Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from .preview_ui import STYLE, prepare_plugins
from .recipe_dialog import RecipeDialog
from .repair_ui import RepairDialog
from .tasks import Tasks
from .workbench import Workbench
from .workspace import write_json


def main():
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepare_plugins()
    app = QApplication(sys.argv[:1])
    app.setStyleSheet(STYLE)
    app.setQuitOnLastWindowClosed(False)
    window = Workbench(Path(sys.argv[1]))
    window.automation = True
    window.show()
    holder = {"window": window}
    state = {"state": "running", "stage": "start", "checks": []}
    deadline = time.monotonic() + 120
    tasks = Tasks(output / "requests", window)

    def done(request, result):
        holder["result"] = result

    def failed(request, message, cancelled):
        if state["stage"] == "cancel" and cancelled:
            state["cancel_finished"] = True
        else:
            state["error"] = message

    tasks.done.connect(done)
    tasks.failed.connect(failed)

    def cancel(event):
        if state["stage"] == "cancel" and not state.get("cancel_time"):
            state["group"] = int(tasks.process.processId())
            state["cancel_time"] = time.monotonic()
            tasks.cancel()

    tasks.progress.connect(cancel)
    timer = QTimer(window)
    timer.setInterval(100)

    def finish(error=None):
        timer.stop()
        state.update(state="failed" if error else "complete")
        if error:
            state["error"] = str(error)
        write_json(output / "verification.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        tasks.shutdown()
        for widget in app.topLevelWidgets():
            widget.close()
        app.exit(1 if error else 0)

    def step():
        try:
            if state.get("error"):
                raise RuntimeError(state["error"])
            if time.monotonic() > deadline:
                raise TimeoutError("补充界面验收超时")
            stage = state["stage"]
            w = holder["window"]
            if stage == "start":
                if w.loaded_key is None:
                    return
                initial = w.workspace.data["jobs"][0]["recipe"]
                d = RecipeDialog([initial["raw"]], w, initial)
                d.ev.setValue(0.5)
                preset = d.preset()
                write_json(output / "preset.json", preset)
                d.set_parameters(
                    {
                        **json.loads((output / "preset.json").read_text())[
                            "parameters"
                        ],
                        "raw": initial["raw"],
                    }
                )
                assert (
                    d.recipes()[0]["exposure_ev"] == 0.5
                    and d.recipes()[0]["raw"] == initial["raw"]
                )
                d.close()
                assert len(w.export_records()) == len(w.samples)
                w.toggle_export_checks()
                assert not w.export_records()
                w.toggle_export_checks()
                w.photo_list.item(0).setCheckState(Qt.CheckState.Unchecked)
                assert len(w.export_records()) == len(w.samples) - 1
                state["checks"].append("处理预设保存/读取及独立批量导出勾选通过")
                record = w.samples[0].record("cool-shared-rgb", "baseline")
                holder["source"] = record
                state["stage"] = "cancel"
                tasks.start(
                    {
                        "action": "repair_prepare",
                        "source": record,
                        "revision": record["sha256"],
                        "directory": str(
                            w.workspace.root
                            / "repairs"
                            / ("cancel-" + uuid.uuid4().hex)
                        ),
                    }
                )
            elif stage == "cancel":
                if (
                    not state.get("cancel_finished")
                    or time.monotonic() - state.get("cancel_time", time.monotonic())
                    < 3.5
                ):
                    return
                try:
                    os.killpg(state["group"], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise AssertionError("PURE Neg 取消后仍有进程组残留")
                state["checks"].append(
                    "PURE Neg 分析实际取消，worker/Node 进程组均回收"
                )
                source = holder["source"]
                holder["receipt_path"] = (
                    w.workspace.root
                    / "repairs"
                    / ("draft-" + uuid.uuid4().hex)
                    / "receipt.json"
                )
                tasks.start(
                    {
                        "action": "repair_prepare",
                        "source": source,
                        "revision": source["sha256"],
                        "directory": str(holder["receipt_path"].parent),
                    }
                )
                state["stage"] = "prepare"
            elif stage == "prepare":
                if tasks.busy or "result" not in holder:
                    return
                d = RepairDialog(holder["receipt_path"], output / "requests", w)
                holder["dialog"] = d
                d.show()
                state["stage"] = "paint"
            elif stage == "paint":
                d = holder["dialog"]
                if d.loading:
                    return
                d.tool.setCurrentIndex(d.tool.findData("erase"))
                region = d.receipt["review"]["regions"][0]
                p = d.canvas.origin + d.canvas.zoom * QPointF(
                    region["x"] + region["width"] / 2,
                    region["y"] + region["height"] / 2,
                )
                QTest.mouseClick(
                    d.canvas,
                    Qt.MouseButton.LeftButton,
                    pos=QPoint(round(p.x()), round(p.y())),
                )
                assert d.dirty and d.draft_path.exists()
                holder["edits"] = deepcopy(d.edits)
                d.close()
                d = RepairDialog(holder["receipt_path"], output / "requests", w)
                holder["dialog"] = d
                assert d.edits == holder["edits"] and d.dirty
                d.show()
                state["stage"] = "restore"
            elif stage == "restore":
                d = holder["dialog"]
                if d.loading:
                    return
                assert not d.confirm_button.isEnabled()
                d.grab().save(str(output / "draft-recovered.png"))
                d.update_button.click()
                state["stage"] = "preview"
            elif stage == "preview":
                d = holder["dialog"]
                if d.tasks.busy or d.loading:
                    return
                assert (
                    not d.dirty
                    and not d.draft_path.exists()
                    and d.confirm_button.isEnabled()
                )
                assert d.receipt["preview"]["outside_mask_changed_pixels"] == 0
                d.grab().save(str(output / "draft-verified.png"))
                d.close()
                w.workspace.data.setdefault("repairs", []).append(
                    str(holder["receipt_path"])
                )
                w.workspace.save()
                state["checks"].append(
                    "真实橡皮笔画草稿退出后恢复；更新预览后才恢复确认能力"
                )
                count = len(w.samples)
                path = w.workspace.root
                w.close()
                w = Workbench(path)
                w.automation = True
                holder["window"] = w
                w.show()
                assert len(w.samples) == count
                state["stage"] = "reopen"
            elif stage == "reopen":
                if w.loaded_key is None:
                    return
                w.grab().save(str(output / "workspace-reopened.png"))
                state["checks"].append("工作区关闭/重开保持完成版本及审核索引")
                finish()
        except Exception as error:
            finish(error)

    timer.timeout.connect(step)
    timer.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
