"""Real GUI and backend acceptance using an existing RAW render and ready review."""

from copy import deepcopy
import json
import time

import numpy as np
from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtTest import QTest

from .preview_data import copy_verified_tiff
from .recipe_dialog import RecipeDialog
from .repair_ui import RepairDialog
from .study import read_rgb
from .workspace import checked_recipe, write_json


def exercise(app, window, output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + 240
    state = {"state": "running", "stage": "initial", "checks": []}
    timer = QTimer(window)
    timer.setInterval(100)
    receipt = window.workspace.root / "repairs/first/receipt.json"
    base = checked_recipe(deepcopy(window.workspace.data["jobs"][0]["recipe"]))

    def finish(error=None):
        timer.stop()
        state.update(state="failed" if error else "complete")
        if error:
            state["error"] = str(error)
        write_json(output / "verification.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        window.close()
        app.exit(1 if error else 0)

    holder = {}

    def step():
        try:
            if time.monotonic() > deadline:
                raise TimeoutError("工作台烟测超时")
            stage = state["stage"]
            if stage == "initial":
                if window.loaded_key is None:
                    if window.last_error:
                        raise RuntimeError(window.last_error)
                    return
                window.grab().save(str(output / "workbench.png"))
                state["stage"] = "import-dialog"
                window.begin_import(base["raw"])
            elif stage == "import-dialog":
                dialog = app.activeModalWidget()
                if not isinstance(dialog, RecipeDialog):
                    return
                assert dialog.recipes()[0] == {**base, "cool_tone": "soft-shoulder"}
                dialog.ev.setValue(0.3)
                assert dialog.recipes()[0]["exposure_ev"] == 0.3
                dialog.grab().save(str(output / "recipe.png"))
                dialog.reject()
                state["checks"].append(
                    "RAW 导入通道检测与真实配方对话框保留 R/G/B、ROI、密度及曝光"
                )
                state["stage"] = "launch-review"
            elif stage == "launch-review":
                review = RepairDialog(
                    receipt, window.workspace.root / "requests", window
                )
                holder["review"] = review
                review.show()
                state["stage"] = "review-load"
            elif stage == "review-load":
                review = holder["review"]
                if review.loading:
                    return
                assert review.confirm_button.isEnabled()
                if review.regions.count():
                    review.regions.item(0).setCheckState(Qt.CheckState.Unchecked)
                review.tool.setCurrentIndex(review.tool.findData("add"))
                canvas = review.canvas
                point = QPoint(int(canvas.width() * 0.3), int(canvas.height() * 0.3))
                expected = (
                    (point.x() - canvas.origin.x()) / canvas.zoom,
                    (point.y() - canvas.origin.y()) / canvas.zoom,
                )
                QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=point)
                assert len(review.edits["strokes"]) == 1
                actual = review.edits["strokes"][0]["points"][0]
                assert np.allclose(actual, expected, atol=0.51)
                assert review.dirty and not review.confirm_button.isEnabled()
                review.view.setCurrentIndex(review.view.findData("split"))
                review.grab().save(str(output / "review-before.png"))
                review.update_button.click()
                state["checks"].append("真实画笔坐标映射正确；修改后禁止未预览确认")
                state["stage"] = "preview"
            elif stage == "preview":
                review = holder["review"]
                if review.tasks.busy or review.loading:
                    return
                assert not review.dirty and review.confirm_button.isEnabled()
                assert review.receipt["preview"]["outside_mask_changed_pixels"] == 0
                review.grab().save(str(output / "review.png"))
                review.tasks.start(
                    {
                        "action": "repair_confirm",
                        "receipt": str(receipt),
                        "revision": review.receipt["revision"],
                        "edits": deepcopy(review.edits),
                        "output": str(output / "published"),
                    }
                )
                review.refresh_actions()
                state["stage"] = "confirm"
            elif stage == "confirm":
                review = holder["review"]
                if review.tasks.busy or review.loading:
                    return
                assert review.receipt["state"] == "confirmed"
                state["repair"] = review.receipt["preview"]
                state["backend_stats"] = review.receipt["review"]["stats"]
                state["checks"].append(
                    "PURE Neg prepare→画笔/排除→preview→confirm；RGB16、ICC、蒙版外逐像素核验通过"
                )
                review.close()
                window.workspace.data.setdefault("repairs", []).append(str(receipt))
                window.workspace.save()
                job = window.workspace.add(base)
                state["cancel_job"] = job["id"]

                def cancel_when_reading(event):
                    if window.tasks.request.get("id") == state["cancel_job"]:
                        window.cancel_task()

                window.tasks.progress.connect(cancel_when_reading)
                window.start_queue()
                state["stage"] = "cancel"
            elif stage == "cancel":
                if window.tasks.busy:
                    return
                assert window.workspace.job(state["cancel_job"])["state"] == "cancelled"
                gain = output / "unity-gain.npz"
                np.savez_compressed(
                    gain, gain=np.ones((32, 32, 3), np.float32), layout="sensor"
                )
                recipe = {**base, "flatfield": str(gain.resolve())}
                job = window.workspace.add(recipe)
                state["flat_job"] = job["id"]
                window.start_queue()
                state["checks"].append("实际 RAW 子进程取消且未接纳部分结果")
                state["stage"] = "flat"
            elif stage == "flat":
                job = window.workspace.job(state["flat_job"])
                if job["state"] == "failed":
                    raise RuntimeError(job["error"])
                if job["state"] != "complete" or window.loaded_key is None:
                    return
                first = window.workspace.sample(window.workspace.data["jobs"][0])
                sample = window.workspace.sample(job)
                manifest = json.loads((sample.directory / "manifest.json").read_text())
                assert manifest["calibration"]["applied_token"]
                exact = {}
                for mode in ("native", "cool-shared-rgb", "flex-endpoint"):
                    a, ai = read_rgb(first.record(mode, "baseline")["path"])
                    b, bi = read_rgb(sample.record(mode, "baseline")["path"])
                    exact[mode] = bool(np.array_equal(a, b) and ai == bi)
                    copy_verified_tiff(
                        sample.record(mode, "baseline"), output / (mode + "-copy.tiff")
                    )
                assert all(exact.values()), exact
                state["unity_flatfield_regression"] = exact
                state["checks"].append(
                    "单位平场配置真实接入，三路仍逐像素/ICC 精确回归；三模式导出完成"
                )
                window.grab().save(str(output / "workbench-final.png"))
                finish()
        except Exception as error:
            finish(error)

    timer.timeout.connect(step)
    timer.start()
