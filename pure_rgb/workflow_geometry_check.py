"""Actual failed auto-ROI → visual drag → revised full-frame render acceptance."""

import json
from pathlib import Path
import sys
import time

from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from .geometry_ui import GeometryDialog
from .preview_data import verified_pixels
from .preview_ui import STYLE, prepare_plugins
from .recipe_dialog import RecipeDialog
from .workbench import Workbench
from .workspace import write_json


def main():
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepare_plugins()
    app = QApplication(sys.argv[:1])
    app.setStyleSheet(STYLE)
    w = Workbench(Path(sys.argv[1]))
    w.automation = True
    w.show()
    base = w.workspace.data["jobs"][0]["recipe"]
    state = {"state": "running", "stage": "start", "checks": []}
    deadline = time.monotonic() + 150
    timer = QTimer(w)
    timer.setInterval(100)

    def finish(error=None):
        timer.stop()
        state.update(state="failed" if error else "complete")
        if error:
            state["error"] = str(error)
        write_json(output / "verification.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        for widget in app.topLevelWidgets():
            widget.close()
        app.exit(1 if error else 0)

    def tick():
        try:
            if time.monotonic() > deadline:
                raise TimeoutError("视觉框选流程超时")
            stage = state["stage"]
            if stage == "start":
                job = w.workspace.add(base)
                state["auto_job"] = job["id"]
                w.start_queue()
                state["stage"] = "auto"
            elif stage == "auto":
                job = w.workspace.job(state["auto_job"])
                if job["state"] == "running" or job["state"] == "queued":
                    return
                assert job["state"] == "failed" and "rectangle" in job["error"], job
                geometry = w.workspace.root / "runs" / job["id"] / "geometry.json"
                assert geometry.exists()
                state["checks"].append(
                    "自动取框真实失败，明确报错并保留合成定位预览，未退回全图"
                )
                for i in range(w.queue_list.count()):
                    if w.queue_list.item(i).data(Qt.ItemDataRole.UserRole) == job["id"]:
                        w.queue_list.setCurrentRow(i)
                state["stage"] = "recipe"
                QTimer.singleShot(0, w.retry_job)
            elif stage == "recipe":
                dialog = app.activeModalWidget()
                if not isinstance(dialog, RecipeDialog):
                    return
                state["stage"] = "geometry"
                QTimer.singleShot(0, dialog.choose_roi)
            elif stage == "geometry":
                dialog = app.activeModalWidget()
                if not isinstance(dialog, GeometryDialog):
                    return
                geo = dialog.geometry_info
                xscale, yscale = (
                    dialog.pixmap.width() / geo["width"],
                    dialog.pixmap.height() / geo["height"],
                )
                a = dialog.view.mapFromScene(QPointF(1000 * xscale, 350 * yscale))
                b = dialog.view.mapFromScene(QPointF(5100 * xscale, 3650 * yscale))
                QTest.mousePress(
                    dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=a
                )
                QTest.mouseMove(dialog.view.viewport(), b, 30)
                QTest.mouseRelease(
                    dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=b
                )
                assert dialog.selection and dialog.ok.isEnabled(), (
                    "No selection",
                    dialog.selection,
                )
                assert (
                    max(
                        abs(x - y)
                        for x, y in zip(dialog.selection, [1000, 350, 5100, 3650])
                    )
                    < 15
                ), dialog.selection
                state["selected_roi"] = dialog.selection
                dialog.grab().save(str(output / "visual-roi.png"))
                dialog.accept()
                state["stage"] = "accept"
            elif stage == "accept":
                dialog = app.activeModalWidget()
                if not isinstance(dialog, RecipeDialog):
                    return
                recipe = dialog.recipes()[0]
                assert recipe["roi"] == list(state["selected_roi"])
                assert (
                    recipe["density"] == "percentile"
                    and recipe["paper"] == "kodak_endura"
                    and recipe["exposure_ev"] == 0.3
                    and recipe["crop"] is False
                )
                state["checks"].append("真实鼠标框选转为原始像素坐标，生成独立配方重试")
                dialog.accept()
                state["stage"] = "render"
            elif stage == "render":
                job = w.workspace.data["jobs"][-1]
                if job["state"] == "failed":
                    raise RuntimeError(job["error"])
                if job["state"] != "complete" or w.loaded_key is None:
                    return
                sample = w.workspace.sample(job)
                manifest = json.loads((sample.directory / "manifest.json").read_text())
                for mode in ("native", "cool-shared-rgb", "flex-endpoint"):
                    pixels, icc = verified_pixels(sample.record(mode, "baseline"))
                    assert list(pixels.shape) == manifest["source_shape"] and icc
                assert manifest["native_config"]["paper_profile"] == "kodak_endura"
                state["checks"].append(
                    "手工框选重试成功：百分位密度、Endura 相纸、+0.3 EV、全画幅三路 RGB16/ICC 均通过"
                )
                w.grab().save(str(output / "full-frame-options.png"))
                finish()
        except Exception as error:
            finish(error)

    timer.timeout.connect(tick)
    timer.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
