"""Real RAW + Cocoa professional-editor regression, in an isolated workspace."""

import json
from pathlib import Path
import sys
import time

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QDialogButtonBox

from .cool_whitepoint_study import lift_codes
from .native import sha256
from .preview_data import ROOT, MODES, verified_pixels
from .preview_ui import STYLE, prepare_plugins
from .professional_controls import NATIVE_SPECS, output_codes
from .recipe_dialog import RecipeDialog
from .workspace import Workspace, write_json


def main():
    import pure_rgb.workbench as workbench

    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    original = (
        ROOT / "outputs/PURE-RGB-Workspace-v01/runs/69c0dfb190a744e6/manifest.json"
    )
    old = json.loads(original.read_text())
    conservative = json.loads(
        (
            ROOT / "outputs/workflow-e2e-v2/runs/00dc6fd9dc584517/manifest.json"
        ).read_text()
    )
    workbench.PREFERENCES = out / "preferences.json"
    Workspace(out / "workspace", create=True)
    prepare_plugins()
    app = QApplication(sys.argv[:1])
    app.setStyleSheet(STYLE)
    app.setFont(QFont("PingFang SC", 12))
    window = workbench.Workbench(out / "workspace")
    window.automation = True
    window.show()
    baseline = window.workspace.add(old["recipe"])
    window.refresh_queue()
    window.start_queue()
    state = {
        "state": "running",
        "stage": "baseline",
        "checks": [],
        "jobs": {"baseline": baseline["id"]},
    }
    deadline = time.monotonic() + 480
    timer = QTimer(window)
    timer.setInterval(150)
    native_values = {
        "exposure.density": 0.9,
        "exposure.wb_cyan": 0.08,
        "exposure.shoulder": 0.15,
        "exposure.highlight_density": 0.06,
        "lab.sharpen": 0.4,
    }
    output_values = {
        "exposure_ev": 0.2,
        "red_ev": 0.04,
        "green_ev": -0.03,
        "blue_ev": -0.08,
        "contrast": 0.95,
        "saturation": 0.9,
    }

    def finish(error=None):
        timer.stop()
        state.update(state="failed" if error else "complete")
        if error:
            state["error"] = str(error)
        write_json(out / "verification.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        window.close()
        app.exit(1 if error else 0)

    def tick():
        try:
            if time.monotonic() > deadline:
                raise TimeoutError("专业控件验收超时")
            if any(j["state"] == "failed" for j in window.workspace.data["jobs"]):
                raise RuntimeError(str(window.workspace.data["jobs"][-1].get("error")))
            stage = state["stage"]
            if stage in ("edit-native", "edit-output"):
                dialog = app.activeModalWidget()
                if not isinstance(dialog, RecipeDialog):
                    return
                assert dialog.tabs.currentIndex() == 1
                if stage == "edit-native":
                    for page in dialog.professional.native_pages:
                        for key, value in native_values.items():
                            if key in page.controls:
                                page.controls[key].set_value(value)
                    dialog.professional.modes.widget(0).setCurrentIndex(1)
                    state["native_control_count"] = len(NATIVE_SPECS)
                    screenshot = "native-controls.png"
                    target = "native"
                else:
                    for page in dialog.professional.output_pages.values():
                        for key, value in output_values.items():
                            page.controls[key].set_value(value)
                    dialog.professional.modes.setCurrentIndex(1)
                    screenshot = "cool-controls.png"
                    target = "output"
                preset = dialog.preset()
                write_json(out / (target + "-preset.json"), preset)
                before = dialog.recipes()[0]
                dialog.professional.set_documents({})
                dialog.set_parameters({**preset["parameters"], "raw": before["raw"]})
                assert dialog.recipes()[0] == before
                dialog.grab().save(str(out / screenshot))
                if target == "output":
                    dialog.professional.modes.setCurrentIndex(2)
                    dialog.grab().save(str(out / "flex-controls.png"))
                state["stage"] = target + "-await"
                dialog.findChild(QDialogButtonBox).button(
                    QDialogButtonBox.StandardButton.Ok
                ).click()
                return
            if stage in ("native-await", "output-await"):
                target = stage.removesuffix("-await")
                # accept() closes the nested modal event loop; edit_recipe adds
                # the job only after that loop has returned to its caller.
                expected_count = 2 if target == "native" else 3
                if len(window.workspace.data["jobs"]) < expected_count:
                    return
                state["jobs"][target] = window.workspace.data["jobs"][-1]["id"]
                state["stage"] = target
                return
            if window.tasks.busy or window.queue_active or window.loaded_key is None:
                return
            job = window.workspace.job(state["jobs"][stage])
            if job["state"] != "complete":
                return
            manifest = json.loads(
                (
                    window.workspace.root / "runs" / job["id"] / "manifest.json"
                ).read_text()
            )
            if stage == "baseline":
                for mode in MODES:
                    current, icc = verified_pixels(manifest["results"][mode])
                    previous, previous_icc = verified_pixels(old["results"][mode])
                    assert icc == previous_icc and np.array_equal(current, previous), (
                        mode
                    )
                state["checks"].append(
                    "新代码零控件从三张原 RW2 重算：三路像素和 ICC 与已接受 soft 默认版本完全相同"
                )
                state["stage"] = "edit-native"
                QTimer.singleShot(0, window.edit_recipe)
            elif stage == "native":
                changed = {}
                for mode in MODES:
                    current, icc = verified_pixels(manifest["results"][mode])
                    previous, previous_icc = verified_pixels(old["results"][mode])
                    assert icc == previous_icc
                    changed[mode] = int(
                        np.count_nonzero(np.any(current != previous, axis=2))
                    )
                    assert (changed[mode] > 0) == (mode == "native"), mode
                state["native_changed_pixels"] = changed
                state["checks"].append(
                    "真实 Native 印相／CMY／肩部／细节控件生效，cool 与 Flex 像素及 ICC 完全不变"
                )
                window.photo_list.setCurrentRow(0)
                state["stage"] = "edit-output"
                QTimer.singleShot(0, window.edit_recipe)
            elif stage == "output":
                for mode in MODES:
                    current, icc = verified_pixels(manifest["results"][mode])
                    source_record = (
                        conservative["results"][mode]
                        if mode == "cool-shared-rgb"
                        else old["results"][mode]
                    )
                    before, previous_icc = verified_pixels(source_record)
                    expected, info = output_codes(
                        before, {"version": 1, "values": output_values}
                    )
                    if mode == "cool-shared-rgb":
                        expected = lift_codes(expected, soft=True)
                    assert icc == previous_icc and np.array_equal(current, expected), (
                        mode
                    )
                    assert not np.array_equal(
                        current, verified_pixels(old["results"][mode])[0]
                    )
                    state.setdefault("output_diagnostics", {})[mode] = info
                assert all(sha256(r["path"]) == r["sha256"] for r in old["inputs"])
                window.grab().save(str(out / "workbench.png"))
                icon = window.windowIcon().pixmap(128, 128)
                assert not icon.isNull()
                icon.save(str(out / "selected-icon-display.png"))
                state["checks"].extend(
                    [
                        "三路独立后期控件实际输出与单次编码计算逐像素相同，soft 叠加顺序和 ICC 正确",
                        "真实 Qt 调色窗口→保存/恢复预设→创建新版本→串行队列→RGB16 输出通过；RAW 未改变",
                    ]
                )
                state["state"] = "complete"
                state["source_sha256"] = {
                    str(p.relative_to(ROOT)): sha256(p)
                    for p in (ROOT / "pure_rgb").glob("*.py")
                }
                finish()
        except Exception as error:
            finish(error)

    timer.timeout.connect(tick)
    timer.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
