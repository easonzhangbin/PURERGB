"""Exercise multi-triplet import and consecutive queue jobs through real GUI signals."""

import json
from pathlib import Path
import sys
import time

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from .native import sha256
from .preview_data import load_samples
from .preview_ui import STYLE, prepare_plugins
from .recipe_dialog import RecipeDialog
from .study import read_rgb
from .workbench import Workbench
from .workspace import write_json


def main():
    root = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepare_plugins()
    app = QApplication(sys.argv[:1])
    app.setStyleSheet(STYLE)
    window = Workbench(root)
    window.automation = True
    window.show()
    primary = load_samples()
    chosen = [primary[0], primary[6]]
    state = {"state": "running", "stage": "begin", "checks": []}
    deadline = time.monotonic() + 180
    timer = QTimer(window)
    timer.setInterval(100)

    def finish(error=None):
        timer.stop()
        state.update(state="failed" if error else "complete")
        if error:
            state["error"] = str(error)
        write_json(output / "verification.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        window.close()
        app.exit(1 if error else 0)

    def tick():
        try:
            if time.monotonic() > deadline:
                raise TimeoutError("批量全流程验收超时")
            if state["stage"] == "begin":
                window.begin_import(
                    [r["path"] for sample in chosen for r in sample.inputs]
                )
                state["stage"] = "dialog"
            elif state["stage"] == "dialog":
                dialog = app.activeModalWidget()
                if not isinstance(dialog, RecipeDialog):
                    return
                recipes = dialog.recipes()
                assert len(recipes) == 2 and all(
                    r["cool_tone"] == "soft-shoulder" for r in recipes
                )
                dialog.grab().save(str(output / "batch-import.png"))
                dialog.accept()
                state["checks"].append(
                    "两组三色文件自动核对后统一确认，均默认 soft-shoulder"
                )
                state["stage"] = "queue"
            elif state["stage"] == "queue":
                jobs = window.workspace.data["jobs"]
                if any(j["state"] == "failed" for j in jobs):
                    raise RuntimeError(
                        [j.get("error") for j in jobs if j["state"] == "failed"]
                    )
                if (
                    len(jobs) != 2
                    or any(j["state"] != "complete" for j in jobs)
                    or window.loaded_key is None
                ):
                    return
                comparisons = {}
                for job in jobs:
                    sample = window.workspace.sample(job)
                    original = next(
                        s
                        for s in chosen
                        if s.inputs[0]["path"] == job["inputs"][0]["path"]
                    )
                    case = {}
                    for mode in ("native", "flex-endpoint", "cool-shared-rgb"):
                        current, ci = read_rgb(sample.record(mode, "baseline")["path"])
                        reference = (
                            Path("outputs/cool-whitepoint-191-v1")
                            / (original.id + "-soft-shoulder.tiff")
                            if mode == "cool-shared-rgb"
                            else original.record(mode, "baseline")["path"]
                        )
                        expected, ei = read_rgb(reference)
                        case[mode] = bool(
                            np.array_equal(current, expected) and ci == ei
                        )
                    assert all(case.values()), case
                    assert all(sha256(r["path"]) == r["sha256"] for r in job["inputs"])
                    comparisons[job["id"]] = case
                state["pixel_icc_regressions"] = comparisons
                state["checks"].append(
                    "真实串行队列连续完成两组；Native/Flex 精确回归，cool 精确匹配已选 soft 候选；RAW 哈希不变"
                )
                window.grab().save(str(output / "batch-complete.png"))
                finish()
        except Exception as error:
            finish(error)

    timer.timeout.connect(tick)
    timer.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
