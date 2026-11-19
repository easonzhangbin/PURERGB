"""Actual sidebar → latest preview → switch/restore → full-size version regression."""

import json
from pathlib import Path
import sys
import time

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QFont
from PyQt6.QtWidgets import QApplication

from .native import sha256
from .preview_data import ROOT
from .preview_ui import STYLE, prepare_plugins, display_image
from .workspace import Workspace, write_json


def image_array(image):
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    data = image.constBits().asstring(image.sizeInBytes())
    return (
        np.frombuffer(data, np.uint8)
        .reshape(image.height(), image.bytesPerLine())[:, : image.width() * 3]
        .reshape(image.height(), image.width(), 3)
        .copy()
    )


def main():
    import pure_rgb.workbench as module

    out = Path(sys.argv[1]).resolve()
    out.mkdir(exist_ok=False, parents=True)
    wspace = Workspace(out / "workspace", create=True)
    source = ROOT / "outputs/PURE-RGB-Workspace-v01"
    old = json.loads((source / "workspace.json").read_text())
    for previous in old["jobs"][:2]:
        job = wspace.add(previous["recipe"])
        directory = wspace.root / "runs" / job["id"]
        directory.mkdir(parents=True)
        manifest = json.loads(
            (source / "runs" / previous["id"] / "manifest.json").read_text()
        )
        write_json(directory / "manifest.json", manifest)
        wspace.set_state(job["id"], "complete")
    module.PREFERENCES = out / "preferences.json"
    prepare_plugins()
    app = QApplication(sys.argv[:1])
    app.setStyleSheet(STYLE)
    app.setFont(QFont("PingFang SC", 12))
    w = module.Workbench(wspace.root)
    w.automation = True
    w.show()
    state = {"state": "running", "stage": "start", "checks": [], "preview_seconds": []}
    baseline_ids = [j["id"] for j in w.workspace.data["jobs"]]
    before_hashes = {
        r["path"]: r["sha256"] for j in w.workspace.data["jobs"] for r in j["inputs"]
    }
    timer = QTimer(w)
    timer.setInterval(100)
    deadline = time.monotonic() + 240
    preview_images = {}
    w.live.ready.connect(
        lambda event: state["preview_seconds"].append(event["seconds"])
    )

    def ready_preview():
        return (
            w.editor.note.text().startswith("预览已更新")
            and not w.preview_timer.isActive()
        )

    def finish(error=None):
        timer.stop()
        state["state"] = "failed" if error else "complete"
        if error:
            state["error"] = str(error)
        write_json(out / "verification.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        w.close()
        app.exit(1 if error else 0)

    def tick():
        try:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "侧栏预览验收超时：" + state["stage"] + " " + w.editor.note.text()
                )
            if w.editor.note.text().startswith("预览未完成"):
                raise RuntimeError(w.editor.note.text())
            assert QApplication.activeModalWidget() is None
            stage = state["stage"]
            if stage == "start":
                if w.loaded_key is None:
                    return
                assert w.canvas.isVisible() and w.canvas.width() > w.width() * 0.5
                w.grab().save(str(out / "sidebar.png"))
                values = w.editor.output_pages["cool-shared-rgb"].controls[
                    "exposure_ev"
                ]
                for ev in (0.2, 0.4, 0.8, -0.2):
                    values.set_value(ev)
                state["token"] = w.edit_token
                assert len(w.workspace.data["jobs"]) == 2
                state["stage"] = "cool"
            elif stage == "cool":
                if not ready_preview():
                    return
                assert w.edit_token >= state["token"]
                assert (
                    w.editing_recipe()["output_controls"]["cool-shared-rgb"]["values"][
                        "exposure_ev"
                    ]
                    == -0.2
                )
                preview_images["cool-shared-rgb"] = image_array(w.canvas.current)
                assert not np.array_equal(
                    preview_images["cool-shared-rgb"], image_array(w.canvas.reference)
                )
                state["checks"].append(
                    "侧栏连续调整仅显示最新参数，无弹窗，无新排队任务；照片占最大区域"
                )
                w.grab().save(str(out / "cool-preview.png"))
                w.choose_mode("flex-endpoint")
                w.editor.focus_group(1)
                for key, value in {
                    "film_gamma": 2.3,
                    "brightness": 4,
                    "contrast": 5,
                    "black_code": 4,
                    "white_code": 250,
                }.items():
                    w.editor.engine_pages["flex-endpoint"].controls[key].set_value(
                        value
                    )
                state["stage"] = "flex"
            elif stage == "flex":
                if not ready_preview():
                    return
                preview_images["flex-endpoint"] = image_array(w.canvas.current)
                w.grab().save(str(out / "flex-engine.png"))
                w.choose_mode("native")
                w.editor.focus_group(1)
                for page in w.editor.native_pages:
                    if "exposure.density" in page.controls:
                        page.controls["exposure.density"].set_value(0.9)
                state["stage"] = "native"
            elif stage == "native":
                if not ready_preview():
                    return
                preview_images["native"] = image_array(w.canvas.current)
                w.grab().save(str(out / "native-engine.png"))
                w.canvas.actual()
                state["stage"] = "full-native"
            elif stage == "full-native":
                if not ready_preview() or w.canvas.sample_scale != 1:
                    return
                assert (
                    w.canvas.current.width() == 4100
                    and w.canvas.current.height() == 3300
                )
                assert w.zoom_label.text() == "100%"
                state["checks"].append(
                    "100% 自动请求全尺寸数据，不把缩略图放大冒充像素预览"
                )
                w.grab().save(str(out / "native-full-size.png"))
                frozen = w.editing_recipe()
                state["frozen_recipe"] = frozen
                # Begin another request, switch photos while it is in flight.
                w.start_edit_preview()
                w.photo_list.setCurrentRow(1)
                state["switch_token"] = w.edit_token
                state["switch_time"] = time.monotonic()
                state["stage"] = "switched"
            elif stage == "switched":
                if w.loaded_key is None or time.monotonic() - state["switch_time"] < 3:
                    return
                assert w.samples[w.index].id == baseline_ids[1] and not w.edit_dirty()
                assert "调色预览" not in w.canvas.mode_text
                w.photo_list.setCurrentRow(0)
                assert w.editing_recipe() == state["frozen_recipe"]
                state["checks"].append("切换照片忽略旧预览；返回时恢复独立调色草稿")
                state["stage"] = "restore"
            elif stage == "restore":
                if not ready_preview():
                    return
                assert len(w.workspace.data["jobs"]) == 2
                w.save_edit()
                assert len(w.workspace.data["jobs"]) == 3
                state["saved_id"] = w.workspace.data["jobs"][-1]["id"]
                state["stage"] = "save"
            elif stage == "save":
                job = w.workspace.job(state["saved_id"])
                if job["state"] == "failed":
                    raise RuntimeError(job.get("error"))
                if job["state"] != "complete" or w.tasks.busy:
                    return
                sample = w.workspace.sample(job)
                for mode, image in preview_images.items():
                    full = image_array(
                        display_image(sample.record(mode, "baseline"), w.display_icc)
                    )
                    step = int(np.ceil(max(full.shape[:2]) / 1400))
                    assert np.array_equal(image, full[::step, ::step]), mode
                assert all(sha256(p) == h for p, h in before_hashes.items())
                state["checks"].append(
                    "三路所见预览逐显示像素匹配保存后的全尺寸版本；引擎 Gamma/端点/亮度/反差生效且无重新拟合抵消"
                )
                state["checks"].append(
                    "完整 RGB16/ICC 新版本保存通过，旧版本与 RAW 保留"
                )
                w.grab().save(str(out / "saved.png"))
                finish()
        except Exception as error:
            finish(error)

    timer.timeout.connect(tick)
    timer.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
