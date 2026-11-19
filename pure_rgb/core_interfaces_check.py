"""Real Cocoa/RW2 shared post, profile controls, full-size preview/export check."""

import json
from pathlib import Path
import plistlib
import sys
import time

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from .editing_check import image_array
from .native import sha256
from .preview_data import ROOT, MODES, verified_pixels
from .preview_ui import STYLE, prepare_plugins, display_image
from .workspace import Workspace, write_json


def main():
    import pure_rgb.workbench as module

    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    old = json.loads(
        (
            ROOT / "outputs/PURE-RGB-Workspace-v01/runs/69c0dfb190a744e6/manifest.json"
        ).read_text()
    )
    workspace = Workspace(out / "workspace", create=True)
    seed = workspace.add(old["recipe"])
    write_json(workspace.root / "runs" / seed["id"] / "manifest.json", old)
    workspace.set_state(seed["id"], "complete")
    profile = out / "synthetic-test-only.plist"
    curve = {"Points": [{"X": 0, "Y": 0}, {"X": 128, "Y": 115}, {"X": 255, "Y": 255}]}
    profile.write_bytes(
        plistlib.dumps(
            {
                "ImageSetting": {
                    "ImageCorrection": {
                        "Gamma": 2.4,
                        "Gradations": [curve] * 3,
                        "Brightness": 4,
                        "Contrast": 6,
                    }
                }
            }
        )
    )
    prepare_plugins()
    app = QApplication(sys.argv[:1])
    app.setFont(QFont("PingFang SC", 12))
    app.setStyleSheet(STYLE)
    module.PREFERENCES = out / "preferences.json"
    w = module.Workbench(workspace.root)
    w.automation = True
    w.show()
    state = {
        "state": "running",
        "stage": "start",
        "checks": [],
        "mode_labels": MODES,
        "profile_scope": "Synthetic test fixture, not a factory film profile",
    }
    modes = list(MODES)
    images = {}
    holder = {}
    deadline = time.monotonic() + 600
    timer = QTimer(w)
    timer.setInterval(100)

    def finish(error=None):
        timer.stop()
        state["state"] = "failed" if error else "complete"
        if error:
            state["error"] = str(error)
        write_json(out / "verification.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        w.close()
        app.exit(1 if error else 0)

    def preview_mode(index):
        holder["index"] = index
        w.choose_mode(modes[index])
        w.full_edit_preview()
        state["stage"] = "preview"

    def tick():
        try:
            if time.monotonic() > deadline:
                raise TimeoutError(state["stage"] + " " + w.editor.note.text())
            if w.editor.note.text().startswith("预览未完成"):
                raise RuntimeError(w.editor.note.text())
            assert QApplication.activeModalWidget() is None
            stage = state["stage"]
            if stage == "start":
                if w.loaded_key is None:
                    return
                w.choose_mode("flex-endpoint")
                w.editor.flex_resource.select_path(profile)
                # Check a value equal to the old synthetic default can override a
                # different profile default, then retain the profile's 2.4 baseline.
                gamma = w.editor.engine_pages["flex-endpoint"].controls["film_gamma"]
                gamma.set_value(2.0)
                assert (
                    w.editing_recipe()["engine_controls"]["flex-endpoint"]["values"][
                        "film_gamma"
                    ]
                    == 2.0
                )
                gamma.set_value(2.4)
                w.editor.loading = True
                for pages in w.editor.post_pages.values():
                    for page in pages:
                        for key, value in {
                            "exposure.wb_cyan": 0.04,
                            "exposure.shoulder": 0.12,
                            "lab.saturation": 0.92,
                            "finish.vignette_stops": 0.15,
                            "geometry.flip_horizontal": True,
                        }.items():
                            if key in page.controls:
                                widget = page.controls[key]
                                widget.setChecked(value) if isinstance(
                                    value, bool
                                ) else widget.set_value(value)
                w.editor.loading = False
                w.editor.edited()
                state["frozen_recipe"] = w.editing_recipe()
                preview_mode(0)
            elif stage == "preview":
                if (
                    not w.editor.note.text().startswith("预览已更新")
                    or w.canvas.sample_scale != 1
                    or w.preview_timer.isActive()
                ):
                    return
                index = holder["index"]
                mode = modes[index]
                images[mode] = image_array(w.canvas.current)
                w.grab().save(str(out / (mode + "-post.png")))
                if index + 1 < len(modes):
                    preview_mode(index + 1)
                else:
                    assert w.editing_recipe() == state["frozen_recipe"]
                    w.save_edit()
                    holder["saved_id"] = w.workspace.data["jobs"][-1]["id"]
                    state["stage"] = "save"
            elif stage == "save":
                job = w.workspace.job(holder["saved_id"])
                if job["state"] == "failed":
                    raise RuntimeError(job.get("error"))
                if job["state"] != "complete" or w.tasks.busy:
                    return
                sample = w.workspace.sample(job)
                manifest = json.loads((sample.directory / "manifest.json").read_text())
                for mode in modes:
                    record = sample.record(mode, "baseline")
                    displayed = image_array(display_image(record, w.display_icc))
                    assert np.array_equal(images[mode], displayed), (
                        mode + " preview/export mismatch"
                    )
                    pixels, icc = verified_pixels(record)
                    assert list(pixels.shape) == old["results"][mode]["shape"]
                    assert manifest["results"][mode]["icc_sha256"] == old["icc_sha256"]
                    assert not np.array_equal(
                        pixels, verified_pixels(old["results"][mode])[0]
                    )
                    if mode != "native":
                        assert (
                            manifest["results"][mode]["fit"]
                            == old["results"][mode]["fit"]
                        )
                assert manifest["results"]["flex-endpoint"]["profile_inspection"][
                    "resource"
                ]["sha256"] == sha256(profile)
                state["checks"].append(
                    "Three full-size on-screen post previews exactly match saved RGB16 display transforms; ICC and frozen fits unchanged"
                )
                state["saved_id"] = job["id"]
                # Direct export of another unsaved shared-post draft.
                w.editor.output_pages[w.mode].controls["exposure_ev"].set_value(0.2)
                target = out / "unsaved-post-export.tiff"
                before = len(w.workspace.data["jobs"])
                w.begin_exports([w.export_item(sample, target)])
                holder.update(
                    before=before,
                    target=target,
                    export_request=dict(w.export_tasks.request),
                )
                state["stage"] = "export"
            elif stage == "export":
                if w.exporting:
                    return
                request = holder["export_request"]
                receipt = json.loads(
                    (Path(request["directory"]) / "export.json").read_text()
                )
                assert (
                    receipt["state"] == "complete"
                    and receipt["saved"][0]["rendered_draft"]
                )
                assert len(w.workspace.data["jobs"]) == holder["before"]
                exported = Path(request["directory"]) / "renders/0/manifest.json"
                manifest = json.loads(exported.read_text())
                expected = verified_pixels(manifest["results"][w.mode])
                from .study import read_rgb

                pixels, icc = read_rgb(holder["target"])
                assert np.array_equal(pixels, expected[0]) and icc == expected[1]
                assert all(sha256(r["path"]) == r["sha256"] for r in old["inputs"])
                state["checks"].append(
                    "Unsaved shared post + profile draft exported directly with exact pixels/ICC, no new workspace job; source RAW unchanged"
                )
                state["log_path"] = str(w.log_path)
                w.grab().save(str(out / "final.png"))
                finish()
        except Exception as error:
            finish(error)

    timer.timeout.connect(tick)
    timer.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
