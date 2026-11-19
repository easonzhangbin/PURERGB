"""Real-widget smoke exercise, usable offscreen or on the desktop."""

import hashlib
import json
import time

from PyQt6.QtCore import QTimer

from .preview_data import copy_verified_tiff


def exercise(app, window, output):
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + 120
    stages = iter(
        [
            ("native", "baseline"),
            ("cool-shared-rgb", "baseline"),
            ("flex-endpoint", "baseline"),
            ("flex-endpoint", "highlight-only"),
            ("flex-endpoint", "shadow-only"),
            ("flex-endpoint", "both"),
            ("cool-shared-rgb", "baseline"),
        ]
    )
    state = {"state": "running", "checks": [], "waiting": False}
    timer = QTimer(window)
    timer.setInterval(80)

    def finish(error=None):
        timer.stop()
        state.pop("origin", None)
        state["state"] = "failed" if error else "complete"
        if error:
            state["error"] = str(error)
        (output / "verification.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(state, ensure_ascii=False), flush=True)
        window.close()
        app.exit(1 if error else 0)

    def step():
        try:
            if time.monotonic() > deadline:
                raise TimeoutError("UI smoke exceeded 120 seconds")
            if window.last_error:
                raise RuntimeError(window.last_error)
            if window.loaded_key is None:
                return
            expected = (window.index, window.mode, window.probe)
            assert window.loaded_key == expected
            assert window.canvas.current.size() == window.canvas.reference.size()
            if state["waiting"]:
                assert window.canvas.zoom == state["zoom"]
                assert window.canvas.origin == state["origin"]
                state["checks"].append(
                    f"loaded {window.mode}/{window.probe}, view preserved"
                )
                state["waiting"] = False
            try:
                mode, probe = next(stages)
            except StopIteration:
                window.compare.setChecked(True)
                window.toggle_split()
                window.split_slider.setValue(43)
                assert window.canvas.split and window.canvas.fraction == 0.43
                window.canvas.fit()
                window.grab().save(str(output / "editor.png"))
                window.toggle_board()
                assert window.pages.currentIndex() == 1
                window.grab().save(str(output / "light-table.png"))
                window.open_sample(2)
                assert window.pages.currentIndex() == 0 and window.index == 2
                # Queue multiple selections; only the final request may be accepted.
                window.choose_mode("native")
                window.choose_mode("flex-endpoint")
                window.open_sample(min(6, len(window.samples) - 1))
                window.choose_mode("cool-shared-rgb")
                timer.timeout.disconnect()
                timer.timeout.connect(final_check)
                return
            window.canvas.actual()
            assert (
                abs(window.canvas.zoom * window.canvas.devicePixelRatioF() - 1) < 1e-9
            )
            state["zoom"], state["origin"] = window.canvas.zoom, window.canvas.origin
            window.choose_mode(mode)
            window.range_box.setCurrentIndex(window.range_box.findData(probe))
            state["waiting"] = True
        except Exception as error:
            finish(error)

    def final_check():
        try:
            if time.monotonic() > deadline:
                raise TimeoutError("Stale selection did not settle")
            if window.last_error:
                raise RuntimeError(window.last_error)
            if window.loaded_key is None:
                return
            assert window.loaded_key == (
                min(6, len(window.samples) - 1),
                "cool-shared-rgb",
                "baseline",
            )
            state["checks"].append(
                "rapid selection settled on final requested sample/mode"
            )
            rec = window.samples[window.index].record(window.mode, window.probe)
            exported = copy_verified_tiff(rec, output / "export-copy.tiff")
            with open(exported, "rb") as stream:
                assert (
                    hashlib.file_digest(stream, "sha256").hexdigest() == rec["sha256"]
                )
            state["checks"].append("real RGB16 TIFF export byte/ICC exact")
            window.canvas.fit()
            window.grab().save(str(output / "editor.png"))
            del state["zoom"], state["origin"]
            state["sample_count"] = len(window.samples)
            state["display"] = window.display_name
            finish()
        except Exception as error:
            state.pop("origin", None)
            finish(error)

    timer.timeout.connect(step)
    timer.start()
