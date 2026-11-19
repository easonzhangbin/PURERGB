"""Isolated process for cancellable rendering and installed PURE Neg commands."""

import json
import os
from pathlib import Path
import sys

from .workspace import write_json
from .diagnostics import setup, event, failure, span, MemorySampler


def emit(**event):
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    setup("worker")
    if (
        os.name == "posix"
        and not os.environ.get("PURE_RGB_INHERIT_PROCESS_GROUP")
        and os.getpgrp() != os.getpid()
    ):
        os.setsid()
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text())
    event(
        "request_start",
        action=request.get("action"),
        request_path=str(request_path),
        job=request.get("id"),
    )
    os.environ["NEGPY_USER_DIR"] = str(request_path.parent / "worker-state")
    try:
        action = request["action"]
        if action == "render":
            from .workflow_render import render

            with (
                MemorySampler() as sample,
                span("raw_three_mode_render", request_path=str(request_path)),
            ):
                render(request)
            write_json(Path(request["directory"]) / "performance.json", sample.report())
            event("render_resources", **sample.report())
        elif action == "probe":
            from negpy.features.rgbscan.logic import place_triplet
            from .raw_import import probe_for_policy, IMPORT_POLICIES

            policy = request.get("raw_import", "legacy-v1")
            if policy not in IMPORT_POLICIES:
                raise ValueError("未知 RAW 导入策略")
            probe = probe_for_policy(policy)

            mapping = {
                str(Path(p).resolve()): probe(str(Path(p).resolve()))
                for p in request["raw"]
            }
            triple, reason = place_triplet(mapping)
            if triple is None:
                raise ValueError("三色核对失败：" + str(reason))
            emit(type="result", raw=[triple.red, triple.green, triple.blue])
        elif action == "flatfield_import":
            from .flatfield import prepare_calibration

            _, report = prepare_calibration(
                request["calibration"], request["directory"], request.get("targets", [])
            )
            if request.get("freeze_gain"):
                import shutil
                from .native import sha256

                directory = Path(request["directory"])
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / "flatfield-gain.npz"
                if Path(report["gain_path"]).resolve() != target.resolve():
                    with Path(report["gain_path"]).open("rb") as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    if sha256(target) != report["gain_sha256"]:
                        raise ValueError("平场增益复制期间发生变化")
                    report["gain_path"] = str(target)
                write_json(directory / "calibration.json", report)
            emit(type="result", calibration=report)
        elif action == "geometry_prepare":
            from .live_worker import Session
            from .native import sha256
            from .workflow_geometry_preview import write_geometry_previews

            directory = Path(request["directory"])
            directory.mkdir(parents=True, exist_ok=False)
            session = Session(
                request["manifest"], request["manifest_sha256"], directory
            )
            session.prepare_raw()
            geometry = write_geometry_previews(session.source, directory)
            if any(
                sha256(r["path"]) != r["sha256"] for r in session.manifest["inputs"]
            ):
                raise ValueError("生成定位图期间原 RAW 发生变化")
            emit(type="result", geometry=geometry)
        elif action == "export_batch":
            from .workflow_exports import export_batch

            report = export_batch(request, lambda event: emit(**event))
            emit(type="result", report=report)
        else:
            from .repair_bridge import prepare, redetect, revise

            if action == "repair_prepare":
                receipt = prepare(request, lambda event: emit(**event))
            elif action == "repair_redetect":
                receipt = redetect(request, lambda event: emit(**event))
            else:
                receipt = revise(
                    request,
                    lambda event: emit(**event),
                    confirm=action == "repair_confirm",
                )
            emit(type="result", receipt=receipt)
    except Exception as error:
        from .raw_registration import RegistrationModelCheckError

        diagnostic = (
            {"registration_check_diagnostic": error.diagnostic}
            if isinstance(error, RegistrationModelCheckError)
            else {}
        )
        failure(
            "request_error",
            error,
            action=request.get("action"),
            request_path=str(request_path),
        )
        write_json(
            request_path.with_suffix(".failure.json"), {"error": str(error), **diagnostic}
        )
        emit(type="error", message=str(error), **diagnostic)
        return 1
    event(
        "request_complete", action=request.get("action"), request_path=str(request_path)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
