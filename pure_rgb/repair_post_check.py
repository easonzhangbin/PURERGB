"""Isolated three-mode shared-post TIFF -> PURE Neg -> verified RGB16/ICC."""

import json
from pathlib import Path
import sys

from .diagnostics import setup, event, ApplicationMemoryMonitor
from .native import sha256
from .repair_bridge import prepare, revise, backend_identity
from .workspace import write_json


def main():
    source = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(source.read_text())
    setup("repair-check")
    monitor = ApplicationMemoryMonitor()
    report = {
        "state": "running",
        "cases": {},
        "scope": "Automated local format/mask regression only; isolated outputs, not user artistic approval",
    }
    initial = backend_identity()
    try:
        for mode, record in manifest["results"].items():
            case = out / mode
            receipt = prepare(
                {
                    "source": record,
                    "revision": record["sha256"],
                    "directory": str(case / "review"),
                },
                lambda data: event("repair_progress", mode=mode, **data),
            )
            assert (
                receipt["state"] == "ready"
                and receipt["preview"]["outside_mask_changed_pixels"] == 0
            )
            request = {
                "receipt": str(case / "review/receipt.json"),
                "revision": record["sha256"],
                "edits": {"excludedRegions": [], "strokes": []},
            }
            preview = revise(
                request, lambda data: event("repair_progress", mode=mode, **data)
            )
            assert (
                preview["state"] == "previewed"
                and preview["preview"]["outside_mask_changed_pixels"] == 0
            )
            final = case / "final"
            final.mkdir()
            confirmed = revise(
                {**request, "output": str(final)},
                lambda data: event("repair_progress", mode=mode, **data),
                confirm=True,
            )
            assert (
                confirmed["state"] == "confirmed"
                and confirmed["preview"]["outside_mask_changed_pixels"] == 0
            )
            assert sha256(record["path"]) == record["sha256"]
            report["cases"][mode] = {
                "state": "complete",
                "preview": preview["preview"],
                "confirmed": confirmed["preview"],
                "receipt": request["receipt"],
                "output": confirmed["published"]["paths"],
            }
            write_json(out / "verification.json", report)
            print(mode + " shared post -> PURE Neg exact outside-mask/ICC", flush=True)
        assert backend_identity()["fingerprint"] == initial["fingerprint"]
        report.update(
            state="complete", backend_unchanged=True, source_tiffs_unchanged=True
        )
    except Exception as error:
        report.update(state="failed", error=str(error))
        raise
    finally:
        monitor.stop()
        report["sampled_peak_tree_rss_bytes"] = monitor.peak
        write_json(out / "verification.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
