"""Re-render all eleven original triplets after interface/memory changes."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .cool_whitepoint_study import lift_codes
from .native import sha256
from .preview_data import ROOT, verified_pixels
from .workspace import write_json


def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cases = json.loads(
        (ROOT / "outputs/primary-range-3f8584ef4c8c/index.json").read_text()
    )["cases"]
    report = {
        "state": "running",
        "cases": [],
        "scope": "All original rgb2 default routes; historical cool plus explicitly selected soft-shoulder",
    }
    write_json(root / "verification.json", report)
    try:
        for case in cases:
            baseline = json.loads(
                (Path(case["baseline"]) / "manifest.json").read_text()
            )
            recipe = {
                "raw": [r["path"] for r in baseline["inputs"]],
                "roi": baseline["picture_region"]["analysis_roi"],
                "density": "negpy",
                "paper": "neutral",
                "crop": True,
                "flatfield": None,
                "exposure_ev": 0.0,
                "cool_tone": "soft-shoulder",
            }
            request = {
                "action": "render",
                "id": case["id"],
                "recipe": recipe,
                "inputs": baseline["inputs"],
                "calibration": None,
                "directory": str(root / case["id"]),
            }
            path = root / (case["id"] + ".request.json")
            write_json(path, request)
            with (root / (case["id"] + ".log")).open("w") as stream:
                completed = subprocess.run(
                    [sys.executable, "-m", "pure_rgb.workflow_worker", str(path)],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            if completed.returncode:
                raise RuntimeError(case["id"] + " worker failed")
            result = json.loads((root / case["id"] / "manifest.json").read_text())
            actuals = {}
            for mode, record in baseline["results"].items():
                expected, icc = verified_pixels(record)
                if mode == "cool-shared-rgb":
                    expected = lift_codes(expected, soft=True)
                actual, other = verified_pixels(result["results"][mode])
                actuals[mode] = {
                    "pixels_exact": bool(np.array_equal(actual, expected)),
                    "icc_exact": icc == other,
                    "shape": list(actual.shape),
                }
                if not actuals[mode]["pixels_exact"] or icc != other:
                    raise AssertionError(case["id"] + " " + mode)
            assert all(sha256(r["path"]) == r["sha256"] for r in baseline["inputs"])
            report["cases"].append(
                {
                    "id": case["id"],
                    "state": "complete",
                    "results": actuals,
                    "performance": json.loads(
                        (root / case["id"] / "performance.json").read_text()
                    ),
                }
            )
            write_json(root / "verification.json", report)
            print(case["id"] + " exact all three", flush=True)
        report.update(
            state="complete",
            case_count=len(cases),
            tiff_count=len(cases) * 3,
            raw_count=len(cases) * 3,
        )
    except Exception as error:
        report.update(state="failed", error=str(error))
        raise
    finally:
        write_json(root / "verification.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
