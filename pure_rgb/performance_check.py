"""Isolated real CPU/Metal exports, path counters, RSS, and RGB16 parity."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import gc
from dataclasses import replace

import numpy as np
import tifffile

from .diagnostics import setup, span, MemorySampler
from .native import native_export, sha256
from .cli import inspect_tiff
from .workspace import write_json


def case(args):
    setup("benchmark")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["NEGPY_USER_DIR"] = str(args.output / "native-state")
    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.image_processor import ImageProcessor

    manifest = json.loads(args.manifest.read_text())
    for source in manifest["inputs"]:
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError("性能测试源 RAW 已变化")
    config = WorkspaceConfig.from_flat_dict(
        manifest.get("native_reference_config", manifest["native_config"])
    )
    if args.variant in ("manual-neutral", "no-auto", "no-auto-details"):
        config = replace(
            config,
            exposure=replace(
                config.exposure,
                auto_exposure=False,
                auto_normalize_contrast=False,
            ),
        )
    if args.variant in ("manual-neutral", "no-cast"):
        config = replace(
            config, exposure=replace(config.exposure, cast_removal_strength=0.0)
        )
    if args.variant in ("manual-neutral", "no-details", "no-auto-details"):
        config = replace(
            config,
            lab=replace(config.lab, sharpen=0.0, chroma_denoise=0.0),
        )
    if config.flatfield.apply:
        raise ValueError("首轮 CPU/GPU 基准请选择不含平场的冻结版本")
    report = {
        "state": "running",
        "backend_requested": args.backend,
        "runs": [],
        "path_calls": {"cpu": 0, "gpu": 0},
        "source_inputs": manifest["inputs"],
        "variant": args.variant,
    }
    processor = None
    try:
        with (
            MemorySampler() as init,
            span("backend_initialization", backend=args.backend),
        ):
            processor = ImageProcessor(use_gpu=args.backend == "gpu")
        report["initialization"] = init.report()
        if args.backend == "gpu":
            from negpy.infrastructure.gpu.device import GPUDevice

            device = GPUDevice.get()
            report["gpu"] = {
                "available": device.is_available,
                "summary": str(device.adapter.summary) if device.adapter else None,
                "backend": device.backend_name,
                "integrated": device.is_integrated,
            }
            if processor.engine_gpu is None:
                report.update(
                    state="unavailable",
                    reason="GPU was not initialized; no CPU fallback timed as GPU",
                )
                return report
        for kind, engine in (
            ("cpu", processor.engine_cpu),
            ("gpu", processor.engine_gpu),
        ):
            if engine is None:
                continue
            original = engine.process

            def counted(*a, _original=original, _kind=kind, **kw):
                report["path_calls"][_kind] += 1
                return _original(*a, **kw)

            engine.process = counted
        first = None
        for index in range(args.iterations):
            with (
                MemorySampler() as measurement,
                span("native_export", backend=args.backend, iteration=index),
            ):
                payload, _ = native_export(
                    processor,
                    config,
                    manifest["inputs"],
                    prefer_gpu=args.backend == "gpu",
                )
            pixels, icc = inspect_tiff(payload)
            if first is None:
                first = pixels
                (args.output / "native.tiff").write_bytes(payload)
            repeat_equal = np.array_equal(first, pixels)
            report["runs"].append(
                {
                    "iteration": index,
                    "cache": "cold-source" if index == 0 else "warm-source",
                    "exact_repeat": repeat_equal,
                    "shape": list(pixels.shape),
                    "icc_sha256": __import__("hashlib").sha256(icc).hexdigest(),
                    **measurement.report(),
                }
            )
            del payload, pixels
            gc.collect()
            from .diagnostics import memory

            report["runs"][-1]["rss_after_iteration_bytes"] = memory()["rss_bytes"]
        if args.backend == "gpu" and not report["path_calls"]["gpu"]:
            raise ValueError("未实际执行 GPU，不能报告 GPU 加速")
        report["state"] = "complete"
    except Exception as error:
        report.update(state="failed", error=str(error))
    finally:
        if processor:
            processor.cleanup()
            processor.engine_cpu.cache.clear()
            processor.destroy_all()
            gc.collect()
            from .diagnostics import memory

            report["after_cleanup_and_destroy"] = memory()
        report["raw_hashes_unchanged"] = all(
            sha256(r["path"]) == r["sha256"] for r in manifest["inputs"]
        )
        write_json(args.output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "gpu"))
    parser.add_argument(
        "--variant",
        choices=(
            "baseline",
            "manual-neutral",
            "no-auto",
            "no-cast",
            "no-details",
            "no-auto-details",
        ),
        default="baseline",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "outputs/PURE-RGB-Workspace-v01/runs/69c0dfb190a744e6/manifest.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.iterations <= 30:
        parser.error("iterations must be between 2 and 30")
    args.output, args.manifest = args.output.resolve(), args.manifest.resolve()
    if args.backend:
        report = case(args)
        print(
            json.dumps(
                {
                    "state": report["state"],
                    "backend": args.backend,
                    "path_calls": report["path_calls"],
                    "error": report.get("error"),
                }
            )
        )
        return 1 if report["state"] == "failed" else 0
    setup("benchmark-controller")
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for backend in ("cpu", "gpu"):
        with (args.output / f"{backend}.log").open("w") as log:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pure_rgb.performance_check",
                    "--backend",
                    backend,
                    "--variant",
                    args.variant,
                    "--iterations",
                    str(args.iterations),
                    "--manifest",
                    str(args.manifest),
                    "--output",
                    str(args.output / backend),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        path = args.output / backend / "report.json"
        reports[backend] = (
            json.loads(path.read_text())
            if path.exists()
            else {"state": "failed", "exit_code": completed.returncode}
        )
        print(
            json.dumps({"backend": backend, "state": reports[backend]["state"]}),
            flush=True,
        )
    summary = {
        "state": "complete",
        "cpu": reports["cpu"],
        "gpu": reports["gpu"],
        "logical_cpu_count": os.cpu_count(),
        "variant": args.variant,
        "iterations": args.iterations,
        "application_default": "CPU unchanged; GPU timing is an isolated experiment",
        "experimental_routes": "cool/Flex NumPy inversion, fitting and CML4 remain CPU; no GPU adapter claimed",
        "memory_scope": "Per process RSS. Unified GPU allocations cannot be separated as dedicated VRAM here.",
    }
    if all(r["state"] == "complete" for r in reports.values()):
        cpu, gpu = [
            tifffile.imread(args.output / b / "native.tiff") for b in ("cpu", "gpu")
        ]
        delta = np.abs(cpu.astype(np.int32) - gpu.astype(np.int32))
        summary["parity"] = {
            "exact": bool(np.array_equal(cpu, gpu)),
            "max_abs_u16": int(delta.max()),
            "mean_abs_u16": float(delta.mean()),
            "p99_abs_u16": float(np.quantile(delta, 0.99)),
            "different_channel_fraction": float(np.mean(delta != 0)),
            "icc_equal": reports["cpu"]["runs"][0]["icc_sha256"]
            == reports["gpu"]["runs"][0]["icc_sha256"],
        }
        c, g = [
            np.median([r["wall_seconds"] for r in reports[b]["runs"][1:]])
            for b in ("cpu", "gpu")
        ]
        summary["native_warm_export_speedup"] = float(c / g)
    else:
        summary["state"] = "partial"
    write_json(args.output / "verification.json", summary)
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k in ("state", "parity", "native_warm_export_speedup")
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
