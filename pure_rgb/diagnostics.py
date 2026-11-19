"""Local bounded JSONL diagnostics and dependency-free process memory sampling."""

from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback
import subprocess
import re

from . import WORKSPACE_ROOT

_LOGGER = logging.getLogger("pure_rgb.diagnostics")
_PATH = None
_DEBUG_EVENTS = {"application_memory", "preview_submit", "preview_dispatch", "preview_stderr", "worker_stderr"}


def set_debug_logging(enabled):
    os.environ["PURE_RGB_DEBUG"] = "1" if enabled else "0"
    _LOGGER.setLevel(logging.DEBUG if enabled else logging.INFO)


class _MachBasic(ctypes.Structure):
    _fields_ = [
        ("virtual", ctypes.c_uint64),
        ("resident", ctypes.c_uint64),
        ("max_resident", ctypes.c_uint64),
        ("user_s", ctypes.c_int32),
        ("user_us", ctypes.c_int32),
        ("system_s", ctypes.c_int32),
        ("system_us", ctypes.c_int32),
        ("policy", ctypes.c_int32),
        ("suspended", ctypes.c_int32),
    ]


_MACH = ctypes.CDLL("/usr/lib/libSystem.B.dylib") if sys.platform == "darwin" else None


def prune_closed_logs(directory, *, limit_bytes=128 * 1024 * 1024, days=14):
    """Retain current processes; cap only our own closed-process log history."""
    candidates = []
    for path in Path(directory).glob("*.jsonl*"):
        match = re.fullmatch(r"[a-z-]+-(\d+)\.jsonl(?:\.[12])?", path.name)
        if not match:
            continue
        try:
            os.kill(int(match[1]), 0)
        except ProcessLookupError:
            stat = path.stat()
            candidates.append((stat.st_mtime, stat.st_size, path))
        except PermissionError:
            pass
    total = sum(size for _, size, _ in candidates)
    cutoff = time.time() - days * 86400
    for stamp, size, path in sorted(candidates):
        if stamp < cutoff or total > limit_bytes:
            path.unlink(missing_ok=True)
            total -= size


class _JSON(logging.Formatter):
    def format(self, record):
        return json.dumps(
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
                "thread": threading.current_thread().name,
                **record.msg,
            },
            ensure_ascii=False,
            default=str,
            allow_nan=False,
        )


def setup(role, directory=None):
    global _PATH
    if _PATH is not None:
        return _PATH
    directory = Path(directory or WORKSPACE_ROOT / "outputs/logs")
    directory.mkdir(parents=True, exist_ok=True)
    try:
        prune_closed_logs(directory)
    except OSError:
        pass
    _PATH = directory / f"{role}-{os.getpid()}.jsonl"
    handler = RotatingFileHandler(
        _PATH, maxBytes=5_000_000, backupCount=2, encoding="utf-8"
    )
    handler.setFormatter(_JSON())
    _LOGGER.addHandler(handler)
    set_debug_logging(os.environ.get("PURE_RGB_DEBUG") == "1")
    _LOGGER.propagate = False
    original = sys.excepthook

    def uncaught(kind, value, tb):
        event(
            "uncaught_exception",
            error=str(value),
            traceback="".join(traceback.format_exception(kind, value, tb)),
        )
        original(kind, value, tb)

    sys.excepthook = uncaught
    event(
        "process_start",
        role=role,
        python=sys.version.split()[0],
        platform=sys.platform,
        logical_cpu_count=os.cpu_count(),
    )
    return _PATH


def memory():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = int(peak if sys.platform == "darwin" else peak * 1024)
    rss = None
    try:
        if sys.platform == "darwin":
            info, count = _MachBasic(), ctypes.c_uint32(ctypes.sizeof(_MachBasic) // 4)
            if (
                _MACH.task_info(
                    _MACH.mach_task_self(),
                    20,
                    ctypes.byref(info),
                    ctypes.byref(count),
                )
                == 0
            ):
                rss = int(info.resident)
        elif Path("/proc/self/statm").exists():
            rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf(
                "SC_PAGE_SIZE"
            )
    except (OSError, ValueError, AttributeError):
        pass
    return {"rss_bytes": rss, "peak_rss_bytes": max(peak, rss or 0)}


def event(name, **fields):
    level = logging.DEBUG if name in _DEBUG_EVENTS else logging.INFO
    if _PATH is not None and _LOGGER.isEnabledFor(level):
        _LOGGER.log(level, {"event": name, **memory(), **fields})


def failure(name, error, **fields):
    event(name, error=str(error), traceback=traceback.format_exc(), **fields)


@contextmanager
def span(name, **fields):
    started = time.perf_counter()
    event("stage_start", stage=name, **fields)
    try:
        yield
    except BaseException as error:
        event(
            "stage_error",
            stage=name,
            seconds=time.perf_counter() - started,
            error=str(error),
            traceback=traceback.format_exc(),
            **fields,
        )
        raise
    else:
        event(
            "stage_complete",
            stage=name,
            seconds=time.perf_counter() - started,
            **fields,
        )


class MemorySampler:
    def __init__(self, interval=0.1):
        self.interval, self.samples, self.peak = interval, 0, 0
        self.stop_event = threading.Event()

    def __enter__(self):
        self.started = time.perf_counter()
        self.cpu_start = resource.getrusage(resource.RUSAGE_SELF)
        self.thread = threading.Thread(
            target=self.run, daemon=True, name="memory-sampler"
        )
        self.thread.start()
        return self

    def run(self):
        while not self.stop_event.is_set():
            value = memory()["rss_bytes"]
            self.peak = max(self.peak, value or 0)
            self.samples += 1
            self.stop_event.wait(self.interval)

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self.seconds = time.perf_counter() - self.started
        now = resource.getrusage(resource.RUSAGE_SELF)
        self.cpu_seconds = (
            now.ru_utime
            + now.ru_stime
            - self.cpu_start.ru_utime
            - self.cpu_start.ru_stime
        )
        self.final_memory = memory()
        self.peak = max(self.peak, self.final_memory["rss_bytes"] or 0)

    def report(self):
        return {
            "wall_seconds": self.seconds,
            "cpu_seconds": self.cpu_seconds,
            "cpu_percent_one_core_100": 100
            * self.cpu_seconds
            / max(self.seconds, 1e-9),
            "sampled_peak_rss_bytes": self.peak,
            "samples": self.samples,
            **self.final_memory,
            "memory_scope": "This process RSS, not a separate VRAM meter or whole application tree",
        }


def process_tree_memory(root_pid):
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True, timeout=2
    )
    result.check_returncode()
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3:
            pid, parent, rss = map(int, fields)
            rows[pid] = (parent, rss * 1024)
    selected = {root_pid}
    for _ in range(16):
        added = {pid for pid, (parent, rss) in rows.items() if parent in selected}
        if added <= selected:
            break
        selected |= added
    return {
        "processes": [
            {"pid": pid, "rss_bytes": rows[pid][1]}
            for pid in sorted(selected)
            if pid in rows
        ],
        "tree_rss_bytes": sum(rows[pid][1] for pid in selected if pid in rows),
    }


class ApplicationMemoryMonitor:
    def __init__(self):
        self.stop_event = threading.Event()
        self.peak = 0
        self.thread = threading.Thread(
            target=self.run, daemon=True, name="application-memory"
        )
        self.thread.start()

    def run(self):
        while not self.stop_event.is_set():
            try:
                tree = process_tree_memory(os.getpid())
                self.peak = max(self.peak, tree["tree_rss_bytes"])
                event(
                    "application_memory", **tree, sampled_peak_tree_rss_bytes=self.peak
                )
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop_event.wait(2)

    def stop(self):
        self.stop_event.set()
        self.thread.join(2.5)
        event("application_memory_summary", sampled_peak_tree_rss_bytes=self.peak)
