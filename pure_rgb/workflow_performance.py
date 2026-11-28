"""Runtime resource policy for isolated workers; it never changes image recipes.

Thread environment must be installed before spawning a worker.  Cache budgets
are soft retention budgets for an idle preview process. Verified metering and
saved rendering use the CPU. Eligible interactive previews use Metal.
"""

from copy import deepcopy
import ctypes
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


MIB = 1024**2
GIB = 1024**3
DEFAULT_SETTINGS = dict(
    version=1,
    mode="balanced",
    backend="auto",
    cpu_threads=0,
    cache_mb=0,
    debug_logging=False,
    idle_previews=True,
)
MODE_LABELS = {"balanced": "均衡", "high": "高性能"}
THREAD_VARIABLES = (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
BLAS_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def checked_performance_settings(document=None):
    if document is None:
        return deepcopy(DEFAULT_SETTINGS)
    if type(document) is not dict or set(document) - DEFAULT_SETTINGS.keys():
        raise ValueError("未知性能设置")
    result = {**DEFAULT_SETTINGS, **document}
    if type(result["version"]) is not int or result["version"] != 1:
        raise ValueError("不支持的性能设置版本")
    if result["mode"] not in MODE_LABELS:
        raise ValueError("请选择均衡或高性能模式")
    if result["backend"] not in {"auto", "cpu", "gpu"}:
        raise ValueError("请选择自动、CPU或GPU预览")
    for name, low, high in (("cpu_threads", 0, 256), ("cache_mb", 0, 65536)):
        value = result[name]
        if type(value) is not int or not low <= value <= high:
            raise ValueError("性能参数超出范围：" + name)
    if 0 < result["cache_mb"] < 256:
        raise ValueError("预览缓存预算至少256MB，或设为自动")
    if any(type(result[key]) is not bool for key in ("debug_logging", "idle_previews")):
        raise ValueError("调试日志与闲置预览开关必须为布尔值")
    return result


def load_performance_settings(path):
    path = Path(path)
    return (
        checked_performance_settings(json.loads(path.read_text()))
        if path.exists()
        else checked_performance_settings()
    )


def save_performance_settings(path, document):
    settings = checked_performance_settings(document)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, encoding="utf-8", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(settings, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return settings


class _MacProcessTaskInfo(ctypes.Structure):
    # PROC_PIDTASKINFO returns struct proc_taskinfo. Only the first two fields
    # are needed here; its remaining 80 bytes contain timing and counters.
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("remaining", ctypes.c_ubyte * 80),
    ]


@lru_cache(maxsize=1)
def _mac_system_functions():
    """Load macOS process and hardware queries once per application process."""
    try:
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        system_info = library.sysctlbyname
        system_info.argtypes = [
            ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p, ctypes.c_size_t,
        ]
        system_info.restype = ctypes.c_int
        process_info = library.proc_pidinfo
        process_info.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int,
        ]
        process_info.restype = ctypes.c_int
        return system_info, process_info
    except (OSError, AttributeError):
        return None


def _mac_physical_memory():
    functions = _mac_system_functions()
    if functions is None:
        return None
    value = ctypes.c_uint64()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if functions[0](b"hw.memsize", ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
        return None
    return int(value.value) if size.value == ctypes.sizeof(value) and value.value else None


def system_memory_pressure():
    """Read macOS's userspace pressure level without running a command."""
    if sys.platform != "darwin":
        return None
    functions = _mac_system_functions()
    if functions is None:
        return None
    value = ctypes.c_uint32()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    status = functions[0](
        b"kern.memorystatus_vm_pressure_level", ctypes.byref(value), ctypes.byref(size), None, 0
    )
    if status != 0 or size.value != ctypes.sizeof(value):
        return None
    # XNU translates its internal level into the userspace event flags.
    return {1: "normal", 2: "warning", 4: "critical"}.get(value.value)


def _mac_process_rss(pid):
    functions = _mac_system_functions()
    if functions is None or pid > 2**31 - 1:
        return None
    info = _MacProcessTaskInfo()
    size = ctypes.sizeof(info)
    # PROC_PIDTASKINFO is 4 in <sys/proc_info.h>.
    if functions[1](pid, 4, 0, ctypes.byref(info), size) != size:
        return None
    return int(info.resident_size)


def hardware_info():
    cores = max(1, os.cpu_count() or 1)
    memory = None
    try:
        if sys.platform == "darwin":
            memory = _mac_physical_memory()
            if memory is None:
                memory = int(
                    subprocess.check_output(
                        ["/usr/sbin/sysctl", "-n", "hw.memsize"], timeout=2, text=True
                    ).strip()
                )
        else:
            memory = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return dict(cpu_count=cores, total_memory_bytes=memory)


def backend_status(document=None):
    settings = checked_performance_settings(document)
    try:
        from .gpu_positive_preview import gpu_info

        gpu = gpu_info()
    except Exception as error:
        gpu = dict(available=False, reason=str(error))
    available = bool(gpu.get("available"))
    return dict(
        requested=settings["backend"],
        effective="cpu" if settings["backend"] == "cpu" or not available else "hybrid-preview",
        gpu_available=available,
        label="CPU + GPU 预览" if available and settings["backend"] != "cpu" else "CPU",
        device=gpu.get("device", ""),
        gpu_unavailable_reason="" if available else gpu.get("reason", "当前设备无法使用GPU预览"),
        scope="Metal accelerates eligible NegPy interactive previews and positive adjustments; metering and saved rendering use verified CPU routes.",
    )


def resolve_performance(document=None, *, hardware=None, include_backend=True):
    settings = checked_performance_settings(document)
    hardware = hardware_info() if hardware is None else hardware
    cores = max(1, int(hardware.get("cpu_count") or 1))
    high = settings["mode"] == "high"
    automatic_threads = cores if high else max(1, min(6, cores // 2))
    threads = min(cores, settings["cpu_threads"] or automatic_threads)
    memory = hardware.get("total_memory_bytes") or 8 * GIB
    automatic_budget = min(memory // (4 if high else 8), (8 if high else 2) * GIB)
    budget = max(256 * MIB, settings["cache_mb"] * MIB or automatic_budget)
    return dict(
        settings=settings,
        mode_label=MODE_LABELS[settings["mode"]],
        cpu_threads=threads,
        cpu_count=cores,
        # Pixel kernels and RAW demosaic own large parallel passes.  The Python
        # matrix work is predominantly RGB triplets / small solves, so a second
        # all-core BLAS pool provides no measured benefit and can contend with
        # parallel work from another library or process.
        blas_threads=1,
        io_threads=threads,
        display_threads=min(8 if high else 4, threads),
        preview_cache_budget_bytes=int(budget),
        preview_idle_release_seconds=300 if high else 30,
        memory_pressure_idle_seconds=15 if high else 8,
        **({"backend": backend_status(settings)} if include_backend else {}),
        memory_budget_is_hard_limit=False,
    )


def worker_environment(document=None, *, hardware=None):
    """Only the explicitly owned variables, for a newly started worker."""
    policy = resolve_performance(document, hardware=hardware, include_backend=False)
    threads = str(policy["cpu_threads"])
    return {
        **{name: str(policy["blas_threads"]) for name in BLAS_VARIABLES},
        "RAYON_NUM_THREADS": threads,
        "OMP_NUM_THREADS": threads,
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "PURE_RGB_IO_THREADS": str(policy["io_threads"]),
        "PURE_RGB_DISPLAY_THREADS": str(policy["display_threads"]),
        "PURE_RGB_COMPUTE_BACKEND": policy["settings"]["backend"],
        "PURE_RGB_PREVIEW_CACHE_BYTES": str(policy["preview_cache_budget_bytes"]),
        "PURE_RGB_PERFORMANCE_MODE": policy["settings"]["mode"],
        "PURE_RGB_DEBUG": "1" if policy["settings"]["debug_logging"] else "0",
    }


def child_environment(document=None, *, hardware=None):
    return worker_environment(document, hardware=hardware)


def install_child_environment(document=None, *, hardware=None):
    """Set inheritance for subsequent QProcess starts, not an existing worker."""
    environment = worker_environment(document, hardware=hardware)
    os.environ.update(environment)
    return environment


def qprocess_environment(document=None, *, hardware=None, base=None):
    """Optionally apply directly at the QProcess creation site."""
    from PyQt6.QtCore import QProcessEnvironment

    environment = (
        QProcessEnvironment(base)
        if base is not None
        else QProcessEnvironment.systemEnvironment()
    )
    for name, value in worker_environment(document, hardware=hardware).items():
        environment.insert(name, value)
    return environment


def process_rss_bytes(pid):
    if type(pid) is not int or pid <= 0:
        return None
    try:
        if sys.platform == "darwin":
            rss = _mac_process_rss(pid)
            if rss is not None:
                return rss
        elif sys.platform.startswith("linux"):
            statm = Path(f"/proc/{pid}/statm").read_text().split()
            return max(0, int(statm[1])) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, OverflowError):
        pass
    try:
        result = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
            timeout=1,
            stderr=subprocess.DEVNULL,
        ).strip()
        return max(0, int(result)) * 1024 if result else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def preview_release_decision(
    document=None, *, idle_seconds, busy, rss_bytes=None, hardware=None,
    retained_source_bytes=0, memory_pressure=None,
):
    """Whether the GUI may release an idle preview process and its whole cache.

    The GUI owns shutdown.  No signals or resource limits are applied here;
    active previews, queued replacements and saved-output jobs must pass busy.
    """
    hardware = hardware_info() if hardware is None else hardware
    policy = resolve_performance(document, hardware=hardware, include_backend=False)
    if type(retained_source_bytes) is not int or retained_source_bytes < 0:
        raise ValueError("保留源缓存字节数必须为非负整数")
    effective_budget = policy["preview_cache_budget_bytes"] + retained_source_bytes
    if memory_pressure == "normal" and retained_source_bytes and not policy["settings"]["cache_mb"]:
        # A qualified RAW session also retains SDK and GPU state outside the
        # source array. On the measured 96 MP workload it occupies 6.22 GiB,
        # while the decoded source accounts for 1.07 GiB. Keep it warm through
        # short idle gaps within a bounded share of known physical memory.
        memory = hardware.get("total_memory_bytes") or 0
        effective_budget = max(effective_budget, min(memory // 8, 8 * GIB))
    release, reason = False, "retained"
    if busy:
        reason = "active-work"
    elif memory_pressure == "critical":
        release, reason = True, "system-memory-pressure"
    elif idle_seconds >= policy["preview_idle_release_seconds"]:
        release, reason = True, "idle-timeout"
    elif (
        rss_bytes is not None
        and rss_bytes > effective_budget
        and idle_seconds >= policy["memory_pressure_idle_seconds"]
    ):
        release, reason = True, "idle-cache-budget"
    return dict(
        release=release,
        reason=reason,
        budget_bytes=policy["preview_cache_budget_bytes"],
        retained_source_bytes=retained_source_bytes,
        effective_budget_bytes=effective_budget,
        memory_pressure=memory_pressure,
        idle_limit_seconds=policy["preview_idle_release_seconds"],
    )


def should_release_preview(document=None, **kwargs):
    return preview_release_decision(document, **kwargs)["release"]
