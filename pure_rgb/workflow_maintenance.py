"""Bounded maintenance of disposable display previews and portable diagnostics."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import time

from .workflow_performance import checked_performance_settings


_SESSION = re.compile(r"[0-9a-f]{32}\Z")
_PREVIEW = re.compile(r"[0-9]+-(?:(?:current|reference)\.tiff|current\.metering\.json)\Z")
_EVENTS = {
    "process_start", "preview_submit", "preview_dispatch", "preview_complete",
    "preview_discard_stale", "preview_shutdown", "task_start", "task_finished",
    "task_failed", "task_cancel_requested", "cache_cleanup", "settings_changed",
    "idle_preview_started", "idle_preview_completed", "idle_preview_cancelled",
    "preview_waiting", "preview_failed",
}
_NUMBERS = {
    "token", "code", "seconds", "count", "bytes", "rss_bytes", "peak_rss_bytes",
    "cancelled", "success", "replaces_pending", "active_token", "cleared_count",
    "cleared_bytes", "generation",
    "worker_pid", "worker_active", "queued", "display_pending",
}
_ENUMS = {
    "action": {
        "render", "probe", "geometry_prepare", "repair_prepare",
        "repair_preview", "repair_redetect", "repair_confirm", "export",
    },
    "mode": {"native", "cool-shared-rgb", "flex-endpoint", "balanced", "high"},
    "status": {"complete", "partial", "failed", "cancelled", "unavailable"},
}


def _age(keep_days):
    if isinstance(keep_days, bool) or not isinstance(keep_days, (int, float)):
        raise ValueError("缓存保留天数无效")
    if not math.isfinite(keep_days) or keep_days < 1:
        raise ValueError("缓存至少保留一天")
    return float(keep_days)


def _path(path):
    return Path(os.path.abspath(os.fspath(path)))


def _cache_root(project_root):
    project = _path(project_root)
    outputs, cache = project / "outputs", project / "outputs" / "live-preview"
    # Never traverse a redirected outputs/cache tree, even if its target is
    # otherwise inside the project. Only this exact disposable layout is owned.
    if project.is_symlink() or outputs.is_symlink() or cache.is_symlink():
        return None
    return cache


def _candidates(project_root, active_directories, keep_days):
    cutoff = time.time() - _age(keep_days) * 86400
    root = _cache_root(project_root)
    active = {_path(item) for item in active_directories}
    entries = []
    if root is None or not root.is_dir():
        return root, entries
    for directory in sorted(root.iterdir()):
        if directory.is_symlink() or not directory.is_dir() or not _SESSION.fullmatch(directory.name):
            continue
        protected = _path(directory) in active
        for path in sorted(directory.iterdir()):
            if path.is_symlink() or not _PREVIEW.fullmatch(path.name):
                continue
            stat = path.lstat()
            if not path.is_file():
                continue
            entries.append(dict(
                path=path, size=stat.st_size, active=protected,
                old=stat.st_mtime < cutoff,
                identity=(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns),
            ))
    return root, entries


def preview_cache_report(project_root, active_directories=(), keep_days=7):
    root, entries = _candidates(project_root, active_directories, keep_days)
    eligible = [entry for entry in entries if entry["old"] and not entry["active"]]
    return dict(
        state="available" if root is not None else "redirected-cache-refused",
        file_count=len(entries), total_bytes=sum(entry["size"] for entry in entries),
        reclaimable_count=len(eligible), reclaimable_bytes=sum(entry["size"] for entry in eligible),
        active_count=sum(entry["active"] for entry in entries), keep_days=_age(keep_days),
        scope="Disposable display previews only",
    )


def _trash_backend():
    try:
        from send2trash import send2trash

        return lambda path: send2trash(str(path))
    except ImportError:
        if sys.platform != "darwin":
            return None
        try:
            from Foundation import NSFileManager, NSURL
        except ImportError:
            return None

        def move(path):
            result = NSFileManager.defaultManager().trashItemAtURL_resultingItemURL_error_(
                NSURL.fileURLWithPath_(str(path)), None, None
            )
            success = result[0] if isinstance(result, tuple) else result
            if not success:
                raise OSError("系统未能将临时预览移至废纸篓")

        return move


def cleanup_preview_cache(project_root, active_directories=(), keep_days=7, *, trash=None):
    """Move only unchanged old preview packets to trash; never unlink as fallback."""
    root, entries = _candidates(project_root, active_directories, keep_days)
    report = preview_cache_report(project_root, active_directories, keep_days)
    report.update(state="complete", cleared_count=0, cleared_bytes=0, failed_count=0, skipped_changed_count=0)
    if root is None:
        report.update(state="unavailable", reason="缓存目录发生重定向，未清理")
        return report
    move = trash if trash is not None else _trash_backend()
    if move is None:
        report.update(state="unavailable", reason="系统废纸篓不可用，未删除任何缓存")
        return report
    for entry in entries:
        if not entry["old"] or entry["active"]:
            continue
        path = entry["path"]
        try:
            if (
                _cache_root(project_root) != root
                or root.is_symlink() or path.parent.is_symlink() or path.is_symlink()
                or not path.is_file()
            ):
                report["skipped_changed_count"] += 1
                continue
            stat = path.lstat()
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if identity != entry["identity"]:
                report["skipped_changed_count"] += 1
                continue
            move(path)
            report["cleared_count"] += 1
            report["cleared_bytes"] += entry["size"]
        except OSError:
            report["failed_count"] += 1
    if report["failed_count"] or report["skipped_changed_count"]:
        report["state"] = "partial"
    return report


def diagnostic_report(settings=None, events=(), hardware=None):
    """Whitelist operating metadata; no paths, pixels, raw logs, or ICC payloads."""
    clean_events = []
    for value in list(events)[-200:]:
        if not isinstance(value, dict) or value.get("event") not in _EVENTS:
            continue
        clean = dict(event=value["event"])
        for key in _NUMBERS:
            item = value.get(key)
            if type(item) in (bool, int) or (type(item) is float and math.isfinite(item)):
                clean[key] = item
        for key, allowed in _ENUMS.items():
            if isinstance(value.get(key), str) and value[key] in allowed:
                clean[key] = value[key]
        clean_events.append(clean)
    clean_hardware = {}
    for key in ("cpu_count", "total_memory_bytes"):
        value = (hardware or {}).get(key)
        if type(value) is int and value > 0:
            clean_hardware[key] = value
    return dict(
        format_version=1, application="PURE RGB",
        created_utc=datetime.now(timezone.utc).isoformat(),
        runtime=dict(python=platform.python_version(), platform=sys.platform),
        performance=checked_performance_settings(settings), hardware=clean_hardware,
        events=clean_events,
        content="Settings and operation results; excludes source paths and image data",
    )


def write_diagnostic_report(path, settings=None, events=(), hardware=None):
    report = diagnostic_report(settings, events, hardware)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report
