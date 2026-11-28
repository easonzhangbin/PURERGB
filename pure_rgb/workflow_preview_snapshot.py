"""Leased, verified display sources independent of unsaved render settings."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import threading
import uuid

from .preview_fit_transport import FIT_TRANSPORT, read_fit_record


@dataclass
class _Entry:
    directory: Path
    current: dict
    reference: dict
    context: dict
    sample_step: int
    epoch: int
    readers: int = 1


class SnapshotLease:
    def __init__(self, store, entry):
        self.store, self.entry = store, entry
        self.released = False

    @property
    def current(self):
        return deepcopy(self.entry.current)

    @property
    def reference(self):
        return deepcopy(self.entry.reference)

    @property
    def context(self):
        return deepcopy(self.entry.context)

    @property
    def sample_step(self):
        return self.entry.sample_step

    def current_tiff_path(self):
        """Materialize a small preview only for consumers that require TIFF."""
        with self.store.lock:
            if self.released:
                raise ValueError("预览快照已释放")
            record = self.entry.current
            path = Path(record["path"])
            if record.get("transport") != FIT_TRANSPORT:
                return path
            destination = self.entry.directory / "light-table.tiff"
            if not destination.is_file():
                import tifffile

                pixels, icc = read_fit_record(record)
                temporary = destination.with_suffix(".tmp")
                try:
                    tifffile.imwrite(
                        temporary, pixels, photometric="rgb", metadata=None,
                        extratags=[(34675, 7, len(icc), icc, False)],
                    )
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
            return destination

    def release(self):
        self.store.release(self)


class PreviewSnapshotStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.latest = None
        self.entries = {}
        self.epoch, self.closed = 0, False

    @staticmethod
    def _copy(record, destination):
        shutil.copyfile(record["path"], destination)
        with destination.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record["sha256"]:
            raise ValueError("预览源在交接期间改变，不能保留为显示快照")
        return {**deepcopy(record), "path": str(destination)}

    def _adopt(self, record, directory, role, worker_directory):
        if record.get("transport") != FIT_TRANSPORT:
            return self._copy(record, directory / f"{role}.tiff")
        source = Path(record["path"])
        destination = directory / f"{role}.rgb16"
        if (source.parent.resolve() != worker_directory or source.is_symlink()
                or source.suffix != ".rgb16" or not source.is_file()):
            raise ValueError("预览帧不在当前工作进程目录")
        try:
            source.replace(destination)
        except OSError:
            return self._copy(record, destination)
        return {**deepcopy(record), "path": str(destination)}

    def prepare(self, current, reference, context, sample_step=1, *, own_files=True, worker_directory=None):
        with self.lock:
            if self.closed:
                raise ValueError("预览快照已关闭")
            epoch = self.epoch
        directory = self.root / uuid.uuid4().hex
        directory.mkdir()
        try:
            if own_files:
                source_directory = self.root.parent if worker_directory is None else Path(worker_directory).resolve()
                first = self._adopt(current, directory, "current", source_directory)
                second = (
                    first if current["sha256"] == reference["sha256"]
                    else self._adopt(reference, directory, "reference", source_directory)
                )
            else:
                # Saved versions are immutable and not owned by this store.
                first, second = deepcopy(current), deepcopy(reference)
            entry = _Entry(directory, first, second, deepcopy(context), sample_step, epoch)
            with self.lock:
                if self.closed or epoch != self.epoch:
                    raise ValueError("预览快照交接已取消")
                self.entries[directory] = entry
            return SnapshotLease(self, entry)
        except BaseException:
            shutil.rmtree(directory)
            raise

    def _drop(self, entry):
        if entry.directory not in self.entries:
            return
        entry.readers -= 1
        if entry.readers == 0:
            self.entries.pop(entry.directory, None)
            shutil.rmtree(entry.directory)

    def publish(self, lease):
        with self.lock:
            entry = lease.entry
            if lease.released or self.closed or entry.epoch != self.epoch:
                return False
            generation = entry.context.get("generation", 0)
            if self.latest and generation < self.latest.context.get("generation", 0):
                return False
            if self.latest is not entry:
                previous, self.latest = self.latest, entry
                entry.readers += 1
                if previous:
                    self._drop(previous)
            return True

    def acquire(self):
        with self.lock:
            if self.closed or self.latest is None:
                return None
            self.latest.readers += 1
            return SnapshotLease(self, self.latest)

    def release(self, lease):
        with self.lock:
            if not lease.released:
                lease.released = True
                self._drop(lease.entry)

    def clear(self):
        with self.lock:
            self.epoch += 1
            previous, self.latest = self.latest, None
            if previous:
                self._drop(previous)

    def close(self, *, force=False):
        with self.lock:
            self.clear()
            self.closed = True
            if force:
                # Host has joined both display pools; queued callbacks no longer
                # perform file IO and may release their lease after this point.
                for entry in self.entries.values():
                    shutil.rmtree(entry.directory)
                self.entries.clear()
