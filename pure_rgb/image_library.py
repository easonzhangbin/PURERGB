"""Persistent linked RAW folders; the catalog never writes to source folders."""

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import time
import uuid

from .flatfield import RAW_EXTENSIONS


class ImageLibrary:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS folders (
                    id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL, scanned REAL);
                CREATE TABLE IF NOT EXISTS media (
                    folder TEXT NOT NULL, relative_path TEXT NOT NULL,
                    size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
                    PRIMARY KEY(folder, relative_path));
                CREATE INDEX IF NOT EXISTS media_folder ON media(folder);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def add_folder(self, path):
        path = Path(path).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ValueError("请选择素材文件夹")
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO folders VALUES (?, ?, ?, NULL)",
                       (uuid.uuid4().hex, str(path), path.name or str(path)))
            return dict(db.execute("SELECT * FROM folders WHERE path=?", (str(path),)).fetchone())

    def folders(self):
        with self.connection() as db:
            return [dict(row) for row in db.execute(
                "SELECT f.*, COUNT(m.relative_path) AS count FROM folders f "
                "LEFT JOIN media m ON f.id=m.folder GROUP BY f.id ORDER BY f.name, f.path")]

    def remove_folder(self, identifier):
        with self.connection() as db:
            db.execute("DELETE FROM media WHERE folder=?", (identifier,))
            db.execute("DELETE FROM folders WHERE id=?", (identifier,))

    def scan(self, identifier, *, cancelled=lambda: False):
        with self.connection() as db:
            row = db.execute("SELECT * FROM folders WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ValueError("目录已从图片库移除")
        root = Path(row["path"])
        if not root.is_dir():
            raise ValueError("目录暂时离线，请连接磁盘后刷新")
        records = []

        def failed(error):
            raise error

        for directory, names, files in os.walk(root, followlinks=False, onerror=failed):
            if cancelled():
                return None
            names[:] = sorted(name for name in names if not name.startswith("."))
            for name in sorted(files):
                path = Path(directory) / name
                if path.suffix.lower() not in RAW_EXTENSIONS or path.is_symlink():
                    continue
                stat = path.stat()
                records.append((identifier, str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
        if cancelled():
            return None
        with self.connection() as db:
            # Take the write lock before checking that the folder still exists.
            # A SELECT by itself does not start a SQLite write transaction.
            db.execute("BEGIN IMMEDIATE")
            if cancelled() or db.execute("SELECT 1 FROM folders WHERE id=?", (identifier,)).fetchone() is None:
                return None
            db.execute("DELETE FROM media WHERE folder=?", (identifier,))
            db.executemany("INSERT INTO media VALUES (?, ?, ?, ?)", records)
            db.execute("UPDATE folders SET scanned=? WHERE id=?", (time.time(), identifier))
        return len(records)

    def entries(self, identifier, source_mode="rgb", *, subdirectory=None, search=""):
        if source_mode not in {"rgb", "mono_rgb", "bw"}:
            raise ValueError("未知输入方式")
        with self.connection() as db:
            folder = db.execute("SELECT * FROM folders WHERE id=?", (identifier,)).fetchone()
            if folder is None:
                return [], 0
            rows = list(db.execute("SELECT * FROM media WHERE folder=? ORDER BY relative_path", (identifier,)))
        root, groups = Path(folder["path"]), {}
        for row in rows:
            relative = Path(row["relative_path"])
            if source_mode in ("rgb", "bw") and relative.suffix.lower() not in {".rw2", ".dng"}:
                continue
            if subdirectory is not None and str(relative.parent) != subdirectory:
                continue
            groups.setdefault((str(relative.parent), relative.suffix.lower()), []).append(row)
        result, remainder = [], 0
        width = 1 if source_mode == "bw" else 3
        query = search.strip().casefold()
        for (parent, _), values in sorted(groups.items()):
            count = len(values) - len(values) % width
            remainder += len(values) - count
            for start in range(0, count, width):
                sample = values[start:start + width]
                paths = [str(root / row["relative_path"]) for row in sample]
                if query and not any(query in row["relative_path"].casefold() for row in sample):
                    continue
                result.append(dict(paths=paths, name=Path(paths[0]).stem, relative_directory=parent,
                                   stamps=[(row["size"], row["mtime_ns"]) for row in sample]))
        return result, remainder

    def subdirectories(self, identifier):
        with self.connection() as db:
            return sorted({str(Path(row[0]).parent) for row in db.execute(
                "SELECT relative_path FROM media WHERE folder=?", (identifier,))})
