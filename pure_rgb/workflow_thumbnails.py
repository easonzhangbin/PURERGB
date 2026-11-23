"""Small source previews for every list row, independent of full processing."""

import hashlib
import json
from pathlib import Path

from PyQt6.QtCore import QObject, QThreadPool, QTimer, Qt
from PyQt6.QtGui import QColor, QIcon, QImage, QPixmap

from .preview_ui import Work


def source_key(photo):
    paths = tuple(photo["base"]["raw"])
    stamps = []
    for path in paths:
        stat = Path(path).stat()
        stamps.append((path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    digest = hashlib.sha256(json.dumps(stamps).encode()).hexdigest()[:24]
    return photo["id"], digest


class PhotoThumbnailLoader(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.start_next)
        self.directory = window.project.workspace.root / "import-thumbnails"
        self.directory.mkdir(exist_ok=True)
        self.pending = {}
        self.current = {}
        self.ready = set()
        self.cache = {}
        self.active = None
        self.closed = False

    def set_icon(self, identifier, icon):
        if self.closed or self.window.closing:
            return
        for index in range(self.window.photo_list.count()):
            item = self.window.photo_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == identifier:
                item.setIcon(icon)
                return

    def sync(self, ready_photo_ids):
        if self.closed or self.window.closing:
            return
        self.ready = set(ready_photo_ids)
        current, pending = {}, {}
        for photo in self.window.project.photos:
            identifier = photo["id"]
            if identifier in self.ready:
                continue
            try:
                key = source_key(photo)
            except OSError:
                continue
            current[identifier] = key
            if key in self.cache:
                self.set_icon(identifier, self.cache[key])
                continue
            path = self.directory / ("-".join(key) + ".png")
            if path.is_file():
                icon = QIcon(str(path))
                if not icon.isNull():
                    self.cache[key] = icon
                    self.set_icon(identifier, icon)
                    continue
            if key != self.active:
                pending[key] = tuple(photo["base"]["raw"])
        self.current, self.pending = current, pending
        if self.active is None:
            self.timer.start(0)

    def start_next(self):
        if self.closed or self.window.closing or self.active is not None or not self.pending:
            return
        key = next(iter(self.pending))
        paths = self.pending.pop(key)
        self.active = key
        path = self.directory / ("-".join(key) + ".png")

        def generate():
            from .workflow_import_preview import import_preview

            snapshot = dict(id=key[0], base=dict(raw=paths))
            if source_key(snapshot) != key:
                raise ValueError("RAW 来源已改变")
            image = import_preview(paths).scaled(96, 96, Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation)
            if image.isNull():
                raise ValueError("RAW 缩略图不可用")
            if source_key(snapshot) != key:
                raise ValueError("读取缩略图时 RAW 已改变")
            image.save(str(path), "PNG")
            return image

        work = Work(key, generate)
        work.signals.done.connect(self.finished)
        work.signals.failed.connect(self.failed)
        self.work = work
        self.pool.start(work)

    def finished(self, key, image):
        self.active = None
        if self.closed or self.window.closing:
            return
        try:
            still_current = source_key(self.window.project.photo(key[0])) == key
        except (StopIteration, OSError):
            still_current = False
        if still_current and self.current.get(key[0]) == key and key[0] not in self.ready:
            icon = QIcon(QPixmap.fromImage(image))
            self.cache[key] = icon
            self.set_icon(key[0], icon)
        self.timer.start(0)

    def failed(self, key, message):
        # An unreadable embedded preview never cancels the full RAW job.
        image = QImage(96, 64, QImage.Format.Format_RGB888)
        image.fill(QColor("#3b3b3b"))
        self.finished(key, image)

    def close(self):
        self.closed = True
        self.timer.stop()
        self.pending.clear()
        self.pool.waitForDone()
