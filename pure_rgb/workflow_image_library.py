"""Folder library with asynchronous scans and bounded RAW thumbnails."""

import hashlib
import json
from pathlib import Path
import threading

from PyQt6.QtCore import QSize, Qt, QThreadPool, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from .image_library import ImageLibrary
from .preview_ui import Work
from .workflow_combo import WorkflowComboBox
from .workflow_notices import center_notice_text


class ImageLibraryDialog(QDialog):
    import_requested = pyqtSignal(list, str)

    def __init__(self, path, parent=None, *, source_mode="rgb"):
        super().__init__(parent)
        self.setWindowTitle("图片库")
        self.resize(1040, 740)
        self.setMinimumSize(740, 520)
        self.library = ImageLibrary(path)
        self.cache = Path(path).parent / "library-thumbnails"
        self.cache.mkdir(exist_ok=True)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.cancelled = threading.Event()
        self.generation = 0
        self.entries = []
        self.pending = []
        self.active = False
        self.scan_active = False
        self.jobs = {}
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.next_thumbnail)
        outer = QVBoxLayout(self)
        actions = QHBoxLayout()
        self.add = QPushButton("关联目录…")
        self.add.clicked.connect(self.add_folder)
        self.refresh = QPushButton("刷新目录")
        self.refresh.clicked.connect(self.scan_current)
        self.remove = QPushButton("移出图片库")
        self.remove.clicked.connect(self.remove_folder)
        self.reveal = QPushButton("在访达中打开")
        self.reveal.clicked.connect(self.reveal_folder)
        for widget in (self.add, self.refresh, self.remove, self.reveal):
            actions.addWidget(widget)
        actions.addStretch()
        outer.addLayout(actions)
        note = QLabel("关联素材目录，保留原文件位置。选中照片即可导入当前项目并自动处理。")
        note.setWordWrap(True)
        outer.addWidget(note)
        split = QSplitter()
        self.folders = QListWidget()
        self.folders.setAccessibleName("图片库目录")
        self.folders.currentItemChanged.connect(self.folder_changed)
        split.addWidget(self.folders)
        contents = QWidget()
        column = QVBoxLayout(contents)
        column.setContentsMargins(0, 0, 0, 0)
        filters = QHBoxLayout()
        self.mode = WorkflowComboBox()
        for label, value in (("拜耳 RGB · 三张一组", "rgb"), ("单色 RGB · 三张一组", "mono_rgb"),
                             ("黑白负片 · 单张", "bw")):
            self.mode.addItem(label, value)
        self.mode.setCurrentIndex(max(0, self.mode.findData(source_mode)))
        self.mode.currentIndexChanged.connect(self.show_entries)
        self.directory = WorkflowComboBox()
        self.directory.currentIndexChanged.connect(self.show_entries)
        self.search = QLineEdit()
        self.search.setPlaceholderText("查找文件名或子目录")
        self.search.textChanged.connect(self.show_entries)
        filters.addWidget(self.mode)
        filters.addWidget(self.directory, 1)
        filters.addWidget(self.search, 1)
        column.addLayout(filters)
        self.photos = QListWidget()
        self.photos.setAccessibleName("图片库照片；单击单选，⌘／Ctrl 多选，Shift 连选，⌘A／Ctrl+A 全选")
        self.photos.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.photos.setViewMode(QListWidget.ViewMode.IconMode)
        self.photos.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.photos.setMovement(QListWidget.Movement.Static)
        self.photos.setIconSize(QSize(144, 108))
        self.photos.setGridSize(QSize(180, 160))
        self.photos.itemSelectionChanged.connect(self.selection_changed)
        self.photos.itemDoubleClicked.connect(lambda _: self.request_import())
        column.addWidget(self.photos, 1)
        split.addWidget(contents)
        split.setStretchFactor(1, 1)
        split.setSizes([230, 780])
        outer.addWidget(split, 1)
        bottom = QHBoxLayout()
        self.select_all = QPushButton("全选")
        self.select_all.clicked.connect(self.photos.selectAll)
        self.select_none = QPushButton("取消选择")
        self.select_none.clicked.connect(self.photos.clearSelection)
        self.status = QLabel()
        self.status.setWordWrap(True)
        center_notice_text(self.status)
        self.import_button = QPushButton("导入所选照片")
        self.import_button.setObjectName("primary")
        self.import_button.clicked.connect(self.request_import)
        close = QPushButton("关闭")
        close.clicked.connect(self.close)
        for widget in (self.select_all, self.select_none):
            bottom.addWidget(widget)
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.import_button)
        bottom.addWidget(close)
        outer.addLayout(bottom)
        self.refresh_folders()

    def current_folder(self):
        item = self.folders.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def refresh_folders(self, selected=None):
        selected = selected or (self.current_folder() or {}).get("id")
        self.folders.blockSignals(True)
        self.folders.clear()
        current = None
        for folder in self.library.folders():
            suffix = " · 离线" if not Path(folder["path"]).is_dir() else ""
            item = QListWidgetItem(f"{folder['name']}\n{folder['count']} 张 RAW{suffix}")
            item.setToolTip(folder["path"])
            item.setData(Qt.ItemDataRole.UserRole, folder)
            self.folders.addItem(item)
            if folder["id"] == selected:
                current = item
        self.folders.blockSignals(False)
        if current is not None:
            self.folders.setCurrentItem(current)
        elif self.folders.count():
            self.folders.setCurrentRow(0)
        else:
            self.folder_changed()

    def add_folder(self):
        path = QFileDialog.getExistingDirectory(self, "关联素材目录", str(Path.home() / "Pictures"))
        if not path:
            return
        try:
            folder = self.library.add_folder(path)
            self.refresh_folders(folder["id"])
            self.scan_current()
        except (OSError, ValueError) as error:
            self.status.setText(str(error))

    def remove_folder(self):
        folder = self.current_folder()
        if folder:
            self.library.remove_folder(folder["id"])
            self.refresh_folders()
            self.status.setText("目录已移出图片库，原始文件和项目编辑记录保留。")

    def reveal_folder(self):
        folder = self.current_folder()
        if folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder["path"]))

    def folder_changed(self, *args):
        folder = self.current_folder()
        self.directory.blockSignals(True)
        self.directory.clear()
        self.directory.addItem("全部子目录", None)
        if folder:
            for relative in self.library.subdirectories(folder["id"]):
                self.directory.addItem("当前目录" if relative == "." else relative, relative)
        self.directory.blockSignals(False)
        self.show_entries()

    def scan_current(self):
        folder = self.current_folder()
        if not folder or self.scan_active:
            return
        self.scan_active = True
        self.refresh.setEnabled(False)
        self.add.setEnabled(False)
        self.status.setText("正在扫描目录…")
        key = ("scan", folder["id"])
        work = Work(key, lambda: self.library.scan(folder["id"], cancelled=self.cancelled.is_set))
        self.jobs[key] = work
        work.signals.done.connect(self.scan_finished)
        work.signals.failed.connect(self.failed)
        self.pool.start(work)

    def scan_finished(self, key, count):
        self.jobs.pop(key, None)
        self.scan_active = False
        self.refresh.setEnabled(True)
        self.add.setEnabled(True)
        if self.cancelled.is_set():
            return
        self.refresh_folders()
        if self.current_folder() and self.current_folder()["id"] == key[1]:
            self.folder_changed()

    def show_entries(self, *args):
        self.generation += 1
        self.pending = []
        self.photos.clear()
        folder = self.current_folder()
        self.entries, self.remainder = (self.library.entries(
            folder["id"], self.mode.currentData(), subdirectory=self.directory.currentData(),
            search=self.search.text()) if folder else ([], 0))
        for entry in self.entries:
            count = len(entry["paths"])
            item = QListWidgetItem(f"{entry['name']}\n{'R · G · B' if count == 3 else '黑白'}")
            item.setToolTip("\n".join(entry["paths"]))
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self.photos.addItem(item)
            digest = hashlib.sha256(json.dumps([entry["paths"], entry["stamps"]]).encode()).hexdigest()
            cached = self.cache / (digest + ".png")
            if cached.is_file() and not QIcon(str(cached)).isNull():
                item.setIcon(QIcon(str(cached)))
            else:
                self.pending.append((self.generation, self.photos.count() - 1, entry, cached))
        self.selection_changed()
        self.timer.start(0)

    def selection_changed(self):
        count = len(self.photos.selectedItems())
        folder = self.current_folder()
        self.import_button.setEnabled(count > 0 and bool(folder) and Path(folder["path"]).is_dir())
        self.import_button.setText(f"导入所选 {count} 张照片" if count else "导入所选照片")
        suffix = f"；另有 {self.remainder} 张 RAW 未组成三色组" if getattr(self, "remainder", 0) else ""
        self.status.setText(f"已选 {count} / {len(self.entries)} 张照片{suffix}" if folder
                            else "点击“关联目录”开始管理素材。")

    def next_thumbnail(self):
        if self.active or not self.pending or self.cancelled.is_set() or not self.isVisible():
            return
        generation, index, entry, cached = self.pending.pop(0)
        self.active = True
        key = ("thumbnail", generation, index)

        def generate():
            if self.cancelled.is_set():
                return None
            from .workflow_import_preview import import_preview
            image = import_preview(entry["paths"]).scaled(
                144, 108, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            if not image.isNull() and not self.cancelled.is_set():
                image.save(str(cached), "PNG")
            return image

        work = Work(key, generate)
        self.jobs[key] = work
        work.signals.done.connect(self.thumbnail_finished)
        work.signals.failed.connect(self.failed)
        self.pool.start(work)

    def thumbnail_finished(self, key, image):
        self.jobs.pop(key, None)
        self.active = False
        if self.cancelled.is_set():
            return
        if key[1] == self.generation and image is not None and not image.isNull():
            item = self.photos.item(key[2])
            if item is not None:
                item.setIcon(QIcon(QPixmap.fromImage(image)))
        self.timer.start(0)

    def failed(self, key, message):
        self.jobs.pop(key, None)
        if self.cancelled.is_set():
            return
        if key[0] == "scan":
            self.scan_active = False
            self.refresh.setEnabled(True)
            self.add.setEnabled(True)
            self.status.setText(message)
        else:
            self.active = False
            self.timer.start(0)

    def request_import(self):
        paths = [path for item in self.photos.selectedItems()
                 for path in item.data(Qt.ItemDataRole.UserRole)["paths"]]
        if paths:
            self.import_requested.emit(paths, self.mode.currentData())

    def closeEvent(self, event):
        # Closing this browser is immediate, even when a large RAW has no
        # embedded thumbnail. Shutdown alone joins decoder work before Qt dies.
        self.pending.clear()
        self.timer.stop()
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if not self.cancelled.is_set():
            self.show_entries()

    def dispose(self):
        self.cancelled.set()
        self.timer.stop()
        self.pool.clear()
        self.pool.waitForDone()
