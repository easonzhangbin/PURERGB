"""Pass read-only project artwork to the native PURE Neg light table."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import uuid

from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtWidgets import QFileDialog

from .workflow_glass import obsidian_enabled
from .workflow_ui_model import RESULT_MODES


class ProjectLightTable:
    def __init__(self, window):
        self.window = window
        self.process = None
        self.directory = None
        self.lease = None
        self.project_ids = {}

    def sync_buttons(self):
        window = self.window
        available = self.process is None
        photo = window.current_photo()
        window.light_table_current.setEnabled(bool(available and photo and photo["initial_job"]))
        window.light_table_roll.setEnabled(available and any(p["initial_job"] for p in window.project.photos))
        window.light_table_files.setEnabled(available)

    def open_current(self):
        photo = self.window.current_photo()
        if photo:
            self._open_project([photo], "single")

    def open_roll(self):
        self._open_project(self.window.project.photos, "roll")

    def _open_project(self, photos, mode):
        if self.process is not None:
            return
        window = self.window
        items = []
        missing = []
        self.lease = window.snapshot_store.acquire()
        try:
            context = window.preview_context() if self.lease else None
            for photo in photos:
                name = Path(photo["base"]["raw"][0]).stem
                try:
                    path, status = self._artwork(photo, context)
                except (KeyError, ValueError, StopIteration, OSError) as error:
                    missing.append(f"{name}（{error}）")
                    continue
                key = str(uuid.uuid5(uuid.NAMESPACE_URL, "pure-rgb:photo:" + photo["id"]))
                self.project_ids[key] = photo["id"]
                items.append(dict(id=key, artwork=str(path), name=name, status=status,
                                  selected=photo["id"] in window.project.data["checked"]))
            if not items:
                raise ValueError("没有可查看的照片。" + "、".join(missing[:3]))
            current = window.project.data["current"]
            current_id = next((k for k, v in self.project_ids.items() if v == current), None)
            self._start(items, mode, current_id, True)
            note = f"已载入 {len(items)} 张照片"
            if missing:
                note += f"，另有 {len(missing)} 张缺少可用版本。"
            window.light_table_note.setText(note)
            window.light_table_note.setToolTip("\n".join(missing))
        except (KeyError, ValueError, OSError, RuntimeError) as error:
            self._cleanup()
            window.error(error)

    def _artwork(self, photo, context):
        window = self.window
        project = window.project
        job = project.saved_job(photo)
        dirty = not job or project.dirty(photo, saved=job)
        if dirty and self.lease and context and photo["id"] == context["photo"]:
            snapshot = self.lease.context
            identity = ("photo", "engine", "recipe_sha256", "repair_plan_sha256")
            path = Path(self.lease.current["path"])
            # A leased worker TIFF is a rendered draft. Immutable saved TIFFs
            # may also have a lease while the next draft is still rendering.
            if (all(snapshot.get(k) == context.get(k) for k in identity)
                    and path.is_relative_to(window.snapshot_store.root) and path.is_file()):
                return self.lease.current_tiff_path(), "当前调色预览"
        if not job:
            raise ValueError("请先生成当前引擎的预览或完整版本")
        baseline = project.workspace.sample(job).record(RESULT_MODES[photo["engine"]], "baseline")
        record = project.confirmed_record(baseline, photo)
        path = Path(record["path"])
        if not path.is_file():
            raise ValueError("成片文件不存在")
        status = "上次保存的完整版本" if dirty else "当前完整版本"
        if record["sha256"] != baseline["sha256"]:
            status += " · 已确认除尘"
        return path, status

    def open_files(self):
        if self.process is not None:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self.window, "打开观片图像", "", "图像 (*.tif *.tiff *.png *.jpg *.jpeg *.heic *.heif)")
        if paths:
            items = [dict(id=str(uuid.uuid4()), artwork=p, name=Path(p).name,
                          status="外部图像", selected=False) for p in paths]
            try:
                self._start(items, "single" if len(items) == 1 else "roll", items[0]["id"], False)
            except (OSError, ValueError, RuntimeError) as error:
                self._cleanup()
                self.window.error(error)

    def _start(self, items, mode, current, can_export):
        from .light_table_launcher import launch_command
        from .workflow_display import window_display_id

        self.directory = TemporaryDirectory(prefix="purergb-light-table-")
        manifest = Path(self.directory.name) / "session.json"
        manifest.write_text(json.dumps(dict(mode=mode, current=current, items=items,
                                            canExport=can_export, displayID=window_display_id(self.window),
                                            obsidian=obsidian_enabled()),
                                       ensure_ascii=False))
        command = launch_command(manifest)
        process = QProcess(self.window)
        self.process = process
        process.readyReadStandardOutput.connect(self._read_events)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._failed)
        process.start(str(command[0]), list(map(str, command[1:])))
        self.sync_buttons()

    def _read_events(self):
        process = self.process
        while process and process.canReadLine():
            try:
                event = json.loads(bytes(process.readLine()))
            except ValueError:
                continue
            if not isinstance(event, dict) or self.window.closing or not self.project_ids:
                continue
            if event.get("action") == "export":
                self.window.navigate_stage(5)
            if event.get("action") != "select" or type(event.get("selected")) is not bool:
                continue
            key = self.project_ids.get(str(event.get("id")).lower())
            if key is None:
                continue
            listing = self.window.photo_list
            for index in range(listing.count()):
                item = listing.item(index)
                if item.data(Qt.ItemDataRole.UserRole) == key:
                    item.setSelected(event["selected"])
                    break

    def _failed(self, error):
        if error == QProcess.ProcessError.FailedToStart and self.process:
            self.window.error("无法打开观片台。" + self.process.errorString())
            self._cleanup()

    def _finished(self, code, _status):
        if self.process is None:
            return
        self._read_events()
        if code and not self.window.closing:
            detail = bytes(self.process.readAllStandardError()).decode(errors="replace").strip()
            self.window.error(detail[-400:] or f"观片台已退出（{code}）")
        self._cleanup()

    def _cleanup(self):
        if self.process:
            self.process.deleteLater()
            self.process = None
        if self.directory:
            self.directory.cleanup()
            self.directory = None
        if self.lease:
            self.lease.release()
            self.lease = None
        self.project_ids.clear()
        if not self.window.closing:
            self.sync_buttons()

    def close(self):
        process = self.process
        if process and process.state() != QProcess.ProcessState.NotRunning:
            process.terminate()
            if not process.waitForFinished(1200):
                process.kill()
                process.waitForFinished(1500)
        self._cleanup()
