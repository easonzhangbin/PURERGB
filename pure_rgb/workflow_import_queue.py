"""Import all registered photos without a second, per-photo confirmation."""

from PyQt6.QtCore import QObject, QTimer

from .workflow_ui_model import RESULT_MODES


class AutoImport(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.pending = list(window.project.data.get("import_queue", []))
        self.active = None
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.pump)

    def save(self):
        self.window.project.data["import_queue"] = list(dict.fromkeys(
            ([self.active] if self.active else []) + self.pending))
        self.window.project.save()

    def enqueue(self, identifiers, *, resume=True):
        project = self.window.project
        if resume:
            project.data["auto_import_paused"] = False
        known = {photo["id"] for photo in project.photos}
        for identifier in identifiers:
            if identifier not in known:
                continue
            photo = project.photo(identifier)
            if (not photo["initial_job"] and not photo.get("import_error")
                    and identifier != self.active and identifier not in self.pending):
                self.pending.append(identifier)
        self.save()
        self.timer.start(0)

    def resume(self):
        self.enqueue([photo["id"] for photo in self.window.project.photos], resume=False)

    def state(self, photo):
        if photo["initial_job"]:
            return None
        if photo.get("import_error"):
            return "需要调整"
        if self.active == photo["id"]:
            return "处理中"
        if self.window.project.data.get("auto_import_paused"):
            return "已暂停处理"
        return "等待处理"

    def pump(self):
        window = self.window
        if window.closing or window.project.data.get("auto_import_paused"):
            return
        if self.active:
            return
        if window.tasks.busy or window.pending_probe or window.geometry_role:
            if self.pending:
                self.timer.start(200)
            return
        project = window.project
        while self.pending:
            identifier = self.pending.pop(0)
            try:
                photo = project.photo(identifier)
            except StopIteration:
                continue
            if photo["initial_job"] or photo.get("import_error"):
                continue
            self.active = identifier
            self.save()
            current = project.data["current"]
            try:
                # Enqueue through the existing recipe/resource validator, while
                # keeping the user's visible photo and thumbnail selection.
                project.data["current"] = identifier
                job = project.enqueue()
            except Exception as error:
                photo["import_error"] = str(error)
                self.active = None
                continue
            finally:
                project.data["current"] = current
                project.save()
            project.workspace.set_state(job["id"], "running")
            request = dict(**job, action="render", auto_import=True,
                only_mode=RESULT_MODES[photo["engine"]],
                directory=str(project.workspace.root / "runs" / job["id"]),
                decode_anchor=None, repair_plan=None)
            try:
                window.tasks.start(request)
            except Exception as error:
                project.workspace.set_state(job["id"], "failed", error=str(error))
                self.failed(request, str(error), False)
                return
            window.refresh_photos()
            window.sync_controls()
            window.status_message(f"正在自动处理 {photo['base']['raw'][0].rsplit('/', 1)[-1]}；其余照片会依次继续。")
            return
        self.save()
        window.refresh_photos()
        window.sync_controls()

    def completed(self, request):
        if not request.get("auto_import"):
            return
        binding = self.window.project.data["jobs"][request["id"]]
        photo = self.window.project.photo(binding["photo"])
        photo.pop("import_error", None)
        self.active = None
        self.save()
        self.timer.start(0)

    def failed(self, request, message, cancelled):
        binding = self.window.project.data["jobs"].get(request.get("id"), {})
        identifier = binding.get("photo")
        if identifier:
            photo = self.window.project.photo(identifier)
            if cancelled:
                if not photo["initial_job"] and identifier not in self.pending:
                    self.pending.insert(0, identifier)
            else:
                photo["import_error"] = str(message)
                photo["import_failed_job"] = request["id"]
        self.active = None
        if cancelled and not self.window.closing:
            self.window.project.data["auto_import_paused"] = True
        self.save()
        if self.window.closing:
            return
        self.window.refresh_photos()
        self.window.sync_controls()
        self.timer.start(0)

    def cancel(self):
        self.timer.stop()
        self.window.project.data["auto_import_paused"] = True
        self.save()

    def close(self):
        self.timer.stop()
        self.save()
