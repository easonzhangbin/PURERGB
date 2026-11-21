"""Choose an existing task or a new workspace before opening either project."""

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import uuid


@dataclass(frozen=True)
class WorkspaceSummary:
    path: Path
    has_task: bool
    photo_count: int | None = None
    latest_photo: str | None = None


@dataclass(frozen=True)
class StartupChoice:
    project: Path | None
    preferences: dict | None


def _existing_path(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        path = Path(value).expanduser()
        return path if (path / "workspace.json").is_file() else None
    except (OSError, RuntimeError, ValueError):
        return None


def inspect_workspace(path):
    """Inspect UI metadata only; never instantiate or write the workspace."""
    path = _existing_path(str(path))
    if path is None:
        return None
    try:
        ui_path = path / "ui-workflow.json"
        ui = json.loads(ui_path.read_text()) if ui_path.exists() else {"photos": []}
        if not isinstance(ui, dict) or not isinstance(ui.get("photos"), list):
            raise ValueError("invalid UI metadata")
        photos = ui["photos"]
        latest = None
        current = ui.get("current")
        selected = next((photo for photo in photos
                         if isinstance(photo, dict) and photo.get("id") == current), None)
        if selected is None:
            selected = next((photo for photo in reversed(photos) if isinstance(photo, dict)), None)
        if selected:
            raw = (selected.get("base") or {}).get("raw") or []
            if isinstance(raw, list) and raw and isinstance(raw[0], str):
                latest = Path(raw[0]).name
        if photos:
            # The UI is sufficient to prove a real task. Avoid reading a large
            # complete workspace/job history merely to show the startup prompt.
            return WorkspaceSummary(path, True, len(photos), latest)
        workspace = json.loads((path / "workspace.json").read_text())
        if not isinstance(workspace, dict) or not isinstance(workspace.get("jobs"), list):
            raise ValueError("invalid workspace metadata")
        return WorkspaceSummary(path, bool(workspace["jobs"]), 0, latest)
    except (OSError, UnicodeError, ValueError, TypeError, AttributeError):
        # A damaged or inaccessible metadata file is not evidence of an empty
        # project. Offer it for recovery instead of silently replacing it.
        return WorkspaceSummary(path, True)


def ask_restore(summary):
    """Present the decision before a splash or workspace/RAW is opened."""
    from PyQt6.QtCore import QSize, Qt
    from PyQt6.QtWidgets import QMessageBox

    from .branding import application_icon
    from .workflow_glass import GlassMessageBox, stylesheet

    box = GlassMessageBox()
    # A native NSAlert before the first main window can leave that window
    # off screen. Keep the decision and its keyboard handling inside Qt.
    box.setOption(QMessageBox.Option.DontUseNativeDialog)
    box.setWindowTitle("载入上次工作区？")
    box.setTextFormat(Qt.TextFormat.PlainText)
    box.setText("是否继续上次任务？")
    detail = "检测到上次工作区。"
    if summary.photo_count:
        detail = f"上次工作区有 {summary.photo_count} 张照片。"
        if summary.latest_photo:
            detail += f"\n当前照片：{summary.latest_photo}"
    box.setInformativeText(detail)
    box.setIconPixmap(application_icon().pixmap(QSize(48, 48), box.devicePixelRatioF()))
    choices = QMessageBox.StandardButton
    box.setStandardButtons(choices.Yes | choices.No | choices.Cancel)
    for standard, label in ((choices.Yes, "是，载入上次任务"), (choices.No, "否，新建空白项目"),
                            (choices.Cancel, "取消启动")):
        button = box.button(standard)
        button.setText(label)
        button.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
    box.button(choices.Yes).setObjectName("primary")
    box.setDefaultButton(choices.Yes)
    box.setEscapeButton(choices.Cancel)
    # The main window's stylesheet is not installed on this top-level prompt.
    # Give Qt the same transparent control faces over the native glass.
    box.setStyleSheet(stylesheet() + """
        QLabel#qt_msgbox_label { font-size:16px; font-weight:600; }
        QPushButton, QPushButton#primary { border-width:2px; }
        QPushButton:focus, QPushButton[nativeMaterial="true"]:focus,
        QPushButton#primary:focus, QPushButton#primary[nativeMaterial="true"]:focus {
            border:2px solid palette(highlight);
        }
    """)
    box.exec()
    answer = box.standardButton(box.clickedButton())
    if answer == QMessageBox.StandardButton.Yes:
        return True
    if answer == QMessageBox.StandardButton.No:
        return False
    return None


def choose_workspace(root, *, explicit_workspace=None, ready_report=None, ask=ask_restore):
    """Choose without opening any project or modifying saved preferences.

    Explicit workspaces and test ready reports do not touch user preferences.
    The returned preference document is committed only after window creation.
    """
    root = Path(root)
    if explicit_workspace is not None:
        return StartupChoice(Path(explicit_workspace), None)
    if ready_report is not None:
        return StartupChoice(Path(tempfile.mkdtemp(prefix="purergb-ready-")) / "project", None)
    path = root / "outputs/workflow-ui-state/preferences.json"
    try:
        preferences = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, UnicodeError, ValueError):
        preferences = {}
    if not isinstance(preferences, dict):
        preferences = {}
    previous_path = _existing_path(preferences.get("workspace"))
    previous = inspect_workspace(previous_path) if previous_path else None
    candidate = previous if previous and previous.has_task else None
    if candidate is None:
        last_task_path = _existing_path(preferences.get("last_task_workspace"))
        remembered = (previous if previous and last_task_path == previous.path else
                      inspect_workspace(last_task_path) if last_task_path else None)
        candidate = remembered if remembered and remembered.has_task else None
    result = dict(preferences)
    if candidate is not None:
        decision = ask(candidate)
        if decision is not True and decision is not False:
            return StartupChoice(None, None)
        result["last_task_workspace"] = str(candidate.path)
        project = (candidate.path if decision else
                   root / "outputs/ui-projects" / ("v35-" + uuid.uuid4().hex[:10]))
    elif previous is not None:
        project = previous.path
    else:
        project = root / "outputs/ui-projects" / ("v35-" + uuid.uuid4().hex[:10])
    result["workspace"] = str(project)
    return StartupChoice(project, result)
