"""Independent, optional RGB film-base meter, with no workflow prerequisites."""

from pathlib import Path
from copy import deepcopy

from PyQt6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QPushButton, QScrollArea, QSizePolicy, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from .rgb_light_balance import CHANNELS, DEFAULT_REGION, FULL_REGION, analyze_triplet, checked_region
from .workflow_combo import WorkflowComboBox
from .workflow_glass import prepare_glass, stylesheet
from .workflow_indicators import ReviewCheckBox
from .workflow_rgb_balance_chart import RgbBalanceChart


class RawReferenceButton(QPushButton):
    """Show a compact basename while preserving the existing path-field API."""

    textChanged = pyqtSignal(str)

    def __init__(self, channel, path="", parent=None):
        super().__init__(parent)
        self.channel = channel
        self._path = ""
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(0)
        self.setText(path)

    def text(self):
        return self._path

    def setText(self, path):
        path = str(path or "")
        changed = path != self._path
        self._path = path
        self.setToolTip(path or f"选择 {self.channel} 灯光下的片基参考 RAW")
        self.setAccessibleName(f"{self.channel} 片基 RAW：{Path(path).name if path else '未选择'}")
        self._update_caption()
        if changed:
            self.textChanged.emit(path)

    def _update_caption(self):
        caption = f"{self.channel} · {Path(self._path).name}" if self._path else f"选择 {self.channel} 参考…"
        width = max(95, self.width() - 28)
        super().setText(self.fontMetrics().elidedText(caption, Qt.TextElideMode.ElideMiddle, width))

    def resizeEvent(self, event):
        self._update_caption()
        super().resizeEvent(event)


class BalanceRegionView(QWidget):
    """One normalized rectangle shared by three independent RAW previews."""

    region_changed = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(170)
        self.setMouseTracking(True)
        self.images = [None] * 3
        self.region = DEFAULT_REGION
        self.local = True
        self._drag = None
        self.setAccessibleName("三路共用片基范围")
        self.setToolTip("在任一路拖动画框；拖动框内移动，拖动边角缩放。三路共用同一范围。")

    def set_previews(self, readings):
        images = []
        for reading in readings:
            pixels = reading.preview
            if pixels is None:
                images.append(None)
            else:
                h, w = pixels.shape
                images.append(QImage(pixels.data, w, h, pixels.strides[0], QImage.Format.Format_Grayscale8).copy())
        self.images = images
        self.update()

    def set_region(self, region, *, emit=True):
        self.region = checked_region(region)
        self.update()
        if emit:
            self.region_changed.emit(self.region)

    def image_rects(self):
        width = (self.width() - 24) / 3
        rects = []
        for index, picture in enumerate(self.images):
            cell = QRectF(8 + index * (width + 4), 26, max(1, width - 8), max(1, self.height() - 35))
            if picture is not None:
                size = picture.size().scaled(cell.size().toSize(), Qt.AspectRatioMode.KeepAspectRatio)
                cell = QRectF(cell.center().x() - size.width() / 2, cell.center().y() - size.height() / 2, size.width(), size.height())
            rects.append(cell)
        return rects

    def region_rect(self, canvas):
        x0, y0, x1, y1 = self.region if self.local else FULL_REGION
        return QRectF(canvas.x() + x0 * canvas.width(), canvas.y() + y0 * canvas.height(),
                      (x1 - x0) * canvas.width(), (y1 - y0) * canvas.height())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#18191b"))
        for index, canvas in enumerate(self.image_rects()):
            painter.setPen(QColor("#e4e4e4"))
            painter.drawText(QRectF(canvas.x(), 0, canvas.width(), 24), Qt.AlignmentFlag.AlignCenter, CHANNELS[index])
            if self.images[index] is None:
                painter.drawText(canvas, Qt.AlignmentFlag.AlignCenter, "选择片基 RAW")
                continue
            painter.drawImage(canvas, self.images[index])
            painter.setPen(QPen(QColor("#ffb347"), 2))
            area = self.region_rect(canvas)
            painter.drawRect(area)
            if self.local:
                for corner in (area.topLeft(), area.topRight(), area.bottomLeft(), area.bottomRight()):
                    painter.fillRect(QRectF(corner.x() - 3, corner.y() - 3, 6, 6), QColor("#ffb347"))

    @staticmethod
    def _fraction(point, canvas):
        return QPointF(max(0, min(1, (point.x() - canvas.x()) / canvas.width())),
                       max(0, min(1, (point.y() - canvas.y()) / canvas.height())))

    def mousePressEvent(self, event):
        if not self.local or event.button() != Qt.MouseButton.LeftButton:
            return
        for index, canvas in enumerate(self.image_rects()):
            if self.images[index] is None or not canvas.contains(event.position()):
                continue
            area, point = self.region_rect(canvas), event.position()
            edges = []
            if abs(point.x() - area.left()) <= 8:
                edges.append("left")
            elif abs(point.x() - area.right()) <= 8:
                edges.append("right")
            if abs(point.y() - area.top()) <= 8:
                edges.append("top")
            elif abs(point.y() - area.bottom()) <= 8:
                edges.append("bottom")
            mode = "resize" if edges and area.adjusted(-8, -8, 8, 8).contains(point) else ("move" if area.contains(point) else "draw")
            self._drag = (canvas, self._fraction(point, canvas), self.region, mode, tuple(edges))
            event.accept()
            return

    def mouseMoveEvent(self, event):
        if self._drag is None:
            return
        canvas, start, original, mode, edges = self._drag
        point = self._fraction(event.position(), canvas)
        x0, y0, x1, y1 = original
        minimum_x, minimum_y = 1 / max(canvas.width(), 1), 1 / max(canvas.height(), 1)
        if mode == "draw":
            x0, x1 = sorted((start.x(), point.x()))
            y0, y1 = sorted((start.y(), point.y()))
            if x1 - x0 < minimum_x or y1 - y0 < minimum_y:
                return
        elif mode == "move":
            dx = max(-x0, min(1 - x1, point.x() - start.x()))
            dy = max(-y0, min(1 - y1, point.y() - start.y()))
            x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
        else:
            if "left" in edges:
                x0 = min(point.x(), x1 - minimum_x)
            if "right" in edges:
                x1 = max(point.x(), x0 + minimum_x)
            if "top" in edges:
                y0 = min(point.y(), y1 - minimum_y)
            if "bottom" in edges:
                y1 = max(point.y(), y0 + minimum_y)
        self.set_region((max(0, x0), max(0, y0), min(1, x1), min(1, y1)))

    def mouseReleaseEvent(self, event):
        self._drag = None


class _MeterSignals(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)


class _MeterWork(QRunnable):
    def __init__(self, paths, region, with_previews, flatfield=None):
        super().__init__()
        self.paths, self.region, self.with_previews = paths, region, with_previews
        self.flatfield = deepcopy(flatfield)
        self.signals = _MeterSignals()

    def run(self):
        try:
            kwargs = dict(with_previews=self.with_previews, progress=self.signals.progress.emit)
            if self.flatfield is not None:
                kwargs["flatfield"] = self.flatfield
            result = analyze_triplet(self.paths, self.region, **kwargs)
        except Exception as error:
            self.signals.failed.emit(str(error))
        else:
            self.signals.done.emit(result)


class RgbBalanceDialog(QDialog):
    def __init__(self, parent=None, source_paths=None, flatfield=None):
        super().__init__(parent)
        self.setWindowTitle("PURE RGB · RGB 拍摄灯光配平")
        self.setObjectName("workflowRoot")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.resize(850, 820)
        prepare_glass(self)
        self.setStyleSheet(stylesheet(False, dark=True) + """
            QDialog#workflowRoot { background:#242322; }
            QDialog#workflowRoot QLineEdit, QDialog#workflowRoot QComboBox {
                background:#303135; color:#f2ede4; }
            QCheckBox#rgbBalanceFlatfield::indicator,
            QCheckBox#rgbBalanceFlatfield::indicator:checked {
                width:17px; height:17px; border:0; background:transparent; }
            QCheckBox#rgbBalanceFlatfield:disabled { color:#98989b; }
            QDialog#workflowRoot QPushButton { background:#3b3a3e; color:#f2ede4; }
            QDialog#workflowRoot QPushButton:disabled { color:#98989b; }
            QDialog#workflowRoot QPushButton#primary { background:#d93722; color:white; }
            QTableWidget, QTextEdit { background:#252628; color:#f2ede4;
                border:1px solid #55565a; border-radius:6px; gridline-color:#55565a; }
            QHeaderView::section { background:#393a3d; color:#f2ede4;
                padding:4px; border:0; }
            QTableWidget::item:selected { background:#63503f; color:#fff5ef; }
            QScrollArea, QWidget#rgbBalanceContent { background:#242322; border:0; }
        """)
        self.analysis = None
        self.disposed = False
        self._work = None
        self._loaded_paths = None
        self.flatfield = None
        self._flatfield_token = None
        self._requested_flatfield_token = None
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        application = QApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self.dispose)
        if parent is not None:
            parent.destroyed.connect(self.dispose)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        body.setObjectName("rgbBalanceContent")
        body.setMinimumWidth(630)
        content = QVBoxLayout(body)
        content.setContentsMargins(4, 4, 4, 4)
        content.setSpacing(8)
        self.scroll.setWidget(body)
        outer.addWidget(self.scroll, 1)
        intro = QLabel("请专门拍摄同卷已冲洗、无画面的片基：固定相机 ISO、快门和光圈，分别拍摄红、绿、蓝三张 RAW。\n当前支持拜耳 / CFA 相机 RAW。可用一小块片基；局部模式下拖动画框，让三路都只包含干净片基。")
        intro.setWordWrap(True)
        intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content.addWidget(intro)
        files = QHBoxLayout()
        self.path_fields, self.file_buttons = [], []
        initial = [source_paths.get(c, "") for c in CHANNELS] if isinstance(source_paths, dict) else list(source_paths or ())
        for index, channel in enumerate(CHANNELS):
            field = RawReferenceButton(channel, str(initial[index]) if len(initial) == 3 else "")
            field.textChanged.connect(self._input_changed)
            field.clicked.connect(lambda checked=False, i=index: self._choose_file(i))
            files.addWidget(field, 1)
            self.path_fields.append(field)
            self.file_buttons.append(field)
        content.addLayout(files)
        controls = QHBoxLayout()
        self.mode = WorkflowComboBox()
        self.mode.addItems(("局部片基", "全幅片基"))
        self.mode.setMinimumWidth(110)
        self.mode.setAccessibleName("片基采样范围")
        self.mode.currentIndexChanged.connect(self._mode_changed)
        controls.addWidget(self.mode)
        self.reset_region = QPushButton("片基框回到中心")
        self.reset_region.clicked.connect(lambda: self.region_view.set_region(DEFAULT_REGION))
        controls.addWidget(self.reset_region)
        controls.addStretch()
        self.analyze_button = QPushButton("载入并分析")
        self.analyze_button.setObjectName("primary")
        self.analyze_button.clicked.connect(self.start_analysis)
        controls.addWidget(self.analyze_button)
        content.addLayout(controls)
        flatfield_row = QHBoxLayout()
        self.flatfield_check = ReviewCheckBox("套用平场", checkmark=True)
        self.flatfield_check.setObjectName("rgbBalanceFlatfield")
        self.flatfield_check.toggled.connect(self._toggle_flatfield)
        self.flatfield_note = QLabel("量化基准：未套用平场")
        self.flatfield_note.setWordWrap(True)
        self.flatfield_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.flatfield_choose = QPushButton("切换平场…")
        self.flatfield_choose.clicked.connect(self._choose_flatfield)
        flatfield_row.addWidget(self.flatfield_check)
        flatfield_row.addWidget(self.flatfield_note, 1)
        flatfield_row.addWidget(self.flatfield_choose)
        content.addLayout(flatfield_row)
        self.region_view = BalanceRegionView()
        self.region_view.region_changed.connect(self._region_changed)
        self.region_view.setFixedHeight(160)
        content.addWidget(self.region_view)
        note = QLabel("三路共用橙色片基框 · 预览按 RAW 方向显示 · 避开齿孔、文字、灰尘与阴影")
        note.setObjectName("muted")
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content.addWidget(note)
        self.status = QLabel("可选参考工具，不改变照片。建议值用于调整灯光后复拍确认。")
        self.status.setWordWrap(True)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content.addWidget(self.status)
        self.chart = RgbBalanceChart()
        content.addWidget(self.chart, 1)
        self.details_expander = QPushButton("数值与报告 ▸")
        self.details_expander.setCheckable(True)
        self.details_expander.toggled.connect(self._toggle_details)
        content.addWidget(self.details_expander)
        self.table = QTableWidget(3, 4)
        self.table.setHorizontalHeaderLabels(("通道", "采样信号", "建议有效曝光", "原 RAW 上限"))
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setFixedHeight(128)
        self.table.hide()
        content.addWidget(self.table)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("分析后显示拍摄设置与参考提示。")
        self.details.setMaximumHeight(110)
        self.details.setMinimumHeight(90)
        self.details.hide()
        content.addWidget(self.details)
        footer = QHBoxLayout()
        self.copy_button, self.save_button = QPushButton("复制报告"), QPushButton("保存报告…")
        self.copy_button.clicked.connect(self.copy_report)
        self.save_button.clicked.connect(self.save_report)
        self.copy_button.setEnabled(False)
        self.save_button.setEnabled(False)
        footer.addWidget(self.copy_button)
        footer.addWidget(self.save_button)
        footer.addStretch()
        close = QPushButton("关闭")
        close.clicked.connect(self.close)
        footer.addWidget(close)
        outer.addLayout(footer)
        self.set_flatfield(flatfield)

    def _toggle_details(self, expanded):
        self.table.setVisible(expanded)
        self.details.setVisible(expanded)
        self.details_expander.setText("数值与报告 ▾" if expanded else "数值与报告 ▸")

    def set_flatfield(self, config):
        """Synchronize a caller-supplied setting without clearing unchanged results."""
        config = dict(config or {})
        path = str(config.get("path") or "")
        strength = float(config.get("strength", 1.0))
        normalized = dict(path=path, strength=strength, enabled=bool(config.get("enabled", bool(path))),
                          name=str(config.get("name") or (Path(path).name if path else "")))
        try:
            stat = Path(path).stat() if path else None
            identity = (stat.st_size, stat.st_mtime_ns) if stat is not None else None
        except OSError:
            identity = None
        token = (path, strength, normalized["enabled"], normalized["name"], identity)
        previous = self._flatfield_token
        changed = token != previous
        self.flatfield, self._flatfield_token = normalized, token
        available = bool(path and Path(path).is_file())
        applied = available and normalized["enabled"] and strength > 0
        self.flatfield_check.blockSignals(True)
        self.flatfield_check.setChecked(applied)
        self.flatfield_check.blockSignals(False)
        self.flatfield_check.setEnabled(available and self._work is None)
        name = normalized["name"] or (Path(path).name if path else "")
        confirmed = bool(not changed and self.analysis is not None and
                         (getattr(self.analysis, "flatfield", None) or {}).get("applied"))
        label = (f"量化基准：{'平场后' if confirmed else '待验证平场'} · {name} · {strength:.0%}" if applied else
                 f"量化基准：未套用平场 · {name}（旁路）" if available else
                 f"量化基准：未套用平场 · {name}（平场文件不可用）" if path else "量化基准：未套用平场")
        self.flatfield_note.setText(label)
        self.flatfield_note.setToolTip(path)
        if changed:
            self._invalidate()
            if previous is not None:
                self.status.setText("平场设置已更新，请重新分析；原 RAW 上限单独检查。")

    def _toggle_flatfield(self, checked):
        config = dict(self.flatfield or {})
        config["enabled"] = bool(checked)
        self.set_flatfield(config)

    def _choose_flatfield(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择已有平场增益", "", "平场增益 (*.npz *.NPZ)")
        if path:
            self.set_flatfield(dict(path=path, strength=(self.flatfield or {}).get("strength", 1.0),
                                    enabled=True, name=Path(path).name))

    def _choose_file(self, index):
        path, _ = QFileDialog.getOpenFileName(self, f"选择 {CHANNELS[index]} 片基 RAW", "", "相机 RAW (*.rw2 *.RW2 *.dng *.DNG *.nef *.NEF *.arw *.ARW *.cr2 *.CR2 *.cr3 *.CR3 *.raf *.RAF *.orf *.ORF *.pef *.PEF *.srw *.SRW);;所有文件 (*)")
        if path:
            self.path_fields[index].setText(path)

    def _invalidate(self):
        self.analysis = None
        self.copy_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.table.clearContents()
        self.details.clear()
        self.chart.set_analysis(None)

    def _input_changed(self):
        self._invalidate()
        self._loaded_paths = None
        self.region_view.images = [None] * 3
        self.region_view.update()
        self.status.setText("参考已改变，请载入并分析。")
        self.analyze_button.setText("载入并分析")

    def _mode_changed(self):
        self.region_view.local = self.mode.currentIndex() == 0
        self.reset_region.setEnabled(self.region_view.local and self._work is None)
        self.region_view.update()
        self._region_changed()

    def _region_changed(self, *args):
        self._invalidate()
        self.status.setText("片基范围已改变，请重新分析。")
        self.analyze_button.setText("分析所选片基")

    def _busy(self, busy):
        self.analyze_button.setEnabled(not busy)
        self.mode.setEnabled(not busy)
        self.region_view.setEnabled(not busy)
        self.reset_region.setEnabled(not busy and self.mode.currentIndex() == 0)
        self.flatfield_check.setEnabled(not busy and bool((self.flatfield or {}).get("path"))
                                        and Path(self.flatfield["path"]).is_file())
        self.flatfield_choose.setEnabled(not busy)
        for control in self.path_fields + self.file_buttons:
            control.setEnabled(not busy)

    def start_analysis(self):
        if self.disposed or self._work is not None:
            return
        self._invalidate()
        paths = tuple(field.text().strip() for field in self.path_fields)
        region = self.region_view.region if self.mode.currentIndex() == 0 else FULL_REGION
        config = deepcopy(self.flatfield) if self.flatfield_check.isChecked() else None
        work = _MeterWork(paths, region, paths != self._loaded_paths, config)
        self._work = work
        self._requested_flatfield_token = self._flatfield_token
        work.signals.progress.connect(self.status.setText)
        work.signals.done.connect(self._finished)
        work.signals.failed.connect(self._failed)
        self._busy(True)
        self.status.setText("正在读取片基参考…")
        self._pool.start(work)

    def _finished(self, analysis):
        if self.disposed:
            return
        self._work = None
        self._busy(False)
        if self._requested_flatfield_token is not None and self._requested_flatfield_token != self._flatfield_token:
            self._requested_flatfield_token = None
            self.status.setText("平场设置已改变，请重新分析。")
            return
        self._requested_flatfield_token = None
        self.analysis = analysis
        self.chart.set_analysis(analysis)
        flatfield = getattr(analysis, "flatfield", None) or {}
        self.flatfield_note.setText(
            f"量化基准：平场后 · {flatfield.get('name', '')} · {flatfield.get('strength', 1.0):.0%}"
            if flatfield.get("applied") else "量化基准：未套用平场")
        self._loaded_paths = tuple(field.text().strip() for field in self.path_fields)
        if any(item.preview is not None for item in analysis.readings):
            self.region_view.set_previews(analysis.readings)
        self.status.setText(f"当前不平衡 {analysis.imbalance_percent:.2f}% · {analysis.grade}；各路采样信号以 90% 为目标。" if analysis.usable else "参考待核对；仅显示已测得信号，展开数值与报告查看原因。")
        for row, (reading, adjustment) in enumerate(zip(analysis.readings, analysis.adjustments_percent)):
            adjustment_issue = getattr(reading, "adjustment_issue", None)
            raw_saturation = getattr(reading, "raw_saturation_fraction", None)
            raw_saturation = reading.saturation_fraction if raw_saturation is None else raw_saturation
            raw_limit = getattr(reading, "raw_limit_signal", None)
            upper = ("目标超出 RAW 余量" if adjustment_issue else "原 RAW 需降低曝光" if raw_saturation > 0.002 else
                     "接近 RAW 上限" if raw_limit is not None and raw_limit >= .998 else "未见明显上限风险")
            cells = (reading.channel, "—" if reading.signal is None else f"{100 * reading.signal:.2f}%",
                     "暂不估算" if adjustment is None else f"{adjustment:+.2f}%", upper)
            notes = list(reading.issues)
            if adjustment_issue:
                notes.append(adjustment_issue)
            if raw_limit is not None:
                notes.append(f"原 RAW 各感光类别最高占用：{raw_limit:.2%}")
            for column, value in enumerate(cells):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setToolTip("\n".join(notes))
                self.table.setItem(row, column, item)
        self.details.setPlainText(analysis.report_text())
        self.copy_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.analyze_button.setText("重新分析片基")

    def _failed(self, message):
        if self.disposed:
            return
        self._work = None
        self._busy(False)
        if self.flatfield_check.isChecked():
            self.flatfield_note.setText("量化基准：未套用平场（分析未完成）")
        self.status.setText(f"暂时无法分析：{message}")

    def copy_report(self):
        if self.analysis is not None:
            QApplication.clipboard().setText(self.analysis.report_text())
            self.status.setText("分析报告已复制。")

    def save_report(self):
        if self.analysis is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存配平报告", "RGB 拍摄灯光配平.txt", "文本报告 (*.txt)")
        if path:
            try:
                Path(path).write_text(self.analysis.report_text(), encoding="utf-8")
            except OSError as error:
                self.status.setText(f"报告未保存：{error}")
            else:
                self.status.setText("分析报告已保存。")

    def dispose(self, *args):
        """Shutdown only: let this dialog's decoder finish before Qt teardown.

        Normal close merely hides the reusable tool. LibRaw cannot safely be
        interrupted mid-decode; this private pool never waits for unrelated work.
        """
        if self.disposed:
            return
        self.disposed = True
        self._pool.waitForDone()
        self._work = None


def show_rgb_balance_dialog(parent=None, source_paths=None, flatfield=None):
    """Show a nonmodal tool. Callers may retain/reuse the returned dialog."""
    dialog = RgbBalanceDialog(parent, source_paths, flatfield)
    dialog.show()
    return dialog
