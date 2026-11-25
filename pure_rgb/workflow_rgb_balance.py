"""Independent, optional RGB film-base meter, with no workflow prerequisites."""

from pathlib import Path
from copy import deepcopy
from dataclasses import replace

from PyQt6.QtCore import QIODevice, QObject, QPointF, QRectF, QRunnable, QSaveFile, Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPalette, QPen
from PyQt6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy, QTableWidget,
    QTableWidgetItem, QTextBrowser, QTextEdit, QVBoxLayout, QWidget,
)

from .rgb_balance_reference import BalanceReference, MAX_LAMP_VALUE_LENGTH
from .rgb_light_balance import (
    CHANNELS, DEFAULT_REGION, FULL_REGION, analyze_triplet, checked_region, combine_readings,
)
from .workflow_combo import WorkflowComboBox
from .workflow_glass import GlassMenu, prepare_glass, stylesheet
from .workflow_indicators import ReviewCheckBox
from .workflow_rgb_balance_chart import RgbBalanceChart


DEFAULT_CAPTURE_REGION = (0.10, 0.10, 0.90, 0.90)
DESIGN_NOTES = Path(__file__).resolve().parents[1] / "docs/RGB_LIGHT_BALANCE_BILINGUAL.md"


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
    """Shared film-base and retained-area rectangles in sensor coordinates."""

    region_changed = pyqtSignal(tuple)
    capture_region_changed = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(170)
        self.setMouseTracking(True)
        self.images = [None] * 3
        self.region = DEFAULT_REGION
        self.capture_region = DEFAULT_CAPTURE_REGION
        self.local = True
        self.capture_visible = False
        self.edit_capture = False
        self._drag = None
        self.setAccessibleName("三路共用片基与保留画幅")
        self.setToolTip("拖动橙框，把它完整放在同一块干净的空白片基内。"
                        "橙框可以在蓝框外。拖框内移动，拖边缘缩放。")

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

    def set_capture_region(self, region):
        self.capture_region = checked_region(region)
        self.update()
        self.capture_region_changed.emit(self.capture_region)

    def set_capture_tool(self, *, visible, editing):
        self.capture_visible, self.edit_capture = visible, editing
        self._drag = None
        self.update()

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

    def region_rect(self, canvas, region=None):
        x0, y0, x1, y1 = region if region is not None else self.region if self.local else FULL_REGION
        return QRectF(canvas.x() + x0 * canvas.width(), canvas.y() + y0 * canvas.height(),
                      (x1 - x0) * canvas.width(), (y1 - y0) * canvas.height())

    @staticmethod
    def _draw_region(painter, area, color, *, active, dashed=False):
        color = QColor(color)
        painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine))
        painter.drawRect(area)
        if active:
            for corner in (area.topLeft(), area.topRight(), area.bottomLeft(), area.bottomRight()):
                painter.fillRect(QRectF(corner.x() - 3, corner.y() - 3, 6, 6), color)

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
            if self.capture_visible:
                self._draw_region(painter, self.region_rect(canvas, self.capture_region), "#6cb9ff",
                                  active=self.edit_capture, dashed=True)
            self._draw_region(painter, self.region_rect(canvas), "#ffb347", active=self.local)

    @staticmethod
    def _fraction(point, canvas):
        return QPointF(max(0, min(1, (point.x() - canvas.x()) / canvas.width())),
                       max(0, min(1, (point.y() - canvas.y()) / canvas.height())))

    @staticmethod
    def _rectangle_hit(area, point):
        # Leave a draggable interior even when a selection is only a few pixels wide.
        x_reach = min(8.0, area.width() / 3)
        y_reach = min(8.0, area.height() / 3)
        if not area.adjusted(-x_reach, -y_reach, x_reach, y_reach).contains(point):
            return None
        edges = tuple(edge for edge, distance, reach in (
            ("left", point.x() - area.left(), x_reach),
            ("right", point.x() - area.right(), x_reach),
            ("top", point.y() - area.top(), y_reach),
            ("bottom", point.y() - area.bottom(), y_reach),
        ) if abs(distance) <= reach)
        return ("resize" if edges else "move"), edges

    def mousePressEvent(self, event):
        if not (self.local or self.edit_capture) or event.button() != Qt.MouseButton.LeftButton:
            return
        for index, canvas in enumerate(self.image_rects()):
            if self.images[index] is None or not canvas.contains(event.position()):
                continue
            point = event.position()
            hit = self._rectangle_hit(self.region_rect(canvas), point) if self.local else None
            capture = self.edit_capture and hit is None
            region = self.capture_region if capture else self.region
            if capture:
                hit = self._rectangle_hit(self.region_rect(canvas, region), point)
            mode, edges = hit or ("draw", ())
            self._drag = (capture, canvas, self._fraction(point, canvas), region, mode, edges)
            event.accept()
            return

    def mouseMoveEvent(self, event):
        if self._drag is None:
            return
        capture, canvas, start, original, mode, edges = self._drag
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
        region = (max(0, x0), max(0, y0), min(1, x1), min(1, y1))
        if capture:
            self.set_capture_region(region)
        else:
            self.set_region(region)

    def mouseReleaseEvent(self, event):
        self._drag = None


class _MeterSignals(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)


class _MeterWork(QRunnable):
    def __init__(self, paths, region, with_previews, flatfield=None, *,
                 capture_region=FULL_REGION, reference_confirmed=False, capture_confirmed=True):
        super().__init__()
        self.paths, self.region, self.with_previews = paths, region, with_previews
        self.flatfield = deepcopy(flatfield)
        self.capture_region = capture_region
        self.reference_confirmed = reference_confirmed
        self.capture_confirmed = capture_confirmed
        self.signals = _MeterSignals()

    def run(self):
        try:
            kwargs = dict(with_previews=self.with_previews, progress=self.signals.progress.emit,
                          capture_region=self.capture_region, reference_confirmed=self.reference_confirmed,
                          capture_confirmed=self.capture_confirmed)
            if self.flatfield is not None:
                kwargs["flatfield"] = self.flatfield
            result = analyze_triplet(self.paths, self.region, **kwargs)
        except Exception as error:
            self.signals.failed.emit(str(error))
        else:
            self.signals.done.emit(result)


class RgbBalanceDialog(QDialog):
    def __init__(self, parent=None, source_paths=None, flatfield=None, flatfield_library=()):
        super().__init__(parent)
        self.setWindowTitle("PURE RGB · RGB 拍摄灯光配平")
        self.setObjectName("workflowRoot")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.resize(850, 850)
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
        self._capture_confirmed = False
        self.saved_reference = None
        self.saved_reference_path = None
        self._design_notes_dialog = None
        self.flatfield = None
        self._project_flatfield = None
        self._flatfield_profiles = ()
        self._local_flatfield = False
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
        intro = QLabel("把同卷已冲洗、未曝光的片基放在翻拍台上，固定 ISO、快门和光圈，分别拍摄 R、G、B 三张 RAW。\n每次调灯后按同一设置复拍，用同一片基区域复测。")
        intro.setWordWrap(True)
        intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content.addWidget(intro)
        basis = QLabel(
            '参考柯达 <a style="color:#ffcc91" href="https://www.kodak.com/content/products-brochures/Film/'
            'LAD-for-KODAK-Color-Negative-Film-H-61a.pdf">D-min 片基参照</a>与'
            '<a style="color:#ffcc91" href="https://www.kodak.com/content/products-brochures/Film/'
            'Scanning-Recommendations-for-Extended-Dynamic-Range-Camera-Films-EN.pdf">扩展动态范围扫描建议</a>设计。'
        )
        basis.setOpenExternalLinks(True)
        basis.setObjectName("muted")
        basis.setWordWrap(True)
        basis.setAlignment(Qt.AlignmentFlag.AlignCenter)
        basis.setToolTip("柯达资料提供片基参照与采集原则。三色配平容差和快门试拍规则由 PURE RGB 设定。")
        basis_row = QHBoxLayout()
        basis_row.addWidget(basis, 1)
        self.design_notes_button = QPushButton("中英文说明")
        self.design_notes_button.setToolTip("在软件内阅读柯达资料的中文论证、英文对照和使用方法。")
        self.design_notes_button.clicked.connect(self.open_design_notes)
        basis_row.addWidget(self.design_notes_button)
        self.guide_button = QPushButton("配平指南")
        self.guide_button.setCheckable(True)
        self.guide_button.setToolTip("查看三色采集、画框、调整与复测步骤。")
        basis_row.addWidget(self.guide_button)
        content.addLayout(basis_row)
        self._add_balance_guide(content)
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
        self.reset_region.setToolTip("将橙框放回起始位置。复位后请将橙框拖到实际的空白片基内。")
        self.reset_region.clicked.connect(lambda: self.region_view.set_region(DEFAULT_REGION))
        controls.addWidget(self.reset_region)
        controls.addStretch()
        self.analyze_button = QPushButton("载入并分析")
        self.analyze_button.setObjectName("primary")
        self.analyze_button.clicked.connect(self.start_analysis)
        controls.addWidget(self.analyze_button)
        content.addLayout(controls)
        reference_controls = QHBoxLayout()
        self.reference_check = ReviewCheckBox("确认是同卷空白片基", checkmark=True)
        self.reference_check.setToolTip("核对三路橙框内只有同卷已冲洗的未曝光片基，避开画面、文字和齿孔。")
        self.reference_check.toggled.connect(self._reference_changed)
        reference_controls.addWidget(self.reference_check)
        reference_controls.addStretch()
        self.capture_scope = WorkflowComboBox()
        self.capture_scope.addItems(("整幅 RAW", "保留画幅"))
        self.capture_scope.setCurrentIndex(1)
        self.capture_scope.ensurePolished()
        self.capture_scope.setMinimumWidth(
            max(self.capture_scope.fontMetrics().horizontalAdvance(self.capture_scope.itemText(index))
                for index in range(self.capture_scope.count())) + 60
        )
        self.capture_scope.setAccessibleName("RAW 曝光检查范围")
        self.capture_scope.currentIndexChanged.connect(self._capture_scope_changed)
        reference_controls.addWidget(self.capture_scope)
        self.edit_capture = QPushButton("确认保留画幅")
        self.edit_capture.setCheckable(True)
        self.edit_capture.setChecked(True)
        self.edit_capture.setToolTip("在预览中拖动蓝框，指定最终要保留的画幅。")
        self.edit_capture.toggled.connect(self._capture_tool_changed)
        self.edit_capture.clicked.connect(self._capture_button_clicked)
        reference_controls.addWidget(self.edit_capture)
        content.addLayout(reference_controls)
        flatfield_row = QHBoxLayout()
        self.flatfield_check = ReviewCheckBox("套用平场", checkmark=True)
        self.flatfield_check.setObjectName("rgbBalanceFlatfield")
        self.flatfield_check.toggled.connect(self._toggle_flatfield)
        self.flatfield_note = QLabel("量化基准：未套用平场")
        self.flatfield_note.setWordWrap(True)
        self.flatfield_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.flatfield_choose = QPushButton("载入平场…")
        self.flatfield_choose.setToolTip("手动载入拍摄的平场 RAW 或已有增益 NPZ。")
        self.flatfield_choose.clicked.connect(self._choose_flatfield)
        self.flatfield_existing = QPushButton("已有平场…")
        self.flatfield_existing.setToolTip("从平场库选用，或恢复跟随当前项目平场。")
        self.flatfield_existing.clicked.connect(self._choose_existing_flatfield)
        flatfield_row.addWidget(self.flatfield_check)
        flatfield_row.addWidget(self.flatfield_note, 1)
        flatfield_row.addWidget(self.flatfield_existing)
        flatfield_row.addWidget(self.flatfield_choose)
        content.addLayout(flatfield_row)
        self.region_view = BalanceRegionView()
        self.region_view.region_changed.connect(self._region_changed)
        self.region_view.capture_region_changed.connect(self._capture_region_changed)
        self.region_view.setFixedHeight(160)
        content.addWidget(self.region_view)
        self.region_note = QLabel("橙框为片基取样范围 · 三路共用 · 按 RAW 方向显示")
        self.region_note.setObjectName("muted")
        self.region_note.setWordWrap(True)
        self.region_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content.addWidget(self.region_note)
        self.status = QLabel("载入三张片基 RAW，检查三路偏差和曝光余量。")
        self.status.setWordWrap(True)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet("font-weight: 600; padding: 3px 6px;")
        content.addWidget(self.status)
        self.chart = RgbBalanceChart()
        content.addWidget(self.chart, 1)
        self._add_capture_reference(content)
        self.details_expander = QPushButton("数值与报告 ▸")
        self.details_expander.setCheckable(True)
        self.details_expander.toggled.connect(self._toggle_details)
        content.addWidget(self.details_expander)
        self.table = QTableWidget(3, 5)
        self.table.setHorizontalHeaderLabels(("通道", "片基中位值", "相对偏差", "空间差／亮尾", "RAW 到顶点数"))
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
        self._capture_tool_changed()
        self.set_project_flatfield(flatfield, flatfield_library)

    def open_design_notes(self):
        if self._design_notes_dialog is not None:
            self._design_notes_dialog.raise_()
            self._design_notes_dialog.activateWindow()
            return
        try:
            text = DESIGN_NOTES.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            self.status.setText("无法读取中英文说明。上方的柯达原文链接仍可打开。")
            self.status.show()
            return

        dialog = QDialog(self)
        dialog.setObjectName("workflowRoot")
        dialog.setWindowTitle("PURE RGB · 配平依据与指南 / Balance references and guide")
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        dialog.resize(740, 685)
        prepare_glass(dialog)
        dialog.setStyleSheet(stylesheet(False, dark=True) + """
            QDialog#workflowRoot { background: #242322; }
            QTextBrowser { background: #252628; color: #f2ede4;
                border: 1px solid #55565a; border-radius: 8px; }
            QPushButton { background: #3b3a3e; color: #f2ede4; }
        """)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 14, 14, 12)
        viewer = QTextBrowser()
        viewer.setAccessibleName("柯达资料与 RGB 配平中英文说明")
        viewer.setOpenExternalLinks(True)
        palette = viewer.palette()
        palette.setColor(QPalette.ColorRole.Link, QColor("#ffcc91"))
        viewer.setPalette(palette)
        viewer.document().setDocumentMargin(18)
        viewer.setMarkdown(text)
        layout.addWidget(viewer, 1)
        close = QPushButton("关闭")
        close.clicked.connect(dialog.close)
        layout.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)
        self._design_notes_dialog = dialog
        dialog.finished.connect(lambda _: setattr(self, "_design_notes_dialog", None))
        dialog.show()

    def _add_balance_guide(self, content):
        self.guide_panel = QWidget()
        layout = QVBoxLayout(self.guide_panel)
        layout.setContentsMargins(8, 4, 8, 8)
        self.guide_next = QLabel()
        self.guide_next.setWordWrap(True)
        self.guide_next.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.guide_next)
        steps = (
            "1. 拍片基　用同卷已冲洗的空白片基。锁定机位、ISO、快门和光圈，仅切换红、绿、蓝灯各拍一张 RAW。",
            "2. 画范围　初始橙框用于定位。把整个橙框移入同一块干净、已冲洗的未曝光片基，避开画面、齿孔、字样和片外。"
            "如果左侧有独立的空白片基，就在左侧均匀区内取样，与影像边界留出距离。"
            "橙框可以在蓝框外。蓝框圈住最终保留的整张照片，再点「确认保留画幅」。",
            "3. 看两图　右图的 100% 是上限。遇到削顶先缩短三路快门。「可试加」供比较密部噪声。"
            "左图正偏差表示该路信号偏强，负偏差表示偏弱。"
            "确认片基后，±5% 内保持灯光，超过 5% 至 10% 先复测。"
            "显示「需调整」时按左图建议小幅调对应灯路。估算以有效曝光表示，旋钮刻度请实测。",
            "4. 复拍并保存　每次只改一种设置。调灯时保持原相机参数，改曝光时三路统一改快门。"
            "用相同两框复测，可填写灯具读数并保存报告。",
        )
        for step in steps:
            label = QLabel(step)
            label.setWordWrap(True)
            label.setObjectName("muted")
            layout.addWidget(label)
        self.guide_panel.hide()
        self.guide_button.toggled.connect(self.guide_panel.setVisible)
        content.addWidget(self.guide_panel)

    def _update_balance_guide(self):
        self.guide_next.setText(f"下一步 · {self._guide_action()}")

    def _guide_action(self):
        result = self.analysis
        if self._work is not None:
            return "正在分析三路 RAW。完成后先看右图的保留画幅曝光。"
        if result is None:
            if all(field.text() for field in self.path_fields):
                return "分析已选的三张 RAW，再核对橙框片基和蓝框画幅。"
            return "按红、绿、蓝加载同一块空白片基的三张 RAW。"
        if not result.capture_confirmed:
            return "把蓝框调整到包含整个保留画面，点击「确认保留画幅」后再看曝光建议。"
        if result.exposure_risk:
            if result.capture_region == FULL_REGION:
                return "全幅有削顶风险。若只保留照片，先圈定蓝框再分析。若保留全幅，先按右图降低曝光。"
            return "蓝框内有削顶风险。先按右图降低三路共用曝光，然后复拍检查。"
        if result.deviations_percent is None:
            return "查看左图列出的量区或拍摄设置问题，修正后再分析。"
        if not result.reference_confirmed:
            return "确认三张的橙框都只包含同一块空白片基，再勾选「确认是同卷空白片基」。"
        if result.imbalance_percent <= 5:
            return "三路配平已进入保持区。保留当前灯光比例，可记录灯具读数并保存报告。"
        if result.imbalance_percent <= 10:
            return "保持当前设置复拍一次，确认偏差是否稳定。"
        if any(value is not None for value in result.adjustments_percent):
            return "按左图建议小幅增减对应灯路，保持同一快门和两处画框，再拍三张复测。"
        return "查看左图的调整限制，先核对片基、拍摄设置与 RAW 余量，再复拍。"

    def _add_capture_reference(self, content):
        row = QHBoxLayout()
        self.reference_expander = QPushButton("拍摄参考（可选） ▸")
        self.reference_expander.setCheckable(True)
        self.reference_expander.toggled.connect(self._toggle_capture_reference)
        self.load_reference_button = QPushButton("载入参考…")
        self.load_reference_button.setToolTip("载入上一次保存的配平报告，核对灯具读数和曝光参数。")
        self.load_reference_button.clicked.connect(self.load_capture_reference)
        row.addWidget(self.reference_expander, 1)
        row.addWidget(self.load_reference_button)
        content.addLayout(row)

        self.reference_panel = QWidget()
        panel = QVBoxLayout(self.reference_panel)
        panel.setContentsMargins(4, 0, 4, 4)
        lamps = QHBoxLayout()
        lamps.addWidget(QLabel("本次灯具读数"))
        self.lamp_fields = []
        for channel in CHANNELS:
            label = QLabel(channel)
            field = QLineEdit()
            field.setMaxLength(MAX_LAMP_VALUE_LENGTH)
            field.setMinimumWidth(0)
            field.setPlaceholderText("可留空")
            field.setAccessibleName(f"{channel} 路灯具读数")
            field.setToolTip("按灯具自己的刻度填写，例如 42% 或 128。")
            field.textChanged.connect(self._lamp_values_changed)
            label.setBuddy(field)
            lamps.addWidget(label)
            lamps.addWidget(field, 1)
            self.lamp_fields.append(field)
        self.save_lamps_button = QPushButton("保存灯光参考…")
        self.save_lamps_button.setToolTip("单独保存本次 R、G、B 灯具读数。有 RAW 分析时会一并记录实拍曝光。")
        self.save_lamps_button.clicked.connect(self.save_lamp_reference)
        lamps.addWidget(self.save_lamps_button)
        panel.addLayout(lamps)
        self.current_exposure_note = QLabel()
        self.saved_reference_note = QLabel()
        self.lamp_adjustment_note = QLabel()
        self.reference_feedback = QLabel()
        for label in (self.current_exposure_note, self.saved_reference_note, self.lamp_adjustment_note, self.reference_feedback):
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setObjectName("muted")
            label.setWordWrap(True)
            panel.addWidget(label)
        self.lamp_adjustment_note.setStyleSheet("font-weight: 600;")
        self.lamp_adjustment_note.setToolTip("使用同一灯具和同一刻度。若拍摄设置或胶片改变，请用三张 RAW 重新测量。")
        self.reference_feedback.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reference_feedback.hide()
        self.reference_actions = QWidget()
        actions = QHBoxLayout(self.reference_actions)
        actions.setContentsMargins(0, 0, 0, 0)
        self.fill_reference_button = QPushButton("填入参考读数")
        self.fill_reference_button.setToolTip("把参考报告中的灯具读数填入本次表单，请核对灯具实际设置。")
        self.fill_reference_button.clicked.connect(self._fill_reference_lamps)
        clear = QPushButton("清除参考")
        clear.clicked.connect(self._clear_capture_reference)
        actions.addWidget(self.fill_reference_button)
        actions.addWidget(clear)
        actions.addStretch()
        panel.addWidget(self.reference_actions)
        self.reference_actions.hide()
        self.reference_panel.hide()
        content.addWidget(self.reference_panel)

    def _toggle_capture_reference(self, expanded):
        self.reference_panel.setVisible(expanded)
        self.reference_expander.setText("拍摄参考（可选） ▾" if expanded else "拍摄参考（可选） ▸")

    def _current_capture_reference(self):
        return BalanceReference.from_analysis(self.analysis, tuple(field.text() for field in self.lamp_fields))

    def _lamp_values_changed(self):
        self.reference_feedback.hide()
        self._update_capture_reference()

    def _update_capture_reference(self):
        lamp_values = tuple(field.text() for field in self.lamp_fields)
        self.save_lamps_button.setEnabled(any(value.strip() for value in lamp_values))
        current = None
        exposure_note = "本次曝光参数将在分析后从 RAW 自动读取。"
        if self.analysis is not None:
            try:
                current = self._current_capture_reference()
                report = current.append_to_report(self.analysis.report_text(), include_reference_data=False)
            except ValueError as error:
                current = None
                exposure_note = f"请核对拍摄参考 · {error}"
                self.details.setPlainText(self.analysis.report_text())
            else:
                exposure_note = f"本次 RAW · {current.exposure_summary_text()}"
                self.details.setPlainText(report)
        self.current_exposure_note.setText(exposure_note)
        self.copy_button.setEnabled(current is not None)
        self.save_button.setEnabled(current is not None)
        saved = self.saved_reference
        if saved is not None:
            note = f"灯光参考 · {self.saved_reference_path.name}\n{saved.summary_text()}"
            if current is not None:
                differences, complete = saved.compare_exposure(current)
                if differences:
                    note += "\n本次与参考不同 · " + "、".join(differences)
                if not complete:
                    note += "\n缺少可对照的完整 RAW 曝光记录。"
                if complete and not differences:
                    note += "\n两次 RAW 拍摄参数一致。"
            self.saved_reference_note.setText(note)
            self.fill_reference_button.setEnabled(any(saved.lamp_values))
        self.saved_reference_note.setVisible(saved is not None)
        self.reference_actions.setVisible(saved is not None)
        lamp_adjustment = saved.lamp_adjustment_text(lamp_values) if saved else ""
        self.lamp_adjustment_note.setText(f"按参考调整灯具（原刻度） · {lamp_adjustment}" if lamp_adjustment else "")
        self.lamp_adjustment_note.setVisible(bool(lamp_adjustment))

    def load_capture_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, "载入拍摄参考", "", "配平报告 (*.txt);;所有文件 (*)")
        if not path:
            return
        self.reference_expander.setChecked(True)
        try:
            reference = BalanceReference.read_report(path)
        except (OSError, ValueError) as error:
            self.reference_feedback.setText(f"参考未载入 · {error}")
            self.reference_feedback.show()
        else:
            self._set_saved_reference(reference, path)
        self.scroll.ensureWidgetVisible(self.reference_panel)

    def save_lamp_reference(self):
        if not self.save_lamps_button.isEnabled():
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存灯光参考", "RGB 灯光参考.txt", "文本参考 (*.txt)")
        if not path:
            return
        try:
            if self.analysis is not None:
                reference = self._current_capture_reference()
            else:
                reference = BalanceReference.from_lamps(tuple(field.text() for field in self.lamp_fields))
            self._write_report_file(path, reference.append_to_report("PURE RGB · RGB 灯光参考"))
        except (OSError, ValueError) as error:
            self.reference_feedback.setText(f"灯光参考未保存 · {error}")
            self.reference_feedback.show()
        else:
            self._set_saved_reference(reference, path)
            self.reference_feedback.setText("当前灯具读数已保存为参考。")
            self.reference_feedback.show()
        self.scroll.ensureWidgetVisible(self.reference_panel)

    def _set_saved_reference(self, reference, path):
        self.saved_reference = reference
        self.saved_reference_path = Path(path)
        self.reference_feedback.hide()
        self._update_capture_reference()

    def _fill_reference_lamps(self):
        if self.saved_reference is not None:
            for field, value in zip(self.lamp_fields, self.saved_reference.lamp_values):
                if value is not None:
                    field.setText(value)

    def _clear_capture_reference(self):
        self.saved_reference = None
        self.saved_reference_path = None
        self.reference_feedback.hide()
        self._update_capture_reference()

    def _toggle_details(self, expanded):
        self.table.setVisible(expanded)
        self.details.setVisible(expanded)
        self.details_expander.setText("数值与报告 ▾" if expanded else "数值与报告 ▸")

    def set_project_flatfield(self, config, library=()):
        self._project_flatfield = deepcopy(config)
        self._flatfield_profiles = tuple(
            entry for entry in library
            if entry.get("gain_path") and Path(entry["gain_path"]).is_file()
        )
        if self._local_flatfield:
            self._update_flatfield_choices()
        else:
            self.set_flatfield(config)

    def _update_flatfield_choices(self):
        self.flatfield_existing.setEnabled(self._work is None and bool(
            self._local_flatfield or self._project_flatfield or self._flatfield_profiles
        ))

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
        self._update_flatfield_choices()
        if changed:
            self._invalidate()
            if previous is not None:
                self.status.setText("平场设置已更新，请重新分析；原 RAW 上限单独检查。")

    def _toggle_flatfield(self, checked):
        config = dict(self.flatfield or {})
        config["enabled"] = bool(checked)
        self.set_flatfield(config)

    def _choose_flatfield(self):
        from .flatfield import FILE_FILTER

        path, _ = QFileDialog.getOpenFileName(self, "载入平场 RAW 或增益 NPZ", "", FILE_FILTER)
        if path:
            self._select_local_flatfield(path, Path(path).name)

    def _select_local_flatfield(self, path, name):
        current, inherited = self.flatfield or {}, self._project_flatfield or {}
        strength = next((item.get("strength", 1.0) for item in (current, inherited)
                         if item.get("path") == path), 1.0)
        self._local_flatfield = True
        self.set_flatfield(dict(path=path, name=name, strength=strength, enabled=True))

    def _follow_project_flatfield(self):
        self._local_flatfield = False
        self.set_flatfield(self._project_flatfield)

    def _choose_existing_flatfield(self):
        menu = GlassMenu(self.flatfield_existing)
        project = self._project_flatfield or {}
        name = project.get("name") or "未启用"
        follow = menu.addAction(f"跟随项目平场 · {name}", self._follow_project_flatfield)
        follow.setEnabled(bool(project or self._local_flatfield))
        if self._flatfield_profiles:
            menu.addSeparator()
        for entry in self._flatfield_profiles:
            path = str(entry["gain_path"])
            name = str(entry.get("name") or Path(path).name)
            label = " · ".join(str(part) for part in (name, *(entry.get("camera") or ())) if part)
            action = menu.addAction(label, lambda checked=False, p=path, n=name: self._select_local_flatfield(p, n))
            action.setToolTip(path)
        menu.exec(self.flatfield_existing.mapToGlobal(self.flatfield_existing.rect().bottomLeft()))
        menu.deleteLater()

    def _choose_file(self, index):
        path, _ = QFileDialog.getOpenFileName(self, f"选择 {CHANNELS[index]} 片基 RAW", "", "相机 RAW (*.rw2 *.RW2 *.dng *.DNG *.nef *.NEF *.arw *.ARW *.cr2 *.CR2 *.cr3 *.CR3 *.raf *.RAF *.orf *.ORF *.pef *.PEF *.srw *.SRW);;所有文件 (*)")
        if path:
            self.path_fields[index].setText(path)

    def _invalidate(self):
        self.analysis = None
        self.status.show()
        self.copy_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.table.clearContents()
        self.details.clear()
        self.chart.set_analysis(None)
        self._update_capture_reference()
        self._update_balance_guide()

    def _forget_reference(self):
        previous = self.reference_check.blockSignals(True)
        self.reference_check.setChecked(False)
        self.reference_check.blockSignals(previous)

    def _input_changed(self):
        self._forget_reference()
        self._capture_confirmed = False
        self._loaded_paths = None
        self.region_view.images = [None] * 3
        self.edit_capture.setChecked(self.capture_scope.currentIndex() == 1)
        self._capture_tool_changed()
        self._invalidate()
        self.status.setText("参考已改变，请载入并分析。")
        self.analyze_button.setText("载入并分析")

    def _mode_changed(self):
        self.region_view.local = self.mode.currentIndex() == 0
        self.reset_region.setEnabled(self.region_view.local and self._work is None)
        self.region_view.update()
        self._region_changed()

    def _region_changed(self, *args):
        self._forget_reference()
        self._invalidate()
        self.status.setText("片基范围已改变，请重新分析。")
        self.analyze_button.setText("分析所选片基")

    def _capture_scope_changed(self):
        self.edit_capture.setChecked(self.capture_scope.currentIndex() == 1)
        self._capture_tool_changed()
        self._capture_region_changed()

    def _capture_tool_changed(self):
        visible = self.capture_scope.currentIndex() == 1
        if not visible and self.edit_capture.isChecked():
            self.capture_scope.setCurrentIndex(1)
            return
        editing = visible and self.edit_capture.isChecked()
        has_previews = all(image is not None for image in self.region_view.images)
        self.edit_capture.setEnabled(self._work is None and (not visible or has_previews))
        self.edit_capture.setText("确认保留画幅" if editing else "调整保留画幅" if visible else "圈定保留画幅")
        self.region_view.set_capture_tool(visible=visible, editing=editing)
        if editing:
            note = "橙框完全放入干净空白片基，可在蓝框外。蓝框圈住要保留的整张照片，确认后显示曝光建议。"
        elif visible:
            note = "橙框用于片基配平 · 蓝框决定 RAW 曝光检查和快门建议"
        else:
            note = "橙框为片基取样范围 · 三路共用 · 按 RAW 方向显示"
        self.region_note.setText(note)

    def _capture_button_clicked(self, editing):
        self._capture_confirmed = not editing
        if self.analysis is not None:
            self._refresh_confirmation(self.reference_check.isChecked())
        elif self._capture_confirmed:
            self.start_analysis()

    def _capture_region_changed(self, *args):
        self._capture_confirmed = False
        if self.capture_scope.currentIndex() == 1:
            self.edit_capture.setChecked(True)
        self._invalidate()
        self.status.setText("RAW 检查范围已改变，请重新分析。")
        self.analyze_button.setText("重新分析")

    def _reference_changed(self, confirmed):
        self._refresh_confirmation(confirmed)

    def _refresh_confirmation(self, reference_confirmed):
        if self.analysis is None:
            return
        current = self.analysis
        updated = combine_readings(current.readings, current.region, capture_region=current.capture_region,
                                   reference_confirmed=reference_confirmed,
                                   capture_confirmed=self.capture_scope.currentIndex() == 0 or self._capture_confirmed)
        self._show_analysis(replace(updated, flatfield=current.flatfield))

    def _busy(self, busy):
        self.analyze_button.setEnabled(not busy)
        self.mode.setEnabled(not busy)
        self.region_view.setEnabled(not busy)
        self.reference_check.setEnabled(not busy)
        self.capture_scope.setEnabled(not busy)
        self._capture_tool_changed()
        self.reset_region.setEnabled(not busy and self.mode.currentIndex() == 0)
        self.flatfield_check.setEnabled(not busy and bool((self.flatfield or {}).get("path"))
                                        and Path(self.flatfield["path"]).is_file())
        self.flatfield_choose.setEnabled(not busy)
        self._update_flatfield_choices()
        for control in self.path_fields + self.file_buttons:
            control.setEnabled(not busy)

    def start_analysis(self):
        if self.disposed or self._work is not None:
            return
        self._invalidate()
        paths = tuple(field.text().strip() for field in self.path_fields)
        region = self.region_view.region if self.mode.currentIndex() == 0 else FULL_REGION
        full_frame = self.capture_scope.currentIndex() == 0
        capture = FULL_REGION if full_frame else self.region_view.capture_region
        config = deepcopy(self.flatfield) if self.flatfield_check.isChecked() else None
        work = _MeterWork(paths, region, paths != self._loaded_paths, config, capture_region=capture,
                          reference_confirmed=self.reference_check.isChecked(),
                          capture_confirmed=full_frame or self._capture_confirmed)
        self._work = work
        self._requested_flatfield_token = self._flatfield_token
        work.signals.progress.connect(self.status.setText)
        work.signals.done.connect(self._finished)
        work.signals.failed.connect(self._failed)
        self._busy(True)
        self.status.setText("正在读取片基参考…")
        self._update_balance_guide()
        self._pool.start(work)

    def _finished(self, analysis):
        if self.disposed:
            return
        self._work = None
        self._busy(False)
        if self._requested_flatfield_token is not None and self._requested_flatfield_token != self._flatfield_token:
            self._requested_flatfield_token = None
            self.status.setText("平场设置已改变，请重新分析。")
            self._update_balance_guide()
            return
        self._requested_flatfield_token = None
        flatfield = getattr(analysis, "flatfield", None) or {}
        self.flatfield_note.setText(
            f"量化基准：平场后 · {flatfield.get('name', '')} · {flatfield.get('strength', 1.0):.0%}"
            if flatfield.get("applied") else "量化基准：未套用平场")
        self._loaded_paths = tuple(field.text().strip() for field in self.path_fields)
        if any(item.preview is not None for item in analysis.readings):
            self.region_view.set_previews(analysis.readings)
        self._capture_tool_changed()
        self._show_analysis(analysis)

    def _show_analysis(self, analysis):
        self.analysis = analysis
        self.chart.set_analysis(analysis)
        self.status.setText(analysis.status)
        self.status.hide()
        deviations = analysis.deviations_percent or (None, None, None)
        for row, (reading, deviation) in enumerate(zip(analysis.readings, deviations)):
            middle = "—" if reading.median is None else f"{reading.median:.2%}"
            spatial = "—" if reading.spatial_range is None else f"{reading.spatial_range:.1%}"
            tail = "—" if reading.bright_tail is None else f"{reading.bright_tail:.1%}"
            count = "—" if reading.capture is None else f"{reading.capture.clipped_count:,}"
            cells = (reading.channel, middle, "—" if deviation is None else f"{deviation:+.2f}%",
                     f"{spatial} / {tail}", count)
            notes = [*reading.issues, *reading.warnings]
            if reading.adjustment_issue:
                notes.append(reading.adjustment_issue)
            if reading.capture is not None and reading.capture.limit_signal is not None:
                notes.append(f"检查范围原 RAW 各感光类别最高 P99.9 为 {reading.capture.limit_signal:.2%}")
            for column, value in enumerate(cells):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setToolTip("\n".join(notes))
                self.table.setItem(row, column, item)
        self._update_capture_reference()
        self._update_balance_guide()
        self.analyze_button.setText("重新分析")

    def _failed(self, message):
        if self.disposed:
            return
        self._work = None
        self._busy(False)
        if self.flatfield_check.isChecked():
            self.flatfield_note.setText("量化基准：未套用平场（分析未完成）")
        self.status.show()
        self.status.setText(f"暂时无法分析：{message}")
        self._update_balance_guide()

    def copy_report(self):
        if self.analysis is not None and self.copy_button.isEnabled():
            QApplication.clipboard().setText(self.details.toPlainText())
            self.status.show()
            self.status.setText("分析报告已复制。")

    def save_report(self):
        if self.analysis is None or not self.save_button.isEnabled():
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存配平报告", "RGB 拍摄灯光配平.txt", "文本报告 (*.txt)")
        if path:
            self.status.show()
            try:
                report = self._current_capture_reference().append_to_report(self.analysis.report_text())
                self._write_report_file(path, report)
            except (OSError, ValueError) as error:
                self.status.setText(f"报告未保存：{error}")
            else:
                self.status.setText("分析报告已保存，可在下次载入拍摄参考。")

    @staticmethod
    def _write_report_file(path, report):
        content = report.encode("utf-8")
        output = QSaveFile(path)
        if not output.open(QIODevice.OpenModeFlag.WriteOnly) or output.write(content) != len(content) or not output.commit():
            raise OSError(output.errorString())

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


def show_rgb_balance_dialog(parent=None, source_paths=None, flatfield=None, flatfield_library=()):
    """Show a nonmodal tool. Callers may retain/reuse the returned dialog."""
    dialog = RgbBalanceDialog(parent, source_paths, flatfield, flatfield_library)
    dialog.show()
    return dialog
