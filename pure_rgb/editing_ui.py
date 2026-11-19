"""Photo-linked sidebar: switch tools, keep the canvas visible."""

from PyQt6.QtCore import pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QScrollArea,
    QLabel,
    QPushButton,
)

from .controls_ui import ControlPage
from .positive_post import POST_SPECS, checked_post
from .workflow_disclosure import DisclosureButton
from .resource_ui import FlexResourceSelector
from .film_ui import NativeFilmSelector
from .professional_controls import (
    NATIVE_SPECS,
    OUTPUT_SPECS,
    ENGINE_SPECS,
    checked_engines,
)


class EditingSidebar(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.loading, self.mode = False, "native"
        self.native_pages, self.output_pages, self.engine_pages = [], {}, {}
        self.post_pages = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.content = QWidget()
        self.column = QVBoxLayout(self.content)
        self.column.setContentsMargins(2, 2, 2, 2)
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll, 1)
        self.groups, self.positions = {}, {}
        self.pages = {}
        self.flex_resource = FlexResourceSelector()
        self.native_film = NativeFilmSelector()
        self.column.addWidget(self.native_film)
        self.native_film.changed.connect(self.edited)
        for mode in ("native", "cool-shared-rgb", "flex-endpoint"):
            page = self.page(OUTPUT_SPECS, "")
            self.output_pages[mode] = page
            self.pages[mode] = [("曝光与色彩", page)]
            self.post_pages[mode] = []
            for group in (
                "后期 · 几何",
                "后期 · 白平衡",
                "后期 · 色调与肩趾",
                "后期 · RGB 曲线",
                "后期 · 色度与细节",
                "后期 · 分离色调",
                "后期 · 暗角",
            ):
                page = self.page(
                    tuple(s for s in POST_SPECS if s.group == group),
                    "三路共用的正像后期；默认旁路。肩趾改变呈现，不会恢复已截断数据。",
                )
                self.post_pages[mode].append(page)
                self.pages[mode].append((group, page))
        for group in dict.fromkeys(s.group for s in NATIVE_SPECS):
            page = self.page(tuple(s for s in NATIVE_SPECS if s.group == group), "")
            self.native_pages.append(page)
            self.pages["native"].append((group, page))
        for mode, specs in ENGINE_SPECS.items():
            page = self.page(
                specs,
                "Flex 参数作用于反演前；曲线微调保持原拟合参考。"
                if mode == "flex-endpoint"
                else "密度与拟合曲线微调；不是 Nikon CMS 参数。",
            )
            self.engine_pages[mode] = page
            # Engine-specific input/curve controls precede common positive post.
            self.pages[mode].insert(0, ("引擎曲线", page))
        # The resource entry stays in the same scroll column as its engine controls.
        self.column.addWidget(self.flex_resource)
        self.flex_resource.changed.connect(self.profile_changed)
        for mode, pages in self.pages.items():
            for index, (title, page) in enumerate(pages):
                group = QWidget()
                group_layout = QVBoxLayout(group)
                group_layout.setContentsMargins(0, 0, 0, 0)
                header = DisclosureButton(title, expanded=True)
                header.setAccessibleName("展开／收起 " + title)
                body = page.takeWidget()
                header.toggled.connect(body.setVisible)
                group_layout.addWidget(header)
                group_layout.addWidget(body)
                self.column.addWidget(group)
                self.groups[(mode, index)] = (group, header)
        self.column.addStretch()
        self.note = QLabel("滑动即可预览；保存时生成完整分辨率新版本。")
        self.note.setWordWrap(True)
        self.note.setObjectName("muted")
        layout.addWidget(self.note)
        self.save = QPushButton("保存调色版本")
        self.save.setObjectName("primary")
        layout.addWidget(self.save)
        self.set_mode("native")

    def page(self, specs, note):
        page = ControlPage(specs, note)
        # Labels above their controls fit a narrow sidebar without stealing canvas space.
        from PyQt6.QtWidgets import QFormLayout

        form = page.widget().layout().itemAt(1).layout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        page.widget().layout().setContentsMargins(2, 2, 2, 2)
        for control in page.controls.values():
            if hasattr(control, "spin"):
                control.spin.setFixedWidth(70)
        page.changed.connect(self.edited)
        page.setParent(self)
        page.hide()
        return page

    def edited(self):
        if not self.loading:
            self.changed.emit()

    def profile_changed(self):
        self.loading = True
        try:
            defaults = (
                self.flex_resource.report["defaults"]
                if self.flex_resource.report
                else {}
            )
            self.engine_pages["flex-endpoint"].set_values(defaults)
        finally:
            self.loading = False
        self.edited()

    def set_mode(self, mode):
        if getattr(self, "shown_mode", None) == mode:
            return
        if getattr(self, "shown_mode", None):
            self.positions[self.shown_mode] = self.scroll.verticalScrollBar().value()
        self.mode = mode
        self.shown_mode = mode
        self.flex_resource.setVisible(mode == "flex-endpoint")
        self.native_film.setVisible(mode == "native")
        for (key, _), (group, header) in self.groups.items():
            group.setVisible(key == mode)
        QTimer.singleShot(
            0,
            lambda: self.scroll.verticalScrollBar().setValue(
                self.positions.get(mode, 0)
            ),
        )

    def focus_group(self, index):
        group, header = self.groups[(self.mode, index)]
        header.setChecked(True)
        self.scroll.ensureWidgetVisible(header, 0, 8)

    def set_documents(self, recipe):
        self.loading = True
        try:
            self.flex_resource.set_resource(recipe.get("flex_profile"))
            self.native_film.set_document(recipe.get("native_film"))
            for page in self.native_pages:
                page.set_values(recipe.get("native_controls", {}).get("values", {}))
            for mode, page in self.output_pages.items():
                page.set_values(
                    recipe.get("output_controls", {}).get(mode, {}).get("values", {})
                )
            for mode, page in self.engine_pages.items():
                defaults = (
                    self.flex_resource.report["defaults"]
                    if mode == "flex-endpoint" and self.flex_resource.report
                    else {}
                )
                page.set_values(
                    {
                        **defaults,
                        **recipe.get("engine_controls", {})
                        .get(mode, {})
                        .get("values", {}),
                    }
                )
            for mode, pages in self.post_pages.items():
                for page in pages:
                    page.set_values(
                        recipe.get("post_controls", {}).get(mode, {}).get("values", {})
                    )
        finally:
            self.loading = False

    def documents(self):
        values = {}
        for page in self.native_pages:
            values.update(page.values())
        result = {}
        if self.native_film.value() is not None:
            result["native_film"] = self.native_film.value()
        if values:
            result["native_controls"] = {"version": 1, "values": values}
        outputs = {
            mode: {"version": 1, "values": page.values()}
            for mode, page in self.output_pages.items()
            if page.values()
        }
        if outputs:
            result["output_controls"] = outputs
        engines = {
            mode: {"version": 1, "values": page.values()}
            for mode, page in self.engine_pages.items()
        }
        if self.flex_resource.resource:
            result["flex_profile"] = self.flex_resource.resource
            engine = engines["flex-endpoint"]
            engine["version"] = 2
            for key in (
                "film_gamma",
                "brightness",
                "contrast",
                "black_code",
                "white_code",
            ):
                engine["values"][key] = (
                    self.engine_pages["flex-endpoint"].controls[key].value()
                )
        engines = checked_engines(engines)
        if engines:
            result["engine_controls"] = engines
        posts = {}
        for mode, pages in self.post_pages.items():
            values = {}
            for page in pages:
                values.update(page.values())
            posts[mode] = {"version": 1, "values": values}
        posts = checked_post(posts)
        if posts:
            result["post_controls"] = posts
        return result
