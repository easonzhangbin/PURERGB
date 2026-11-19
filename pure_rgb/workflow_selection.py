"""Draw source-coordinate selections on a current positive or source locator."""

from copy import deepcopy
import math
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath, QPainterPathStroker, QPen, QPixmap, QTransform
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QGraphicsItem,
    QGraphicsPathItem,
    QToolTip,
)

from .metering_hover_v45 import grid_hover, hover_text
from .selection_mapping_v43 import SelectionMapping
from .workflow_crop_handles import CropHandles


_UNCHANGED = object()


class _GridClip(QGraphicsPathItem):
    def shape(self):
        return self.path()


class SelectionView(QGraphicsView):
    selected = pyqtSignal(object)
    zone_selected = pyqtSignal(object, object)
    crop_rotated = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#17181a"))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMinimumSize(240, 220)
        self.geometry_info = None
        self.source_photo_id = None
        self.drawing_enabled = True
        self.shape_kind = "rounded_rectangle"
        self.points = []
        self.temporary = None
        self.overlays = []
        self.grid_document = None
        self.mapping = None
        self.temporary_shape = None
        self._fit_mode = True
        self.hover_model = dict(regions=[], grid=None, fresh=False)
        self.hover_region = None
        self._hover_cache = None
        self.selected_zone = None
        self.crop_handles = CropHandles(self)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def set_geometry(self, geometry, overlays=(), grid=None, *, source_id=None, positive=False):
        self.crop_handles.configure(None, 0)
        pixmap = QPixmap(geometry["positive_preview" if positive else "preview"])
        if pixmap.isNull():
            raise ValueError("完整源定位图不可用")
        mapping = SelectionMapping((geometry["height"], geometry["width"], 3))
        self.source_photo_id = source_id
        self._set_frame(pixmap, mapping, dict(geometry), overlays, grid, preserve_view=False)

    def refresh_source_overlays(self, overlays=(), grid=None):
        """Replace committed source outlines while retaining locator and stroke.

        Crop edits must keep pixels outside their old bounds reachable. This
        refresh deliberately never substitutes a cropped or rotated positive.
        """
        if (not self.geometry_info or self.geometry_info.get("preview_kind") == "positive"
                or self.mapping is None):
            return False
        self._set_frame(self.pixmap, self.mapping, dict(self.geometry_info), overlays, grid,
                        preserve_view=True)
        return True

    def set_preview(self, image, source_shape, output_roi=None, output_orientation=None,
                    overlays=None, grid=_UNCHANGED, preserve_view=True, *, crop_locator=False):
        """Draw on the current positive without changing the metering domain.

        The caller must exclude non-D4 common post geometry. Unsupported
        geometry belongs on the original locator, never a guessed transform.
        Passing no overlays/grid retains their current source-coordinate data.
        """
        if not isinstance(image, QImage) or image.isNull():
            raise ValueError("当前正像预览不可用")
        self.source_photo_id = None
        mapping = SelectionMapping(tuple(source_shape), output_roi, output_orientation)
        geometry = dict(width=source_shape[1], height=source_shape[0],
                        source_shape=list(source_shape),
                        preview_kind="crop-positive" if crop_locator else "positive",
                        output_roi=list(mapping.output_roi),
                        output_orientation=deepcopy(mapping.output_orientation))
        self._set_frame(QPixmap.fromImage(image), mapping, geometry,
                        self.overlays if overlays is None else overlays,
                        self.grid_document if grid is _UNCHANGED else grid,
                        preserve_view=preserve_view)

    def _set_frame(self, pixmap, mapping, geometry, overlays, grid, *, preserve_view):
        self._hide_hover()
        self._hover_cache = None
        previous_center = previous_scale = None
        previous_fit = self._fit_mode
        if preserve_view and self.mapping is not None:
            center = self.mapToScene(self.viewport().rect().center())
            previous_center = self.mapping.preview_to_source(
                (center.x(), center.y()), (self.pixmap.width(), self.pixmap.height()))
            previous_scale = self.transform().m11() * self.pixmap.width() / self.mapping.output_size[0]
        points = self.points if preserve_view else []
        temporary_shape = self.temporary_shape if preserve_view else None
        self.geometry_info, self.pixmap, self.mapping = geometry, pixmap, mapping
        self.scene().clear()
        self.grid_clip = None
        self.image_item = self.scene().addPixmap(pixmap)
        clip_path = QPainterPath()
        clip_path.addRect(QRectF(pixmap.rect()))
        self.overlay_root = _GridClip(clip_path)
        self.overlay_root.setPen(QPen(Qt.PenStyle.NoPen))
        self.overlay_root.setFlag(QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape, True)
        self.scene().addItem(self.overlay_root)
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        self.points, self.temporary = points, None
        self.temporary_shape = temporary_shape
        self.overlays, self.grid_document = deepcopy(list(overlays)), deepcopy(grid)
        for shape, color in (() if geometry.get("preview_kind") == "crop-positive" else self.overlays):
            pen = QPen(QColor(color), 1.8)
            pen.setCosmetic(True)
            if color == "#e7c06e":
                pen.setStyle(Qt.PenStyle.DashLine)
            item = QGraphicsPathItem(self.path(shape), self.overlay_root)
            item.setPen(pen)
        if grid:
            roi, meter, automatic_excluded = grid
            clip = self.path(dict(type="rectangle", bounds=roi))
            review_grid = bool((self.hover_model.get("grid") or {}).get("review"))
            if not review_grid and meter.get("include"):
                included = QPainterPath()
                for shape in meter["include"]:
                    included = included.united(self.path(shape))
                clip = clip.intersected(included)
            for shape in (() if review_grid else meter.get("exclude", [])):
                clip = clip.subtracted(self.path(shape))
            exclusions = [] if review_grid else meter.get("excluded_rois", []) + automatic_excluded
            for bounds in exclusions:
                a, b, c, d = bounds
                padding = meter.get("padding", 0)
                expanded = [a - padding, b - padding, c + padding, d + padding]
                clip = clip.subtracted(
                    self.path(dict(type="rectangle", bounds=expanded))
                )
            parent = _GridClip(clip, self.overlay_root)
            parent.setPen(QPen(Qt.PenStyle.NoPen))
            parent.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape, True
            )
            self.grid_clip = parent
            lines = QPainterPath()
            x0, y0, x1, y1 = roi
            rows, columns = meter.get("zones", [8, 12])
            for col in range(1, columns):
                x = x0 + (x1 - x0) * col // columns
                lines.moveTo(*self.mapping.source_to_preview((x, y0), (pixmap.width(), pixmap.height())))
                lines.lineTo(*self.mapping.source_to_preview((x, y1), (pixmap.width(), pixmap.height())))
            for row in range(1, rows):
                y = y0 + (y1 - y0) * row // rows
                lines.moveTo(*self.mapping.source_to_preview((x0, y), (pixmap.width(), pixmap.height())))
                lines.lineTo(*self.mapping.source_to_preview((x1, y), (pixmap.width(), pixmap.height())))
            item = QGraphicsPathItem(lines, parent)
            pen = QPen(QColor(98, 204, 223, 90), 0.7)
            pen.setCosmetic(True)
            item.setPen(pen)
        if self.points and (self.shape_kind != "polygon" or len(self.points) >= 3):
            self.temporary_shape = self.document()
        if self.temporary_shape is not None:
            self._paint_temporary(self.temporary_shape)
        if previous_center is not None and not previous_fit:
            zoom = previous_scale * mapping.output_size[0] / pixmap.width()
            self.setTransform(QTransform.fromScale(zoom, zoom))
            center = mapping.source_to_preview(previous_center, (pixmap.width(), pixmap.height()))
            self.centerOn(*center)
            self._fit_mode = False
        else:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._fit_mode = True

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        self.crop_handles.paint(painter)
        grid = self.hover_model.get("grid") or {}
        records = grid.get("records", ()) if grid.get("review") else ()
        selected = self.selected_zone if grid.get("review") else None
        manual_exclusions = grid.get("excluded_rectangles", ())
        if self.mapping is None or not (records or selected or manual_exclusions):
            return
        transform = painter.worldTransform()
        painter.save()
        painter.resetTransform()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = painter.font()
        font.setPixelSize(11)
        painter.setFont(font)
        for bounds in manual_exclusions:
            shape = dict(type="rectangle", bounds=bounds)
            box = transform.mapRect(self.path(shape).boundingRect())
            painter.fillRect(box, QColor(200, 51, 42, 115))
            painter.setPen(QPen(QColor(249, 108, 69, 215), 1))
            painter.drawRect(box.adjusted(0.5, 0.5, -0.5, -0.5))
        for record in records:
            shape = dict(type="rectangle", bounds=record["source_xyxy"])
            cell = self.path(shape).boundingRect()
            if not rect.intersects(cell):
                continue
            box = transform.mapRect(cell)
            median = record["median_log_rgb"]
            excluded = record["source_xyxy"] in manual_exclusions
            if median is None and not excluded:
                painter.fillRect(box, QColor(220, 100, 65, 90))
            if box.width() < 30 or box.height() < 16:
                continue
            label = "排除" if excluded else f"{-sum(median) / 3:.2f}" if median else "—"
            badge = QRectF(box.center().x() - 15, box.center().y() - 8, 30, 16)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(13, 16, 18, 150))
            painter.drawRoundedRect(badge, 4, 4)
            painter.setPen(QColor("#f3eee9"))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, label)
        if selected:
            shape = dict(type="rectangle", bounds=selected["bounds"])
            box = transform.mapRect(self.path(shape).boundingRect())
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#ffb478"), 2))
            painter.drawRect(box.adjusted(1, 1, -1, -1))
        painter.restore()

    def set_hover_model(self, model):
        """Install display-only definitions; no recipe edits or analysis signals."""
        self._hide_hover()
        self.hover_model = deepcopy(model or dict(regions=[], grid=None, fresh=False))
        self._hover_cache = None

    def _hide_hover(self):
        if self.hover_region is not None:
            QToolTip.hideText()
        self.hover_region = None

    def hover_at(self, position):
        """Hit visible source boundaries with a constant seven screen-pixel halo."""
        if not self.geometry_info or self.mapping is None or self.points:
            return None
        position = QPointF(position)
        transform = self.viewportTransform()
        inverse, valid = transform.inverted()
        if not valid:
            return None
        scene_point = inverse.map(position)
        if not QRectF(self.pixmap.rect()).contains(scene_point):
            return None
        key = (transform.m11(), transform.m12(), transform.m21(), transform.m22(),
               transform.dx(), transform.dy())
        if self._hover_cache is None or self._hover_cache[0] != key:
            stroker = QPainterPathStroker()
            stroker.setWidth(14)
            priority = {"manual_exclusion": 0, "protected": 1,
                        "auto_exclusion": 2, "auto_candidate": 2,
                        "include": 3, "roi": 4}
            regions = sorted(self.hover_model.get("regions", []),
                             key=lambda region: priority.get(region["kind"], 5))
            paths = [(region, stroker.createStroke(transform.map(self.path(region["shape"]))))
                     for region in regions]
            self._hover_cache = key, paths
        for region, border in self._hover_cache[1]:
            if border.contains(position):
                result = deepcopy(region)
                grid = self.hover_model.get("grid")
                if grid and grid.get("review"):
                    source = self.mapping.preview_to_source(
                        (scene_point.x(), scene_point.y()), (self.pixmap.width(), self.pixmap.height()))
                    zone = grid_hover(grid, source)
                    if zone:
                        result["details"] += [zone["title"], *zone["details"]]
                return result
        if self.grid_clip is not None and self.grid_clip.path().contains(scene_point):
            source = self.mapping.preview_to_source(
                (scene_point.x(), scene_point.y()), (self.pixmap.width(), self.pixmap.height()))
            return grid_hover(self.hover_model.get("grid"), source)
        return None

    def _show_hover(self, event):
        region = self.hover_at(event.position())
        if region == self.hover_region:
            return
        self._hide_hover()
        if region is not None:
            self.hover_region = region
            QToolTip.showText(self.viewport().mapToGlobal(event.position().toPoint() + QPoint(12, 18)),
                              hover_text(region), self.viewport())

    def _source_transform(self):
        size = self.pixmap.width(), self.pixmap.height()
        zero = self.mapping.source_to_preview((0, 0), size)
        x = self.mapping.source_to_preview((1, 0), size)
        y = self.mapping.source_to_preview((0, 1), size)
        return QTransform(x[0] - zero[0], x[1] - zero[1],
                          y[0] - zero[0], y[1] - zero[1], zero[0], zero[1])

    def path(self, shape):
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.OddEvenFill)
        if shape["type"] == "polygon":
            points = [QPointF(x, y) for x, y in shape["points"]]
            path.moveTo(points[0])
            for point in points[1:]:
                path.lineTo(point)
            path.closeSubpath()
        else:
            x0, y0, x1, y1 = shape["bounds"]
            rect = QRectF(x0, y0, x1 - x0, y1 - y0)
            if shape["type"] == "ellipse":
                path.addEllipse(rect)
            elif shape["type"] == "rounded_rectangle":
                path.addRoundedRect(rect, shape["radius"], shape["radius"])
            else:
                path.addRect(rect)
        return self._source_transform().map(path)

    def source_point(self, event, *, clamp=True):
        inverse, valid = self.viewportTransform().inverted()
        if not valid:
            raise ValueError("当前选区视图无法映射回原始坐标")
        point = inverse.map(event.position())
        return list(self.mapping.preview_to_source(
            (point.x(), point.y()), (self.pixmap.width(), self.pixmap.height()), clamp=clamp))

    def _paint_temporary(self, shape):
        if self.temporary is None:
            pen = QPen(QColor("#62ccdf"), 2)
            pen.setCosmetic(True)
            self.temporary = QGraphicsPathItem(self.path(shape), self.overlay_root)
            self.temporary.setPen(pen)
        else:
            self.temporary.setPath(self.path(shape))

    def document(self):
        if self.shape_kind == "polygon":
            return dict(type="polygon", points=self.points[:2048])
        x0, y0 = self.points[0]
        x1, y1 = self.points[-1]
        bounds = [
            math.floor(min(x0, x1)),
            math.floor(min(y0, y1)),
            math.ceil(max(x0, x1)),
            math.ceil(max(y0, y1)),
        ]
        shape = dict(type=self.shape_kind, bounds=bounds)
        if self.shape_kind == "rounded_rectangle":
            shape["radius"] = min(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.08
        return shape

    def mousePressEvent(self, event):
        self._hide_hover()
        if event.button() == Qt.MouseButton.LeftButton and self.crop_handles.press(event):
            event.accept()
            return
        grid = self.hover_model.get("grid")
        if (self.mapping is not None and grid and grid.get("review")
                and QRectF(self.pixmap.rect()).contains(self.mapToScene(event.position().toPoint()))
                and event.button() == Qt.MouseButton.LeftButton):
            zone = grid_hover(grid, self.source_point(event))
            if zone:
                self.selected_zone = zone
                self.viewport().update()
                self.zone_selected.emit(zone, event.globalPosition().toPoint() + QPoint(10, 14))
                event.accept()
                return
        if (
            self.geometry_info
            and self.drawing_enabled
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self.points = [self.source_point(event)]
            self.temporary_shape = None
            if self.temporary is not None:
                self.scene().removeItem(self.temporary)
                self.temporary = None
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.crop_handles.move(event):
            event.accept()
            return
        if self.points and event.buttons() & Qt.MouseButton.LeftButton:
            point = self.source_point(event)
            if self.shape_kind == "polygon":
                if len(self.points) < 2048 and math.dist(point, self.points[-1]) > 3:
                    self.points.append(point)
            else:
                self.points = [self.points[0], point]
            shape = self.document()
            if shape["type"] == "polygon" and len(shape["points"]) < 3:
                return
            self.temporary_shape = deepcopy(shape)
            self._paint_temporary(shape)
            event.accept()
        else:
            if event.buttons() == Qt.MouseButton.NoButton:
                self._show_hover(event)
            else:
                self._hide_hover()
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.crop_handles.release(event):
            event.accept()
            return
        if self.points and event.button() == Qt.MouseButton.LeftButton:
            if self.shape_kind != "polygon":
                self.points = [self.points[0], self.source_point(event)]
            shape = self.document()
            self.points = []
            if shape["type"] == "polygon":
                valid = len(shape["points"]) >= 3
            else:
                x0, y0, x1, y1 = shape["bounds"]
                valid = x1 - x0 >= 2 and y1 - y0 >= 2
            if valid:
                self.temporary_shape = deepcopy(shape)
                self._paint_temporary(shape)
                self.selected.emit(shape)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.geometry_info and self._fit_mode:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def leaveEvent(self, event):
        self._hide_hover()
        self.viewport().unsetCursor()
        super().leaveEvent(event)

    def hideEvent(self, event):
        self._hide_hover()
        self.crop_handles.cancel()
        super().hideEvent(event)

    def wheelEvent(self, event):
        self._hide_hover()
        if not self.geometry_info or self.geometry_info.get("preview_kind") != "positive":
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta:
            self.scale(1.2 if delta > 0 else 1 / 1.2,
                       1.2 if delta > 0 else 1 / 1.2)
            self._fit_mode = False
        event.accept()
