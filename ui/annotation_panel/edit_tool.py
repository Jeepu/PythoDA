import math
from PySide6.QtCore import Qt, QObject, Signal, QPointF
from PySide6.QtGui import QColor, QPen, QBrush, QPolygonF
from PySide6.QtWidgets import QGraphicsPolygonItem, QGraphicsRectItem


class EditTool(QObject):
    status_message = Signal(str)

    def __init__(self, canvas):
        super().__init__()
        self.canvas = canvas
        self.scene = canvas.scene

        self.active_poly_item = None
        self.vertex_handles = []
        self.dragging_idx = -1

        self.default_hint = (
            "🛠️ Edit: [L-Click] select. [Drag] move node. "
            "[R-Click] delete node. [Dbl-Click] add node on edge.")

    def show_hint(self):
        self.status_message.emit(self.default_hint)

    def mouse_press(self, event, pos):
        if event.button() == Qt.LeftButton:
            for i, handle in enumerate(self.vertex_handles):
                if handle.contains(handle.mapFromScene(pos)):
                    self.dragging_idx = i
                    return

            items = self.scene.items(
                pos, Qt.IntersectsItemShape, Qt.DescendingOrder,
                self.canvas.transform())
            for item in items:
                if isinstance(item, QGraphicsPolygonItem) and \
                   item.data(0) is not None:
                    self._set_active_polygon(item)
                    return

            self._clear_handles()

    def mouse_drag(self, event, pos):
        """Drag a vertex handle to reshape the polygon in real time."""
        if self.dragging_idx >= 0 and self.active_poly_item:
            poly = self.active_poly_item.polygon()

            pts = [poly.at(i) for i in range(poly.count())]
            pts[self.dragging_idx] = pos
            self.active_poly_item.setPolygon(QPolygonF(pts))

            rect = self.vertex_handles[self.dragging_idx].rect()
            rect.moveCenter(pos)
            self.vertex_handles[self.dragging_idx].setRect(rect)

    def mouse_release(self, event, pos):
        self.dragging_idx = -1

    def context_menu(self, event):
        """Handle right-click forwarded from canvas for vertex deletion."""
        scene_pos = self.canvas.mapToScene(event.pos())
        for i, handle in enumerate(self.vertex_handles):
            if handle.contains(handle.mapFromScene(scene_pos)):
                self._delete_vertex(i)
                return
        self._clear_handles()

    def mouse_double_click(self, event, pos):
        """Double-click on a polygon edge to insert a new vertex."""
        if event.button() == Qt.LeftButton:
            if not self.active_poly_item:
                items = self.scene.items(
                    pos, Qt.IntersectsItemShape, Qt.DescendingOrder,
                    self.canvas.transform())
                for item in items:
                    if isinstance(item, QGraphicsPolygonItem) and \
                       item.data(0) is not None:
                        self._set_active_polygon(item)
                return

            poly = self.active_poly_item.polygon()
            count = poly.count()
            if count < 3:
                return

            min_dist = float('inf')
            insert_idx = -1

            for i in range(count):
                p1 = poly.at(i)
                p2 = poly.at((i + 1) % count)

                dx = p2.x() - p1.x()
                dy = p2.y() - p1.y()
                l2 = dx * dx + dy * dy

                if l2 == 0:
                    dist = math.hypot(pos.x() - p1.x(),
                                       pos.y() - p1.y())
                else:
                    t = max(0, min(1, ((pos.x() - p1.x()) * dx
                                        + (pos.y() - p1.y()) * dy) / l2))
                    proj_x = p1.x() + t * dx
                    proj_y = p1.y() + t * dy
                    dist = math.hypot(pos.x() - proj_x,
                                       pos.y() - proj_y)

                if dist < min_dist:
                    min_dist = dist
                    insert_idx = i + 1

            if min_dist < 15.0:
                pts = [poly.at(i) for i in range(count)]

                if insert_idx == count:
                    pts.append(pos)
                else:
                    pts.insert(insert_idx, pos)

                self.active_poly_item.setPolygon(QPolygonF(pts))
                self._set_active_polygon(self.active_poly_item)
                self.status_message.emit("✨ New node added.")
            else:
                self.status_message.emit(
                    "Double-click closer to the edge to add a node.")

    def _set_active_polygon(self, poly_item):
        self._clear_handles()
        self.active_poly_item = poly_item
        poly = poly_item.polygon()

        for i in range(poly.count()):
            pt = poly.at(i)
            handle = QGraphicsRectItem(pt.x() - 4, pt.y() - 4, 8, 8)
            handle.setPen(QPen(Qt.black, 1))
            handle.setBrush(QBrush(Qt.white))
            handle.setZValue(9999)
            self.scene.addItem(handle)
            self.vertex_handles.append(handle)
        self.status_message.emit(f"Editing {poly_item.data(0)}.")

    def _delete_vertex(self, idx):
        if not self.active_poly_item:
            return
        poly = self.active_poly_item.polygon()
        if poly.count() <= 3:
            self.status_message.emit(
                "⚠️ Cannot delete: Polygon must have at least 3 vertices.")
            return

        pts = [poly.at(i) for i in range(poly.count())]
        pts.pop(idx)
        self.active_poly_item.setPolygon(QPolygonF(pts))

        self._set_active_polygon(self.active_poly_item)
        self.status_message.emit("🗑️ Node deleted.")

    def _clear_handles(self):
        for h in self.vertex_handles:
            self.scene.removeItem(h)
        self.vertex_handles = []
        self.active_poly_item = None
        self.dragging_idx = -1

    def mouse_move_no_drag(self, pos):
        pass
