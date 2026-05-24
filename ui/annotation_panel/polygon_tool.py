from PySide6.QtCore import Qt, QObject, Signal, QPointF
from PySide6.QtGui import QColor, QPen, QBrush, QPolygonF
from PySide6.QtWidgets import QGraphicsPolygonItem, QGraphicsEllipseItem

class PolygonTool(QObject):
    status_message = Signal(str)

    def __init__(self, scene, on_finish_callback):
        super().__init__()
        self.scene = scene
        self.callback = on_finish_callback
        
        self.points = []           
        self.temp_poly = None
        self.temp_points = []      
        self.current_pos = QPointF()

    def mouse_press(self, event, pos):
        self.points.append(pos)
        dot = self.scene.addEllipse(pos.x()-2, pos.y()-2, 4, 4, QPen(Qt.red), QBrush(Qt.red))
        self.temp_points.append(dot)
        self._update_temp_poly()
        self.status_message.emit(f"Point added: {len(self.points)}")

    def mouse_move_no_drag(self, pos):
        self.current_pos = pos
        if len(self.points) > 0:
            self._update_temp_poly()

    def context_menu(self, event):
        if len(self.points) < 3:
            self.status_message.emit("Need at least 3 points to close polygon.")
            return

        poly = QPolygonF(self.points)
        poly_item = QGraphicsPolygonItem(poly)
        
        pen = QPen(QColor(0, 255, 0), 2)
        brush = QBrush(QColor(0, 255, 0, 100))
        poly_item.setPen(pen)
        poly_item.setBrush(brush)
        poly_item.setFlags(QGraphicsPolygonItem.ItemIsSelectable | QGraphicsPolygonItem.ItemIsMovable)
        
        self.scene.addItem(poly_item)
        if self.callback: self.callback(poly_item)

        self._reset()
        self.status_message.emit("Polygon closed.")

    def mouse_drag(self, event, pos): pass
    def mouse_release(self, event, pos): pass
    def mouse_double_click(self, event, pos): pass

    def _update_temp_poly(self):
        """Draw a live dashed preview polygon while adding points."""
        if self.temp_poly:
            self.scene.removeItem(self.temp_poly)

        pts = self.points + [self.current_pos]
        poly = QPolygonF(pts)
        self.temp_poly = QGraphicsPolygonItem(poly)

        self.temp_poly.setPen(QPen(QColor(0, 255, 255), 1.5, Qt.DashLine))
        self.temp_poly.setBrush(QBrush(QColor(0, 255, 255, 40)))
        
        self.scene.addItem(self.temp_poly)

    def _reset(self):
        self.points = []
        if self.temp_poly:
            self.scene.removeItem(self.temp_poly)
            self.temp_poly = None
        for dot in self.temp_points:
            self.scene.removeItem(dot)
        self.temp_points = []