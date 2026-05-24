import math
from PySide6.QtCore import Qt, QObject, Signal, QPointF
from PySide6.QtGui import QColor, QPen, QBrush, QPolygonF
from PySide6.QtWidgets import QGraphicsPolygonItem

class FreehandPolyTool(QObject):
    status_message = Signal(str)

    def __init__(self, canvas, on_finish_callback):
        super().__init__()
        self.canvas = canvas
        self.scene = canvas.scene
        self.callback = on_finish_callback
        
        self.pts = []
        self.is_drawing = False
        self.preview_item = None
        self.min_dist = 15.0 
        self.default_hint = "〰️ Freehand Poly: [Left Drag] to draw contour. [Release] to close and save."

    def show_hint(self):
        self.status_message.emit(self.default_hint)

    def mouse_press(self, event, pos):
        if event.button() == Qt.LeftButton:
            self.is_drawing = True
            self.pts = [pos]
            
            self.preview_item = QGraphicsPolygonItem(QPolygonF(self.pts))
            self.preview_item.setPen(QPen(QColor(0, 255, 255), 2, Qt.DashLine))
            self.preview_item.setBrush(QBrush(QColor(0, 255, 255, 40)))
            self.scene.addItem(self.preview_item)

    def mouse_drag(self, event, pos):
        if self.is_drawing and self.pts:
            last_pt = self.pts[-1]
            dist = math.hypot(pos.x() - last_pt.x(), pos.y() - last_pt.y())
            
            if dist >= self.min_dist:
                self.pts.append(pos)
                self.preview_item.setPolygon(QPolygonF(self.pts))

    def mouse_release(self, event, pos):
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.is_drawing = False
            if len(self.pts) > 2:
                self._finish()
            else:
                self._reset()

    def _finish(self):
        if self.preview_item:
            self.scene.removeItem(self.preview_item)
            self.preview_item = None
            
        poly = QPolygonF(self.pts)
        poly_item = QGraphicsPolygonItem(poly)
        poly_item.setFlags(QGraphicsPolygonItem.ItemIsSelectable | QGraphicsPolygonItem.ItemIsMovable)
        self.scene.addItem(poly_item)
        
        if self.callback: self.callback(poly_item)
        self.pts = []
        self.status_message.emit("✅ Freehand Polygon saved.")

    def _reset(self):
        if self.preview_item:
            self.scene.removeItem(self.preview_item)
            self.preview_item = None
        self.pts = []

    def context_menu(self, event):
        self._reset()
        self.is_drawing = False
        self.status_message.emit("Drawing canceled.")

    def mouse_double_click(self, event, pos): pass
    def mouse_move_no_drag(self, pos): pass