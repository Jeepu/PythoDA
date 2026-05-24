import cv2
import numpy as np
from PySide6.QtCore import Qt, QObject, Signal, QPointF
from PySide6.QtGui import QColor, QPen, QBrush, QPolygonF
from PySide6.QtWidgets import QGraphicsPolygonItem

class SAMPointTool(QObject):
    status_message = Signal(str)

    def __init__(self, canvas, on_finish_callback, display_img_rgb, predictor):
        super().__init__()
        self.canvas = canvas
        self.scene = canvas.scene
        self.callback = on_finish_callback
        
        self.input_points = []
        self.input_labels = []
        self.point_items = []
        self.preview_poly_item = None
        self.predictor = predictor
        
        self.default_hint = "🎯 SAM: [L-Click]: Include (Green). [R-Click]: Exclude (Red). [Dbl-Click]: Save."
        
        try:
            self.predictor.set_image(display_img_rgb)
            self.show_hint()
        except Exception as e:
            self.status_message.emit(f"❌ SAM Error: {str(e)}")

    def show_hint(self): self.status_message.emit(self.default_hint)

    def mouse_press(self, event, pos):
        if event.button() == Qt.LeftButton: self._add_point(pos, label=1)

    def context_menu(self, event):
        scene_pos = self.canvas.mapToScene(event.pos())
        self._add_point(scene_pos, label=0)

    def mouse_double_click(self, event, pos):
        if event.button() == Qt.LeftButton: self.confirm_polygon()

    def _add_point(self, pos, label):
        ix, iy = int(pos.x()), int(pos.y())
        pm = self.canvas.pixmap_item.pixmap()
        if ix < 0 or ix >= pm.width() or iy < 0 or iy >= pm.height(): return

        self.input_points.append([ix, iy])
        self.input_labels.append(label) 
        
        color = Qt.green if label == 1 else Qt.red
        dot = self.scene.addEllipse(ix-3, iy-3, 6, 6, QPen(color), QBrush(color))
        self.point_items.append(dot)
        self._predict()

    def _predict(self):
        if not self.input_points: return
        pts, lbls = np.array(self.input_points), np.array(self.input_labels)
        try:
            masks, scores, _ = self.predictor.predict(point_coords=pts, point_labels=lbls, multimask_output=False)
            mask = masks[0].astype(np.uint8) * 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: return
            
            best_cnt = max(cnts, key=cv2.contourArea)
            approx = cv2.approxPolyDP(best_cnt, 0.0015 * cv2.arcLength(best_cnt, True), True)
            if len(approx) < 3: return 
            
            poly = QPolygonF([QPointF(pt[0][0], pt[0][1]) for pt in approx])
            if self.preview_poly_item: self.preview_poly_item.setPolygon(poly)
            else:
                self.preview_poly_item = QGraphicsPolygonItem(poly)
                self.preview_poly_item.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
                self.preview_poly_item.setBrush(QBrush(QColor(255, 255, 0, 80)))
                self.scene.addItem(self.preview_poly_item)
        except Exception as e: print(f"SAM Predict Error: {e}")

    def confirm_polygon(self):
        if not self.preview_poly_item: return
        poly_item = QGraphicsPolygonItem(self.preview_poly_item.polygon())
        poly_item.setFlags(QGraphicsPolygonItem.ItemIsSelectable | QGraphicsPolygonItem.ItemIsMovable)
        self.scene.addItem(poly_item)
        if self.callback: self.callback(poly_item)
        self._reset()
        self.status_message.emit("✅ Saved!")

    def _reset(self):
        for pt in self.point_items: self.scene.removeItem(pt)
        if self.preview_poly_item: self.scene.removeItem(self.preview_poly_item)
        self.input_points, self.input_labels, self.point_items, self.preview_poly_item = [], [], [], None

    def mouse_move_no_drag(self, pos): pass
    def mouse_drag(self, event, pos): pass
    def mouse_release(self, event, pos): pass