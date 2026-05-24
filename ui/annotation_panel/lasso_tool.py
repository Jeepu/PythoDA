import cv2
import numpy as np
from PySide6.QtCore import Qt, QObject, Signal, QPointF
from PySide6.QtGui import QColor, QPen, QBrush, QPolygonF, QImage
from PySide6.QtWidgets import QGraphicsPolygonItem, QGraphicsPathItem, QApplication

def qimage_to_numpy(qimage):
    qimage = qimage.convertToFormat(QImage.Format_RGB888)
    width = qimage.width()
    height = qimage.height()
    ptr = qimage.constBits()
    arr = np.array(ptr).reshape((height, width, 3))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

class MagneticLassoTool(QObject):
    status_message = Signal(str)

    def __init__(self, canvas, on_finish_callback):
        super().__init__()
        self.canvas = canvas
        self.scene = canvas.scene
        self.callback = on_finish_callback
        
        self.scissors = None
        self.is_active = False
        self.anchor_pts = []
        self.current_contour = []
        self.scale_factor = 1.0
        self.history_contours = []
        
        self.pen_confirmed = QPen(QColor(0, 255, 0), 2)
        self.pen_confirmed.setCosmetic(True)
        self.pen_preview = QPen(QColor(255, 255, 0), 1.5, Qt.DashLine)
        self.pen_preview.setCosmetic(True)
        
        self.preview_item = None
        self.confirmed_item = None
        
        self.default_hint = "🧲 Lasso: [L-Click] start/add anchor. [Hover] snap to edge. [R-Click] undo. [Dbl-Click] close."

    def show_hint(self):
        self.status_message.emit(self.default_hint)
        
    def _to_calc(self, x, y): return (int(x * self.scale_factor), int(y * self.scale_factor))
    def _to_orig(self, x, y): return (int(x / self.scale_factor), int(y / self.scale_factor))

    def _init_scissors(self, pos):
        if not self.canvas.pixmap_item: return False
        pm = self.canvas.pixmap_item.pixmap()
        img_src = qimage_to_numpy(pm.toImage())
        h, w = img_src.shape[:2]
        
        max_calc_dim = 800.0
        if max(h, w) > max_calc_dim:
            self.scale_factor = max_calc_dim / max(h, w)
            new_w, new_h = int(w * self.scale_factor), int(h * self.scale_factor)
            img_calc = cv2.resize(img_src, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            self.scale_factor = 1.0
            img_calc = img_src

        self.scissors = cv2.segmentation.IntelligentScissorsMB()
        self.scissors.setEdgeFeatureCannyParameters(32, 100)
        self.scissors.setGradientMagnitudeMaxLimit(200)
        
        self.status_message.emit("⏳ Initializing Magnetic Engine...")
        self.canvas.setCursor(Qt.WaitCursor)
        QApplication.processEvents()
        
        self.scissors.applyImage(img_calc)
        self.canvas.setCursor(Qt.CrossCursor)
        self.show_hint()
        return True

    def mouse_press(self, event, pos):
        ix, iy = int(pos.x()), int(pos.y())
        pm = self.canvas.pixmap_item.pixmap() if self.canvas.pixmap_item else None
        if not pm: return
        w, h = pm.width(), pm.height()
        if ix < 0 or ix >= w or iy < 0 or iy >= h: return

        if event.button() == Qt.LeftButton:
            if not self.is_active:
                if self._init_scissors(pos):
                    self.is_active = True
                    self.anchor_pts = [(ix, iy)]
                    self.current_contour = [(ix, iy)]
                    cx, cy = self._to_calc(ix, iy)
                    self.scissors.buildMap((cx, cy))
                    self._update_confirmed_drawing()
            else:
                if self.scissors:
                    cx, cy = self._to_calc(ix, iy)
                    contour = self.scissors.getContour((cx, cy))
                    if contour is not None:
                        self.history_contours.append(list(self.current_contour))
                        pts = contour.reshape(-1, 2)
                        for pt in pts:
                            ox, oy = self._to_orig(pt[0], pt[1])
                            self.current_contour.append((ox, oy))
                        self.anchor_pts.append((ix, iy))
                        self.scissors.buildMap((cx, cy))
                        self._update_confirmed_drawing()

    def mouse_double_click(self, event, pos):
        if event.button() == Qt.LeftButton and self.is_active and len(self.anchor_pts) > 2:
            self._close_polygon()

    def context_menu(self, event):
        if self.is_active and len(self.anchor_pts) > 1:
            self.anchor_pts.pop()
            self.current_contour = self.history_contours.pop()
            cx, cy = self._to_calc(self.anchor_pts[-1][0], self.anchor_pts[-1][1])
            self.scissors.buildMap((cx, cy))
            self._update_confirmed_drawing()
            if self.preview_item:
                self.scene.removeItem(self.preview_item)
                self.preview_item = None
        elif self.is_active and len(self.anchor_pts) <= 1:
            self._reset()

    def mouse_move_no_drag(self, pos):
        if not self.is_active or not self.scissors: return
        ix, iy = int(pos.x()), int(pos.y())
        pm = self.canvas.pixmap_item.pixmap()
        ix = max(0, min(ix, pm.width() - 1))
        iy = max(0, min(iy, pm.height() - 1))
        
        cx, cy = self._to_calc(ix, iy)
        contour = self.scissors.getContour((cx, cy))
        if contour is not None:
            from PySide6.QtGui import QPainterPath
            pts = contour.reshape(-1, 2)
            path = QPainterPath()
            start_ox, start_oy = self._to_orig(pts[0][0], pts[0][1])
            path.moveTo(start_ox, start_oy)
            for pt in pts[1:]:
                ox, oy = self._to_orig(pt[0], pt[1])
                path.lineTo(ox, oy)
                
            if not self.preview_item:
                self.preview_item = QGraphicsPathItem(path)
                self.preview_item.setPen(self.pen_preview)
                self.scene.addItem(self.preview_item)
            else:
                self.preview_item.setPath(path)

    def _close_polygon(self):
        if not self.current_contour or len(self.current_contour) < 3: return
        
        poly = QPolygonF([QPointF(pt[0], pt[1]) for pt in self.current_contour])
        
        poly_item = QGraphicsPolygonItem(poly)
        poly_item.setFlags(QGraphicsPolygonItem.ItemIsSelectable | QGraphicsPolygonItem.ItemIsMovable)
        self.scene.addItem(poly_item)
        
        if self.callback:
            self.callback(poly_item)
            
        self._reset()
        self.status_message.emit("✅ Polygon saved.")

    def _update_confirmed_drawing(self):
        if len(self.current_contour) < 2: return
        from PySide6.QtGui import QPainterPath
        path = QPainterPath()
        path.moveTo(self.current_contour[0][0], self.current_contour[0][1])
        for pt in self.current_contour[1:]:
            path.lineTo(pt[0], pt[1])
            
        if not self.confirmed_item:
            self.confirmed_item = QGraphicsPathItem(path)
            self.confirmed_item.setPen(self.pen_confirmed)
            self.scene.addItem(self.confirmed_item)
        else:
            self.confirmed_item.setPath(path)

    def _reset(self):
        if self.preview_item: self.scene.removeItem(self.preview_item)
        if self.confirmed_item: self.scene.removeItem(self.confirmed_item)
        self.is_active = False
        self.scissors = None
        self.anchor_pts = []
        self.current_contour = []
        self.history_contours = []
        self.preview_item = None
        self.confirmed_item = None

    def mouse_drag(self, event, pos): pass
    def mouse_release(self, event, pos): pass