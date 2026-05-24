import cv2
import numpy as np
from PySide6.QtCore import Qt, QObject, Signal, QPointF, QRectF, QThread
from PySide6.QtGui import QColor, QPen, QBrush, QPolygonF
from PySide6.QtWidgets import QGraphicsPolygonItem, QGraphicsRectItem


class SamBoxInferenceThread(QThread):
    """Offloads SAM inference and contour extraction to a background thread."""
    finished_signal = Signal(object)
    error_signal = Signal(str)

    def __init__(self, predictor, input_box):
        super().__init__()
        self.predictor = predictor
        self.input_box = input_box

    def run(self):
        try:
            masks, _, _ = self.predictor.predict(
                point_coords=None, point_labels=None,
                box=self.input_box[None, :], multimask_output=False)
            mask = masks[0].astype(np.uint8) * 255

            cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                self.finished_signal.emit(None)
                return

            best_cnt = max(cnts, key=cv2.contourArea)
            approx = cv2.approxPolyDP(
                best_cnt, 0.0015 * cv2.arcLength(best_cnt, True), True)
            if len(approx) < 3:
                self.finished_signal.emit(None)
                return

            self.finished_signal.emit(approx)
        except Exception as e:
            self.error_signal.emit(str(e))


class SAMBoxTool(QObject):
    status_message = Signal(str)

    def __init__(self, canvas, on_finish_callback, display_img_rgb, predictor):
        super().__init__()
        self.canvas = canvas
        self.scene = canvas.scene
        self.callback = on_finish_callback
        
        self.start_pos = None
        self.temp_rect_item = None
        self.preview_poly_item = None

        # Hold reference to background thread to prevent premature GC
        self.inference_thread = None 
        
        self.predictor = predictor
        self.default_hint = "⬜ SAM Box: [Left Drag]: Draw Box. [Right Click]: Cancel. [Double Click]: Save."
        
        try:
            self.show_hint()
        except Exception as e:
            self.status_message.emit(f"❌ SAM Error: {str(e)}")

    def show_hint(self):
        self.status_message.emit(self.default_hint)

    def mouse_press(self, event, pos):
        # Guard: ignore clicks while a previous inference is still running
        if self.inference_thread and self.inference_thread.isRunning():
            return
            
        if event.button() == Qt.LeftButton:
            self.start_pos = pos
            if self.temp_rect_item:
                self.scene.removeItem(self.temp_rect_item)
            self.temp_rect_item = QGraphicsRectItem(QRectF(pos, pos))
            self.temp_rect_item.setPen(QPen(QColor(0, 255, 255), 2, Qt.DashLine))
            self.scene.addItem(self.temp_rect_item)

    def mouse_drag(self, event, pos):
        """Update the live rubber-band rectangle during drag."""
        if self.start_pos and self.temp_rect_item:
            rect = QRectF(self.start_pos, pos).normalized()
            self.temp_rect_item.setRect(rect)

    def mouse_release(self, event, pos):
        """On release, hand off the box to the background SAM thread."""
        if self.start_pos and self.temp_rect_item and event.button() == Qt.LeftButton:
            rect = self.temp_rect_item.rect()
            self.start_pos = None

            self._trigger_predict(rect)

    def mouse_double_click(self, event, pos):
        if event.button() == Qt.LeftButton:
            self.confirm_polygon()

    def context_menu(self, event):
        """Right-click to cancel the current box."""
        self._reset()
        self.status_message.emit("Canceled current box. Try again.")

    def _trigger_predict(self, rect):
        """Submit the bounding box to the async SAM inference thread."""
        x1, y1, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        if w < 5 or h < 5:
            return

        pm = self.canvas.pixmap_item.pixmap()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(pm.width() - 1, x1 + w), min(pm.height() - 1, y1 + h)
        input_box = np.array([x1, y1, x2, y2])

        self.canvas.setCursor(Qt.WaitCursor)
        self.status_message.emit(
            "⏳ SAM is generating mask in background...")

        self.inference_thread = SamBoxInferenceThread(
            self.predictor, input_box)
        self.inference_thread.finished_signal.connect(
            self._on_predict_finished)
        self.inference_thread.error_signal.connect(
            self._on_predict_error)
        self.inference_thread.start()

    def _on_predict_finished(self, approx):
        """Callback slot when the background thread finishes computing."""
        self.canvas.setCursor(Qt.CrossCursor)
        self.status_message.emit(self.default_hint)

        if approx is None:
            return

        poly = QPolygonF([QPointF(pt[0][0], pt[0][1]) for pt in approx])

        if self.preview_poly_item:
            self.preview_poly_item.setPolygon(poly)
        else:
            self.preview_poly_item = QGraphicsPolygonItem(poly)
            self.preview_poly_item.setPen(
                QPen(QColor(255, 255, 0), 2, Qt.DashLine))
            self.preview_poly_item.setBrush(
                QBrush(QColor(255, 255, 0, 80)))
            self.scene.addItem(self.preview_poly_item)

    def _on_predict_error(self, err_msg):
        self.canvas.setCursor(Qt.CrossCursor)
        self.status_message.emit(f"❌ SAM Predict Error: {err_msg}")

    def confirm_polygon(self):
        if not self.preview_poly_item: return
        final_poly = self.preview_poly_item.polygon()
        poly_item = QGraphicsPolygonItem(final_poly)
        poly_item.setFlags(QGraphicsPolygonItem.ItemIsSelectable | QGraphicsPolygonItem.ItemIsMovable)
        self.scene.addItem(poly_item)
        if self.callback: self.callback(poly_item)
        self._reset()
        self.status_message.emit("✅ Polygon saved!")

    def _reset(self):
        if self.temp_rect_item: 
            self.scene.removeItem(self.temp_rect_item)
        if self.preview_poly_item: 
            self.scene.removeItem(self.preview_poly_item)
        self.temp_rect_item = None
        self.preview_poly_item = None
        self.start_pos = None
        
        # Release the thread reference
        if self.inference_thread and self.inference_thread.isRunning():
            self.inference_thread = None

    def mouse_move_no_drag(self, pos): pass