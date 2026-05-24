import cv2
import numpy as np
from PySide6.QtCore import Qt, QObject, Signal, QPointF, QTimer
from PySide6.QtGui import QColor, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsPolygonItem, QApplication


class Auto3DTool(QObject):
    status_message = Signal(str)

    def __init__(self, canvas, on_finish_callback, display_img_rgb,
                 depth_map, d_min, d_max, predictor):
        super().__init__()
        self.canvas = canvas
        self.scene = canvas.scene
        self.callback = on_finish_callback
        self.predictor = predictor

        self.status_message.emit(
            "✨ Analyzing 3D Depth bounds for whole object...")
        QApplication.processEvents()

        valid_mask = (depth_map >= d_min) & (depth_map <= d_max)
        y_indices, x_indices = np.where(valid_mask)

        if len(x_indices) == 0 or len(y_indices) == 0:
            self.status_message.emit(
                "⚠️ No prominent objects found in this depth slice.")
            return

        min_x, max_x = np.min(x_indices), np.max(x_indices)
        min_y, max_y = np.min(y_indices), np.max(y_indices)

        pad = 10
        h, w = depth_map.shape
        min_x, max_x = max(0, min_x - pad), min(w - 1, max_x + pad)
        min_y, max_y = max(0, min_y - pad), min(h - 1, max_y + pad)

        input_box = np.array([min_x, min_y, max_x, max_y])

        self.status_message.emit(
            f"🎯 Auto-Bounding Box generated: {input_box}. "
            f"Sending to SAM...")
        QApplication.processEvents()

        try:
            self.predictor.set_image(display_img_rgb)

            masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False)
            mask = masks[0].astype(np.uint8) * 255

            cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if cnts:
                best_cnt = max(cnts, key=cv2.contourArea)
                approx = cv2.approxPolyDP(
                    best_cnt, 0.0015 * cv2.arcLength(best_cnt, True), True)

                if len(approx) >= 3:
                    poly = QPolygonF(
                        [QPointF(pt[0][0], pt[0][1]) for pt in approx])
                    poly_item = QGraphicsPolygonItem(poly)
                    poly_item.setFlags(
                        QGraphicsPolygonItem.ItemIsSelectable
                        | QGraphicsPolygonItem.ItemIsMovable)
                    self.scene.addItem(poly_item)

                    # Flash a red bounding box to show how the result was derived
                    rect_item = self.scene.addRect(
                        min_x, min_y, max_x - min_x, max_y - min_y,
                        QPen(QColor(255, 0, 0, 150), 2, Qt.DashLine))
                    QTimer.singleShot(
                        1000, lambda: self.scene.removeItem(rect_item)
                        if rect_item in self.scene.items() else None)

                    if self.callback:
                        self.callback(poly_item)
                    self.status_message.emit(
                        "✅ Whole Plant Auto-Segmentation Success!")
                    return
        except Exception as e:
            self.status_message.emit(f"❌ Auto 3D Error: {str(e)}")

    def show_hint(self): pass
    def mouse_press(self, event, pos): pass
    def context_menu(self, event): pass
    def mouse_double_click(self, event, pos): pass
    def mouse_move_no_drag(self, pos): pass
    def mouse_drag(self, event, pos): pass
    def mouse_release(self, event, pos): pass
    def _reset(self): pass
    def _clear_handles(self): pass