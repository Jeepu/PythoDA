from PySide6.QtWidgets import QGraphicsView, QGraphicsScene
from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QPixmap, QPainter, QWheelEvent, QMouseEvent

class GraphicsCanvas(QGraphicsView):
    mouse_moved = Signal(int, int)
    tool_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self.setMouseTracking(True)

        self.pixmap_item = None
        self._is_panning = False
        self._last_mouse_pos = QPointF()

        self.current_tool = None

    def set_image(self, pixmap: QPixmap):
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.setSceneRect(self.pixmap_item.boundingRect())
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def set_tool(self, tool):
        """
        Switch the active tool, safely disconnecting the previous one first.
        """
        if self.current_tool:
            try:
                self.current_tool.status_message.disconnect(
                    self.tool_message.emit)
            except Exception:
                pass

        self.current_tool = tool

        if self.current_tool:
            self.current_tool.status_message.connect(
                self.tool_message.emit)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton:
            self._is_panning = True
            self._last_mouse_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.RightButton and self.current_tool:
            if hasattr(self.current_tool, 'context_menu'):
                self.current_tool.context_menu(event)
                event.accept()
                return

        if event.button() == Qt.LeftButton and self.current_tool:
            scene_pos = self.mapToScene(event.pos())
            self.current_tool.mouse_press(event, scene_pos)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._is_panning:
            delta = event.position() - self._last_mouse_pos
            self._last_mouse_pos = event.position()
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
            event.accept()
            return

        scene_pos = self.mapToScene(event.pos())

        if self.current_tool:
            if event.buttons() & Qt.LeftButton:
                self.current_tool.mouse_drag(event, scene_pos)
            elif hasattr(self.current_tool, 'mouse_move_no_drag'):
                self.current_tool.mouse_move_no_drag(scene_pos)

        if self.pixmap_item:
            x, y = int(scene_pos.x()), int(scene_pos.y())
            img_rect = self.pixmap_item.boundingRect()
            if img_rect.contains(scene_pos):
                self.mouse_moved.emit(x, y)
            else:
                self.mouse_moved.emit(-1, -1)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton and self.current_tool:
            scene_pos = self.mapToScene(event.pos())
            self.current_tool.mouse_release(event, scene_pos)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.current_tool:
            if hasattr(self.current_tool, 'mouse_double_click'):
                scene_pos = self.mapToScene(event.pos())
                self.current_tool.mouse_double_click(event, scene_pos)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self.scale(zoom_factor, zoom_factor)