from PySide6.QtWidgets import (QGraphicsLineItem, QGraphicsPathItem, QMenu, 
                               QGraphicsItem, QGraphicsPixmapItem, QGraphicsEllipseItem, 
                               QGraphicsSimpleTextItem, QGraphicsRectItem, QGraphicsPolygonItem,
                               QInputDialog, QGraphicsTextItem)
from PySide6.QtCore import Qt, QPointF, Signal, QObject, QRectF
from PySide6.QtGui import QPen, QColor, QPainterPath, QBrush, QFont, QPolygonF, QPainterPathStroker, QImage
import math
import numpy as np
import cv2

from .dialogs import SetScaleDialog
from core.global_state import GlobalState

# -- Utility functions and metrics extraction --

def dist(p1, p2): return math.sqrt((p1.x() - p2.x())**2 + (p1.y() - p2.y())**2)

def get_angle(p1, center, p2):
    v1 = (p1.x() - center.x(), p1.y() - center.y()); v2 = (p2.x() - center.x(), p2.y() - center.y())
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2); mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 * mag2 == 0: return 0.0
    val = max(min(dot / (mag1 * mag2), 1.0), -1.0)
    return math.degrees(math.acos(val))

def polygon_area(points):
    area = 0.0
    for i in range(len(points)):
        j = (i + 1) % len(points)
        area += points[i].x() * points[j].y() - points[j].x() * points[i].y()
    return abs(area) / 2.0

def get_item_metrics(item):
    """Extract metrics from item data tags. Four categories only."""
    if not hasattr(item, "data") or not item.data(0): return None
    
    gs = GlobalState(); scale = gs.pixels_per_unit; unit = gs.unit_name
    cat = item.data(0); tool_name = item.data(1)
    
    if cat not in ["Closed Shape", "Line Segment", "Angle", "Count"]: return None
    
    metrics = {"Type": cat, "Tool": tool_name, "Length": "-", "Width": "-", "Area": "-", "Perim": "-", "RGB": "-", "Swatch": "-"}
    
    if cat == "Count":
        metrics["Length"] = f"ID: {item.data(2)}"; return metrics
        
    elif cat == "Angle":
        metrics["Length"] = f"{item.data(2):.2f}°"; return metrics
        
    elif cat == "Line Segment":
        if isinstance(item, QGraphicsLineItem): d_px = dist(item.line().p1(), item.line().p2())
        elif isinstance(item, QGraphicsPathItem): 
            poly = item.path().toFillPolygon(); pts = [poly[i] for i in range(poly.count())]
            d_px = sum(dist(pts[i], pts[i+1]) for i in range(len(pts)-1)) if len(pts)>1 else 0
        else: d_px = 0
        metrics["Length"] = f"{d_px:.0f}" if unit == "px" else f"{d_px/scale:.2f}"; return metrics
        
    elif cat == "Closed Shape":
        area_px = 0; perim_px = 0; pts = []
        if isinstance(item, QGraphicsRectItem):
            r = item.rect(); area_px = r.width() * r.height(); perim_px = 2 * (r.width() + r.height())
            pts = [r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()]
        elif isinstance(item, QGraphicsEllipseItem):
            r = item.rect(); a = r.width() / 2; b = r.height() / 2; area_px = math.pi * a * b
            if a + b != 0: h = ((a - b)**2) / ((a + b)**2); perim_px = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
            path = QPainterPath(); path.addEllipse(r); poly = path.toFillPolygon(); pts = [poly[i] for i in range(poly.count())]
        elif isinstance(item, (QGraphicsPolygonItem, QGraphicsPathItem)):
            poly = item.polygon() if isinstance(item, QGraphicsPolygonItem) else item.path().toFillPolygon()
            pts = [poly[i] for i in range(poly.count())]
            area_px = polygon_area(pts)
            if len(pts) > 1: perim_px = sum(dist(pts[i], pts[(i+1)%len(pts)]) for i in range(len(pts)))

        len_px, wid_px = 0, 0
        if len(pts) >= 3:
            np_pts = np.array([[p.x(), p.y()] for p in pts], dtype=np.float32)
            rect = cv2.minAreaRect(np_pts)
            len_px = max(rect[1][0], rect[1][1]); wid_px = min(rect[1][0], rect[1][1])
            
            cx, cy = int(rect[0][0]), int(rect[0][1])
            if item.scene():
                bg_items = [i for i in item.scene().items() if isinstance(i, QGraphicsPixmapItem)]
                if bg_items:
                    pm = bg_items[-1].pixmap().toImage() 
                    if pm.valid(cx, cy):
                        c = pm.pixelColor(cx, cy)
                        metrics["RGB"] = f"{c.red()}, {c.green()}, {c.blue()}"
                        metrics["Swatch"] = f"#{c.red():02X}{c.green():02X}{c.blue():02X}"

        if unit == "px":
            metrics["Area"] = f"{area_px:.0f}"; metrics["Perim"] = f"{perim_px:.0f}"
            metrics["Length"] = f"{len_px:.1f}" if len_px else "-"; metrics["Width"] = f"{wid_px:.1f}" if wid_px else "-"
        else:
            metrics["Area"] = f"{area_px/(scale*scale):.2f}"; metrics["Perim"] = f"{perim_px/scale:.2f}"
            metrics["Length"] = f"{len_px/scale:.2f}" if len_px else "-"; metrics["Width"] = f"{wid_px/scale:.2f}" if wid_px else "-"
            
    return metrics

def qimage_to_numpy(qimage):
    qimage = qimage.convertToFormat(QImage.Format_RGB888); w = qimage.width(); h = qimage.height(); bpl = qimage.bytesPerLine() 
    arr = np.array(qimage.constBits()).reshape((h, bpl))[:, :w * 3].reshape((h, w, 3))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def convert_to_path_item(item, scene):
    if isinstance(item, QGraphicsPathItem): return item
    path = QPainterPath()
    if isinstance(item, QGraphicsRectItem): path.addRect(item.rect())
    elif isinstance(item, QGraphicsEllipseItem): path.addEllipse(item.rect())
    elif isinstance(item, QGraphicsPolygonItem): path.addPolygon(item.polygon())
    else: return None
    new_item = QGraphicsPathItem(path)
    new_item.setPen(item.pen()); new_item.setBrush(item.brush())
    scene.removeItem(item); scene.addItem(new_item)
    return new_item

# -- Tool base class --
class ToolBase(QObject):
    status_message = Signal(str)
    tool_exit_requested = Signal()
    reset_view_requested = Signal()

    def __init__(self, c):
        super().__init__()
        self.canvas = c
        self.scene = c.scene
        self.pen = QPen(QColor(255, 255, 0), 2)
        self.pen.setCosmetic(True)
        self.default_hint = "Ready."

    def keep_item(self, item):
        """Keep a hard Python reference to prevent immediate GC of the item."""
        if not hasattr(self.scene, "kept_items"):
            self.scene.kept_items = []
        self.scene.kept_items.append(item)

    def show_hint(self): self.status_message.emit(self.default_hint)
    def mouse_press(self, e, p): pass
    def mouse_drag(self, e, p): pass
    def mouse_release(self, e, p): pass
    def mouse_double_click(self, e, p): pass 
    def mouse_move_no_drag(self, p): pass
    def context_menu(self, e): pass
    def set_size(self, s): pass 

# --- Select ---
class SelectTool(ToolBase):
    def __init__(self, c): 
        super().__init__(c)
        self.default_hint = "Left Click: Select. Drag: Move. Click & Drag on background: Box Selection."
        self.dragging_item = None; self.last_mouse_pos = None; self.selection_box = None; self.box_start_pos = None

    def mouse_press(self, e, p):
        if e.button() != Qt.LeftButton: return
        item = self.scene.itemAt(p, self.canvas.transform())
        if item and not isinstance(item, QGraphicsPixmapItem): 
            item.setFlags(item.flags() | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)
            if not item.isSelected(): self.scene.clearSelection(); item.setSelected(True)
            self.dragging_item = item; self.last_mouse_pos = p
        else:
            self.scene.clearSelection(); self.dragging_item = None; self.box_start_pos = p
            self.selection_box = QGraphicsRectItem(); pen = QPen(QColor(0, 120, 215), 1, Qt.DashLine); pen.setCosmetic(True)
            self.selection_box.setPen(pen); self.selection_box.setBrush(QBrush(QColor(0, 120, 215, 50)))
            self.scene.addItem(self.selection_box); self.selection_box.setRect(QRectF(p, p))

    def mouse_drag(self, e, p):
        if self.dragging_item and self.last_mouse_pos:
            dx = p.x() - self.last_mouse_pos.x(); dy = p.y() - self.last_mouse_pos.y()
            for selected_item in self.scene.selectedItems(): selected_item.moveBy(dx, dy)
            self.last_mouse_pos = p
        elif self.selection_box and self.box_start_pos:
            self.selection_box.setRect(QRectF(self.box_start_pos, p).normalized())

    def mouse_release(self, e, p):
        if e.button() != Qt.LeftButton: return
        if self.dragging_item:
            self.dragging_item = None; self.last_mouse_pos = None; self.status_message.emit("refresh_table") 
        elif self.selection_box:
            rect = self.selection_box.rect()
            self.scene.removeItem(self.selection_box); self.selection_box = None; self.box_start_pos = None
            for item in self.scene.items(rect, Qt.IntersectsItemShape):
                if not isinstance(item, QGraphicsPixmapItem): item.setFlags(item.flags() | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable); item.setSelected(True)

    def _reset_tool(self):
        if hasattr(self, 'selection_box') and self.selection_box: self.scene.removeItem(self.selection_box); self.selection_box = None
        self.dragging_item = None; self.last_mouse_pos = None; self.box_start_pos = None

# --- Line ---
class LineTool(ToolBase):
    def __init__(self, c): 
        super().__init__(c); self.cur=None; self.sp=None; self.default_hint = "Left Drag: Draw line. Right Click on line: Set Scale."
    def mouse_press(self, e, p): 
        if e.button() == Qt.LeftButton: 
            self.sp = p; self.cur = QGraphicsLineItem(p.x(), p.y(), p.x(), p.y()); self.cur.setPen(self.pen); self.scene.addItem(self.cur)
    def mouse_drag(self, e, p):
        if self.cur and self.sp:
            end = p 
            if e.modifiers() & Qt.ShiftModifier: 
                dx = abs(end.x() - self.sp.x()); dy = abs(end.y() - self.sp.y()); end.setY(self.sp.y()) if dx > dy else end.setX(self.sp.x())
            self.cur.setLine(self.sp.x(), self.sp.y(), end.x(), end.y())
    def mouse_release(self, e, p): 
        if self.cur: 
            self.cur.setData(0, "Line Segment"); self.cur.setData(1, "Line")
            self.keep_item(self.cur)
            self.status_message.emit("refresh_table")
            self.cur = None
    def context_menu(self, e):
        scene_pos = self.canvas.mapToScene(e.pos())
        item = self.scene.itemAt(scene_pos, self.canvas.transform())

        if not isinstance(item, QGraphicsLineItem) or \
           item.data(1) != "Line":
            return

        m = QMenu(self.canvas)
        a = m.addAction("📏 Set Scale")
        if m.exec(e.globalPos()) == a:
            d = dist(item.line().p1(), item.line().p2())
            dlg = SetScaleDialog(d, self.canvas) 
            if dlg.exec(): 
                k, u = dlg.get_values()
                GlobalState().set_scale(k, d, u)
                self.status_message.emit("refresh_table")

# --- Polyline ---
class PolylineTool(ToolBase):
    def __init__(self, c): 
        super().__init__(c); self.cur=None; self.pts=[]; self.tm=None; self.default_hint = "Left Click: Add point. Double-click/Right Click: Finish."
    def mouse_press(self, e, p):
        if e.button() == Qt.LeftButton:
            if not self.pts: self.path = QPainterPath(p); self.cur = QGraphicsPathItem(self.path); self.cur.setPen(self.pen); self.scene.addItem(self.cur)
            else: self.path.lineTo(p); self.cur.setPath(self.path)
            self.pts.append(p)
    def context_menu(self, e):
        if self.pts: self._fin() 
    def mouse_move_no_drag(self, p):
        if self.pts: 
            if self.tm: self.scene.removeItem(self.tm)
            self.tm = QGraphicsLineItem(self.pts[-1].x(), self.pts[-1].y(), p.x(), p.y()); self.tm.setPen(QPen(Qt.white, 1, Qt.DotLine)); self.scene.addItem(self.tm)
    def mouse_double_click(self, e, p): self._fin() 
    def _fin(self):
        if self.tm: self.scene.removeItem(self.tm); self.tm = None
        if not self.pts: return
        self.cur.setData(0, "Line Segment"); self.cur.setData(1, "Polyline")
        self.keep_item(self.cur)
        self.pts = []; self.cur = None 
        self.status_message.emit("refresh_table")

# --- Freehand ---
class FreehandTool(ToolBase):
    def __init__(self, c): 
        super().__init__(c); self.cur=None; self.pts=[]; self.default_hint = "Left Drag: Draw freehand path."
    def mouse_press(self, e, p): 
        if e.button() == Qt.LeftButton: self.pts = [p]; self.path = QPainterPath(p); self.cur = QGraphicsPathItem(self.path); self.cur.setPen(self.pen); self.scene.addItem(self.cur)
    def mouse_drag(self, e, p): 
        if self.cur: self.path.lineTo(p); self.cur.setPath(self.path); self.pts.append(p)
    def mouse_release(self, e, p):
        if self.pts: 
            self.cur.setData(0, "Line Segment"); self.cur.setData(1, "Freehand")
            self.keep_item(self.cur)
            self.pts = []; self.cur = None 
            self.status_message.emit("refresh_table")

# --- Angle ---
class AngleTool(ToolBase):
    def __init__(self, c): 
        super().__init__(c); self.pts=[]; self.tls=[]; self.default_hint = "Left Click: Point 1 -> Vertex(Center) -> Point 2."
    def mouse_press(self, e, p):
        if e.button() == Qt.LeftButton:
            self.pts.append(p)
            if len(self.pts) == 1: self._draw()
            elif len(self.pts) == 2: self._draw()
            elif len(self.pts) == 3: 
                a = get_angle(self.pts[0], self.pts[1], self.pts[2])
                self._draw(angle_val=a)
                self.status_message.emit("refresh_table") 
    def mouse_move_no_drag(self, p):
        if 0 < len(self.pts) < 3: self._draw(p)
    def _draw(self, pp=None, angle_val=None):
        for l in self.tls: self.scene.removeItem(l)
        self.tls = []; d = self.pts + [pp] if pp else self.pts
        if len(d) == 3 and angle_val is not None:
            parent = QGraphicsPolygonItem(QPolygonF(d))
            parent.setPen(Qt.NoPen); parent.setBrush(Qt.NoBrush)
            parent.setData(0, "Angle"); parent.setData(1, "Angle Tool"); parent.setData(2, angle_val)
            parent.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)
            self.scene.addItem(parent)
            
            l1 = QGraphicsLineItem(d[0].x(), d[0].y(), d[1].x(), d[1].y(), parent)
            l2 = QGraphicsLineItem(d[1].x(), d[1].y(), d[2].x(), d[2].y(), parent)
            l1.setPen(self.pen); l2.setPen(self.pen)
            
            self.keep_item(parent)
            self.pts = []
        else:
            for i in range(len(d)-1): 
                l = QGraphicsLineItem(d[i].x(), d[i].y(), d[i+1].x(), d[i+1].y())
                l.setPen(self.pen); self.scene.addItem(l); self.tls.append(l)

# --- Point ---
class PointTool(ToolBase):
    def __init__(self, c): 
        super().__init__(c); self.sz=10; self.br=QBrush(QColor(255,0,0)); self.default_hint = "Left Click: Place point marker (auto-numbered)."
    def set_size(self, s): self.sz = s
    def mouse_press(self, e, p):
        if e.button() == Qt.LeftButton:
            # Count existing markers for auto-numbering
            current_id = len([i for i in self.scene.items()
                            if hasattr(i, "data")
                            and i.data(0) == "Count"]) + 1
            r = self.sz / 2
            el = QGraphicsEllipseItem(p.x()-r, p.y()-r, self.sz, self.sz)
            el.setPen(QPen(Qt.white, 2))
            el.setBrush(self.br)
            el.setZValue(100)
            el.setData(0, "Count")
            el.setData(1, "Point")
            el.setData(2, current_id)

            t = QGraphicsSimpleTextItem(str(current_id), el)
            t.setBrush(QBrush(QColor(255, 255, 0)))
            t.setFont(QFont("Arial", max(10, int(self.sz * 0.8)),
                           QFont.Bold))
            t.setPos(p.x() + r, p.y() - r * 2)

            self.scene.addItem(el)
            self.keep_item(el)
            self.status_message.emit("refresh_table")

# --- Eraser ---
class EraserTool(ToolBase):
    def __init__(self, canvas): 
        super().__init__(canvas); self.size = 20; self.default_hint = "Left Click/Drag: Erase drawings under cursor."
    def set_size(self, s): self.size = s
    def mouse_press(self, e, p): 
        if e.button() == Qt.LeftButton: self._e(p)
    def mouse_drag(self, e, p): self._e(p)
    def _e(self, p):
        offset = self.size/2; rect = QRectF(p.x()-offset, p.y()-offset, self.size, self.size); items = self.scene.items(rect); c = 0
        for i in items:
            if isinstance(i, QGraphicsPixmapItem) or (i.parentItem() and isinstance(i.parentItem(), QGraphicsPixmapItem)): continue
            t = i.parentItem() if i.parentItem() else i
            if t.scene() == self.scene: 
                self.scene.removeItem(t)
                if hasattr(self.scene, "kept_items") and t in self.scene.kept_items: self.scene.kept_items.remove(t)
                c += 1
        if c > 0: self.status_message.emit("refresh_table")

# --- Rectangle ---
class RectTool(ToolBase):
    def __init__(self, c): 
        super().__init__(c); self.cur=None; self.sp=None; self.pen=QPen(QColor(0,255,255), 2); self.pen.setCosmetic(True)
        self.default_hint = "Left Drag: Draw rectangle. Hold Shift for Square."
    def mouse_press(self, e, p): 
        if e.button() == Qt.LeftButton: self.sp = p; self.cur = QGraphicsRectItem(p.x(), p.y(), 0, 0); self.cur.setPen(self.pen); self.scene.addItem(self.cur)
    def mouse_drag(self, e, p):
        if self.cur and self.sp:
            x, y = min(self.sp.x(), p.x()), min(self.sp.y(), p.y()); w, h = abs(p.x()-self.sp.x()), abs(p.y()-self.sp.y())
            if e.modifiers() & Qt.ShiftModifier: s = max(w, h); w, h = s, s; x = self.sp.x()-s if p.x() < self.sp.x() else self.sp.x(); y = self.sp.y()-s if p.y() < self.sp.y() else self.sp.y()
            self.cur.setRect(x, y, w, h)
    def mouse_release(self, e, p): 
        if self.cur:
            self.cur.setData(0, "Closed Shape"); self.cur.setData(1, "Rectangle")
            self.keep_item(self.cur); self.cur = None
            self.status_message.emit("refresh_table")

# --- Rounded Rectangle ---
class RoundedRectTool(ToolBase):
    def __init__(self, canvas): 
        super().__init__(canvas); self.cur=None; self.sp=None; self.r=20; self.pen=QPen(QColor(0,255,255), 2); self.pen.setCosmetic(True)
        self.default_hint = "Left Drag: Draw rounded rectangle."
    def set_size(self, r): self.r = r
    def mouse_press(self, e, p): self.sp = p; pa = QPainterPath(); pa.addRoundedRect(QRectF(p, p), self.r, self.r); self.cur = QGraphicsPathItem(pa); self.cur.setPen(self.pen); self.scene.addItem(self.cur)
    def mouse_drag(self, e, p):
        if self.cur and self.sp:
            x, y = min(self.sp.x(), p.x()), min(self.sp.y(), p.y()); w, h = abs(p.x()-self.sp.x()), abs(p.y()-self.sp.y()); sr = min(self.r, w/2, h/2)
            pa = QPainterPath(); pa.addRoundedRect(x, y, w, h, sr, sr); self.cur.setPath(pa)
    def mouse_release(self, e, p): 
        if self.cur: 
            self.cur.setData(0, "Closed Shape"); self.cur.setData(1, "Rounded Rect")
            self.keep_item(self.cur); self.cur = None
            self.status_message.emit("refresh_table")

# --- Oval ---
class OvalTool(RectTool):
    def __init__(self, c): super().__init__(c); self.default_hint = "Left Drag: Draw oval. Hold Shift for Circle."
    def mouse_press(self, e, p):
        if e.button() == Qt.LeftButton: self.sp = p; self.cur = QGraphicsEllipseItem(p.x(), p.y(), 0, 0); self.cur.setPen(self.pen); self.scene.addItem(self.cur)
    def mouse_release(self, e, p): 
        if self.cur:
            self.cur.setData(0, "Closed Shape"); self.cur.setData(1, "Oval")
            self.keep_item(self.cur); self.cur = None
            self.status_message.emit("refresh_table")

# --- Polygon ---
class PolygonSelectionTool(ToolBase):
    def __init__(self, c): 
        super().__init__(c); self.cur=None; self.pts=[]; self.tm=None; self.pen=QPen(QColor(0,255,255), 2); self.pen.setCosmetic(True)
        self.default_hint = "Left Click: Add vertices. Double-click/Right Click: Close polygon."
    def mouse_press(self, e, p):
        if e.button() == Qt.LeftButton:
            if self.pts and len(self.pts) > 2 and dist(p, self.pts[0]) < 10: self._fin(); return
            if not self.pts: self.pts = [p]; self.cur = QGraphicsPolygonItem(QPolygonF(self.pts)); self.cur.setPen(self.pen); self.scene.addItem(self.cur)
            else: self.pts.append(p); self.cur.setPolygon(QPolygonF(self.pts))
    def context_menu(self, e): self._fin() 
    def mouse_move_no_drag(self, p):
        if self.pts: 
            if self.tm: self.scene.removeItem(self.tm)
            self.tm = QGraphicsLineItem(self.pts[-1].x(), self.pts[-1].y(), p.x(), p.y()); self.tm.setPen(QPen(Qt.white, 1, Qt.DotLine)); self.scene.addItem(self.tm)
    def mouse_double_click(self, e, p): self._fin()
    def _fin(self):
        if self.tm: self.scene.removeItem(self.tm); self.tm = None
        if len(self.pts) < 3: return
        self.cur.setData(0, "Closed Shape"); self.cur.setData(1, "Polygon")
        self.keep_item(self.cur); self.pts = []; self.cur = None
        self.status_message.emit("refresh_table")

# --- Freehand Selection ---
class FreehandSelectionTool(ToolBase):
    def __init__(self, c): 
        super().__init__(c); self.p=None; self.pts=[]; self.pen=QPen(QColor(0,255,255), 2); self.pen.setCosmetic(True)
        self.default_hint = "Left Drag: Draw freehand boundary. Release to auto-close."
    def mouse_press(self, e, p): 
        if e.button() == Qt.LeftButton: self.pts = [p]; self.path = QPainterPath(p); self.p = QGraphicsPathItem(self.path); self.p.setPen(self.pen); self.scene.addItem(self.p)
    def mouse_drag(self, e, p): 
        if self.p: self.path.lineTo(p); self.p.setPath(self.path); self.pts.append(p)
    def mouse_release(self, e, p):
        if self.pts:
            self.pts.append(self.pts[0]); self.path.closeSubpath(); self.scene.removeItem(self.p)
            poly = QPolygonF(self.pts); fi = QGraphicsPolygonItem(poly); fi.setPen(self.pen)
            fi.setData(0, "Closed Shape"); fi.setData(1, "Freehand")
            self.scene.addItem(fi); self.keep_item(fi)
            self.p = None; self.pts = []
            self.status_message.emit("refresh_table")

# --- Brush Selection ---
class BrushSelectionTool(ToolBase):
    def __init__(self, c): 
        super().__init__(c); self.target=None; self.bw=20; self.default_hint = "Left Drag: Add to existing area. Hold Alt+Drag: Subtract area."
    def set_size(self, w): self.bw = w
    def mouse_press(self, e, p):
        if e.button() == Qt.LeftButton:
            r = self.bw / 2; detect_rect = QRectF(p.x() - r, p.y() - r, self.bw, self.bw)
            its = self.scene.items(detect_rect); found = None
            for i in its:
                if isinstance(i, (QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPolygonItem, QGraphicsPathItem)): found = convert_to_path_item(i, self.scene); break
            if found: self.target = found
            else:
                if not (e.modifiers() & Qt.AltModifier): self.target = QGraphicsPathItem(); self.target.setPen(QPen(QColor(0,255,255), 2)); self.scene.addItem(self.target)
                else: self.target = None
            if self.target: self._paint(p, e.modifiers())
    def mouse_drag(self, e, p): 
        if self.target: self._paint(p, e.modifiers())
    def mouse_release(self, e, p): 
        if self.target:
            self.target.setData(0, "Closed Shape"); self.target.setData(1, "Brush")
            self.keep_item(self.target); self.target = None
            self.status_message.emit("refresh_table")
    def _paint(self, p, mods):
        bs = QPainterPath(); bs.addEllipse(p, self.bw/2, self.bw/2); cp = self.target.path()
        np = (cp.subtracted(bs) if mods & Qt.AltModifier else cp.united(bs)); self.target.setPath(np)

# --- Magnetic Lasso (Intelligent Scissors) ---
class WandTool(ToolBase):
    def __init__(self, c):
        super().__init__(c); self.scissors = None; self.is_active = False; self.anchor_pts = []; self.current_contour = []; self.scale_factor = 1.0; self.history_contours = [] 
        self.pen_confirmed = QPen(QColor(0, 255, 255), 2); self.pen_confirmed.setCosmetic(True)
        self.pen_preview = QPen(QColor(0, 255, 255), 1.5, Qt.DashLine); self.pen_preview.setCosmetic(True)
        self.preview_item = None; self.confirmed_item = None
        self.default_hint = "Left Click: Start/Add Anchor. Right Click: Undo anchor. Dbl-click: Close."

    def _to_calc(self, x, y): return (int(x * self.scale_factor), int(y * self.scale_factor))
    def _to_orig(self, x, y): return (int(x / self.scale_factor), int(y / self.scale_factor))

    def _init_scissors(self, p):
        if not self.canvas.pixmap_item: return False
        pm = self.canvas.pixmap_item.pixmap(); img_src = qimage_to_numpy(pm.toImage()); h, w = img_src.shape[:2]; max_calc_dim = 800.0
        if max(h, w) > max_calc_dim:
            self.scale_factor = max_calc_dim / max(h, w); img_calc = cv2.resize(img_src, (int(w * self.scale_factor), int(h * self.scale_factor)), interpolation=cv2.INTER_AREA)
        else: self.scale_factor = 1.0; img_calc = img_src
        self.scissors = cv2.segmentation.IntelligentScissorsMB()
        self.scissors.setEdgeFeatureCannyParameters(32, 100); self.scissors.setGradientMagnitudeMaxLimit(200)
        self.canvas.setCursor(Qt.WaitCursor); self.scissors.applyImage(img_calc); self.canvas.setCursor(Qt.CrossCursor)
        return True

    def mouse_press(self, e, p): 
        ix, iy = int(p.x()), int(p.y())
        pm = self.canvas.pixmap_item.pixmap() if self.canvas.pixmap_item else None
        if not pm or ix < 0 or ix >= pm.width() or iy < 0 or iy >= pm.height(): return
        if not self.is_active:
            if self._init_scissors(p):
                self.is_active = True; self.anchor_pts = [(ix, iy)]; self.current_contour = [(ix, iy)]
                cx, cy = self._to_calc(ix, iy); self.scissors.buildMap((cx, cy)); self._update_confirmed_drawing()
        else:
            if self.scissors:
                cx, cy = self._to_calc(ix, iy); contour = self.scissors.getContour((cx, cy))
                if contour is not None:
                    self.history_contours.append(list(self.current_contour))
                    pts = contour.reshape(-1, 2)
                    for pt in pts: ox, oy = self._to_orig(pt[0], pt[1]); self.current_contour.append((ox, oy))
                    self.anchor_pts.append((ix, iy)); self.scissors.buildMap((cx, cy)); self._update_confirmed_drawing()

    def mouse_double_click(self, e, p):
        if self.is_active and len(self.anchor_pts) >= 2: self._close_polygon()

    def context_menu(self, e):
        if self.is_active and len(self.anchor_pts) > 1:
            self.anchor_pts.pop(); self.current_contour = self.history_contours.pop()
            cx, cy = self._to_calc(self.anchor_pts[-1][0], self.anchor_pts[-1][1]); self.scissors.buildMap((cx, cy)); self._update_confirmed_drawing()
            if self.preview_item: self.scene.removeItem(self.preview_item); self.preview_item = None
        elif self.is_active and len(self.anchor_pts) == 1: self._reset_tool()

    def mouse_move_no_drag(self, p):
        if not self.is_active or not self.scissors: return
        ix, iy = int(p.x()), int(p.y()); pm = self.canvas.pixmap_item.pixmap()
        ix = max(0, min(ix, pm.width() - 1)); iy = max(0, min(iy, pm.height() - 1))
        cx, cy = self._to_calc(ix, iy); contour = self.scissors.getContour((cx, cy))
        if contour is not None:
            pts = contour.reshape(-1, 2); path = QPainterPath(); start_ox, start_oy = self._to_orig(pts[0][0], pts[0][1])
            path.moveTo(start_ox, start_oy)
            for pt in pts[1:]: ox, oy = self._to_orig(pt[0], pt[1]); path.lineTo(ox, oy)
            if not self.preview_item: self.preview_item = QGraphicsPathItem(path); self.preview_item.setPen(self.pen_preview); self.scene.addItem(self.preview_item)
            else: self.preview_item.setPath(path)

    def _close_polygon(self):
        if not self.current_contour: return
        pts = [QPointF(pt[0], pt[1]) for pt in self.current_contour]; poly = QPolygonF(pts)
        self._clear_temp_items()
        new_item = QGraphicsPolygonItem(poly); new_item.setPen(self.pen_confirmed)
        new_item.setBrush(Qt.NoBrush) 
        new_item.setFlags(QGraphicsPolygonItem.ItemIsSelectable | QGraphicsPolygonItem.ItemIsMovable)
        new_item.setData(0, "Closed Shape"); new_item.setData(1, "Magic Wand")
        self.scene.addItem(new_item)
        self.keep_item(new_item)
        self._reset_tool(); self.status_message.emit("refresh_table")

    def _update_confirmed_drawing(self):
        if len(self.current_contour) < 2: return
        path = QPainterPath(); path.moveTo(self.current_contour[0][0], self.current_contour[0][1])
        for pt in self.current_contour[1:]: path.lineTo(pt[0], pt[1])
        if not self.confirmed_item: self.confirmed_item = QGraphicsPathItem(path); self.confirmed_item.setPen(self.pen_confirmed); self.scene.addItem(self.confirmed_item)
        else: self.confirmed_item.setPath(path)

    def _clear_temp_items(self):
        if self.preview_item: self.preview_item.setPath(QPainterPath()); self.scene.removeItem(self.preview_item); self.preview_item = None
        if self.confirmed_item: self.confirmed_item.setPath(QPainterPath()); self.scene.removeItem(self.confirmed_item); self.confirmed_item = None

    def _reset_tool(self):
        self._clear_temp_items(); self.is_active = False; self.scissors = None; self.anchor_pts = []; self.current_contour = []; self.history_contours = []

# --- Arrow ---
class ArrowTool(ToolBase):
    def __init__(self, c): 
        super().__init__(c); self.sp=None; self.cur=None; self.width=2; self.default_hint = "Left Drag: Draw arrow pointing from start to end."
    def set_size(self, w): self.width = w
    def mouse_press(self, e, p):
        if e.button() == Qt.LeftButton: self.sp = p; self.cur = QGraphicsLineItem(p.x(), p.y(), p.x(), p.y()); pen = QPen(GlobalState().draw_color, self.width); pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin); self.cur.setPen(pen); self.scene.addItem(self.cur)
    def mouse_drag(self, e, p):
        if self.cur: self.cur.setLine(self.sp.x(), self.sp.y(), p.x(), p.y())
    def mouse_release(self, e, p):
        if self.cur: 
            self.scene.removeItem(self.cur); line = self._draw_arrow(self.sp, p)
            self.keep_item(line); self.cur = None; self.status_message.emit("refresh_table")
            
    def _draw_arrow(self, start, end):
        color = GlobalState().draw_color; angle = math.atan2(end.y() - start.y(), end.x() - start.x()); arrow_len = 15 + self.width * 2; arrow_angle = math.pi / 6
        p1 = QPointF(end.x() - arrow_len * math.cos(angle - arrow_angle), end.y() - arrow_len * math.sin(angle - arrow_angle))
        p2 = QPointF(end.x() - arrow_len * math.cos(angle + arrow_angle), end.y() - arrow_len * math.sin(angle + arrow_angle))
        neck = QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2) 
        pen = QPen(color, self.width); pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin)
        line = QGraphicsLineItem(start.x(), start.y(), neck.x(), neck.y()); line.setPen(pen); self.scene.addItem(line)
        head = QGraphicsPolygonItem(QPolygonF([end, p1, p2])); head.setPen(Qt.NoPen); head.setBrush(QBrush(color)); self.scene.addItem(head); head.setParentItem(line); line.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)
        line.setData(0, "Line Segment"); line.setData(1, "Arrow")
        return line

# --- Text ---
class TextTool(ToolBase):
    def __init__(self, c): 
        super().__init__(c); self.font_size = 12; self.default_hint = "Left Click: Add text label at cursor."
    def set_size(self, s): self.font_size = s
    def mouse_press(self, e, p):
        if e.button() == Qt.LeftButton:
            text, ok = QInputDialog.getText(None, "Add Text", "Enter text:")
            if ok and text:
                ti = QGraphicsTextItem(text); font = QFont("Arial", self.font_size); ti.setFont(font); ti.setDefaultTextColor(GlobalState().draw_color); ti.setPos(p); ti.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable); self.scene.addItem(ti)
                ti.setData(0, "Closed Shape"); ti.setData(1, "Text")
                self.keep_item(ti); self.status_message.emit("refresh_table")

# --- Pencil ---
class PencilTool(ToolBase):
    def __init__(self, c): 
        super().__init__(c); self.cur=None; self.path=None; self.width=2; self.default_hint = "Left Drag: Freehand drawing."
    def set_size(self, w): self.width = w
    def mouse_press(self, e, p):
        if e.button() == Qt.LeftButton: self.path = QPainterPath(p); self.cur = QGraphicsPathItem(self.path); pen = QPen(GlobalState().draw_color, self.width); pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin); self.cur.setPen(pen); self.scene.addItem(self.cur)
    def mouse_drag(self, e, p):
        if self.cur: self.path.lineTo(p); self.cur.setPath(self.path)
    def mouse_release(self, e, p):
        if self.cur: 
            self.cur.setData(0, "Line Segment"); self.cur.setData(1, "Pencil")
            self.keep_item(self.cur); self.cur = None
            self.status_message.emit("refresh_table")

# --- Flood Fill ---
class FloodFillTool(ToolBase):
    def __init__(self, c): 
        super().__init__(c); self.tol=20; self.default_hint = "Left Click: Flood fill areas of similar color."
    def set_size(self, t): self.tol = t
    def mouse_press(self, e, p): 
        if e.button() == Qt.LeftButton: self._fill(p)
    def _fill(self, p):
        if not self.canvas.pixmap_item: return
        pm = self.canvas.pixmap_item.pixmap(); ix, iy = int(p.x()), int(p.y())
        if ix < 0 or ix >= pm.width() or iy < 0 or iy >= pm.height(): return
        bgr = qimage_to_numpy(pm.toImage()); h, w = bgr.shape[:2]; mask = np.zeros((h+2, w+2), np.uint8); flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE
        cv2.floodFill(bgr, mask, (ix, iy), (0, 0, 0), (self.tol,)*3, (self.tol,)*3, flags)
        cnts, _ = cv2.findContours(mask[1:-1, 1:-1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); fill_color = GlobalState().draw_color; fill_color.setAlpha(200) 
        for c in cnts:
            pts = [QPointF(pt[0][0], pt[0][1]) for pt in cv2.approxPolyDP(c, 1.0, True)]; pi = QGraphicsPolygonItem(QPolygonF(pts)); pi.setPen(Qt.NoPen); pi.setBrush(QBrush(fill_color)); self.scene.addItem(pi)
            pi.setData(0, "Closed Shape"); pi.setData(1, "Flood Fill")
            self.keep_item(pi)
        self.status_message.emit("refresh_table")

# --- Color Picker ---
class ColorPickerTool(ToolBase):
    def __init__(self, c):
        super().__init__(c); self.default_hint = "Left Click: Pick color from image to use for drawing."
    def mouse_press(self, e, p):
        if e.button() == Qt.LeftButton:
            if not self.canvas.pixmap_item: return
            pm = self.canvas.pixmap_item.pixmap()
            if pm.toImage().valid(int(p.x()), int(p.y())):
                c = pm.toImage().pixelColor(int(p.x()), int(p.y())); GlobalState().draw_color = c; self.status_message.emit("Color Picked")

# --- Zoom ---
class ZoomTool(ToolBase):
    def __init__(self, c):
        super().__init__(c); self.default_hint = "Left Click: Zoom In. Alt+Click: Zoom Out. Right Click: Reset View."
    def mouse_press(self, e, p):
        if e.button() == Qt.LeftButton:
            f = 1.25; 
            if e.modifiers() & Qt.AltModifier: f = 1.0/1.25
            self.canvas.scale(f, f)
    def context_menu(self, e): self.reset_view_requested.emit()

# --- Hand ---
class HandTool(ToolBase):
    def __init__(self, c): 
        super().__init__(c); self.lp=None; self.default_hint = "Left Drag: Pan image. Right Click: Reset View."
    def mouse_press(self, e, p): 
        if e.button() == Qt.LeftButton: self.lp=e.globalPos(); self.canvas.setCursor(Qt.ClosedHandCursor)
    def mouse_drag(self, e, p):
        if self.lp: d=e.globalPos()-self.lp; h=self.canvas.horizontalScrollBar(); v=self.canvas.verticalScrollBar(); h.setValue(h.value()-d.x()); v.setValue(v.value()-d.y()); self.lp=e.globalPos()
    def mouse_release(self, e, p): self.lp=None; self.canvas.setCursor(Qt.ArrowCursor)
    def context_menu(self, e): self.reset_view_requested.emit()