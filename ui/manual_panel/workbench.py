import os
import csv
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QLabel, QFileDialog, QToolBar, QSlider, QDialog, 
                               QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, 
                               QGraphicsPixmapItem, QColorDialog, QGraphicsTextItem, QGraphicsLineItem,
                               QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPolygonItem, QGraphicsPathItem,
                               QSplitter, QAbstractItemView, QMenu,QGraphicsItem)
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPixmap, QAction, QActionGroup, QColor, QPainter, QImage

from ui import MENU_STYLESHEET
from .canvas import GraphicsCanvas
from .tools import (LineTool, PolylineTool, FreehandTool, AngleTool, EraserTool, PointTool,
                    RectTool, RoundedRectTool, OvalTool, PolygonSelectionTool,
                    FreehandSelectionTool, BrushSelectionTool, WandTool, SelectTool,
                    ArrowTool, TextTool, PencilTool, FloodFillTool, ColorPickerTool,
                    ZoomTool, HandTool, get_item_metrics)
from core.global_state import GlobalState

class ManualWorkbench(QWidget):
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.active_tool = None
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # -- Left panel --
        self.left_panel = QWidget()
        self.left_panel.setFixedWidth(250)
        self.left_panel.setStyleSheet("QWidget { background-color: #ffffff; border-right: 1px solid #e0e0e0; }")
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(10, 15, 10, 15)
        left_layout.setSpacing(10)

        # --- Buttons ---
        from PySide6.QtWidgets import QGridLayout 
        btn_layout = QGridLayout()
        btn_layout.setSpacing(8)
        
        self.btn_load = QPushButton("📂 Open Image")
        self.btn_load.clicked.connect(self.load_image)
        self.btn_load.setStyleSheet("QPushButton { font-weight: bold; }")
        self.btn_save = QPushButton("💾 Export CSV")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_save.setStyleSheet("QPushButton { background-color: #1976d2; color: white; border: none; padding: 6px; border-radius: 4px; font-weight: bold; }")
        self.btn_reset = QPushButton("🔍 1:1 Reset")
        self.btn_reset.clicked.connect(self.reset_scale)
        self.btn_reset.setStyleSheet("QPushButton { font-weight: bold; }")
        self.btn_clear = QPushButton("🗑️ Clear All")
        self.btn_clear.clicked.connect(self.clear_all_items)
        self.btn_clear.setStyleSheet("QPushButton { font-weight: bold; }")
        
        btn_layout.addWidget(self.btn_load, 0, 0); btn_layout.addWidget(self.btn_save, 0, 1)
        btn_layout.addWidget(self.btn_reset, 1, 0); btn_layout.addWidget(self.btn_clear, 1, 1)
        left_layout.addLayout(btn_layout)

        # --- Table ---
        self.table_details = QTableWidget()
        self.table_details.setColumnCount(9)
        self.table_details.setHorizontalHeaderLabels(["ID", "Type", "Tool", "Len", "Wid", "Area", "Perim", "RGB", "Swatch"])
        init_widths = [28, 46, 50, 48, 42, 50, 50, 62, 42]
        for i, w in enumerate(init_widths):
            self.table_details.setColumnWidth(i, w)

        self.table_details.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table_details.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table_details.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table_details.horizontalHeader().setStretchLastSection(False)
        
        self.table_details.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_details.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        self.table_details.setStyleSheet("""
            QTableWidget { background-color: white; border: 1px solid #ccc; font-size: 12px; color: #333; }
            QHeaderView::section { background-color: #f5f5f5; font-weight: bold; border: 1px solid #ddd; padding: 4px; font-size: 12px; color: #333; }
            QTableWidget::item:selected { background-color: #1976d2; color: white; } 
        """)
        
        self.table_details.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_details.customContextMenuRequested.connect(self.show_table_context_menu)
        self.table_details.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_details.horizontalHeader().customContextMenuRequested.connect(self.show_header_context_menu)
        self.table_details.itemDoubleClicked.connect(self.highlight_canvas_item)
        
        left_layout.addWidget(self.table_details)
        main_layout.addWidget(self.left_panel)

        # -- Right panel --
        self.right_panel = QWidget()
        self.right_panel.setStyleSheet("QWidget { background-color: #e8e8e8; }")
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0); right_layout.setSpacing(0)

        # -- Upper toolbar --
        self.toolbar_upper = QToolBar()
        self.toolbar_upper.setFixedHeight(36)
        self.toolbar_upper.setStyleSheet("""
            QToolBar { spacing: 5px; padding: 5px; background: #ffffff; border-bottom: 1px solid #eeeeee; }
            QToolButton { 
                padding: 4px; font-weight: bold; border-radius: 4px; 
                background-color: #ffffff; /* default white */
                border: 1px solid #cccccc; /* border prevents size jump on check */
            }
            QToolButton:hover { background-color: #f5f5f5; }
            QToolButton:checked {
                background-color: #e0e0e0; /* selected state */
                border: 1px solid #999999; /* same width, deeper color */
            }
        """)
        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)
        
        tools_config_upper = [
            ("↖ Select", "select"), ("🔍 Zoom", "zoom"), ("✋ Hand", "hand"), ("|", "sep"),
            ("⬇️ Arrow", "arrow"), ("T Text", "text"), ("✏️ Pencil", "pencil"),
            ("💧 Fill", "flood"), ("💉 Picker", "picker"), ("🧽 Erase", "eraser")
        ]
        self.add_tools_to_bar(self.toolbar_upper, tools_config_upper)
        
        self.toolbar_upper.addSeparator()
        self.lbl_param = QLabel(" Size: ")
        self.lbl_param.setStyleSheet("font-weight: bold; color: #555; font-size: 12px; background-color: #ffffff;")
        self.toolbar_upper.addWidget(self.lbl_param)

        self.slider_param = QSlider(Qt.Horizontal)
        pal = self.slider_param.palette()
        pal.setColor(self.slider_param.backgroundRole(), QColor("#ffffff"))
        self.slider_param.setAutoFillBackground(True)
        self.slider_param.setPalette(pal)
        
        self.slider_param.setRange(1, 100)
        self.slider_param.setValue(10)
        self.slider_param.setFixedWidth(100)
        self.slider_param.valueChanged.connect(self.update_tool_param)
        self.toolbar_upper.addWidget(self.slider_param)

        self.toolbar_upper.addSeparator()
        self.btn_color = QPushButton()
        self.btn_color.setFixedWidth(24); self.btn_color.setFixedHeight(24)
        self.btn_color.setStyleSheet(f"background-color: {GlobalState().draw_color.name()}; border: 1px solid #999; border-radius: 4px;")
        self.btn_color.clicked.connect(self.choose_color)
        self.toolbar_upper.addWidget(self.btn_color)

        right_layout.addWidget(self.toolbar_upper)

        # -- Lower toolbar --
        self.toolbar_lower = QToolBar()
        self.toolbar_lower.setFixedHeight(36)
        self.toolbar_lower.setStyleSheet("""
            QToolBar { spacing: 5px; padding: 5px; background: #ffffff; border-bottom: 1px solid #eeeeee; } 
            QToolButton { 
                padding: 4px; font-weight: bold; border-radius: 4px; 
                background-color: #ffffff; 
                border: 1px solid #cccccc; 
            } 
            QToolButton:hover { background-color: #f5f5f5; }
            QToolButton:checked { 
                background-color: #e0e0e0; 
                border: 1px solid #999999; 
            }
        """)
        tools_config_lower = [
            ("📏 Line", "line"), ("📉 Poly", "poly"), ("〰 Free", "free"),
            ("📐 Angle", "angle"), ("🎯 Point", "point"), ("|", "sep"),
            ("⬜ Rect", "rect"), ("⬜ R-Rect", "round_rect"), ("⚪ Oval", "oval"),
            ("⬠ PolySel", "poly_sel"), ("〰 FreeSel", "free_sel"),
            ("🖌️ Brush", "brush_sel"), ("🪄 Magic", "wand")
        ]
        self.add_tools_to_bar(self.toolbar_lower, tools_config_lower)
        right_layout.addWidget(self.toolbar_lower)

        # --- Canvas ---
        self.canvas = GraphicsCanvas()
        self.canvas.mouse_moved.connect(self.update_coords)
        self.canvas.tool_message.connect(self.update_message)
        right_layout.addWidget(self.canvas)

        # --- Status bar ---
        status_bar = QWidget()
        status_bar.setStyleSheet("background-color: #f5f5f5; border-top: 1px solid #ccc;")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(10, 5, 10, 5)
        self.lbl_status = QLabel()
        self.lbl_status.setStyleSheet("color: #333; font-weight: bold;")
        status_layout.addWidget(self.lbl_status)
        status_layout.addStretch()
        self.lbl_hints = QLabel("")
        self.lbl_hints.setStyleSheet("color: #666; font-style: italic; font-family: Arial;")
        status_layout.addWidget(self.lbl_hints)
        right_layout.addWidget(status_bar)

        main_layout.addWidget(self.right_panel)

    # --- Context menu & double-click highlight ---
    def show_table_context_menu(self, pos):
        item = self.table_details.itemAt(pos)
        if not item: return
        row = item.row()
        menu = QMenu(self)
        menu.setStyleSheet(MENU_STYLESHEET)
        del_action = menu.addAction("🗑️ Delete This Object")
        if menu.exec(self.table_details.viewport().mapToGlobal(pos)) == del_action:
            item_obj = self.table_details.item(row, 0).data(Qt.UserRole)
            if item_obj and item_obj.scene(): 
                self.canvas.scene.removeItem(item_obj)
                if hasattr(self.canvas.scene, "kept_items") and item_obj in self.canvas.scene.kept_items:
                    self.canvas.scene.kept_items.remove(item_obj)
            self.refresh_measurement_table()

    def show_header_context_menu(self, pos):
        header = self.table_details.horizontalHeader()
        col = header.logicalIndexAt(pos)
        if col <= 0: return 
        col_name = self.table_details.horizontalHeaderItem(col).text()
        menu = QMenu(self)
        menu.setStyleSheet(MENU_STYLESHEET)
        hide_action = menu.addAction(f"🗑️ Hide Column: {col_name}")
        if menu.exec(header.mapToGlobal(pos)) == hide_action:
            self.table_details.setColumnHidden(col, True)

    def highlight_canvas_item(self, table_item):
        row = table_item.row()
        id_item = self.table_details.item(row, 0)
        target_item = id_item.data(Qt.UserRole)
        if target_item and target_item.scene():
            self.canvas.scene.clearSelection()
            target_item.setFlags(target_item.flags() | QGraphicsItem.ItemIsSelectable)
            target_item.setSelected(True)
            self.canvas.centerOn(target_item)

    # --- Measurement table refresh ---
    def refresh_measurement_table(self):
        if not hasattr(self, 'canvas') or not self.canvas.scene: return
        self.table_details.setRowCount(0)
        
        valid_items = []
        point_count = 0
        first_point_item = None  # used for double-click table->canvas navigation
        
        for item in self.canvas.scene.items():
            if isinstance(item, QGraphicsPixmapItem) or item.parentItem() is not None: continue
            m = get_item_metrics(item)
            if not m or m.get("Type") == "-": continue 
            
            # Point tool: accumulate count, merge into a single summary row later
            if m.get("Type") == "Count":
                point_count += 1
                if first_point_item is None: first_point_item = item 
                continue
                
            valid_items.append({"item": item, "data": m})
            
        valid_items.reverse() 
        
        # Merge all point counts into a single summary row at the bottom
        if point_count > 0:
            summary_data = {
                "Type": "Count", "Tool": "Point", "Length": f"Total: {point_count}", 
                "Width": "-", "Area": "-", "Perim": "-", "RGB": "-", "Swatch": "-"
            }
            valid_items.append({"item": first_point_item, "data": summary_data})

        self.table_details.setRowCount(len(valid_items))
        for row, info in enumerate(valid_items):
            item_obj = info["item"]; d = info["data"]
            id_item = QTableWidgetItem(str(row + 1)); id_item.setData(Qt.UserRole, item_obj)
            self.table_details.setItem(row, 0, id_item)
            self.table_details.setItem(row, 1, QTableWidgetItem(d.get("Type", "-")))
            self.table_details.setItem(row, 2, QTableWidgetItem(d.get("Tool", "-")))
            self.table_details.setItem(row, 3, QTableWidgetItem(d.get("Length", "-")))
            self.table_details.setItem(row, 4, QTableWidgetItem(d.get("Width", "-")))
            self.table_details.setItem(row, 5, QTableWidgetItem(d.get("Area", "-")))
            self.table_details.setItem(row, 6, QTableWidgetItem(d.get("Perim", "-")))
            self.table_details.setItem(row, 7, QTableWidgetItem(d.get("RGB", "-")))
            swatch_item = QTableWidgetItem("")
            hex_code = d.get("Swatch", "-")
            if hex_code.startswith("#"): swatch_item.setBackground(QColor(hex_code))
            self.table_details.setItem(row, 8, swatch_item)
            
        # Resize columns to fit content, preventing truncation
        self.table_details.resizeColumnsToContents()

    # --- Utilities ---
    def add_tools_to_bar(self, bar, config):
        for name, key in config:
            if key == "sep": bar.addSeparator(); continue
            action = QAction(name, self); action.setCheckable(True)
            action.triggered.connect(lambda checked, k=key: self.set_tool(k))
            bar.addAction(action); self.tool_group.addAction(action)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.jpg *.png *.bmp *.tif)")
        if path:
            self.current_image_path = path; pixmap = QPixmap(path)
            self.canvas.set_image(pixmap)
            self.refresh_measurement_table()

    def save_result(self):
        if not self.canvas.scene or not self.canvas.pixmap_item: return
        default_path = ""
        if self.current_image_path:
            base_dir = os.path.dirname(self.current_image_path)
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            default_path = os.path.join(base_dir, f"{base_name}_result.png")
            
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Result", default_path, "PNG (*.png);;JPG (*.jpg)")
        if save_path:
            orig_pixmap = self.canvas.pixmap_item.pixmap()
            image = QImage(orig_pixmap.size(), QImage.Format_ARGB32); image.fill(Qt.transparent)
            painter = QPainter(image)
            self.canvas.scene.render(painter, QRectF(image.rect()), self.canvas.pixmap_item.sceneBoundingRect())
            painter.end(); image.save(save_path)
            
            csv_path = os.path.splitext(save_path)[0] + ".csv"
            headers, visible_cols = [], []
            for c in range(self.table_details.columnCount()):
                if not self.table_details.isColumnHidden(c):
                    headers.append(self.table_details.horizontalHeaderItem(c).text())
                    visible_cols.append(c)
            export_data = []
            for r in range(self.table_details.rowCount()):
                row_data = {}
                for h, c in zip(headers, visible_cols):
                    row_data[h] = self.table_details.item(r, c).text()
                export_data.append(row_data)
            if export_data:
                with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader(); writer.writerows(export_data)
            self.lbl_status.setText(f"Saved: Image & CSV exported to {os.path.basename(save_path)}")

    def reset_scale(self): GlobalState().reset(); self.refresh_measurement_table()
    def fit_view(self):
        if self.canvas.pixmap_item: self.canvas.fitInView(self.canvas.pixmap_item, Qt.KeepAspectRatio)
    def choose_color(self):
        c = QColorDialog.getColor(GlobalState().draw_color, self, "Select Color")
        if c.isValid():
            GlobalState().draw_color = c
            self.btn_color.setStyleSheet(f"background-color: {c.name()}; border: 1px solid #999; border-radius: 4px;")
            items = self.canvas.scene.selectedItems()
            for item in items:
                if isinstance(item, QGraphicsTextItem): item.setDefaultTextColor(c)
                elif isinstance(item, QGraphicsLineItem):
                    pen = item.pen(); pen.setColor(c); item.setPen(pen)
                    for child in item.childItems(): 
                        if hasattr(child, 'setBrush'): child.setBrush(QColor(c))
    def update_coords(self, x, y): pass
    
    def update_message(self, msg): 
        if msg == "refresh_table": self.refresh_measurement_table(); return
        
        # Route hints to the right side, keep status line for completion messages
        if "Click" in msg or "Drag" in msg or "Hover" in msg or "Ready." in msg or "Active:" in msg: 
            self.lbl_hints.setText(f"💡 {msg}")
            self.lbl_status.clear() 
        elif "Color Picked" in msg:
            c = GlobalState().draw_color
            self.btn_color.setStyleSheet(f"background-color: {c.name()}; border: 1px solid #999; border-radius: 4px;")
            self.lbl_status.clear()
        else: 
            self.lbl_status.setText(msg)

    def set_tool(self, tool_name):
        msg = ""; slider_val = 10
        if tool_name == "select": self.active_tool = SelectTool(self.canvas); msg="Selection Cursor"
        elif tool_name == "zoom": self.active_tool = ZoomTool(self.canvas); msg="Zoom Tool"
        elif tool_name == "hand": self.active_tool = HandTool(self.canvas); msg="Hand Tool"
        elif tool_name == "rect": self.active_tool=RectTool(self.canvas); msg="Rect Active"
        elif tool_name == "round_rect": self.active_tool=RoundedRectTool(self.canvas); msg="Rounded Rect Active"; slider_val=20
        elif tool_name == "oval": self.active_tool=OvalTool(self.canvas); msg="Oval Active"
        elif tool_name == "poly_sel": self.active_tool=PolygonSelectionTool(self.canvas); msg="Polygon Active"
        elif tool_name == "free_sel": self.active_tool=FreehandSelectionTool(self.canvas); msg="Freehand Active"
        elif tool_name == "brush_sel": self.active_tool=BrushSelectionTool(self.canvas); msg="Brush Active"; slider_val=20
        elif tool_name == "wand": self.active_tool=WandTool(self.canvas); msg="Magic Lasso Active"; slider_val=1
        elif tool_name == "line": self.active_tool=LineTool(self.canvas); msg="Line Active"
        elif tool_name == "poly": self.active_tool=PolylineTool(self.canvas); msg="Polyline Active"
        elif tool_name == "free": self.active_tool=FreehandTool(self.canvas); msg="Freehand Line Active"
        elif tool_name == "angle": self.active_tool=AngleTool(self.canvas); msg="Angle Tool Active"
        elif tool_name == "point": self.active_tool=PointTool(self.canvas); msg="Point Tool Active"
        elif tool_name == "arrow": self.active_tool = ArrowTool(self.canvas); msg="Arrow Active"; slider_val=2; self.slider_param.setRange(1, 20)
        elif tool_name == "text": self.active_tool = TextTool(self.canvas); msg="Text Active"; slider_val=20; self.slider_param.setRange(8, 72)
        elif tool_name == "pencil": self.active_tool = PencilTool(self.canvas); msg="Pencil Active"; slider_val=3; self.slider_param.setRange(1, 50)
        elif tool_name == "flood": self.active_tool = FloodFillTool(self.canvas); msg="Flood Fill Active"; slider_val=20
        elif tool_name == "picker": self.active_tool = ColorPickerTool(self.canvas); msg="Color Picker Active"
        elif tool_name == "eraser": self.active_tool = EraserTool(self.canvas); msg="Eraser Active"; slider_val=20; self.slider_param.setRange(5, 100)
        
        if tool_name not in ["arrow", "text", "pencil", "eraser"]: self.slider_param.setRange(1, 100)
        if hasattr(self.active_tool, 'reset_view_requested'): self.active_tool.reset_view_requested.connect(self.fit_view)
        
        self.canvas.set_tool(self.active_tool)
        
        if hasattr(self.active_tool, 'show_hint'): self.active_tool.show_hint()
        if hasattr(self.active_tool, 'set_size'):
            self.slider_param.blockSignals(True); self.slider_param.setValue(slider_val); self.slider_param.blockSignals(False); self.active_tool.set_size(slider_val)

    def update_tool_param(self, val):
        if hasattr(self.active_tool, 'set_size'): self.active_tool.set_size(val)
        self.lbl_status.clear()  # suppress slider parameter echo in status bar

    def clear_all_items(self):
        if QMessageBox.question(self, 'Clear All', 'Remove ALL items?', QMessageBox.Yes|QMessageBox.No)==QMessageBox.Yes:
            for i in self.canvas.scene.items():
                if not isinstance(i, QGraphicsPixmapItem) and not (i.parentItem() and isinstance(i.parentItem(), QGraphicsPixmapItem)): 
                    self.canvas.scene.removeItem(i)
            # Clear the underlying item-tracking list as well
            if hasattr(self.canvas.scene, "kept_items"): self.canvas.scene.kept_items.clear()
            self.lbl_status.setText("All items cleared."); self.refresh_measurement_table()