import os
import glob
import json
import random
import numpy as np
import cv2
from core.utils import imread_unicode

from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
                               QListWidget, QListWidgetItem, QLabel, QGroupBox,
                               QSplitter, QSlider, QToolButton, QAbstractItemView,
                               QColorDialog, QInputDialog, QFileDialog, QMessageBox,
                               QComboBox, QGraphicsPolygonItem, QDialog, QLineEdit,
                               QMenu, QGridLayout, QCheckBox, QGraphicsPixmapItem)
from PySide6.QtCore import Qt, QThread, Signal, QPointF, QRectF
from PySide6.QtGui import (QIcon, QColor, QFont, QPixmap, QImage, QPolygonF,
                           QKeySequence, QShortcut, QPainter, QPen, QBrush)

from ui import MENU_STYLESHEET
from ui.manual_panel.canvas import GraphicsCanvas
from ui.manual_panel.tools import HandTool
from .lasso_tool import MagneticLassoTool
from .sam_point_tool import SAMPointTool
from .sam_box_tool import SAMBoxTool
from .edit_tool import EditTool
from .freehand_poly_tool import FreehandPolyTool
from .polygon_tool import PolygonTool
from .auto_3d_tool import Auto3DTool


class RangeSlider(QWidget):
    """Dual-handle range slider for depth filtering."""
    valueChanged = Signal(int, int)
    sliderReleased = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(24)
        self._min = 0
        self._max = 255
        self._low = 0
        self._high = 255
        self._active_handle = None
        self.last_active_handle = 'high'

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()

        painter.setBrush(QBrush(QColor(40, 40, 40)))
        painter.setPen(Qt.NoPen)
        painter.drawRect(QRectF(10, h / 2 - 2, w - 20, 4))

        low_x = 10 + (self._low / self._max) * (w - 20)
        high_x = 10 + (self._high / self._max) * (w - 20)
        painter.setBrush(QBrush(QColor(120, 120, 120)))
        painter.drawRect(QRectF(low_x, h / 2 - 2, high_x - low_x, 4))

        hw, hh = 10, 16

        if self.last_active_handle == 'low':
            painter.setPen(QPen(QColor("#1976d2"), 2))
            painter.setBrush(QBrush(QColor(220, 220, 220)))
        else:
            painter.setPen(QPen(QColor(20, 20, 20), 1))
            painter.setBrush(QBrush(QColor(150, 150, 150)))
        painter.drawRect(QRectF(low_x - hw / 2, h / 2 - hh / 2, hw, hh))

        if self.last_active_handle == 'high':
            painter.setPen(QPen(QColor("#1976d2"), 2))
            painter.setBrush(QBrush(QColor(220, 220, 220)))
        else:
            painter.setPen(QPen(QColor(20, 20, 20), 1))
            painter.setBrush(QBrush(QColor(150, 150, 150)))
        painter.drawRect(QRectF(high_x - hw / 2, h / 2 - hh / 2, hw, hh))

    def mousePressEvent(self, event):
        w = self.width()
        low_x = 10 + (self._low / self._max) * (w - 20)
        high_x = 10 + (self._high / self._max) * (w - 20)
        mx = event.position().x()

        if abs(mx - low_x) < 12:
            self._active_handle = 'low'
            self.last_active_handle = 'low'
        elif abs(mx - high_x) < 12:
            self._active_handle = 'high'
            self.last_active_handle = 'high'
        else:
            self._active_handle = None

        self.update()

    def mouseMoveEvent(self, event):
        if not self._active_handle:
            return
        w = self.width()
        val = int(((event.position().x() - 10) / (w - 20)) * self._max)
        val = max(self._min, min(self._max, val))

        changed = False
        if self._active_handle == 'low':
            new_low = min(val, self._high)
            if new_low != self._low:
                self._low = new_low
                changed = True
        else:
            new_high = max(val, self._low)
            if new_high != self._high:
                self._high = new_high
                changed = True

        if changed:
            self.update()
            self.valueChanged.emit(self._low, self._high)

    def mouseReleaseEvent(self, event):
        if self._active_handle:
            self._active_handle = None
            self.sliderReleased.emit(self._low, self._high)

    def setValues(self, low, high):
        self._low = max(self._min, min(high, low))
        self._high = min(self._max, max(low, high))
        self.update()


class CategoryEditDialog(QDialog):
    def __init__(self, name, color, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Category")
        self.setFixedSize(300, 150)
        self.name = name
        self.color = color
        layout = QVBoxLayout(self)
        self.name_input = QLineEdit(self.name)
        layout.addWidget(QLabel("Category Name:"))
        layout.addWidget(self.name_input)
        self.color_btn = QPushButton()
        self.update_color_btn()
        self.color_btn.clicked.connect(self.choose_color)
        layout.addWidget(QLabel("Category Color:"))
        layout.addWidget(self.color_btn)
        btn_box = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        layout.addLayout(btn_box)

    def update_color_btn(self):
        self.color_btn.setText(f"Color: {self.color.name()}")
        fg = 'white' if self.color.lightness() < 128 else 'black'
        self.color_btn.setStyleSheet(
            f"background-color: {self.color.name()}; color: {fg}; "
            f"font-weight: bold; padding: 5px; border-radius: 4px;")

    def choose_color(self):
        c = QColorDialog.getColor(self.color, self, "Select Color")
        if c.isValid():
            c.setAlpha(self.color.alpha())
            self.color = c
            self.update_color_btn()


class HotkeyDialog(QDialog):
    def __init__(self, current_hotkeys, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⌨️ Shortcut Settings")
        self.resize(550, 400)
        self.hotkeys = current_hotkeys
        self.inputs = {}
        layout = QGridLayout(self)
        row, col = 0, 0
        for action, key in self.hotkeys.items():
            layout.addWidget(QLabel(action), row, col * 2)
            line_edit = QLineEdit(key)
            self.inputs[action] = line_edit
            layout.addWidget(line_edit, row, col * 2 + 1)
            col += 1
            if col > 1:
                col = 0
                row += 1
        btn_save = QPushButton("Save & Apply Hotkeys")
        btn_save.setStyleSheet(
            "background: #1976d2; color: white; padding: 8px; "
            "font-weight: bold;")
        btn_save.clicked.connect(self.save)
        layout.addWidget(btn_save, row + 1, 0, 1, 4)

    def save(self):
        for action, line_edit in self.inputs.items():
            self.hotkeys[action] = line_edit.text().strip().upper()
        self.accept()


class FormatConverterDialog(QDialog):
    def __init__(self, categories, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔄 Dataset Format Converter")
        self.resize(500, 200)
        self.categories = categories
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Convert all JSON files in the current directory to "
            "YOLO Segmentation format."))
        self.btn_convert = QPushButton(
            "🚀 Start Conversion (LabelMe -> YOLO-Seg txt)")
        self.btn_convert.setStyleSheet(
            "background: #d32f2f; color: white; font-size: 14px; "
            "font-weight: bold; padding: 15px;")
        self.btn_convert.clicked.connect(self.run_conversion)
        layout.addWidget(self.btn_convert)

    def run_conversion(self):
        parent_panel = self.parent()
        if not parent_panel or parent_panel.list_files.count() == 0:
            return QMessageBox.warning(
                self, "Error",
                "Please load an image directory first!")
        class_mapping = {
            name: idx for idx, name in enumerate(self.categories.keys())}
        first_img_path = parent_panel.list_files.item(0).data(Qt.UserRole)
        base_dir = os.path.dirname(first_img_path)
        json_files = glob.glob(os.path.join(base_dir, "*.json"))
        if not json_files:
            return QMessageBox.warning(
                self, "Error", "No JSON files found.")
        converted_count = 0
        for j_path in json_files:
            try:
                with open(j_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                img_w, img_h = data['imageWidth'], data['imageHeight']
                txt_path = os.path.splitext(j_path)[0] + ".txt"
                with open(txt_path, 'w', encoding='utf-8') as f_out:
                    for shape in data.get('shapes', []):
                        cls_name = shape['label']
                        if cls_name not in class_mapping:
                            continue
                        cls_id = class_mapping[cls_name]
                        normalized_pts = []
                        for pt in shape['points']:
                            normalized_pts.extend([
                                f"{max(0.0, min(1.0, pt[0] / img_w)):.6f}",
                                f"{max(0.0, min(1.0, pt[1] / img_h)):.6f}"])
                        f_out.write(
                            f"{cls_id} " + " ".join(normalized_pts) + "\n")
                converted_count += 1
            except Exception:
                pass
        with open(os.path.join(base_dir, "classes.txt"), 'w',
                  encoding='utf-8') as f:
            for name in class_mapping.keys():
                f.write(name + "\n")
        QMessageBox.information(
            self, "Success",
            f"Converted {converted_count} files to YOLO format!\n"
            f"Classes saved to classes.txt")
        self.accept()


class DepthInferenceThread(QThread):
    finished_signal = Signal(object, object)
    error_signal = Signal(str)

    def __init__(self, img_path, sam_type_to_keep):
        super().__init__()
        self.img_path = img_path
        self.sam_type = sam_type_to_keep

    def run(self):
        try:
            img_bgr = imread_unicode(self.img_path)
            if img_bgr is None:
                raise ValueError("Failed to load image")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            from core.model_loader import ModelLoader
            loader = ModelLoader()
            depth_model, _, _ = loader.get_models(sam_type=self.sam_type)
            if depth_model is None:
                raise RuntimeError("Depth model not loaded.")
            depth = depth_model.infer_image(img_bgr, 518)
            depth_resized = cv2.resize(
                depth, (img_rgb.shape[1], img_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR)
            d_min, d_max = depth_resized.min(), depth_resized.max()
            depth_map = ((depth_resized - d_min) / (d_max - d_min)
                         * 255.0).astype(np.uint8)
            self.finished_signal.emit(img_rgb, depth_map)
        except Exception as e:
            self.error_signal.emit(str(e))


class SamWorkerThread(QThread):
    finished_signal = Signal(object)
    error_signal = Signal(str)

    def __init__(self, sam_type, display_img_rgb, existing_predictor=None):
        super().__init__()
        self.sam_type = sam_type
        self.display_img_rgb = display_img_rgb
        self.existing_predictor = existing_predictor

    def run(self):
        try:
            from core.model_loader import ModelLoader
            from segment_anything import SamPredictor

            predictor = self.existing_predictor

            if predictor is None:
                loader = ModelLoader()
                _, sam_model, _ = loader.get_models(sam_type=self.sam_type)
                predictor = SamPredictor(sam_model)

            if self.display_img_rgb is not None:
                predictor.set_image(self.display_img_rgb)

            self.finished_signal.emit(predictor)
        except Exception as e:
            self.error_signal.emit(str(e))


class AnnotationPanel(QWidget):
    CONFIG_FILE = "categories_config.json"
    HOTKEY_FILE = "hotkeys_config.json"

    log_requested = Signal(str)
    copilot_message_requested = Signal(str)

    def __init__(self):
        super().__init__()
        self.current_tool = None
        self.sam_predictor = None
        self.shortcut_objects = []
        self.categories = {}
        self.image_cache = {}
        self.unsaved_changes = set()
        self.image_dims = {}
        self._current_loaded_path = None
        self.load_categories_config()

        self.hotkeys = {
            "Prev Image": "A", "Next Image": "D",
            "Delete Object": "DEL", "Save JSON": "S",
            "Tool: Select": "V", "Tool: Pan": "H",
            "Tool: SAM Point": "Q", "Tool: SAM Box": "W",
            "Tool: Poly": "C", "Tool: Free Poly": "F",
            "Tool: Lasso": "L", "Tool: Edit": "E",
            "Tool: Auto 3D": "T", "Cancel/Reset": "ESC",
            "Depth Minus": "[", "Depth Plus": "]"
        }
        self.load_hotkeys()

        self.annotations = []
        self.base_img_rgb = None
        self.depth_map = None
        self.current_display_img_rgb = None
        self._is_working = False
        self.global_opacity = 128
        self.depth_overlay_item = None

        self._depth_thread = None
        self._sam_thread = None

        self.init_ui()
        self.setup_shortcuts()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # -- Left toolbar --
        left_bar = QWidget()
        left_bar.setFixedWidth(160)
        left_bar.setStyleSheet(
            "background-color: #2e2e2e; border-right: 1px solid #111;")
        left_layout = QVBoxLayout(left_bar)
        left_layout.setContentsMargins(10, 15, 10, 15)
        left_layout.setSpacing(10)

        self.tool_buttons = {}
        tools_config = [
            ("select", f"↖ Select ({self.hotkeys['Tool: Select']})",
             "Select objects"),
            ("hand", f"✋ Pan ({self.hotkeys['Tool: Pan']})",
             "Pan image"),
            ("sam_point", f"🎯 Point ({self.hotkeys['Tool: SAM Point']})",
             "SAM Point"),
            ("sam_box", f"⬜ Box ({self.hotkeys['Tool: SAM Box']})",
             "SAM Box"),
            ("auto_3d",
             f"✨ Auto 3D ({self.hotkeys['Tool: Auto 3D']})",
             "Depth-guided Zero-Click Seg"),
            ("lasso", f"🧲 Lasso ({self.hotkeys['Tool: Lasso']})",
             "Magnetic Lasso"),
            ("poly", f"⬠ Poly ({self.hotkeys['Tool: Poly']})",
             "Click point by point"),
            ("free_poly",
             f"〰️ Free ({self.hotkeys['Tool: Free Poly']})",
             "Drag editable polygon"),
            ("edit", f"🛠️ Edit ({self.hotkeys['Tool: Edit']})",
             "Edit vertices"),
            ("sep", "", ""),
            ("delete", f"🗑️ Delete ({self.hotkeys['Delete Object']})",
             "Delete item")
        ]

        for key, text, tooltip in tools_config:
            if key == "sep":
                sep = QWidget()
                sep.setFixedHeight(1)
                sep.setStyleSheet("background-color: #555;")
                left_layout.addWidget(sep)
                continue
            btn = QToolButton()
            btn.setText(text)
            btn.setToolTip(tooltip)
            btn.setFixedSize(140, 40)
            btn.setCheckable(key not in ["delete", "auto_3d"])
            btn.setFont(QFont("Arial", 10, QFont.Bold))
            btn.setStyleSheet("""
                QToolButton {
                    background-color: #444; color: #fff;
                    border-radius: 6px; text-align: left; padding-left: 10px;
                }
                QToolButton:checked {
                    background-color: #1976d2;
                    border: 2px solid #64b5f6;
                }
                QToolButton:hover { background-color: #555; }
                QToolButton:pressed { background-color: #333; }
            """)

            left_layout.addWidget(btn)
            self.tool_buttons[key] = btn
            if key == "delete":
                btn.clicked.connect(self.delete_selected)
            elif key == "auto_3d":
                btn.clicked.connect(self.run_auto_3d)
            else:
                btn.clicked.connect(
                    lambda checked, k=key: self.activate_tool(k))

        left_layout.addStretch()

        # -- Center: canvas --
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        top_bar = QWidget()
        top_bar.setFixedHeight(45)
        top_bar.setStyleSheet(
            "background-color: #383838; color: white; "
            "border-bottom: 1px solid #111;")
        top_layout = QHBoxLayout(top_bar)

        lbl_depth = QLabel("Depth Slice:")
        lbl_depth.setFont(QFont("Arial", 10, QFont.Bold))

        btn_depth_minus = QPushButton("-")
        btn_depth_minus.setFixedSize(22, 22)
        btn_depth_minus.setStyleSheet(
            "background: #555; color: white; border-radius: 3px;")
        btn_depth_minus.setToolTip(
            f"Decrease active slider "
            f"({self.hotkeys['Depth Minus']})")
        btn_depth_minus.clicked.connect(self.depth_step_minus)

        self.slider_depth_range = RangeSlider()
        self.slider_depth_range.setFixedWidth(100)
        self.slider_depth_range.valueChanged.connect(
            self.on_depth_slider_changed_live)
        self.slider_depth_range.sliderReleased.connect(
            self.on_depth_slider_released)

        btn_depth_plus = QPushButton("+")
        btn_depth_plus.setFixedSize(22, 22)
        btn_depth_plus.setStyleSheet(
            "background: #555; color: white; border-radius: 3px;")
        btn_depth_plus.setToolTip(
            f"Increase active slider "
            f"({self.hotkeys['Depth Plus']})")
        btn_depth_plus.clicked.connect(self.depth_step_plus)

        self.edit_depth_val = QLineEdit("0-255")
        self.edit_depth_val.setFixedWidth(55)
        self.edit_depth_val.setAlignment(Qt.AlignCenter)
        self.edit_depth_val.setStyleSheet(
            "background: #222; color: white; border: 1px solid #555; "
            "border-radius: 3px;")
        self.edit_depth_val.returnPressed.connect(
            self.apply_manual_depth)

        lbl_model = QLabel(" | SAM:")
        lbl_model.setFont(QFont("Arial", 10, QFont.Bold))
        self.combo_sam = QComboBox()
        self.combo_sam.addItems(
            ["Fast (ViT-B)", "Balanced (ViT-L)", "Precision (ViT-H)"])
        self.combo_sam.setStyleSheet(
            "background: white; color: black; border: 1px solid #ccc; "
            "border-radius: 4px; padding: 2px;")

        self.btn_load_sam = QPushButton("Load")
        self.btn_load_sam.setStyleSheet(
            "background: #2e7d32; color: white; padding: 4px; "
            "font-weight: bold; border-radius: 3px;")
        self.btn_load_sam.clicked.connect(self.load_global_sam)

        btn_hotkey = QPushButton("⌨️ Hotkeys")
        btn_hotkey.clicked.connect(self.open_hotkey_dialog)
        btn_convert = QPushButton("🔄 YOLO Exporter")
        btn_convert.clicked.connect(self.open_converter_dialog)

        self.show_depth_checkbox = QCheckBox("Show Depth")
        self.show_depth_checkbox.setStyleSheet(
            "color: #64b5f6; font-weight: bold; margin-right: 5px;")
        self.show_depth_checkbox.stateChanged.connect(
            self.toggle_depth_overlay)

        top_layout.addWidget(self.show_depth_checkbox)
        top_layout.addWidget(lbl_depth)
        top_layout.addWidget(btn_depth_minus)
        top_layout.addWidget(self.slider_depth_range)
        top_layout.addWidget(btn_depth_plus)
        top_layout.addWidget(self.edit_depth_val)

        top_layout.addWidget(lbl_model)
        top_layout.addWidget(self.combo_sam)
        top_layout.addWidget(self.btn_load_sam)
        top_layout.addStretch()
        top_layout.addWidget(btn_hotkey)
        top_layout.addWidget(btn_convert)

        self.canvas = GraphicsCanvas()
        self.canvas.setStyleSheet(
            "border: none; background-color: #222;")

        status_bar = QWidget()
        status_bar.setFixedHeight(30)
        status_bar.setStyleSheet(
            "background-color: #e0e0e0; border-top: 1px solid #ccc;")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(10, 0, 10, 0)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet(
            "color: #333; font-weight: bold;")
        self.lbl_status.setMinimumWidth(10)

        self.lbl_hints = QLabel("💡 Select a tool from left.")
        self.lbl_hints.setStyleSheet(
            "color: #2e7d32; font-style: italic; font-weight: bold;")
        self.lbl_hints.setMinimumWidth(10)

        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(0, 255)
        self.slider_opacity.setValue(self.global_opacity)
        self.slider_opacity.setFixedWidth(80)
        self.slider_opacity.valueChanged.connect(
            self.on_opacity_changed)

        status_layout.addWidget(self.lbl_status, stretch=1)
        status_layout.addWidget(self.lbl_hints, stretch=2)
        status_layout.addWidget(QLabel(
            f" | Opacity ({self.hotkeys['Cancel/Reset']} to Cancel):"))
        status_layout.addWidget(self.slider_opacity)
        center_layout.addWidget(top_bar)
        center_layout.addWidget(self.canvas)
        center_layout.addWidget(status_bar)

        # -- Right panel --
        right_bar = QWidget()
        right_bar.setFixedWidth(200)
        right_bar.setStyleSheet(
            "background-color: #f8f9fa; border-left: 1px solid #ccc;")
        right_layout = QVBoxLayout(right_bar)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_splitter = QSplitter(Qt.Vertical)

        widget_files = QWidget()
        files_layout = QVBoxLayout(widget_files)
        files_layout.setContentsMargins(0, 0, 0, 0)
        btn_open_dir = QPushButton("📁 Open Directory")
        btn_open_dir.setStyleSheet(
            "background: #1976d2; color: white; font-weight: bold; "
            "padding: 8px; border-radius: 4px;")
        btn_open_dir.clicked.connect(self.open_directory)
        self.list_files = QListWidget()
        self.list_files.itemClicked.connect(self.on_file_selected)
        files_layout.addWidget(btn_open_dir)
        files_layout.addWidget(self.list_files)

        widget_labels = QWidget()
        labels_layout = QVBoxLayout(widget_labels)
        labels_layout.setContentsMargins(0, 0, 0, 0)
        group_classes = QGroupBox("🏷️ Categories")
        class_layout = QVBoxLayout(group_classes)
        self.list_classes = QListWidget()
        self.list_classes.setFixedHeight(120)
        self.list_classes.itemDoubleClicked.connect(self.edit_class)
        self.list_classes.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_classes.customContextMenuRequested.connect(
            self.show_class_context_menu)
        self.refresh_class_list()
        btn_add_class = QPushButton("+ Add Class")
        btn_add_class.clicked.connect(self.add_new_class)
        class_layout.addWidget(self.list_classes)
        class_layout.addWidget(btn_add_class)

        group_objects = QGroupBox("📦 Instances")
        obj_layout = QVBoxLayout(group_objects)
        self.list_objects = QListWidget()
        self.list_objects.setSelectionMode(
            QAbstractItemView.ExtendedSelection)
        self.list_objects.itemClicked.connect(
            self.on_list_item_clicked)
        self.list_objects.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_objects.customContextMenuRequested.connect(
            self.show_object_context_menu)
        btn_export = QPushButton(
            f"💾 Save JSON ({self.hotkeys['Save JSON']})")
        btn_export.setStyleSheet(
            "background: #2e7d32; color: white; padding: 10px; "
            "font-weight: bold; border-radius: 6px;")
        btn_export.clicked.connect(self.export_dataset)
        obj_layout.addWidget(self.list_objects)
        obj_layout.addWidget(btn_export)

        labels_layout.addWidget(group_classes)
        labels_layout.addWidget(group_objects)
        right_splitter.addWidget(widget_files)
        right_splitter.addWidget(widget_labels)
        right_splitter.setSizes([350, 500])
        right_layout.addWidget(right_splitter)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_bar)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_bar)
        splitter.setSizes([160, 800, 200])
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)
        self.activate_tool("select")

    # ------------------------------------------------------------------
    # Depth slider logic
    # ------------------------------------------------------------------

    def on_depth_slider_changed_live(self, low, high):
        self.edit_depth_val.setText(f"{low}-{high}")

    def on_depth_slider_released(self, low, high):
        if self.base_img_rgb is not None and self.depth_map is not None:
            self.apply_depth_mask()
            if self.sam_predictor is not None and \
               self.current_display_img_rgb is not None:
                self.update_message(
                    "⏳ SAM is encoding the depth slice in background...")
                self._sam_thread = SamWorkerThread(
                    self.get_current_sam_type(),
                    self.current_display_img_rgb,
                    self.sam_predictor)
                self._sam_thread.finished_signal.connect(
                    lambda p: self.update_message(
                        "✅ Depth features encoded. AI tools ready."))
                self._sam_thread.start()

    def depth_step_minus(self):
        low, high = \
            self.slider_depth_range._low, self.slider_depth_range._high
        target = self.slider_depth_range.last_active_handle
        if target == 'low':
            if low > 0:
                low -= 1
        else:
            if high > low:
                high -= 1
        self.slider_depth_range.setValues(low, high)
        self.on_depth_slider_changed_live(low, high)
        self.on_depth_slider_released(low, high)

    def depth_step_plus(self):
        low, high = \
            self.slider_depth_range._low, self.slider_depth_range._high
        target = self.slider_depth_range.last_active_handle
        if target == 'low':
            if low < high:
                low += 1
        else:
            if high < 255:
                high += 1
        self.slider_depth_range.setValues(low, high)
        self.on_depth_slider_changed_live(low, high)
        self.on_depth_slider_released(low, high)

    def apply_manual_depth(self):
        txt = self.edit_depth_val.text().strip()
        try:
            parts = txt.split('-')
            if len(parts) == 2:
                low = max(0, min(255, int(parts[0])))
                high = max(0, min(255, int(parts[1])))
                if low > high:
                    low, high = high, low
                self.slider_depth_range.setValues(low, high)
                self.on_depth_slider_changed_live(low, high)
                self.on_depth_slider_released(low, high)
        except ValueError:
            self.on_depth_slider_changed_live(
                self.slider_depth_range._low, self.slider_depth_range._high)

    # ------------------------------------------------------------------
    # SAM loading
    # ------------------------------------------------------------------

    def get_current_sam_type(self):
        sam_text = self.combo_sam.currentText()
        if "ViT-H" in sam_text:
            return "vit_h"
        elif "ViT-L" in sam_text:
            return "vit_l"
        return "vit_b"

    def load_global_sam(self):
        sam_type = self.get_current_sam_type()

        self.btn_load_sam.setText("Loading...")
        self.btn_load_sam.setEnabled(False)
        self.combo_sam.setEnabled(False)
        self.update_message(
            f"⏳ Loading SAM ({sam_type}) in background... "
            f"UI is fully responsive!")

        self._sam_thread = SamWorkerThread(
            sam_type, self.current_display_img_rgb, None)
        self._sam_thread.finished_signal.connect(
            self.on_sam_loaded_async)
        self._sam_thread.error_signal.connect(
            self.on_sam_error_async)
        self._sam_thread.start()

    def on_sam_loaded_async(self, predictor):
        self.sam_predictor = predictor
        self.update_message(
            f"✅ SAM ({self.get_current_sam_type()}) is ready! "
            f"You can use Point/Auto3D now.")
        self.btn_load_sam.setText("Loaded")
        self.btn_load_sam.setStyleSheet(
            "background: #005005; color: white; padding: 4px; "
            "font-weight: bold; border-radius: 3px;")
        self.btn_load_sam.setEnabled(True)
        self.combo_sam.setEnabled(True)

    def on_sam_error_async(self, err_msg):
        QMessageBox.critical(self, "SAM Error", f"Failed: {err_msg}")
        self.update_message("❌ SAM load failed.")
        self.btn_load_sam.setText("Load")
        self.btn_load_sam.setEnabled(True)
        self.combo_sam.setEnabled(True)

    def run_auto_3d(self):
        if self.sam_predictor is None:
            QMessageBox.warning(
                self, "Notice",
                "Please click 'Load' next to SAM Model first!")
            return
        if self.depth_map is None:
            return
        d_min, d_max = \
            self.slider_depth_range._low, self.slider_depth_range._high
        self.current_tool = Auto3DTool(
            self.canvas, self.on_polygon_created,
            self.current_display_img_rgb, self.depth_map,
            d_min, d_max, self.sam_predictor)

    # ------------------------------------------------------------------
    # Shortcut system
    # ------------------------------------------------------------------

    def load_hotkeys(self):
        if os.path.exists(self.HOTKEY_FILE):
            try:
                with open(self.HOTKEY_FILE, 'r') as f:
                    self.hotkeys.update(json.load(f))
            except Exception:
                pass

    def setup_shortcuts(self):
        for sc in self.shortcut_objects:
            sc.setParent(None)
            sc.deleteLater()
        self.shortcut_objects = []

        mapping = {
            "Prev Image": lambda: self.navigate_image(-1),
            "Next Image": lambda: self.navigate_image(1),
            "Delete Object": self.delete_selected,
            "Save JSON": self.export_dataset,
            "Tool: Select": lambda: self.activate_tool("select"),
            "Tool: Pan": lambda: self.activate_tool("hand"),
            "Tool: SAM Point": lambda: self.activate_tool("sam_point"),
            "Tool: SAM Box": lambda: self.activate_tool("sam_box"),
            "Tool: Auto 3D": self.run_auto_3d,
            "Tool: Lasso": lambda: self.activate_tool("lasso"),
            "Tool: Poly": lambda: self.activate_tool("poly"),
            "Tool: Free Poly": lambda: self.activate_tool("free_poly"),
            "Tool: Edit": lambda: self.activate_tool("edit"),
            "Cancel/Reset": self.cancel_current_action,
            "Depth Minus": self.depth_step_minus,
            "Depth Plus": self.depth_step_plus
        }

        for name, func in mapping.items():
            key = self.hotkeys.get(name, "")
            if key:
                sc = QShortcut(QKeySequence(key), self)
                sc.activated.connect(func)
                self.shortcut_objects.append(sc)

    def cancel_current_action(self):
        if self.current_tool and hasattr(self.current_tool, '_reset'):
            self.current_tool._reset()
            self.update_message("Action canceled.")

    def open_hotkey_dialog(self):
        dlg = HotkeyDialog(self.hotkeys, self)
        if dlg.exec():
            with open(self.HOTKEY_FILE, 'w') as f:
                json.dump(self.hotkeys, f)
            self.setup_shortcuts()

            mapping = {
                "select": ("↖ Select", "Tool: Select"),
                "hand": ("✋ Pan", "Tool: Pan"),
                "sam_point": ("🎯 Point", "Tool: SAM Point"),
                "sam_box": ("⬜ Box", "Tool: SAM Box"),
                "auto_3d": ("✨ Auto 3D", "Tool: Auto 3D"),
                "lasso": ("🧲 Lasso", "Tool: Lasso"),
                "poly": ("⬠ Poly", "Tool: Poly"),
                "free_poly": ("〰️ Free", "Tool: Free Poly"),
                "edit": ("🛠️ Edit", "Tool: Edit"),
                "delete": ("🗑️ Delete", "Delete Object")
            }
            for btn_key, (base_text, hk_key) in mapping.items():
                if btn_key in self.tool_buttons:
                    self.tool_buttons[btn_key].setText(
                        f"{base_text} ({self.hotkeys[hk_key]})")

            QMessageBox.information(
                self, "Applied",
                "Hotkeys saved and applied immediately!")

    def open_converter_dialog(self):
        FormatConverterDialog(self.categories, self).exec()

    def navigate_image(self, direction):
        if self.list_files.count() == 0:
            return
        new_row = self.list_files.currentRow() + direction
        if 0 <= new_row < self.list_files.count():
            self.list_files.setCurrentRow(new_row)
            self.on_file_selected(self.list_files.item(new_row))

    # ------------------------------------------------------------------
    # Tool activation
    # ------------------------------------------------------------------

    def update_message(self, msg):
        if "Click" in msg or "Drag" in msg or "Ready" in msg or \
           "[" in msg or "Double-click" in msg:
            self.lbl_hints.setText(f"{msg}")
        else:
            self.lbl_status.setText(msg)

    def fit_view(self):
        if self.canvas.pixmap_item:
            self.canvas.fitInView(
                self.canvas.pixmap_item, Qt.KeepAspectRatio)

    def activate_tool(self, tool_key):
        if getattr(self, '_is_working', False):
            return
        self._is_working = True
        try:
            if hasattr(self.current_tool, '_reset'):
                self.current_tool._reset()
            if hasattr(self.current_tool, '_clear_handles'):
                self.current_tool._clear_handles()

            for k, btn in self.tool_buttons.items():
                if k not in ["delete", "auto_3d"] and k != tool_key:
                    btn.setChecked(False)
            if tool_key in self.tool_buttons:
                self.tool_buttons[tool_key].setChecked(True)

            self.current_tool = None
            hint_text = ""

            if tool_key == "select":
                hint_text = ("💡 [L-Click]: Select. [Drag]: Move. "
                             "[R-Click List]: Re-Classify.")
            elif tool_key == "hand":
                self.current_tool = HandTool(self.canvas)
                hint_text = ("💡 [Left Drag]: Pan. "
                             "[Right Click]: Reset View.")
            elif tool_key == "lasso":
                self.current_tool = MagneticLassoTool(
                    self.canvas, self.on_polygon_created)
            elif tool_key == "poly":
                self.current_tool = PolygonTool(
                    self.canvas.scene, self.on_polygon_created)
            elif tool_key == "free_poly":
                self.current_tool = FreehandPolyTool(
                    self.canvas, self.on_polygon_created)
            elif tool_key == "edit":
                self.current_tool = EditTool(self.canvas)
            elif tool_key in ["sam_point", "sam_box"]:
                if self.sam_predictor is None:
                    QMessageBox.warning(
                        self, "Notice",
                        "Please load SAM first via the top bar "
                        "'Load' button!")
                    self.tool_buttons[tool_key].setChecked(False)
                    return
                if tool_key == "sam_point":
                    self.current_tool = SAMPointTool(
                        self.canvas, self.on_polygon_created,
                        self.current_display_img_rgb, self.sam_predictor)
                else:
                    self.current_tool = SAMBoxTool(
                        self.canvas, self.on_polygon_created,
                        self.current_display_img_rgb, self.sam_predictor)

            if hasattr(self.current_tool, 'reset_view_requested'):
                self.current_tool.reset_view_requested.connect(
                    self.fit_view)

            if self.current_tool:
                self.canvas.set_tool(self.current_tool)
                self.canvas.setCursor(Qt.CrossCursor)
                if hasattr(self.current_tool, 'show_hint'):
                    self.current_tool.show_hint()
                elif hint_text:
                    self.update_message(hint_text)
            else:
                self.canvas.set_tool(None)
                self.canvas.setCursor(Qt.ArrowCursor)
                if hint_text:
                    self.update_message(hint_text)
        finally:
            self._is_working = False

    # ------------------------------------------------------------------
    # Categories, depth, and image loading
    # ------------------------------------------------------------------

    def load_categories_config(self):
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.categories[k] = QColor(
                            v[0], v[1], v[2], v[3])
                return
            except Exception:
                pass
        self.categories = {
            "Leaf": QColor(0, 255, 0, 128),
            "Tomato": QColor(255, 0, 0, 128)
        }

    def save_categories_config(self):
        data = {k: [v.red(), v.green(), v.blue(), v.alpha()]
                for k, v in self.categories.items()}
        try:
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def on_opacity_changed(self, val):
        self.global_opacity = val
        for ann in self.annotations:
            color = QColor(self.categories.get(
                ann["class"], QColor(0, 255, 0)))
            color.setAlpha(val)
            ann["item"].setBrush(QBrush(color))

    def open_directory(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Image Directory")
        if not dir_path:
            return
        self.list_files.clear()
        img_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            img_paths.extend(glob.glob(os.path.join(dir_path, ext)))
        for path in sorted(img_paths):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, path)
            self.list_files.addItem(item)

        self.update_file_list_ui()

        if self.list_files.count() > 0:
            self.list_files.setCurrentRow(0)
            self.on_file_selected(self.list_files.item(0))

    def on_file_selected(self, item):
        self.load_image_with_depth(item.data(Qt.UserRole))

    def load_image_with_depth(self, img_path):
        if self._depth_thread is not None and \
           self._depth_thread.isRunning():
            self.update_message(
                "⏳ Previous image is still processing!")
            return

        self.save_current_to_cache()
        self.cancel_current_action()
        self.canvas.scene.clear()
        self.canvas.pixmap_item = None
        self.depth_overlay_item = None

        self.list_objects.clear()
        self.annotations = []
        self.base_img_rgb = None
        self.depth_map = None
        self.canvas.scene.addText(
            "🤖 Computing Depth Map... Please Wait."
        ).setDefaultTextColor(Qt.white)

        self._depth_thread = DepthInferenceThread(
            img_path, self.get_current_sam_type())
        self._depth_thread.finished_signal.connect(
            lambda r, d: self.on_depth_computed_and_load_json(
                img_path, r, d))
        self._depth_thread.start()

    def on_depth_computed_and_load_json(self, img_path, img_rgb, depth_map):
        self.base_img_rgb = img_rgb
        self.depth_map = depth_map
        self.image_dims[img_path] = (
            img_rgb.shape[0], img_rgb.shape[1])
        self._current_loaded_path = img_path

        self.slider_depth_range.setValues(0, 255)
        self.edit_depth_val.setText("0-255")
        self.apply_depth_mask()

        if img_path in self.image_cache:
            self.load_annotations_from_cache(img_path)
        else:
            json_path = os.path.splitext(img_path)[0] + ".json"
            if os.path.exists(json_path):
                self.load_annotations_from_json(json_path)

        self.activate_tool("select")

        if self.sam_predictor is not None and \
           self.current_display_img_rgb is not None:
            self.update_message(
                "⏳ SAM is encoding the new image in background...")
            self._sam_thread = SamWorkerThread(
                self.get_current_sam_type(),
                self.current_display_img_rgb,
                self.sam_predictor)
            self._sam_thread.finished_signal.connect(
                lambda p: self.update_message(
                    "✅ Image encoded. AI tools ready."))
            self._sam_thread.start()

    def apply_depth_mask(self):
        d_min, d_max = \
            self.slider_depth_range._low, self.slider_depth_range._high
        display_img = self.base_img_rgb.copy()
        mask = (self.depth_map >= d_min) & (self.depth_map <= d_max)
        display_img[~mask] = [20, 20, 20]
        self.current_display_img_rgb = display_img
        h, w, ch = display_img.shape
        pixmap = QPixmap.fromImage(
            QImage(display_img.data, w, h, ch * w, QImage.Format_RGB888))
        if self.canvas.pixmap_item:
            self.canvas.pixmap_item.setPixmap(pixmap)
        else:
            self.canvas.scene.clear()
            self.depth_overlay_item = None
            self.canvas.pixmap_item = self.canvas.scene.addPixmap(pixmap)
            self.canvas.setSceneRect(
                self.canvas.pixmap_item.boundingRect())
            self.canvas.fitInView(
                self.canvas.pixmap_item, Qt.KeepAspectRatio)

    def load_annotations_from_json(self, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for shape in data.get("shapes", []):
                cls_name = shape.get("label", "Uncategorized")
                pts = shape.get("points", [])
                if len(pts) < 3:
                    continue
                if cls_name not in self.categories:
                    self.categories[cls_name] = QColor(
                        random.randint(50, 250),
                        random.randint(50, 250),
                        random.randint(50, 250), 128)
                    self.save_categories_config()
                    self.refresh_class_list()
                current_color = QColor(self.categories[cls_name])
                current_color.setAlpha(self.global_opacity)
                poly_item = QGraphicsPolygonItem(
                    QPolygonF([QPointF(p[0], p[1]) for p in pts]))
                poly_item.setBrush(QBrush(current_color))
                poly_item.setPen(
                    QPen(current_color.darker(), 2))
                poly_item.setFlags(
                    QGraphicsPolygonItem.ItemIsSelectable
                    | QGraphicsPolygonItem.ItemIsMovable)
                self.canvas.scene.addItem(poly_item)
                name = f"{cls_name}_{self.list_objects.count() + 1}"
                poly_item.setData(0, name)
                self.annotations.append({
                    "name": name, "item": poly_item,
                    "class": cls_name})
                self.list_objects.addItem(name)
        except Exception as e:
            print(f"Error loading JSON: {e}")

    def refresh_class_list(self):
        self.list_classes.clear()
        for c_name, color in self.categories.items():
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(
                color.red(), color.green(), color.blue(), 255))
            self.list_classes.addItem(
                QListWidgetItem(QIcon(pixmap), c_name))

    def add_new_class(self):
        dlg = CategoryEditDialog(
            "New_Category",
            QColor(random.randint(50, 250), random.randint(50, 250),
                   random.randint(50, 250), 128),
            self)
        if dlg.exec() and dlg.name_input.text().strip():
            self.categories[
                dlg.name_input.text().strip()] = dlg.color
            self.save_categories_config()
            self.refresh_class_list()

    def show_class_context_menu(self, pos):
        item = self.list_classes.itemAt(pos)
        if not item:
            return

        menu = QMenu(self)
        menu.setStyleSheet(MENU_STYLESHEET)

        edit_action = menu.addAction("Edit")
        delete_action = menu.addAction("Delete")

        action = menu.exec(self.list_classes.mapToGlobal(pos))

        if action == edit_action:
            self.edit_class(item)
        elif action == delete_action:
            class_name = item.text()
            reply = QMessageBox.question(
                self, 'Delete Category',
                f"Are you sure you want to delete the category "
                f"'{class_name}'?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                if class_name in self.categories:
                    del self.categories[class_name]
                    self.save_categories_config()
                    self.refresh_class_list()

    def edit_class(self, item):
        old_name = item.text()
        old_color = self.categories.get(
            old_name, QColor(0, 255, 0, 128))
        dlg = CategoryEditDialog(old_name, old_color, self)
        if dlg.exec() and dlg.name_input.text().strip():
            new_name = dlg.name_input.text().strip()
            if new_name != old_name:
                del self.categories[old_name]
            self.categories[new_name] = dlg.color
            self.save_categories_config()
            self.refresh_class_list()
            for ann in self.annotations:
                if ann["class"] == old_name:
                    ann["class"] = new_name
                    apply_color = QColor(dlg.color)
                    apply_color.setAlpha(self.global_opacity)
                    ann["item"].setBrush(QBrush(apply_color))
                    ann["item"].setPen(
                        QPen(apply_color.darker(), 2))
            self.list_objects.clear()
            for ann in self.annotations:
                self.list_objects.addItem(ann["name"])

    def show_object_context_menu(self, pos):
        item = self.list_objects.itemAt(pos)
        if not item:
            return
        menu = QMenu(self)

        change_menu = menu.addMenu("Change Category To...")
        for c_name in self.categories.keys():
            change_menu.addAction(c_name).triggered.connect(
                lambda checked, n=c_name, it=item:
                self.change_object_category(it, n))

        menu.addSeparator()
        menu.addAction("Rename").triggered.connect(
            lambda checked, it=item: self.rename_object(it))
        menu.addAction("Delete Selected").triggered.connect(
            self.delete_selected)
        menu.exec(self.list_objects.mapToGlobal(pos))

    def rename_object(self, list_item):
        old_name = list_item.text()
        new_name, ok = QInputDialog.getText(
            self, "Rename Object", "Enter new name:", text=old_name)

        if ok and new_name.strip() and new_name != old_name:
            new_name = new_name.strip()
            if any(a["name"] == new_name for a in self.annotations):
                QMessageBox.warning(
                    self, "Name Exists",
                    f"The name '{new_name}' already exists!")
                return

            for ann in self.annotations:
                if ann["name"] == old_name:
                    ann["name"] = new_name
                    ann["item"].setData(0, new_name)
                    list_item.setText(new_name)
                    break
        self.mark_dirty()

    def change_object_category(self, list_item, new_class_name):
        old_name = list_item.text()
        new_color = QColor(self.categories[new_class_name])
        new_color.setAlpha(self.global_opacity)

        existing_indices = []
        for ann in self.annotations:
            if ann["class"] == new_class_name:
                parts = ann["name"].split('_')
                if len(parts) > 1 and parts[-1].isdigit():
                    existing_indices.append(int(parts[-1]))

        next_index = max(existing_indices) + 1 if existing_indices else 1
        new_name = f"{new_class_name}_{next_index}"

        while any(a["name"] == new_name for a in self.annotations):
            next_index += 1
            new_name = f"{new_class_name}_{next_index}"

        for ann in self.annotations:
            if ann["name"] == old_name:
                ann["class"] = new_class_name
                ann["name"] = new_name

                ann["item"].setBrush(QBrush(new_color))
                ann["item"].setPen(QPen(new_color.darker(), 2))
                ann["item"].setData(0, new_name)

                list_item.setText(new_name)
                break
        self.mark_dirty()

    def on_polygon_created(self, poly_item):
        current_class = (
            self.list_classes.currentItem().text()
            if self.list_classes.currentItem() else "Uncategorized")
        if current_class not in self.categories:
            self.categories[current_class] = QColor(0, 255, 255, 128)
            self.save_categories_config()
            self.refresh_class_list()
        current_color = QColor(self.categories[current_class])
        current_color.setAlpha(self.global_opacity)
        poly_item.setBrush(QBrush(current_color))
        poly_item.setPen(QPen(current_color.darker(), 2))
        name = f"{current_class}_{self.list_objects.count() + 1}"
        poly_item.setData(0, name)
        self.annotations.append({
            "name": name, "item": poly_item, "class": current_class})
        self.list_objects.addItem(name)
        self.list_objects.setCurrentRow(self.list_objects.count() - 1)
        self.mark_dirty()

    def delete_selected(self):
        names_to_delete = set()
        for item in self.canvas.scene.selectedItems():
            name = item.data(0)
            if name:
                names_to_delete.add(name)
        for item in self.list_objects.selectedItems():
            name = item.text()
            if name:
                names_to_delete.add(name)
        if not names_to_delete:
            return

        for name in names_to_delete:
            items_to_remove = self.list_objects.findItems(
                name, Qt.MatchExactly)
            for i in items_to_remove:
                self.list_objects.takeItem(self.list_objects.row(i))
            for ann in self.annotations:
                if ann["name"] == name:
                    self.canvas.scene.removeItem(ann["item"])
                    break
        self.annotations = [a for a in self.annotations
                            if a['name'] not in names_to_delete]
        self.mark_dirty()

    def on_list_item_clicked(self, list_item):
        self.canvas.scene.clearSelection()
        for ann in self.annotations:
            if ann["name"] == list_item.text():
                ann["item"].setSelected(True)
                break

    def export_dataset(self):
        self.save_current_to_cache()

        if not self.unsaved_changes:
            QMessageBox.information(
                self, "Export",
                "No new or modified annotations to save.")
            return

        saved_count = 0
        for img_path in list(self.unsaved_changes):
            shapes = self.image_cache.get(img_path, [])
            img_h, img_w = self.image_dims.get(
                img_path, (1000, 1000))

            data = {
                "version": "5.2.1",
                "flags": {},
                "shapes": [{
                    "label": s["class"],
                    "points": s["points"],
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                } for s in shapes],
                "imagePath": os.path.basename(img_path),
                "imageHeight": img_h,
                "imageWidth": img_w,
                "imageData": None
            }
            try:
                json_path = os.path.splitext(img_path)[0] + ".json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                saved_count += 1
            except Exception as e:
                print(f"Error saving {img_path}: {e}")

        self.unsaved_changes.clear()
        self.update_file_list_ui()
        self.lbl_status.setText(f"✅ Batch Saved {saved_count} files!")

    def toggle_depth_overlay(self, state):
        if state == Qt.Checked.value or state == Qt.Checked:
            self.show_depth()
        else:
            self.hide_depth()

    def show_depth(self):
        if self.depth_map is None:
            return

        colored_depth = cv2.applyColorMap(
            self.depth_map.astype(np.uint8), cv2.COLORMAP_JET)
        colored_depth = cv2.cvtColor(
            colored_depth, cv2.COLOR_BGR2RGB)

        h, w, c = colored_depth.shape
        qimage = QImage(
            colored_depth.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        if self.depth_overlay_item is None:
            self.depth_overlay_item = QGraphicsPixmapItem(pixmap)
            self.depth_overlay_item.setOpacity(0.55)
            self.depth_overlay_item.setZValue(0.5)
            self.canvas.scene.addItem(self.depth_overlay_item)
        else:
            self.depth_overlay_item.setPixmap(pixmap)
            self.depth_overlay_item.show()

    def hide_depth(self):
        if self.depth_overlay_item is not None:
            self.depth_overlay_item.hide()

    # ------------------------------------------------------------------
    # Copilot interface
    # ------------------------------------------------------------------

    def get_copilot_context(self):
        return {
            "ui_mode": "🔬 Annotation Studio",
            "has_image_loaded": self.base_img_rgb is not None,
            "is_sam_loaded": self.sam_predictor is not None,
            "existing_categories": list(self.categories.keys()),
            "current_annotation_count": len(self.annotations)
        }

    def execute_copilot_action(self, action, params):
        if action == "auto_labeling":
            label_name = params.get("label_name", "Object")

            if self.base_img_rgb is None:
                self.copilot_message_requested.emit(
                    "⚠️ Please import an image directory first "
                    "before I can help you label!")
                return

            if label_name not in self.categories:
                self.categories[label_name] = QColor(
                    random.randint(50, 250), random.randint(50, 250),
                    random.randint(50, 250), 128)
                self.save_categories_config()
                self.refresh_class_list()

            items = self.list_classes.findItems(
                label_name, Qt.MatchExactly)
            if items:
                self.list_classes.setCurrentItem(items[0])

            if self.sam_predictor is None:
                self.copilot_message_requested.emit(
                    f"🔄 I am loading the SAM engine to prepare "
                    f"for labeling '{label_name}'...")
                self.load_global_sam()

            self.activate_tool("auto_3d")

            self.log_requested.emit(
                f"✨ UI Action: Prepared SAM and selected category "
                f"'{label_name}'.")
            self.copilot_message_requested.emit(
                f"✅ I have created/selected the category "
                f"**'{label_name}'** and activated the **Auto 3D "
                f"Tool** for you!\n\n👉 Just click anywhere on the "
                f"target in the image, and it will be automatically "
                f"segmented and labeled as '{label_name}'!")

        elif action == "export_dataset":
            export_format = params.get("format", "json").lower()
            if not self.annotations:
                self.copilot_message_requested.emit(
                    "⚠️ There are no annotations to export yet. "
                    "Please label some objects first.")
                return

            if export_format == "yolo":
                self.copilot_message_requested.emit(
                    "🔄 Opening the YOLO Format Converter for you...")
                self.open_converter_dialog()
            else:
                self.copilot_message_requested.emit(
                    "💾 Opening the JSON save dialog...")
                self.export_dataset()

    # ------------------------------------------------------------------
    # Cache and state management
    # ------------------------------------------------------------------

    def update_file_list_ui(self):
        for i in range(self.list_files.count()):
            item = self.list_files.item(i)
            img_path = item.data(Qt.UserRole)
            base_name = os.path.basename(img_path)

            if img_path in self.unsaved_changes:
                item.setText("✏️ " + base_name)
            elif os.path.exists(
                    os.path.splitext(img_path)[0] + ".json"):
                item.setText("✅ " + base_name)
            else:
                item.setText("📄 " + base_name)

    def mark_dirty(self):
        if self.list_files.currentItem():
            img_path = self.list_files.currentItem().data(
                Qt.UserRole)
            self.unsaved_changes.add(img_path)
            self.update_file_list_ui()

    def save_current_to_cache(self):
        if self.base_img_rgb is not None and \
           getattr(self, '_current_loaded_path', None):
            shapes_data = []
            for ann in self.annotations:
                poly = ann["item"].polygon()
                pts = [[poly.at(i).x(), poly.at(i).y()]
                       for i in range(poly.count())]
                shapes_data.append({
                    "class": ann["class"],
                    "name": ann["name"],
                    "points": pts})
            self.image_cache[self._current_loaded_path] = shapes_data

    def load_annotations_from_cache(self, img_path):
        shapes = self.image_cache.get(img_path, [])
        for shape in shapes:
            cls_name, pts, name = \
                shape["class"], shape["points"], shape["name"]
            if cls_name not in self.categories:
                continue

            current_color = QColor(self.categories[cls_name])
            current_color.setAlpha(self.global_opacity)
            poly_item = QGraphicsPolygonItem(
                QPolygonF([QPointF(p[0], p[1]) for p in pts]))

            poly_item.setBrush(QBrush(current_color))
            poly_item.setPen(QPen(current_color.darker(), 2))
            poly_item.setFlags(
                QGraphicsPolygonItem.ItemIsSelectable
                | QGraphicsPolygonItem.ItemIsMovable)
            poly_item.setData(0, name)

            self.canvas.scene.addItem(poly_item)
            self.annotations.append({
                "name": name, "item": poly_item,
                "class": cls_name})
            self.list_objects.addItem(name)

    def check_unsaved_changes(self):
        self.save_current_to_cache()
        if self.unsaved_changes:
            reply = QMessageBox.warning(
                self, 'Unsaved Changes',
                f"You have {len(self.unsaved_changes)} unsaved images "
                f"(marked with ✏️).\nDo you want to save them before "
                f"proceeding?",
                QMessageBox.Save | QMessageBox.Discard
                | QMessageBox.Cancel, QMessageBox.Save)
            if reply == QMessageBox.Save:
                self.export_dataset()
                return True
            elif reply == QMessageBox.Discard:
                return True
            else:
                return False
        return True
