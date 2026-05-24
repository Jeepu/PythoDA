import os
import csv
import re
import math
import cv2
from core.utils import imread_unicode
import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QComboBox, QFileDialog,
                               QProgressBar, QTableWidget, QTableWidgetItem,
                               QHeaderView, QMessageBox, QStackedWidget, QGridLayout,
                               QSizePolicy, QTabWidget, QAbstractItemView,
                               QDoubleSpinBox, QMenu, QInputDialog)
from PySide6.QtGui import QPixmap, QImage, QColor
from PySide6.QtCore import Qt, QThread, Signal

from core.worker import WorkerThread
from core.model_loader import ModelLoader
from core.global_state import GlobalState
from ui import MENU_STYLESHEET
from ui.widgets import ToggleSwitch

EXPORT_COLUMN_ORDER = [
    "ID", "Row",
    "Length", "Width", "Diameter", "Height", "Radius", "Distance", "Gap",
    "Area", "Size", "Perimeter",
    "Volume",
    "L/W Ratio", "Ratio", "Circularity", "Angle",
    "Color", "ColorName", "RGB", "RGB Val", "Swatch"
]

class InteractiveImageLabel(QLabel):
    """Custom QLabel that emits right-click coordinates."""
    right_clicked = Signal(int, int)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.right_clicked.emit(event.pos().x(), event.pos().y())
        super().mousePressEvent(event)


class ModelInitThread(QThread):
    progress_signal = Signal(int, str)
    finished_signal = Signal()

    def __init__(self, target_sam_type="vit_b"):
        super().__init__()
        self.target_sam_type = target_sam_type

    def run(self):
        loader = ModelLoader()
        loader.load_models(callback_signal=self.progress_signal,
                           target_sam_type=self.target_sam_type)
        self.finished_signal.emit()


class AutoPanel(QWidget):
    log_requested = Signal(str)
    copilot_message_requested = Signal(str)
    def __init__(self):
        super().__init__()
        self.image_queue = []
        self.current_idx = -1
        self.analysis_results = {}
        self.is_batch_running = False
        self.worker = None
        self.cache_main_img = None
        self.cache_steps_imgs = []
        self.pending_target = ""
        self.pending_highlight_color = "#ffcdd2"
        self.pending_show_cols = []
        self.pending_hide_cols = []

        self.init_ui()
        self.start_model_loading()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # -- Left column: controls --
        control_panel = QWidget()
        control_panel.setFixedWidth(320)
        control_panel.setStyleSheet(
            "background-color: #f8f9fa; border-right: 1px solid #ddd;")
        ctrl_layout = QVBoxLayout(control_panel)
        ctrl_layout.setContentsMargins(15, 15, 15, 15)

        title = QLabel("PhytoDA")
        title.setStyleSheet(
            "font-size: 24px; font-weight: 900; color: #1b5e20; "
            "margin-bottom: 10px;")
        ctrl_layout.addWidget(title)

        ctrl_layout.addWidget(QLabel("<b>Task & Model:</b>"))
        self.combo_task = QComboBox()
        self.combo_task.addItems([
            "Seed Analysis", "Leaf Phenotyping", "Corn Analysis",
            "Tomato Phenotyping", "Wheat Leaf Angle", "Wheat Ear Analysis",
            "Canopy Analysis"])
        self.combo_task.setFixedHeight(35)
        self.combo_task.setStyleSheet(
            "QComboBox::item:hover {"
            "  font-weight: bold; background-color: #e8f5e9; color: #1b5e20;"
            "}"
            "QComboBox::item:selected {"
            "  font-weight: bold; background-color: #c8e6c9; color: #1b5e20;"
            "}")
        ctrl_layout.addWidget(self.combo_task)

        sam_row = QHBoxLayout()
        self.combo_sam_weight = QComboBox()
        self.combo_sam_weight.addItems([
            "High Precision (ViT-H)", "Balanced (ViT-L)", "Fast (ViT-B)"])
        self.combo_sam_weight.setCurrentIndex(2)
        self.combo_sam_weight.setFixedHeight(35)
        self.combo_sam_weight.setStyleSheet(
            "QComboBox::item:hover {"
            "  font-weight: bold; background-color: #e8f5e9; color: #1b5e20;"
            "}"
            "QComboBox::item:selected {"
            "  font-weight: bold; background-color: #c8e6c9; color: #1b5e20;"
            "}")

        self.btn_reload_sam = QPushButton("Load")
        self.btn_reload_sam.setFixedWidth(60)
        self.btn_reload_sam.setFixedHeight(35)
        self.btn_reload_sam.setStyleSheet(
            "background-color: #0d47a1; color: white; "
            "font-weight: bold; border-radius: 4px;")
        self.btn_reload_sam.clicked.connect(self.on_manual_model_load)
        sam_row.addWidget(self.combo_sam_weight)
        sam_row.addWidget(self.btn_reload_sam)
        ctrl_layout.addLayout(sam_row)
        ctrl_layout.addSpacing(10)

        self.btn_load = QPushButton("📂 Import Images")
        self.btn_load.setFixedHeight(40)
        self.btn_load.setStyleSheet(
            "background-color: white; border: 1px solid #ccc; "
            "border-radius: 6px;")
        self.btn_load.clicked.connect(self.batch_load_images)
        ctrl_layout.addWidget(self.btn_load)

        self.lbl_file_count = QLabel("No images loaded")
        self.lbl_file_count.setStyleSheet("color: #666; font-size: 11px;")
        self.lbl_file_count.setAlignment(Qt.AlignCenter)
        ctrl_layout.addWidget(self.lbl_file_count)
        ctrl_layout.addSpacing(5)

        self.btn_run = QPushButton("⚡ START ANALYSIS")
        self.btn_run.setFixedHeight(50)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32; color: white;
                font-weight: bold; font-size: 16px; border-radius: 8px;
            }
            QPushButton:hover { background-color: #388e3c; }
            QPushButton:disabled { background-color: #a5d6a7; }
        """)
        self.btn_run.clicked.connect(self.start_batch_analysis)
        self.btn_run.setEnabled(False)
        ctrl_layout.addWidget(self.btn_run)
        ctrl_layout.addSpacing(10)

        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_prev.clicked.connect(self.show_prev_image)
        self.btn_prev.setEnabled(False)
        self.lbl_index = QLabel("0 / 0")
        self.lbl_index.setAlignment(Qt.AlignCenter)
        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(self.show_next_image)
        self.btn_next.setEnabled(False)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_index)
        nav_layout.addWidget(self.btn_next)
        ctrl_layout.addLayout(nav_layout)

        action_layout = QHBoxLayout()
        self.btn_export = QPushButton("💾 Export Results")
        self.btn_export.setFixedHeight(35)
        self.btn_export.setStyleSheet(
            "background-color: #1565c0; color: white; "
            "font-weight: bold; border-radius: 4px;")
        self.btn_export.clicked.connect(self.export_all_results)
        self.btn_export.setEnabled(False)

        self.btn_clear = QPushButton("🗑️ Clear")
        self.btn_clear.setFixedHeight(35)
        self.btn_clear.setFixedWidth(80)
        self.btn_clear.setStyleSheet(
            "background-color: #616161; color: white; "
            "font-weight: bold; border-radius: 4px;")
        self.btn_clear.clicked.connect(self.clear_workspace)
        self.btn_clear.setEnabled(False)

        action_layout.addWidget(self.btn_export)
        action_layout.addWidget(self.btn_clear)
        ctrl_layout.addLayout(action_layout)
        ctrl_layout.addSpacing(15)

        self.stats_tabs = QTabWidget()
        self.stats_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #ddd; background: white; }
            QTabBar::tab { padding: 8px 15px; background: #eee; }
            QTabBar::tab:selected {
                background: white; border-bottom: 2px solid #2e7d32;
                font-weight: bold;
            }
        """)

        self.table_summary = QTableWidget()
        self.table_summary.setColumnCount(2)
        self.table_summary.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table_summary.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.table_summary.verticalHeader().setVisible(False)
        self.stats_tabs.addTab(self.table_summary, "Summary")

        self.table_details = QTableWidget()
        self.table_details.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.table_details.setAlternatingRowColors(True)
        self.table_details.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_details.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_details.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_details.customContextMenuRequested.connect(
            self.show_table_context_menu)
        self.table_details.horizontalHeader().setContextMenuPolicy(
            Qt.CustomContextMenu)
        self.table_details.horizontalHeader().customContextMenuRequested.connect(
            self.show_header_context_menu)
        self.stats_tabs.addTab(self.table_details, "Details")

        ctrl_layout.addWidget(self.stats_tabs)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(
            "QProgressBar { height: 4px; border-radius: 2px; } "
            "QProgressBar::chunk { background-color: #4caf50; }")
        self.progress_bar.setTextVisible(False)
        ctrl_layout.addWidget(self.progress_bar)

        # -- Right column: image display --
        display_panel = QWidget()
        display_panel.setStyleSheet("background-color: #2b2b2b;")
        display_panel.setSizePolicy(QSizePolicy.Expanding,
                                     QSizePolicy.Expanding)
        disp_layout = QVBoxLayout(display_panel)
        disp_layout.setContentsMargins(10, 10, 10, 10)

        disp_top_bar = QHBoxLayout()

        self.lbl_style_prefix = QLabel("Annotation Style |")
        self.lbl_style_prefix.setStyleSheet(
            "color: #ccc; font-weight: 900; margin-right: 5px; font-size: 13px;")
        disp_top_bar.addWidget(self.lbl_style_prefix)

        self.lbl_font = QLabel("Text size:")
        self.lbl_font.setStyleSheet("color: #ccc; font-weight: bold;")
        self.spin_font = QDoubleSpinBox()
        self.spin_font.setRange(0, 5.0)
        self.spin_font.setSingleStep(0.1)
        self.spin_font.setValue(1.0)
        self.spin_font.setFixedWidth(55)
        self.spin_font.setStyleSheet("""
            QDoubleSpinBox {
                background: #444; color: white; border: 1px solid #666;
                border-radius: 3px;
            }
            QDoubleSpinBox:focus { border: 1px solid #1976d2; }
        """)
        self.spin_font.valueChanged.connect(self.redraw_main_image)

        self.lbl_pos = QLabel("Position:")
        self.lbl_pos.setStyleSheet(
            "color: #ccc; font-weight: bold; margin-left: 10px;")

        self.combo_pos = QComboBox()
        self.combo_pos.addItems(["Center", "Top", "Bottom", "Left", "Right"])
        self.combo_pos.setFixedWidth(75)
        self.combo_pos.setStyleSheet("""
            QComboBox {
                background: #444; color: white; border: 1px solid #666;
                border-radius: 3px; padding-left: 5px;
            }
            QComboBox:focus { border: 1px solid #1976d2; }
            QComboBox::drop-down { border: none; }
        """)
        self.combo_pos.currentIndexChanged.connect(self.redraw_main_image)

        self.lbl_thick = QLabel("Thickness:")
        self.lbl_thick.setStyleSheet(
            "color: #ccc; font-weight: bold; margin-left: 10px;")

        self.spin_thick = QDoubleSpinBox()
        self.spin_thick.setRange(0.5, 10.0)
        self.spin_thick.setSingleStep(0.5)
        self.spin_thick.setValue(2.0)
        self.spin_thick.setFixedWidth(55)
        self.spin_thick.setStyleSheet("""
            QDoubleSpinBox {
                background: #444; color: white; border: 1px solid #666;
                border-radius: 3px;
            }
            QDoubleSpinBox:focus { border: 1px solid #1976d2; }
        """)
        self.spin_thick.valueChanged.connect(self.redraw_main_image)

        disp_top_bar.addWidget(self.lbl_font)
        disp_top_bar.addWidget(self.spin_font)
        disp_top_bar.addWidget(self.lbl_pos)
        disp_top_bar.addWidget(self.combo_pos)
        disp_top_bar.addWidget(self.lbl_thick)
        disp_top_bar.addWidget(self.spin_thick)
        disp_top_bar.addStretch()

        self.switch_label = QLabel("👁️ Show Process Details")
        self.switch_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #ccc;")
        self.toggle_details = ToggleSwitch()
        self.toggle_details.stateChanged.connect(self.toggle_view_mode)
        self.toggle_details.setEnabled(False)
        disp_top_bar.addWidget(self.switch_label)
        disp_top_bar.addWidget(self.toggle_details)
        disp_layout.addLayout(disp_top_bar)

        self.stack = QStackedWidget()
        self.page_single = QWidget()
        layout_single = QVBoxLayout(self.page_single)
        self.lbl_main_image = InteractiveImageLabel("Waiting for input...")
        self.lbl_main_image.right_clicked.connect(self.on_image_right_clicked)
        self.lbl_main_image.setAlignment(Qt.AlignCenter)
        self.lbl_main_image.setStyleSheet("color: #888; font-size: 18px;")
        self.lbl_main_image.setSizePolicy(QSizePolicy.Ignored,
                                           QSizePolicy.Ignored)
        layout_single.addWidget(self.lbl_main_image)

        self.page_grid = QWidget()
        self.grid_layout = QGridLayout(self.page_grid)
        self.grid_labels = []
        titles = ["1. Original", "2. Depth Map", "3. Depth Mask",
                  "4. Depth Cutout", "5. SAM Seg", "6. Final Result"]
        for i in range(6):
            r, c = i // 3, i % 3
            box = QWidget()
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(0, 0, 0, 0)
            box_layout.setSpacing(2)
            lbl_title = QLabel(titles[i])
            lbl_title.setStyleSheet(
                "color: #ddd; font-weight: bold; background-color: #444; "
                "padding: 4px;")
            lbl_title.setAlignment(Qt.AlignCenter)
            lbl_title.setFixedHeight(24)
            lbl_img = QLabel()
            lbl_img.setAlignment(Qt.AlignCenter)
            lbl_img.setStyleSheet("border: 1px solid #555;")
            lbl_img.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            box_layout.addWidget(lbl_title)
            box_layout.addWidget(lbl_img)
            self.grid_layout.addWidget(box, r, c)
            self.grid_labels.append(lbl_img)

        self.stack.addWidget(self.page_single)
        self.stack.addWidget(self.page_grid)
        disp_layout.addWidget(self.stack)

        main_layout.addWidget(control_panel)
        main_layout.addWidget(display_panel)

    # ------------------------------------------------------------------
    # Logging & Copilot interface
    # ------------------------------------------------------------------

    def log(self, msg):
        self.log_requested.emit(msg)

    def get_copilot_context(self):
        context_data = None
        if self.image_queue and self.current_idx >= 0:
            path = self.image_queue[self.current_idx]
            if path in self.analysis_results:
                res = self.analysis_results[path]
                details = res.get('details', [])
                context_data = {
                    "summary_stats": res.get('stats', {}),
                    "available_columns": list(details[0].keys()) if details else [],
                    "details_sample": details[:15]
                }
        return context_data

    def execute_copilot_action(self, action, params):
        self.log(f"🤖 [Copilot Dispatch] Action: {action}")

        # --- System & Execution ---
        if action == "run_analysis":
            task_name = params.get("task_name")
            if task_name:
                idx = self.combo_task.findText(task_name, Qt.MatchContains)
                if idx >= 0:
                    self.combo_task.setCurrentIndex(idx)

            model_type = params.get("model_type", "")
            if model_type:
                m_str = str(model_type).lower()
                target_idx = -1
                if "vit_h" in m_str or "high" in m_str:
                    target_idx = 0
                elif "vit_l" in m_str or "balance" in m_str:
                    target_idx = 1
                elif "vit_b" in m_str or "fast" in m_str:
                    target_idx = 2
                if target_idx != -1 and \
                   self.combo_sam_weight.currentIndex() != target_idx:
                    self.combo_sam_weight.setCurrentIndex(target_idx)

            self.pending_target = params.get("target", "")
            self.pending_highlight_color = params.get("highlight_color",
                                                       "#1976d2")
            self.pending_show_cols = params.get("show_columns", [])
            self.pending_hide_cols = params.get("hide_columns", [])

            if self.image_queue:
                self.log("➡️ [UI Action] Initiating analysis automatically...")
                self.start_batch_analysis(from_copilot=True)
            else:
                self.copilot_message_requested.emit(
                    "⚠️ No images loaded. Please load an image first "
                    "before running analysis.")

        elif action == "load_model":
            model_type = params.get("model_type", "")
            m_str = str(model_type).lower()
            target_idx = -1
            if "vit_h" in m_str:
                target_idx = 0
            elif "vit_l" in m_str:
                target_idx = 1
            elif "vit_b" in m_str:
                target_idx = 2
            if target_idx != -1:
                self.combo_sam_weight.setCurrentIndex(target_idx)
                self.on_manual_model_load()
            else:
                self.copilot_message_requested.emit(
                    "⚠️ Invalid model type specified.")

        elif action == "clear_workspace":
            self._clear_workspace_internal()
            self.log("➡️ [UI Action] Workspace memory cleared by Copilot.")
            self.copilot_message_requested.emit(
                "🧹 Workspace cleared successfully. "
                "I'm ready for new tasks!")

        elif action == "export_results":
            if self.analysis_results:
                self.log("➡️ [UI Action] Initiating AUTO-export from AI command...")
                self.export_all_results(auto_export=True)
            else:
                self.copilot_message_requested.emit(
                    "⚠️ There are no analysis results to save yet. "
                    "Please run the analysis first.")

        elif action == "load_image":
            file_path = params.get("file_path", "")
            target_path = file_path.strip().strip("'\"")
            if target_path and os.path.exists(target_path):
                valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
                loaded_paths = []
                if os.path.isdir(target_path):
                    for f in os.listdir(target_path):
                        if f.lower().endswith(valid_extensions):
                            loaded_paths.append(
                                os.path.normpath(os.path.join(target_path, f)))
                    if not loaded_paths:
                        self.copilot_message_requested.emit(
                            f"⚠️ Directory opened, but no supported images "
                            f"(.jpg, .png, .bmp) were found in:\n📁 {target_path}")
                        return
                    self.copilot_message_requested.emit(
                        f"📂 Got it! Batch loaded {len(loaded_paths)} images "
                        f"from the directory.")
                else:
                    if target_path.lower().endswith(valid_extensions):
                        loaded_paths = [os.path.normpath(target_path)]
                        self.copilot_message_requested.emit(
                            f"📂 Loaded image successfully:\n"
                            f"📄 {os.path.basename(target_path)}")
                    else:
                        self.copilot_message_requested.emit(
                            "❌ Unsupported image format. Please provide "
                            ".jpg, .png or .bmp files.")
                        return

                self.image_queue = loaded_paths
                self.current_idx = 0
                self.analysis_results = {}
                self.lbl_file_count.setText(
                    f"{len(loaded_paths)} images loaded (By Copilot)")
                self.update_nav_ui()
                self.show_current_image_data()
                self.btn_run.setEnabled(True)
                self.btn_export.setEnabled(False)
                self.btn_clear.setEnabled(True)
                self.toggle_details.setEnabled(False)
                self.toggle_details.setChecked(False)
            else:
                self.copilot_message_requested.emit(
                    f"❌ Sorry, I couldn't find the path you specified:\n"
                    f"{target_path}\nPlease verify the spelling.")
                
        elif action == "add_summary_metric":
            metric_name = params.get("metric_name", "")
            metric_value = params.get("metric_value", "")
            
            if metric_name and hasattr(self, 'table_summary'):  
                
                row_count = self.table_summary.rowCount()
                
                self.table_summary.insertRow(row_count)
                self.table_summary.setItem(row_count, 0, QTableWidgetItem(str(metric_name)))
                self.table_summary.setItem(row_count, 1, QTableWidgetItem(str(metric_value)))

                if self.image_queue and self.current_idx < len(self.image_queue):
                    current_path = self.image_queue[self.current_idx]
                    if current_path in self.analysis_results:
                        self.analysis_results[current_path]['stats'][metric_name] = metric_value

                if hasattr(self, 'log_requested'):
                    self.log_requested.emit(f"📊 Copilot Added Summary Metric: {metric_name} = {metric_value}")

        # --- Navigation & View ---
        elif action == "navigate_image":
            direction = params.get("direction", "next")
            if direction == "prev" and self.btn_prev.isEnabled():
                self.show_prev_image()
                self.copilot_message_requested.emit(
                    "◀️ Moved to the previous image.")
            elif direction == "next" and self.btn_next.isEnabled():
                self.show_next_image()
                self.copilot_message_requested.emit(
                    "▶️ Moved to the next image.")
            else:
                self.copilot_message_requested.emit(
                    "⚠️ Cannot navigate further in that direction. "
                    "You are at the end of the queue.")

        elif action == "toggle_details_view":
            show_details = params.get("show_details", False)
            if self.toggle_details.isEnabled():
                self.toggle_details.setChecked(show_details)
                status = "enabled" if show_details else "disabled"
                self.copilot_message_requested.emit(
                    f"👁️ Process details view has been {status}.")
            else:
                self.copilot_message_requested.emit(
                    "⚠️ Process details are not available yet. "
                    "Run analysis first.")

        elif action == "set_visual_style":
            text_size = params.get("text_size")
            text_pos = params.get("text_position")
            thickness = params.get("thickness")
            changes_made = False
            if text_size is not None:
                self.spin_font.setValue(float(text_size))
                changes_made = True
            if thickness is not None:
                self.spin_thick.setValue(float(thickness))
                changes_made = True
            if text_pos is not None:
                idx = self.combo_pos.findText(text_pos, Qt.MatchContains)
                if idx >= 0:
                    self.combo_pos.setCurrentIndex(idx)
                    changes_made = True
            if changes_made:
                self.copilot_message_requested.emit(
                    "🎨 Visual styling updated dynamically on the canvas.")

        # --- Data Editing & Tables ---
        elif action == "highlight_item":
            target = params.get("target", "")
            color_hex = params.get("highlight_color", "#1976d2")
            if target:
                self._execute_copilot_highlight(target, color_hex)

        elif action == "toggle_columns":
            show_cols = params.get("show_columns", [])
            hide_cols = params.get("hide_columns", [])
            self._execute_copilot_toggle_columns(show_cols, hide_cols)

        elif action == "delete_item":
            target_id = params.get("target_id", "")
            if not self.image_queue or self.current_idx < 0:
                self.copilot_message_requested.emit(
                    "⚠️ Please load an image and run the analysis first.")
                return
            current_path = self.image_queue[self.current_idx]
            if current_path not in self.analysis_results:
                self.copilot_message_requested.emit(
                    "⚠️ The current image has not been analyzed yet.")
                return
            details = self.analysis_results[current_path].get('details', [])
            if not details:
                return

            target_row_idx = -1
            match = re.search(r'\d+', str(target_id))
            if match:
                clean_id = match.group(0)
                for i, row in enumerate(details):
                    if str(row.get('ID', '')) == clean_id:
                        target_row_idx = i
                        break
            if target_row_idx >= 0:
                self.log(f"➡️ [UI Action] Copilot initiated deletion for ID: {clean_id}")
                self.delete_object_from_table(target_row_idx)
            else:
                self.copilot_message_requested.emit(
                    f"❌ I couldn't find a target with ID {target_id} "
                    f"in the current data.")

        elif action == "edit_item_id":
            old_id = params.get("old_id")
            new_id = params.get("new_id")
            if not self.image_queue or self.current_idx < 0:
                return
            current_path = self.image_queue[self.current_idx]
            if current_path not in self.analysis_results:
                return
            details = self.analysis_results[current_path].get('details', [])
            found = False
            for item in details:
                if str(item.get('ID', '')) == str(old_id):
                    item['ID'] = int(new_id)
                    found = True
                    break
            if found:
                res = self.analysis_results[current_path]
                self.populate_tables(res['stats'], details)
                self.redraw_main_image()
                self.log(f"➡️ [UI Action] Copilot changed ID {old_id} to {new_id}")
                self.copilot_message_requested.emit(
                    f"✅ Object ID successfully updated from {old_id} to {new_id}.")
            else:
                self.copilot_message_requested.emit(
                    f"⚠️ Could not locate ID {old_id} to modify.")

        elif action == "reorder_ids":
            if self.image_queue and self.current_idx >= 0:
                current_path = self.image_queue[self.current_idx]
                if current_path in self.analysis_results:
                    self.auto_reorder_ids()
                    self.copilot_message_requested.emit(
                        "✅ All IDs have been sequentially reordered.")

        elif action == "delete_column":
            col_name = params.get("column_name", "")
            if not col_name or not self.image_queue or self.current_idx < 0:
                return
            target_idx = -1
            self.stats_tabs.setCurrentWidget(self.table_details)
            for col in range(self.table_details.columnCount()):
                header_item = self.table_details.horizontalHeaderItem(col)
                if header_item and \
                   col_name.lower() in header_item.text().lower():
                    target_idx = col
                    break
            if target_idx >= 0:
                actual_name = \
                    self.table_details.horizontalHeaderItem(target_idx).text()
                self.delete_column_from_table(target_idx, actual_name)
                self.copilot_message_requested.emit(
                    f"🗑️ Column '{actual_name}' has been deleted from the dataset.")
            else:
                self.copilot_message_requested.emit(
                    f"⚠️ Could not find a column matching '{col_name}'.")

    def _clear_workspace_internal(self):
        """Clear workspace state without confirmation dialog."""
        self.image_queue = []
        self.current_idx = -1
        self.analysis_results.clear()
        self.cache_main_img = None
        self.cache_steps_imgs = []
        self.lbl_file_count.setText("No images loaded")
        self.lbl_index.setText("0 / 0")
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.progress_bar.setValue(0)
        self.table_summary.setRowCount(0)
        self.table_details.setRowCount(0)
        self.toggle_details.setChecked(False)
        self.toggle_details.setEnabled(False)
        self.stack.setCurrentIndex(0)
        self.lbl_main_image.setText("Waiting for input...")
        self.lbl_main_image.setPixmap(QPixmap())

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def start_model_loading(self):
        self.log("System Initializing...")
        self.init_thread = ModelInitThread()
        self.init_thread.progress_signal.connect(self.update_progress)
        self.init_thread.finished_signal.connect(self.on_models_loaded)
        self.init_thread.start()

    def on_manual_model_load(self):
        text = self.combo_sam_weight.currentText()
        if "ViT-H" in text:
            sam_type = "vit_h"
        elif "ViT-L" in text:
            sam_type = "vit_l"
        else:
            sam_type = "vit_b"

        self.copilot_message_requested.emit(
            "🔄 Manual Model Switch Initiated...")
        self.log(f"Initiating manual load for SAM {sam_type}...")
        self.btn_run.setEnabled(False)
        self.btn_reload_sam.setEnabled(False)

        self.init_thread = ModelInitThread(target_sam_type=sam_type)
        self.init_thread.progress_signal.connect(self.update_progress)
        self.init_thread.finished_signal.connect(self.on_models_loaded)
        self.init_thread.start()

    def on_models_loaded(self):
        self.log("AI Engine Ready.")
        self.copilot_message_requested.emit(
            "Engine is fully loaded and ready!")
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)
        if hasattr(self, 'btn_reload_sam'):
            self.btn_reload_sam.setEnabled(True)

    # ------------------------------------------------------------------
    # Copilot helper actions
    # ------------------------------------------------------------------

    def _execute_copilot_toggle_columns(self, show_cols, hide_cols):
        self.stats_tabs.setCurrentWidget(self.table_details)
        for col in range(self.table_details.columnCount()):
            header_item = self.table_details.horizontalHeaderItem(col)
            if not header_item:
                continue
            header_text = header_item.text().strip().lower()
            for h in hide_cols:
                h_clean = h.strip().lower()
                if h_clean in header_text or header_text in h_clean:
                    self.table_details.setColumnHidden(col, True)
            for s in show_cols:
                s_clean = s.strip().lower()
                if s_clean in header_text or header_text in s_clean:
                    self.table_details.setColumnHidden(col, False)
        self.log("✨ [UI Action] Columns updated.")

    def _execute_copilot_highlight(self, target_str, color_hex="#bbdefb"):
        if not self.image_queue or self.current_idx < 0:
            return
        target_str = str(target_str)
        color_hex = str(color_hex)

        current_path = self.image_queue[self.current_idx]
        if current_path not in self.analysis_results:
            return
        details = self.analysis_results[current_path].get('details', [])
        if not details:
            return

        target_row_idx = -1
        match_minmax = re.search(
            r'(max|min)\s*\(\s*(.+?)\s*\)', target_str, re.IGNORECASE)
        if match_minmax:
            op = match_minmax.group(1).lower()
            field = match_minmax.group(2).strip()
            actual_key = None
            for k in details[0].keys():
                if field.lower() in k.lower() or k.lower() in field.lower():
                    actual_key = k
                    break
            if actual_key:
                try:
                    if op == 'max':
                        target_row_idx = max(
                            range(len(details)),
                            key=lambda i: float(details[i][actual_key] or 0))
                    else:
                        target_row_idx = min(
                            range(len(details)),
                            key=lambda i: float(
                                details[i][actual_key] or float('inf')))
                except Exception:
                    pass

        if target_row_idx == -1:
            match_id = re.search(
                r'(?:ID|Row|No\.?)?\s*[:=]?\s*(\d+)', target_str,
                re.IGNORECASE)
            target_id = None
            if match_id:
                target_id = match_id.group(1)
            elif re.search(r'\b(\d+)\b', target_str):
                target_id = re.search(r'\b(\d+)\b', target_str).group(1)

            if target_id:
                for i, row in enumerate(details):
                    if str(row.get('ID', '')) == target_id:
                        target_row_idx = i
                        break

        if target_row_idx >= 0:
            self.stats_tabs.setCurrentWidget(self.table_details)
            self.table_details.clearSelection()
            self.table_details.scrollToItem(
                self.table_details.item(target_row_idx, 0))

            for r in range(self.table_details.rowCount()):
                for c in range(self.table_details.columnCount()):
                    item = self.table_details.item(r, c)
                    if item:
                        item.setBackground(Qt.NoBrush)

            for r, row_data in enumerate(details):
                for c in range(self.table_details.columnCount()):
                    header = self.table_details.horizontalHeaderItem(c)
                    if header and header.text() == "Swatch":
                        item = self.table_details.item(r, c)
                        hex_code = row_data.get("ColorHex")
                        if hex_code and isinstance(hex_code, str):
                            item.setBackground(QColor(hex_code))
                        elif "RGB" in row_data and \
                             isinstance(row_data["RGB"], (tuple, list)):
                            item.setBackground(QColor(*row_data["RGB"]))

            for col in range(self.table_details.columnCount()):
                item = self.table_details.item(target_row_idx, col)
                if item:
                    item.setBackground(QColor(color_hex))

            self.table_details.viewport().update()
            self.log(f"✨ [UI Action] Highlighted Row {target_row_idx + 1} "
                     f"with semantic color ({color_hex}).")
        else:
            self.log(f"⚠️ [UI Action] Failed to match target "
                     f"'{target_str}' in data.")

    # ------------------------------------------------------------------
    # Image loading & navigation
    # ------------------------------------------------------------------

    def batch_load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.jpg *.png *.bmp *.jpeg)")
        if paths:
            self.copilot_message_requested.emit(
                "📂 Importing images into workspace...")
            self.image_queue = paths
            self.current_idx = 0
            self.analysis_results = {}
            self.lbl_file_count.setText(f"{len(paths)} images loaded")
            self.update_nav_ui()
            self.show_current_image_data()
            self.btn_run.setEnabled(True)
            self.btn_export.setEnabled(False)
            self.btn_clear.setEnabled(True)
            self.toggle_details.setEnabled(False)
            self.toggle_details.setChecked(False)
            self.log(f"Batch loaded: {len(paths)} images.")

    def update_nav_ui(self):
        if not self.image_queue:
            self.lbl_index.setText("0 / 0")
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            return
        self.lbl_index.setText(
            f"{self.current_idx + 1} / {len(self.image_queue)}")
        self.btn_prev.setEnabled(self.current_idx > 0)
        self.btn_next.setEnabled(
            self.current_idx < len(self.image_queue) - 1)

    def show_prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_nav_ui()
            self.show_current_image_data()

    def show_next_image(self):
        if self.current_idx < len(self.image_queue) - 1:
            self.current_idx += 1
            self.update_nav_ui()
            self.show_current_image_data()

    def show_current_image_data(self):
        if not self.image_queue:
            return
        path = self.image_queue[self.current_idx]
        if path in self.analysis_results:
            res = self.analysis_results[path]
            self.cache_steps_imgs = res['pack']['steps']
            self.populate_tables(res['stats'], res['details'])
            self.toggle_details.setEnabled(True)
            if self.toggle_details.isChecked():
                self.stack.setCurrentIndex(1)
                self.update_grid_display()
            else:
                self.stack.setCurrentIndex(0)
                self.redraw_main_image()
        else:
            self.cache_main_img = None
            self.cache_steps_imgs = []
            self.toggle_details.setEnabled(False)
            self.toggle_details.setChecked(False)
            self.stack.setCurrentIndex(0)
            self.show_single_image(path)
            self.table_summary.setRowCount(0)
            self.table_details.setRowCount(0)

    # ------------------------------------------------------------------
    # Batch analysis engine
    # ------------------------------------------------------------------

    def start_batch_analysis(self, from_copilot=False):
        if not self.image_queue:
            return
        if not from_copilot:
            self.copilot_message_requested.emit(
                "⚙️ Manual Batch Analysis Started")

        self.is_batch_running = True
        self.btn_run.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_reload_sam.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.process_next_in_queue()

    def process_next_in_queue(self):
        target_idx = -1
        for i in range(len(self.image_queue)):
            if self.image_queue[i] not in self.analysis_results:
                target_idx = i
                break
        if target_idx == -1:
            self.batch_finished()
            return

        self.current_idx = target_idx
        self.update_nav_ui()
        self.show_current_image_data()

        path = self.image_queue[target_idx]
        task = self.combo_task.currentText()

        weight_text = self.combo_sam_weight.currentText()
        if "ViT-H" in weight_text:
            sam_type = "vit_h"
        elif "ViT-L" in weight_text:
            sam_type = "vit_l"
        else:
            sam_type = "vit_b"
        user_params = {"sam_type": sam_type}

        self.log(f"Processing ({target_idx + 1}/{len(self.image_queue)}): "
                 f"{os.path.basename(path)}...")
        self.worker = WorkerThread(task, path, params=user_params)
        self.worker.signals.progress.connect(self.update_progress)
        self.worker.signals.result.connect(self.batch_handle_result)
        self.worker.signals.error.connect(self.handle_error)
        self.worker.signals.finished.connect(self.on_worker_finished)
        self.worker.start()

    def batch_handle_result(self, images_packet, summary_stats, detailed_data):
        current_path = self.image_queue[self.current_idx]
        clean_bgr = imread_unicode(current_path)
        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
        self.analysis_results[current_path] = {
            'pack': images_packet,
            'stats': summary_stats,
            'details': detailed_data,
            'clean_main': clean_rgb
        }
        self.show_current_image_data()

        if getattr(self, 'pending_show_cols', []) or \
           getattr(self, 'pending_hide_cols', []):
            self._execute_copilot_toggle_columns(
                self.pending_show_cols, self.pending_hide_cols)

        if getattr(self, 'pending_target', ""):
            color = getattr(self, 'pending_highlight_color', '#ffcdd2')
            self._execute_copilot_highlight(self.pending_target, color)

        self.pending_target = ""
        self.pending_show_cols = []
        self.pending_hide_cols = []
        self.pending_highlight_color = "#ffcdd2"

    def on_worker_finished(self):
        if self.is_batch_running:
            self.process_next_in_queue()

    def batch_finished(self):
        self.is_batch_running = False
        self.log("✅ Batch Analysis Complete!")
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_reload_sam.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.update_nav_ui()

    def handle_error(self, err):
        self.log(f"ERROR: {err}")
        QMessageBox.critical(self, "Error", str(err))
        self.is_batch_running = False
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.btn_reload_sam.setEnabled(True)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_all_results(self, auto_export=False):
        if not self.analysis_results:
            return
        default_dir = os.path.dirname(self.image_queue[0]) \
            if self.image_queue else ""

        save_dir = ""
        custom_base_name = None

        if auto_export:
            save_dir = default_dir
        else:
            if len(self.image_queue) == 1:
                orig_name = os.path.splitext(
                    os.path.basename(self.image_queue[0]))[0]
                default_path = os.path.join(default_dir,
                                             f"{orig_name}_result")
                save_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Results", default_path, "All Files (*)")
                if not save_path:
                    return
                save_dir = os.path.dirname(save_path)
                custom_base_name = os.path.splitext(
                    os.path.basename(save_path))[0]
            else:
                save_dir = QFileDialog.getExistingDirectory(
                    self, "Select Output Directory For Batch", default_dir)
                if not save_dir:
                    return

        try:
            count = 0
            for path in self.image_queue:
                if path not in self.analysis_results:
                    continue
                res = self.analysis_results[path]

                if len(self.image_queue) == 1 and custom_base_name:
                    base_name = custom_base_name
                else:
                    orig_name = os.path.splitext(
                        os.path.basename(path))[0]
                    base_name = f"{orig_name}_result"

                img_save_path = os.path.join(save_dir, f"{base_name}.jpg")
                cv2.imwrite(img_save_path,
                            cv2.cvtColor(res['pack']['main'],
                                         cv2.COLOR_RGB2BGR))

                details = res.get('details', [])
                summary_stats = res.get('stats', {})

                if details and len(details) > 0:
                    keys = details[0].keys()


                    ui_visible_headers = []
                    if hasattr(self, 'table_details') and self.table_details.columnCount() > 0:
                        for col in range(self.table_details.columnCount()):
                            if not self.table_details.isColumnHidden(col):
                                header_item = self.table_details.horizontalHeaderItem(col)
                                if header_item:
                                    ui_visible_headers.append(header_item.text())

                    visible_keys = []
                    for k in keys:
                        if str(k).startswith("_") or k in ["Contour Global", "Draw Axises", "Center", "ColorHex", "Box"]:
                            continue

                        if ui_visible_headers:
                            if str(k) in ui_visible_headers:
                                visible_keys.append(k)
                        else:
                            visible_keys.append(k)

                    export_keys = [k for k in EXPORT_COLUMN_ORDER if k in visible_keys]
                    remaining = [k for k in visible_keys if k not in export_keys]
                    export_keys += remaining

                    csv_save_path = os.path.join(save_dir, f"{base_name}.csv")
                    with open(csv_save_path, 'w', newline='', encoding='utf-8') as csvfile:
                        
                        if summary_stats:
                            basic_writer = csv.writer(csvfile)
                            basic_writer.writerow(["[Summary]"]) 
                            
                            for stat_key, stat_value in summary_stats.items():
                                basic_writer.writerow([stat_key, stat_value])
                                
                            basic_writer.writerow([]) 
                            basic_writer.writerow(["[Details]"])  

                        dict_writer = csv.DictWriter(csvfile, fieldnames=export_keys)
                        dict_writer.writeheader()
                        for row in details:
                            dict_writer.writerow({k: row.get(k, "") for k in export_keys})
                            
                count += 1

            self.log(f"Exported {count} items to {save_dir}")

            if auto_export:
                self.copilot_message_requested.emit(
                    f"✅ I have automatically exported the results "
                    f"(Images & CSV) to your original folder:\n📁 {save_dir}")
            else:
                QMessageBox.information(
                    self, "Export Success",
                    f"Successfully exported {count} images and CSV files "
                    f"to:\n{save_dir}")

        except Exception as e:
            self.log(f"Export Failed: {e}")
            if auto_export:
                self.copilot_message_requested.emit(
                    f"❌ Failed to export automatically: {str(e)}")
            else:
                QMessageBox.critical(self, "Export Error", str(e))

    # ------------------------------------------------------------------
    # Table rendering
    # ------------------------------------------------------------------

    def populate_tables(self, summary_stats, detailed_data):
        gs = GlobalState()
        scale = gs.pixels_per_unit
        unit = gs.unit_name
        metric_rules = {
            "Length": 1, "Width": 1, "Diameter": 1, "Perim": 1,
            "Height": 1, "Radius": 1, "Distance": 1, "Gap": 1,
            "Area": 2, "Size": 2, "Volume": 3,
            "Ratio": 0, "Circularity": 0, "Angle": 0
        }

        def smart_convert(key, val):
            if any(k in key for k in
                   ["RGB", "Color", "Row", "ID", "Swatch"]):
                return str(val)
            dimension = -1
            for rule_key, dim in metric_rules.items():
                if rule_key.lower() in key.lower():
                    dimension = dim
                    break
            if dimension == -1:
                return str(val)
            try:
                if isinstance(val, (int, float)):
                    num_val = float(val)
                else:
                    matches = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
                    if not matches:
                        return str(val)
                    num_val = float(matches[0])
            except Exception:
                return str(val)

            if unit != "px" and scale != 1.0 and dimension > 0:
                if dimension == 1:
                    return f"{num_val / scale:.2f} {unit}"
                elif dimension == 2:
                    return f"{num_val / (scale * scale):.2f} {unit}²"
                elif dimension == 3:
                    return f"{num_val / (scale * scale * scale):.2f} {unit}³"
            return f"{num_val:.2f}"

        self.table_summary.setRowCount(len(summary_stats))
        for i, (k, v) in enumerate(summary_stats.items()):
            self.table_summary.setItem(i, 0, QTableWidgetItem(str(k)))
            self.table_summary.setItem(
                i, 1, QTableWidgetItem(str(smart_convert(k, v))))

        if detailed_data and len(detailed_data) > 0:
            detailed_data.sort(key=lambda x: x.get('ID', 0))
            all_keys = list(detailed_data[0].keys())
            ignore_list = ["ColorHex", "Box", "Contour Global",
                           "Draw Axises", "Center"]
            visible_keys = [k for k in all_keys
                            if not str(k).startswith("_")
                            and k not in ignore_list]
            priority_order = [
                "ID", "Row", "Length", "Width", "Area", "L/W Ratio",
                "Diameter", "Volume", "ColorName", "Color", "RGB",
                "RGB Val", "Swatch"]
            sorted_keys = [k for k in priority_order if k in visible_keys]
            remaining_keys = [k for k in visible_keys
                              if k not in sorted_keys]
            final_headers = sorted_keys + remaining_keys

            self.table_details.setColumnCount(len(final_headers))
            self.table_details.setHorizontalHeaderLabels(final_headers)
            self.table_details.setRowCount(len(detailed_data))

            for row_idx, row_data in enumerate(detailed_data):
                for col_idx, header in enumerate(final_headers):
                    val = row_data.get(header, "")
                    if header == "Swatch":
                        item = QTableWidgetItem("")
                        hex_code = row_data.get("ColorHex")
                        if hex_code and isinstance(hex_code, str):
                            item.setBackground(QColor(hex_code))
                        elif "RGB" in row_data and \
                             isinstance(row_data["RGB"], (tuple, list)):
                            item.setBackground(QColor(*row_data["RGB"]))
                        item.setFlags(
                            item.flags() ^ Qt.ItemIsEditable)
                        self.table_details.setItem(
                            row_idx, col_idx, item)
                    else:
                        item = QTableWidgetItem(
                            str(smart_convert(header, val)))
                        item.setTextAlignment(Qt.AlignCenter)
                        self.table_details.setItem(
                            row_idx, col_idx, item)

            for col in range(self.table_details.columnCount()):
                self.table_details.setColumnHidden(col, False)
        else:
            self.table_details.setRowCount(0)

    # ------------------------------------------------------------------
    # Image display
    # ------------------------------------------------------------------

    def toggle_view_mode(self, checked):
        if not self.cache_steps_imgs:
            return
        if checked:
            self.stack.setCurrentIndex(1)
            self.update_grid_display()
        else:
            self.stack.setCurrentIndex(0)
            self.redraw_main_image()

    def show_single_image(self, img_source):
        pixmap = self._convert_to_pixmap(img_source)
        if pixmap:
            self.lbl_main_image.setPixmap(
                pixmap.scaled(self.lbl_main_image.size(),
                              Qt.KeepAspectRatio,
                              Qt.SmoothTransformation))

    def update_grid_display(self):
        for i, lbl in enumerate(self.grid_labels):
            if i < len(self.cache_steps_imgs):
                pixmap = self._convert_to_pixmap(self.cache_steps_imgs[i])
                if pixmap:
                    lbl.setPixmap(
                        pixmap.scaled(lbl.size(),
                                      Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation))

    def resizeEvent(self, event):
        if self.current_idx >= 0 and self.image_queue:
            if self.stack.currentIndex() == 0:
                self.show_single_image(
                    self.cache_main_img or self.image_queue[self.current_idx])
            else:
                self.update_grid_display()
        super().resizeEvent(event)

    def _convert_to_pixmap(self, img_source):
        if isinstance(img_source, str):
            return QPixmap(img_source)
        elif isinstance(img_source, np.ndarray):
            h, w, ch = img_source.shape
            return QPixmap.fromImage(
                QImage(img_source.data, w, h, ch * w, QImage.Format_RGB888))
        return None

    def update_progress(self, val, msg):
        self.progress_bar.setValue(val)
        if msg:
            self.log(msg)

    # ------------------------------------------------------------------
    # Clear workspace
    # ------------------------------------------------------------------

    def clear_workspace(self):
        reply = QMessageBox.question(
            self, 'Clear Workspace',
            "Are you sure you want to clear all images and results?\n"
            "This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return
        self._clear_workspace_internal()
        self.log("Workspace memory completely cleared.")
        self.copilot_message_requested.emit(
            "🧹 Workspace cleared. I'm ready for a new batch of tasks!")

    # ------------------------------------------------------------------
    # Context menu (table right-click)
    # ------------------------------------------------------------------

    def show_table_context_menu(self, pos):
        if not self.image_queue or self.current_idx < 0:
            return
        item = self.table_details.itemAt(pos)
        if not item:
            return

        menu = QMenu(self)
        menu.setStyleSheet(MENU_STYLESHEET)

        row = item.row()
        details = self.analysis_results[
            self.image_queue[self.current_idx]]['details']
        current_id = details[row].get('ID', '?')

        edit_action = menu.addAction(f"✏️ Edit ID ({current_id})")
        menu.addSeparator()
        reorder_action = menu.addAction("🔄 Auto-Reorder All IDs")
        delete_action = menu.addAction("🗑️ Delete This Object")

        action = menu.exec(self.table_details.viewport().mapToGlobal(pos))

        if action == edit_action:
            self.edit_object_id(row)
        elif action == reorder_action:
            self.auto_reorder_ids()
        elif action == delete_action:
            self.delete_object_from_table(row)

    def delete_object_from_table(self, row):
        current_path = self.image_queue[self.current_idx]
        res = self.analysis_results[current_path]
        details = res['details']
        stats = res['stats']

        deleted_item = details.pop(row)

        if details:
            for k in list(stats.keys()):
                if "Total" in k:
                    stats[k] = len(details)
                elif k == "Avg Area":
                    stats[k] = int(
                        sum([d["Area"] for d in details]) / len(details))
                elif k == "Avg L/W":
                    stats[k] = round(
                        sum([d["L/W Ratio"] for d in details])
                        / len(details), 2)
                elif k == "Avg Length":
                    stats[k] = round(
                        sum([d["Length"] for d in details])
                        / len(details), 1)
                elif k == "Avg Volume":
                    stats[k] = int(
                        sum([d["Volume"] for d in details]) / len(details))
        else:
            res['stats'] = {"Status": "All objects deleted manually"}

        self.populate_tables(res['stats'], details)
        self.redraw_main_image()

        deleted_id = deleted_item.get('ID', '?')
        self.log(f"🗑️ Cleaned up ID {deleted_id}. "
                 f"Summary statistics recalculated.")
        self.copilot_message_requested.emit(
            f"✅ I have removed **ID {deleted_id}** from the dataset "
            f"and recalculated the averages for you!")

    def on_image_right_clicked(self, label_x, label_y):
        if not self.image_queue or self.current_idx < 0:
            return
        current_path = self.image_queue[self.current_idx]
        if current_path not in self.analysis_results:
            return

        res = self.analysis_results[current_path]
        pixmap = self.lbl_main_image.pixmap()
        if not pixmap:
            return

        lbl_w = self.lbl_main_image.width()
        lbl_h = self.lbl_main_image.height()
        scaled_w = pixmap.width()
        scaled_h = pixmap.height()
        offset_x = (lbl_w - scaled_w) / 2
        offset_y = (lbl_h - scaled_h) / 2

        if label_x < offset_x or label_x > offset_x + scaled_w or \
           label_y < offset_y or label_y > offset_y + scaled_h:
            return

        real_h, real_w = res['clean_main'].shape[:2]
        img_x = int((label_x - offset_x) / scaled_w * real_w)
        img_y = int((label_y - offset_y) / scaled_h * real_h)

        from PySide6.QtGui import QCursor

        for row_idx, item in enumerate(res['details']):
            cnt = item.get('_contour')
            if cnt is not None:
                dist = cv2.pointPolygonTest(cnt, (img_x, img_y), False)
                if dist >= 0:
                    menu = QMenu(self)
                    menu.setStyleSheet(MENU_STYLESHEET)

                    target_id = item.get('ID', '?')
                    edit_action = menu.addAction(
                        f"✏️ Edit ID ({target_id})")
                    menu.addSeparator()
                    reorder_action = menu.addAction(
                        "🔄 Auto-Reorder All IDs")
                    delete_action = menu.addAction(
                        f"🗑️ Delete Object ID: {target_id}")

                    action = menu.exec(QCursor.pos())

                    if action == edit_action:
                        self.edit_object_id(row_idx)
                    elif action == reorder_action:
                        self.auto_reorder_ids()
                    elif action == delete_action:
                        self.delete_object_from_table(row_idx)

                    break

    def edit_object_id(self, row):
        current_path = self.image_queue[self.current_idx]
        res = self.analysis_results[current_path]
        details = res['details']

        old_id = details[row].get('ID', 0)

        new_id, ok = QInputDialog.getInt(
            self, "Edit ID",
            f"Enter a new ID for the selected object "
            f"(Current: {old_id}):",
            int(old_id), 1, 99999, 1)
        if ok and new_id != old_id:
            details[row]['ID'] = new_id
            self.populate_tables(res['stats'], details)
            self.redraw_main_image()
            self.log(f"✏️ Object ID manually updated from {old_id} to {new_id}.")

    def auto_reorder_ids(self):
        current_path = self.image_queue[self.current_idx]
        res = self.analysis_results[current_path]
        details = res['details']

        for index, item in enumerate(details):
            item['ID'] = index + 1

        self.populate_tables(res['stats'], details)
        self.redraw_main_image()
        self.log(f"🔄 Automatically reordered {len(details)} objects.")

    # ------------------------------------------------------------------
    # Dynamic annotation rendering
    # ------------------------------------------------------------------

    def redraw_main_image(self):
        if not self.image_queue or self.current_idx < 0:
            return
        current_path = self.image_queue[self.current_idx]
        if current_path not in self.analysis_results:
            return
        res = self.analysis_results[current_path]

        canvas = res['clean_main'].copy()

        user_font_mult = self.spin_font.value()
        user_thick_val = self.spin_thick.value()
        pos_text = self.combo_pos.currentText()

        img_h, img_w = canvas.shape[:2]
        diagonal = math.sqrt(img_w**2 + img_h**2)
        base_scale = math.sqrt(diagonal / 1500.0)

        final_font_scale = base_scale * user_font_mult
        calculated_thick = int(round(user_thick_val * base_scale))
        contour_thickness = max(1, calculated_thick)
        text_thickness = contour_thickness
        text_outline = text_thickness + max(1, int(base_scale * 2))
        offset_dist = int(35 * base_scale * user_font_mult)

        for item in res['details']:
            cnt = item.get('_contour')
            raw_center = item.get('_center')
            display_text = str(item.get('ID', ''))

            if raw_center is not None:
                cx, cy = raw_center

                if cnt is not None:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                if "Top" in pos_text:
                    cy -= offset_dist
                elif "Bottom" in pos_text:
                    cy += offset_dist
                elif "Left" in pos_text:
                    cx -= offset_dist
                elif "Right" in pos_text:
                    cx += offset_dist

                # Render contours and geometric overlays
                if cnt is not None:
                    if '_viz_color' in item:
                        bgr = item['_viz_color']
                        viz_color = (bgr[2], bgr[1], bgr[0])
                    else:
                        viz_color = (34, 139, 34)

                    current_thickness = contour_thickness
                    if '_thickness_boost' in item:
                        current_thickness += item['_thickness_boost']

                    cv2.drawContours(canvas, [cnt], -1, viz_color,
                                     current_thickness)

                    hull = item.get('_hull')
                    if hull is not None:
                        cv2.drawContours(canvas, [hull], -1,
                                         (255, 0, 0), contour_thickness)
                    box = item.get('_box')
                    if box is not None:
                        cv2.drawContours(canvas, [box], -1,
                                         (0, 0, 255), contour_thickness)
                    circle = item.get('_circle')
                    if circle is not None:
                        cv2.circle(canvas, circle[0], circle[1],
                                   (200, 200, 200),
                                   max(1, contour_thickness - 1))

                    major_axis = item.get('_major_axis')
                    if major_axis is not None:
                        cv2.line(canvas, major_axis[0], major_axis[1],
                                 (255, 0, 0), contour_thickness)
                    minor_axis = item.get('_minor_axis')
                    if minor_axis is not None:
                        cv2.line(canvas, minor_axis[0], minor_axis[1],
                                 (0, 0, 255), contour_thickness)

                # Wheat Leaf Angle rendering
                angle_data = item.get('_angle_data')
                if angle_data is not None:
                    j_orig = angle_data['junction']
                    s_end_orig = angle_data['stem_end']
                    l_end_orig = angle_data['leaf_end']
                    angle_val = angle_data['angle_val']

                    cv2.line(canvas, j_orig, s_end_orig,
                             (0, 255, 0), contour_thickness)
                    cv2.line(canvas, j_orig, l_end_orig,
                             (255, 0, 0), contour_thickness)
                    cv2.circle(canvas, j_orig, max(3, contour_thickness),
                               (0, 0, 255), -1)

                    radius = int(45 * base_scale * user_font_mult)
                    vec_s = np.array(s_end_orig) - np.array(j_orig)
                    vec_l = np.array(l_end_orig) - np.array(j_orig)
                    ang_s = np.degrees(np.arctan2(vec_s[1], vec_s[0]))
                    ang_l = np.degrees(np.arctan2(vec_l[1], vec_l[0]))

                    if ang_s < 0:
                        ang_s += 360
                    if ang_l < 0:
                        ang_l += 360

                    if abs(ang_s - ang_l) < 180:
                        start, end = min(ang_s, ang_l), max(ang_s, ang_l)
                    else:
                        start, end = \
                            max(ang_s, ang_l), min(ang_s, ang_l) + 360

                    cv2.ellipse(canvas, j_orig, (radius, radius),
                                0, start, end, (255, 255, 0), -1)

                    display_text = f"{angle_val:.2f} deg"

                    if "Center" in pos_text:
                        cx += int(30 * base_scale)
                        cy -= int(30 * base_scale)

                # Draw centered text label
                (text_w, text_h), _ = cv2.getTextSize(
                    display_text, cv2.FONT_HERSHEY_SIMPLEX,
                    final_font_scale, text_thickness)
                draw_x = int(cx - text_w / 2)
                draw_y = int(cy + text_h / 2)
                draw_point = (draw_x, draw_y)

                cv2.putText(canvas, display_text, draw_point,
                            cv2.FONT_HERSHEY_SIMPLEX, final_font_scale,
                            (0, 0, 0), text_outline)
                cv2.putText(canvas, display_text, draw_point,
                            cv2.FONT_HERSHEY_SIMPLEX, final_font_scale,
                            (255, 255, 255), text_thickness)

        self.cache_main_img = canvas
        res['pack']['main'] = canvas
        self.show_single_image(self.cache_main_img)

    # ------------------------------------------------------------------
    # Header context menu (delete column)
    # ------------------------------------------------------------------

    def show_header_context_menu(self, pos):
        if not self.image_queue or self.current_idx < 0:
            return

        header = self.table_details.horizontalHeader()
        col_idx = header.logicalIndexAt(pos)
        if col_idx < 0:
            return

        col_name = self.table_details.horizontalHeaderItem(col_idx).text()

        menu = QMenu(self)
        menu.setStyleSheet(MENU_STYLESHEET)

        delete_action = menu.addAction(f"🗑️ Delete Column: {col_name}")
        action = menu.exec(header.mapToGlobal(pos))

        if action == delete_action:
            self.delete_column_from_table(col_idx, col_name)

    def delete_column_from_table(self, col_idx, col_name):
        current_path = self.image_queue[self.current_idx]
        res = self.analysis_results[current_path]

        for item in res['details']:
            if col_name in item:
                del item[col_name]

        self.table_details.removeColumn(col_idx)
        self.log(f"🗑️ Column '{col_name}' deleted successfully.")
        self.redraw_main_image()
