import json
import os
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QComboBox,
                               QStackedWidget, QHBoxLayout, QLabel, QPushButton,
                               QTextEdit, QSizePolicy, QScrollArea,
                               QLineEdit, QApplication,QMessageBox, QDialog, QFormLayout, 
                               QDialogButtonBox, QMessageBox, QVBoxLayout)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QIcon

from ui.auto_panel import AutoPanel
from ui.manual_panel.workbench import ManualWorkbench
from ui.annotation_panel.panel import AnnotationPanel


class APISettingsDialog(QDialog):
    """A sleek dialog for users to configure their AI provider settings."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Copilot API Settings")
        self.setMinimumWidth(450)
        self.config_file = "api_config.json"
        
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        self.input_url = QLineEdit()
        self.input_url.setPlaceholderText("e.g., https://api.deepseek.com/chat/completions")
        
        self.input_model = QLineEdit()
        self.input_model.setPlaceholderText("e.g., deepseek-chat")
        
        self.input_key = QLineEdit()
        self.input_key.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        self.input_key.setPlaceholderText("sk-...")
        
        form_layout.addRow("Full API URL:", self.input_url)
        form_layout.addRow("Model Name:", self.input_model)
        form_layout.addRow("API Key:", self.input_key)
        
        layout.addLayout(form_layout)
        
        self.btn_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.btn_box.accepted.connect(self.save_config)
        self.btn_box.rejected.connect(self.reject)
        layout.addWidget(self.btn_box)
        
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.input_url.setText(config.get("base_url", ""))
                    self.input_model.setText(config.get("model_name", ""))
                    self.input_key.setText(config.get("api_key", ""))
            except Exception:
                pass

    def save_config(self):
        url = self.input_url.text().strip()
        if url.endswith("/"):
            url = url[:-1]
            
        config = {
            "base_url": url,
            "model_name": self.input_model.text().strip(),
            "api_key": self.input_key.text().strip()
        }
            
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
            
        QMessageBox.information(self, "Success", "API Configuration saved successfully!\nThe Copilot will use this immediately.")
        self.accept()

class CopilotWorkerThread(QThread):
    """Background thread for AI Copilot intent processing."""
    result_ready = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, user_input, context_data=None):
        super().__init__()
        self.user_input = user_input
        self.context_data = context_data

    def run(self):
        try:
            from core.copilot import PhytoDACopilot
            copilot = PhytoDACopilot()
            res = copilot.process_intent(self.user_input,
                                         context_data=self.context_data)
            self.result_ready.emit(res)
        except Exception as e:
            self.error_occurred.emit(str(e))


class UserMessageWidget(QWidget):
    """Chat bubble for user messages."""
    def __init__(self, text):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addStretch()
        lbl_msg = QLabel(text)
        lbl_msg.setWordWrap(True)
        lbl_msg.setStyleSheet(
            "background-color: #2e7d32; color: white; padding: 10px; "
            "border-radius: 8px; font-size: 13px; max-width: 250px;")
        layout.addWidget(lbl_msg)
        lbl_avatar = QLabel("🧑‍🌾")
        lbl_avatar.setStyleSheet("font-size: 20px;")
        layout.addWidget(lbl_avatar)


class CopilotSessionWidget(QWidget):
    """Expandable AI response bubble with collapsible internal log."""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 15)
        layout.setSpacing(5)

        top_layout = QHBoxLayout()
        top_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lbl_avatar = QLabel("🤖")
        lbl_avatar.setStyleSheet("font-size: 20px;")
        top_layout.addWidget(lbl_avatar)

        self.lbl_reply = QLabel()
        self.lbl_reply.setWordWrap(True)
        self.lbl_reply.setStyleSheet(
            "background-color: #e8f5e9; color: #1b5e20; padding: 10px; "
            "border-radius: 8px; font-size: 13px; max-width: 250px; "
            "border: 1px solid #c8e6c9;")
        top_layout.addWidget(self.lbl_reply)
        layout.addLayout(top_layout)

        self.btn_toggle = QPushButton("▶ Show Internal Process (0 steps)")
        self.btn_toggle.setStyleSheet(
            "text-align: left; border: none; color: #558b2f; "
            "font-size: 11px; padding-left: 35px; font-weight: bold;")
        self.btn_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_toggle.clicked.connect(self.toggle_logs)
        layout.addWidget(self.btn_toggle)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet(
            "background-color: #1e1e1e; color: #4af626; "
            "font-family: Consolas, monospace; font-size: 10px; "
            "border-radius: 6px; padding: 5px; margin-left: 35px;")
        self.log_box.setVisible(False)
        self.log_box.setFixedHeight(120)
        layout.addWidget(self.log_box)
        self.log_count = 0

    def set_reply(self, text):
        self.lbl_reply.setText(text)

    def append_log(self, text):
        self.log_box.append(text)
        self.log_count += 1
        arrow = "▼" if self.log_box.isVisible() else "▶"
        self.btn_toggle.setText(
            f"{arrow} Internal Process ({self.log_count} steps)")
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum())

    def toggle_logs(self):
        visible = not self.log_box.isVisible()
        self.log_box.setVisible(visible)
        arrow = "▼" if visible else "▶"
        self.btn_toggle.setText(
            f"{arrow} Internal Process ({self.log_count} steps)")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhytoDA")
        self.setWindowIcon(QIcon("ui/resources/Logo.png"))
        self.resize(1600, 800)
        self.setMinimumSize(1000, 600) 
        self.current_copilot_session = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # -- Left: main work area --
        self.main_container = QWidget()
        main_layout = QVBoxLayout(self.main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_bar = QWidget()
        top_bar.setStyleSheet(
            "background-color: #e0e0e0; border-bottom: 1px solid #ccc;")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(10, 5, 10, 5)

        lbl_mode = QLabel("Working Mode:")
        lbl_mode.setStyleSheet("font-weight: bold; color: #333;")

        self.combo_mode = QComboBox()
        self.combo_mode.addItems([
            "🤖 Auto Phenotyping",
            "✏️ Image Workbench",
            "🔬 Annotation Studio"])
        self.combo_mode.setFixedWidth(250)
        self.combo_mode.setStyleSheet("font-size: 14px; padding: 5px;")
        self.combo_mode.currentIndexChanged.connect(self.switch_mode)

        top_layout.addWidget(lbl_mode)
        top_layout.addWidget(self.combo_mode)
        self.btn_settings = QPushButton("API Settings")
        self.btn_settings.setStyleSheet(
            "background-color: #f5f5f5; border: 1px solid #ccc; "
            "padding: 5px 15px; border-radius: 4px; font-weight: bold; color: #333;"
        )
        self.btn_settings.setCursor(Qt.PointingHandCursor)
        self.btn_settings.clicked.connect(self.open_api_settings)
        
        top_layout.addWidget(self.btn_settings)
        top_layout.addStretch()
        main_layout.addWidget(top_bar)

        self.stack = QStackedWidget()
        self.auto_panel = AutoPanel()
        self.auto_panel.log_requested.connect(self.log_from_module)
        self.auto_panel.copilot_message_requested.connect(
            self._add_copilot_session)

        self.manual_panel = ManualWorkbench()
        self.anno_panel = AnnotationPanel()

        self.stack.addWidget(self.auto_panel)
        self.stack.addWidget(self.manual_panel)
        self.stack.addWidget(self.anno_panel)
        main_layout.addWidget(self.stack)

        root_layout.addWidget(self.main_container, stretch=1)

        self.btn_toggle = QPushButton("▶")
        self.btn_toggle.setFixedWidth(16)
        self.btn_toggle.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.btn_toggle.setStyleSheet("""
            QPushButton {
                background-color: #dcdcdc; border: 1px solid #bbb;
                border-top: none; border-bottom: none;
                color: #555; font-size: 12px;
            }
            QPushButton:hover {
                background-color: #c0c0c0; color: #000;
            }
        """)
        self.btn_toggle.clicked.connect(self.toggle_copilot)
        root_layout.addWidget(self.btn_toggle)

        # -- Right: Copilot panel --
        self.copilot_panel = QWidget()
        self.copilot_panel.setFixedWidth(350)
        self.copilot_panel.setStyleSheet(
            "background-color: #f4f9f5; border-left: 1px solid #c8e6c9;")
        copilot_layout = QVBoxLayout(self.copilot_panel)
        copilot_layout.setContentsMargins(0, 0, 0, 0)
        copilot_layout.setSpacing(0)

        copilot_header = QLabel("🤖 PhytoDA Copilot")
        copilot_header.setStyleSheet(
            "background-color: #e8f5e9; font-size: 16px; font-weight: bold; "
            "color: #1b5e20; padding: 15px; border-bottom: 1px solid #c8e6c9;")
        copilot_layout.addWidget(copilot_header)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; } "
            "QScrollBar:vertical { width: 8px; }")

        self.chat_container = QWidget()
        self.chat_container.setStyleSheet("background: transparent;")
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setContentsMargins(10, 15, 10, 15)
        self.chat_layout.setSpacing(15)
        self.chat_scroll.setWidget(self.chat_container)
        copilot_layout.addWidget(self.chat_scroll)

        input_container = QWidget()
        input_container.setStyleSheet(
            "background-color: #e8f5e9; border-top: 1px solid #c8e6c9;")
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(10, 10, 10, 10)

        self.input_intent = QLineEdit()
        self.input_intent.setPlaceholderText("Ask AI to analyze or filter...")
        self.input_intent.setStyleSheet(
            "background: white; border: 1px solid #a5d6a7; padding: 8px; "
            "border-radius: 15px; font-size: 13px;")
        self.input_intent.returnPressed.connect(self.run_ai_routing)

        btn_ask = QPushButton("⬆️")
        btn_ask.setFixedSize(32, 32)
        btn_ask.setStyleSheet(
            "background-color: #2e7d32; color: white; "
            "border-radius: 16px; font-weight: bold; font-size: 16px;")
        btn_ask.clicked.connect(self.run_ai_routing)

        input_layout.addWidget(self.input_intent)
        input_layout.addWidget(btn_ask)
        copilot_layout.addWidget(input_container)

        root_layout.addWidget(self.copilot_panel)

        self.copilot_visible = True
        self._add_copilot_session(
            "👋 Hello! I am your PhytoDA Copilot. I can control the software, "
            "run analysis, and filter data for you. Try asking me to run the "
            "fast model on tomatoes!")

    def open_api_settings(self):
        dialog = APISettingsDialog(self)
        dialog.exec()

    def switch_mode(self, index):
        self.stack.setCurrentIndex(index)

    def toggle_copilot(self):
        self.copilot_visible = not self.copilot_visible      
        self.copilot_panel.setVisible(self.copilot_visible)
        self.btn_toggle.setText("▶" if not self.copilot_visible else "◀")


    def _add_user_message(self, text):
        msg_widget = UserMessageWidget(text)
        self.chat_layout.addWidget(msg_widget)
        self._scroll_chat_to_bottom()

    def _add_copilot_session(self, initial_text=""):
        if not self.copilot_visible:
            return
        session = CopilotSessionWidget()
        if initial_text:
            session.set_reply(initial_text)
        self.chat_layout.addWidget(session)
        self.current_copilot_session = session
        self._scroll_chat_to_bottom()

    def _scroll_chat_to_bottom(self):
        QApplication.processEvents()
        QTimer.singleShot(50, lambda:
            self.chat_scroll.verticalScrollBar().setValue(
                self.chat_scroll.verticalScrollBar().maximum()))

    def log_from_module(self, msg):
        if not self.copilot_visible:
            return
        if not self.current_copilot_session:
            self._add_copilot_session("⚙️ System Process Started")
        self.current_copilot_session.append_log(msg)
        QApplication.processEvents()
        vbar = self.chat_scroll.verticalScrollBar()
        vbar.setValue(vbar.maximum())

    def run_ai_routing(self):
        user_text = self.input_intent.text().strip()
        if not user_text:
            return
        self.input_intent.clear()

        self._add_user_message(user_text)
        self._add_copilot_session("Thinking...")

        context_data = {"ui_mode": self.combo_mode.currentText()}

        current_idx = self.stack.currentIndex()
        if current_idx == 0 and hasattr(self.auto_panel, 'get_copilot_context'):
            auto_ctx = self.auto_panel.get_copilot_context()
            if auto_ctx:
                context_data.update(auto_ctx)
        elif current_idx == 1 and hasattr(self.manual_panel, 'get_copilot_context'):
            man_ctx = self.manual_panel.get_copilot_context()
            if man_ctx:
                context_data.update(man_ctx)
        elif current_idx == 2 and hasattr(self.anno_panel, 'get_copilot_context'):
            anno_ctx = self.anno_panel.get_copilot_context()
            if anno_ctx:
                context_data.update(anno_ctx)

        self.copilot_thread = CopilotWorkerThread(user_text, context_data)
        self.copilot_thread.result_ready.connect(self.handle_copilot_result)
        self.copilot_thread.error_occurred.connect(
            lambda e: self.current_copilot_session.set_reply(f"❌ Error: {e}"))
        self.copilot_thread.start()

    def handle_copilot_result(self, result_dict):
        reply = result_dict.get("dialogue_reply", "")
        action = result_dict.get("ui_action", "none")
        params = result_dict.get("action_params", {})

        self.current_copilot_session.set_reply(reply)
        self._scroll_chat_to_bottom()

        # Global command: mode switching
        if action == "switch_mode":
            target = params.get("target_mode", "").lower()
            if "workbench" in target or "image" in target:
                self.combo_mode.setCurrentIndex(1)
            elif "annotation" in target or "label" in target:
                self.combo_mode.setCurrentIndex(2)
            else:
                self.combo_mode.setCurrentIndex(0)
            self.log_from_module(
                f"🔄 UI Action: Switched to {self.combo_mode.currentText()}")
            return

        # Dispatch action to the active panel
        current_idx = self.stack.currentIndex()
        if current_idx == 0:
            if hasattr(self.auto_panel, 'execute_copilot_action'):
                self.auto_panel.execute_copilot_action(action, params)
        elif current_idx == 1:
            if hasattr(self.manual_panel, 'execute_copilot_action'):
                self.manual_panel.execute_copilot_action(action, params)
            else:
                self.log_from_module(
                    "⚠️ Image Workbench API hook is not implemented yet.")
        elif current_idx == 2:
            if hasattr(self.anno_panel, 'execute_copilot_action'):
                self.anno_panel.execute_copilot_action(action, params)
            else:
                self.log_from_module(
                    "⚠️ Annotation Studio API hook is not implemented yet.")
