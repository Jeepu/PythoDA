#core/base_plugin.py
from PySide6.QtCore import QObject, Signal

class BasePhenoPlugin(QObject):
    
    update_progress = Signal(int, str)        
    result_ready = Signal(object, dict, list) 
    error_occurred = Signal(str)              

    def __init__(self):
        super().__init__()
        self.is_running = False

    def run(self, image_path, params=None):
        raise NotImplementedError("Plugins must implement the run() method.")

    def stop(self):
        self.is_running = False