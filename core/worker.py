# core/worker.py
from PySide6.QtCore import QThread, QObject, Signal
import traceback
from plugins.plugin_seed import SeedPlugin
from plugins.plugin_leaf import LeafPlugin 
from plugins.plugin_corn import CornPlugin 
from plugins.plugin_tomato import TomatoPlugin
from plugins.plugin_leafangle import WheatLeafAnglePlugin
from plugins.plugin_wheatear import WheatEarPlugin
from plugins.plugin_canopy import CanopyPlugin

class WorkerSignals(QObject):
    progress = Signal(int, str)
    result = Signal(object, dict, list) 
    error = Signal(str)
    finished = Signal()

class WorkerThread(QThread):
    def __init__(self, task_type, image_path, params=None):
        super().__init__()
        self.task_type = task_type
        self.image_path = image_path
        self.params = params  
        self.signals = WorkerSignals()
        self.plugin = None

    def run(self):
        try:
            if "Seed" in self.task_type:
                self.plugin = SeedPlugin()
            elif "Leaf Phenotyping" in self.task_type: 
                self.plugin = LeafPlugin()
            elif "Corn" in self.task_type:
                self.plugin = CornPlugin()
            elif "Tomato" in self.task_type:
                self.plugin = TomatoPlugin()
            elif "Leaf Angle" in self.task_type:
                self.plugin = WheatLeafAnglePlugin()
            elif "Wheat Ear" in self.task_type:
                self.plugin = WheatEarPlugin()
            elif "Canopy" in self.task_type:
                self.plugin = CanopyPlugin()
            else:
                raise ValueError(f"Unknown task: {self.task_type}")

            if self.plugin is None:
                raise ValueError("Plugin Init Failed")

            self.plugin.update_progress.connect(self.signals.progress.emit)
            self.plugin.result_ready.connect(self.signals.result.emit)
            self.plugin.error_occurred.connect(self.signals.error.emit)
            self.plugin.run(self.image_path, params=self.params)
            
        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()

    def stop(self):
        if self.plugin:
            self.plugin.stop()
        self.quit()