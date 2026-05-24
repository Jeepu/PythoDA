import os
import importlib
import sys
import inspect
from core.base_plugin import BasePhenoPlugin

class PluginManager:
    def __init__(self, plugin_dir="plugins"):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.plugin_dir = os.path.join(base_path, plugin_dir)
        self.plugins = {} # { "Corn Analysis": CornPluginClass, ... }
        self._discover_plugins()

    def _discover_plugins(self):
        if not os.path.exists(self.plugin_dir):
            os.makedirs(self.plugin_dir)
            
        project_root = os.path.dirname(self.plugin_dir)

        for filename in os.listdir(self.plugin_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3]
                try:
                    module = importlib.import_module(f"plugins.{module_name}")
                    
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and issubclass(obj, BasePhenoPlugin) and obj is not BasePhenoPlugin:
                            display_name = name.replace("Plugin", " Analysis")
                            self.plugins[display_name] = obj
                except Exception as e:
                    print(f"Failed to load plugin {filename}: {e}")

    def get_available_plugins(self):
        return list(self.plugins.keys())

    def get_plugin_class(self, name):
        return self.plugins.get(name)