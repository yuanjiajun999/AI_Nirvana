import importlib
import os
from typing import Any, Dict


class PluginManager:
    def __init__(self, plugin_dir: str = "src/plugins"):
        self.plugin_dir = plugin_dir
        self.plugins: Dict[str, Any] = {}

    def load_plugins(self):
        for filename in os.listdir(self.plugin_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3]
                module = importlib.import_module(f"src.plugins.{module_name}")
                if hasattr(module, 'register_plugin'):
                    plugin = module.register_plugin()
                    self.plugins[module_name] = plugin

    def get_plugin(self, name: str) -> Any:
        return self.plugins.get(name)

    def execute_plugin(self, name: str, *args, **kwargs) -> Any:
        plugin = self.get_plugin(name)
        if plugin and hasattr(plugin, 'execute'):
            return plugin.execute(*args, **kwargs)
        return None

    def unload_plugin(self, name: str):
        if name in self.plugins:
            del self.plugins[name]