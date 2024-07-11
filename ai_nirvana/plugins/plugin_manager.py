# 这个文件将在未来实现插件系统
class PluginManager:
    def __init__(self):
        self.plugins = {}

    def load_plugin(self, name, plugin):
        self.plugins[name] = plugin

    def unload_plugin(self, name):
        if name in self.plugins:
            del self.plugins[name]

    def get_plugin(self, name):
        return self.plugins.get(name)

    def execute_plugin(self, name, *args, **kwargs):
        plugin = self.get_plugin(name)
        if plugin:
            return plugin.execute(*args, **kwargs)
        else:
            return f"Plugin {name} not found"