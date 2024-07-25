from src.plugins.plugin_manager import PluginManager


def main():
    plugin_manager = PluginManager()

    # 加载插件
    plugin_manager.load_plugins()
    print("Plugins loaded.")

    # 获取并执行翻译插件
    translator_plugin = plugin_manager.get_plugin("translator")
    if translator_plugin:
        translation = plugin_manager.execute_plugin(
            "translator", text="Hello, world!", source_lang="en", target_lang="fr"
        )
        print(f"Translation: {translation}")
    else:
        print("Translator plugin not found.")

    # 尝试获取不存在的插件
    nonexistent_plugin = plugin_manager.get_plugin("nonexistent")
    if nonexistent_plugin is None:
        print("Nonexistent plugin correctly returns None.")

    # 卸载插件
    plugin_manager.unload_plugin("translator")
    print("Translator plugin unloaded.")

    # 验证插件已被卸载
    if plugin_manager.get_plugin("translator") is None:
        print("Translator plugin successfully unloaded.")


if __name__ == "__main__":
    main()
