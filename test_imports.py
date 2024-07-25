def test_imports():
    modules_to_test = [
        "absl",
        "aiohttp",
        "aiosignal",
        "requests",  # 新添加的包
        # ... 添加其他关键包
    ]

    for module in modules_to_test:
        try:
            __import__(module)
            print(f"Successfully imported {module}")
        except ImportError as e:
            print(f"Failed to import {module}: {e}")


if __name__ == "__main__":
    test_imports()
