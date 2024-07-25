from src.config import Config


def main():
    # 初始化配置
    config = Config()

    # 获取配置值
    model_name = config.get("model")
    log_level = config.get("log_level")
    max_input_length = config.get("max_input_length")

    print("Current configuration:")
    print(f"Model: {model_name}")
    print(f"Log level: {log_level}")
    print(f"Max input length: {max_input_length}")

    # 设置新的配置值
    config.set("model", "gpt-4")
    config.set("max_input_length", 150)

    print("\nUpdated configuration:")
    print(f"Model: {config.get('model')}")
    print(f"Max input length: {config.get('max_input_length')}")

    # 获取预定义响应
    intro_response = config.get_predefined_response("introduce_yourself")
    print("\nPredefined response for 'introduce_yourself':")
    print(intro_response)

    # 验证配置
    is_valid = config.validate_config()
    print(f"\nIs the current configuration valid? {is_valid}")


if __name__ == "__main__":
    main()
