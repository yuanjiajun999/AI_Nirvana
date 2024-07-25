from src.utils.error_handler import (
    error_handler,
    logger,
    AIAssistantException,
    InputValidationError,
    ModelError,
)


@error_handler
def validate_input(text):
    if len(text) < 5:
        raise InputValidationError("Input text is too short")
    return text


@error_handler
def process_with_model(text):
    if "error" in text.lower():
        raise ModelError("Model encountered an error")
    return f"Processed: {text}"


def main():
    try:
        # 测试正常情况
        result = validate_input("Hello, world!")
        print("Validated input:", result)

        result = process_with_model("This is a test")
        print("Model output:", result)

        # 测试输入验证错误
        validate_input("Hi")
    except InputValidationError as e:
        print("Caught InputValidationError:", str(e))

    try:
        # 测试模型错误
        process_with_model("This will cause an error")
    except ModelError as e:
        print("Caught ModelError:", str(e))

    # 测试未预期的错误
    @error_handler
    def unexpected_error():
        raise ValueError("Unexpected error occurred")

    try:
        unexpected_error()
    except AIAssistantException as e:
        print("Caught AIAssistantException:", str(e))

    # 检查日志文件
    print("\nCheck the 'ai_assistant.log' file for logged errors.")


if __name__ == "__main__":
    main()
