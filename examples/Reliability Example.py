from src.core.reliability import setup_logging, handle_exception


@handle_exception
def risky_function(x):
    if x == 0:
        raise ValueError("Cannot divide by zero")
    return 10 / x


def main():
    # 设置日志
    setup_logging()

    # 测试异常处理装饰器
    print("Testing exception handling:")
    result = risky_function(2)
    print(f"Result for x=2: {result}")

    result = risky_function(0)
    print(f"Result for x=0: {result}")  # 这里会打印出错误消息

    print("\nCheck the log file for more details on the handled exception.")


if __name__ == "__main__":
    main()
