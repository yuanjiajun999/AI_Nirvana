from src.utils.logger import setup_logger


def main():
    # 设置日志记录器
    logger = setup_logger("ai_nirvana", "ai_nirvana.log")

    # 记录不同级别的日志
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    print("Log messages have been written to 'ai_nirvana.log'")
    print("Check the log file to see the recorded messages.")


if __name__ == "__main__":
    main()
