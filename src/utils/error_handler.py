import logging
from functools import wraps
from typing import Any, Callable


class AIAssistantException(Exception):
    """AI助手相关的自定义异常基类"""

    pass


class InputValidationError(AIAssistantException):
    """输入验证错误"""

    pass


class ModelError(AIAssistantException):
    """模型相关错误"""

    pass


class ConfigurationError(AIAssistantException):
    """配置相关错误"""

    pass


class SecurityException(Exception):
    pass


def error_handler(func):
    # 保留现有的 error_handler 函数实现
    pass


# 如果 logger 是在这个文件中定义的，保留它的定义


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    设置并返回一个logger

    Args:
        name (str): Logger的名称
        log_file (str): 日志文件的路径
        level (int): 日志级别

    Returns:
        logging.Logger: 配置好的logger对象
    """

    formatter = logging.Formatter(
        "% (asctime)s [ % (levelname)s] % (name)s:% (message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger("ai_assistant", "ai_assistant.log")


def error_handler(func: Callable) -> Callable:
    """
    装饰器: 处理函数执行过程中的错误，并记录日志

    Args:
        func (Callable): 被装饰的函数

    Returns:
        Callable: 装饰后的函数
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except InputValidationError as e:
            logger.error(f"Input validation error in {func.__name__}: {str(e)}")
            raise
        except ModelError as e:
            logger.error(f"Model error in {func.__name__}: {str(e)}")
            raise
        except ConfigurationError as e:
            logger.error(f"Configuration error in {func.__name__}: {str(e)}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in {func.__name__}: {str(e)}")
            raise AIAssistantException(f"An unexpected error occurred: {str(e)}")

    return wrapper
