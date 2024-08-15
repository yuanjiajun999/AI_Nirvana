import logging  
import traceback  
import functools  
from functools import wraps  
from typing import Any, Callable, Dict, Optional  

logger = logging.getLogger(__name__)

class AIAssistantException(Exception):  
    """AI助手相关的自定义异常基类"""  
    def __init__(self, message: str, error_code: Optional[int] = None):  
        super().__init__(message)  
        self.error_code = error_code  

class InputValidationError(AIAssistantException):  
    """输入验证错误"""  
    pass  

class ModelError(AIAssistantException):  
    """模型相关错误"""  
    pass  

class ConfigurationError(AIAssistantException):  
    """配置相关错误"""  
    pass  

class SecurityException(AIAssistantException):  
    """安全相关错误"""  
    pass  

class DataProcessingError(AIAssistantException):  
    """数据处理错误"""  
    pass  

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
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')  

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

def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            # 返回一个错误字典，而不是引发异常
            return {"error": str(e), "positive": 0.0, "neutral": 0.0, "negative": 0.0}
    return wrapper

def log_function_call(func: Callable) -> Callable:  
    """  
    装饰器: 记录函数调用的日志  

    Args:  
        func (Callable): 被装饰的函数  

    Returns:  
        Callable: 装饰后的函数  
    """  
    @wraps(func)  
    def wrapper(*args: Any, **kwargs: Any) -> Any:  
        logger.info(f"Calling function: {func.__name__}")  
        result = func(*args, **kwargs)  
        logger.info(f"Function {func.__name__} completed")  
        return result  
    return wrapper  

def get_error_details(e: Exception) -> Dict[str, Any]:  
    """  
    获取异常的详细信息  

    Args:  
        e (Exception): 异常对象  

    Returns:  
        Dict[str, Any]: 包含异常详细信息的字典  
    """  
    return {  
        "type": type(e).__name__,  
        "message": str(e),  
        "traceback": traceback.format_exc()  
    }  

def handle_exception(e: Exception) -> None:  
    """  
    处理异常并记录日志  

    Args:  
        e (Exception): 需要处理的异常  
    """  
    error_details = get_error_details(e)  
    logger.error(f"Exception occurred: {error_details['type']}")  
    logger.error(f"Error message: {error_details['message']}")  
    logger.debug(f"Traceback:\n{error_details['traceback']}")  

def set_log_level(level: int) -> None:  
    """  
    设置日志级别  

    Args:  
        level (int): 日志级别 (e.g., logging.INFO, logging.DEBUG)  
    """  
    logger.setLevel(level)  
    for handler in logger.handlers:  
        handler.setLevel(level)