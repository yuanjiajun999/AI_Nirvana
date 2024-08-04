import logging  
from src.utils.error_handler import (  
    setup_logger, error_handler, log_function_call,   
    InputValidationError, ModelError, handle_exception  
)  

# 设置日志  
logger = setup_logger("example_logger", "example.log", logging.DEBUG)  

@error_handler  
@log_function_call  
def example_function(x):  
    if not isinstance(x, int):  
        raise InputValidationError("Input must be an integer")  
    if x < 0:  
        raise ModelError("Input must be non-negative")  
    return x * 2  

def main():  
    try:  
        result = example_function(5)  
        logger.info(f"Result: {result}")  

        result = example_function("not an integer")  
    except Exception as e:  
        handle_exception(e)  

    try:  
        result = example_function(-1)  
    except Exception as e:  
        handle_exception(e)  

if __name__ == "__main__":  
    main()