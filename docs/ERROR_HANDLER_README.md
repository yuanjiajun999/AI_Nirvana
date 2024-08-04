# Error Handler Module  

The Error Handler module provides a set of utilities for managing errors and logging in the AI Assistant project.  

## Features  

- Custom exception classes  
- Error handling decorator  
- Function call logging decorator  
- Logging setup utility  
- Exception handling utility  

## Usage  

### Setting up logging  

```python  
from src.utils.error_handler import setup_logger  
import logging  

logger = setup_logger("my_logger", "my_log_file.log", logging.DEBUG)  
Using the error handler decorator
from src.utils.error_handler import error_handler  

@error_handler  
def my_function():  
    # Your code here  
    pass  
Using the log function call decorator
from src.utils.error_handler import log_function_call  

@log_function_call  
def my_function():  
    # Your code here  
    pass  
Handling exceptions
from src.utils.error_handler import handle_exception  

try:  
    # Your code here  
    pass  
except Exception as e:  
    handle_exception(e)  
Custom exceptions
from src.utils.error_handler import InputValidationError, ModelError  

def my_function(x):  
    if not isinstance(x, int):  
        raise InputValidationError("Input must be an integer")  
    if x < 0:  
        raise ModelError("Input must be non-negative")  
    return x * 2  
Example
See example_error_handler_usage.py for a complete example of how to use the Error Handler module.

Testing
The test_error_handler.py file contains unit tests for all functionalities of the Error Handler module. Run these tests to ensure the module is working correctly:

pytest tests/test_error_handler.py  
Note
Make sure to import and use these error handling utilities throughout your project to maintain consistent error handling and logging practices.