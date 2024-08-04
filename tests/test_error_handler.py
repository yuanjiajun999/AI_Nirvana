import unittest  
import logging  
import os  
import tempfile  
from io import StringIO  
from src.utils.error_handler import (  
    AIAssistantException, InputValidationError, ModelError, ConfigurationError,  
    SecurityException, DataProcessingError, error_handler, log_function_call,  
    get_error_details, handle_exception, set_log_level, setup_logger  
)  

class TestErrorHandler(unittest.TestCase):  
    def setUp(self):  
        self.logger = logging.getLogger('ai_assistant')  
        self.logger.setLevel(logging.DEBUG)  
        self.log_capture = StringIO()  
        self.handler = logging.StreamHandler(self.log_capture)  
        self.logger.addHandler(self.handler)  

    def tearDown(self):  
        self.logger.removeHandler(self.handler)  
        self.log_capture.close()  

    def test_setup_logger(self):  
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.log') as temp_file:  
            test_log_file = temp_file.name  
        
        try:  
            test_logger = setup_logger("test_logger", test_log_file, logging.DEBUG)  
            
            self.assertEqual(test_logger.name, "test_logger")  
            self.assertEqual(test_logger.level, logging.DEBUG)  
            self.assertEqual(len(test_logger.handlers), 2)  # File handler and console handler  
            
            # Test logging  
            test_message = "Test log message"  
            test_logger.debug(test_message)  
            
            # Close all handlers to ensure file is released  
            for handler in test_logger.handlers:  
                handler.close()  
            
            # Check if message is in log file  
            with open(test_log_file, 'r') as f:  
                log_content = f.read()  
                self.assertIn(test_message, log_content)  
        finally:  
            # Clean up  
            try:  
                os.remove(test_log_file)  
            except PermissionError:  
                pass  # If we still can't delete, we'll let the OS clean it up later  

    def test_custom_exceptions(self):  
        with self.assertRaises(InputValidationError):  
            raise InputValidationError("Invalid input")  
        
        with self.assertRaises(ModelError):  
            raise ModelError("Model failed")  

        with self.assertRaises(ConfigurationError):  
            raise ConfigurationError("Invalid configuration")  

        with self.assertRaises(SecurityException):  
            raise SecurityException("Security breach")  

        with self.assertRaises(DataProcessingError):  
            raise DataProcessingError("Data processing failed")  

    def test_error_handler_decorator(self):  
        @error_handler  
        def faulty_function():  
            raise ValueError("Test error")  

        with self.assertRaises(AIAssistantException):  
            faulty_function()  

        log_output = self.log_capture.getvalue()  
        self.assertIn("Unexpected error in faulty_function", log_output)  

    def test_log_function_call_decorator(self):  
        @log_function_call  
        def sample_function():  
            return "Hello, World!"  

        sample_function()  
        log_output = self.log_capture.getvalue()  
        self.assertIn("Calling function: sample_function", log_output)  
        self.assertIn("Function sample_function completed", log_output)  

    def test_get_error_details(self):  
        try:  
            raise ValueError("Test error")  
        except ValueError as e:  
            details = get_error_details(e)  
            self.assertEqual(details['type'], 'ValueError')  
            self.assertEqual(details['message'], 'Test error')  
            self.assertIn('Traceback', details['traceback'])  

    def test_handle_exception(self):  
        try:  
            raise RuntimeError("Test runtime error")  
        except RuntimeError as e:  
            handle_exception(e)  
        
        log_output = self.log_capture.getvalue()  
        self.assertIn("Exception occurred: RuntimeError", log_output)  
        self.assertIn("Error message: Test runtime error", log_output)  

    def test_set_log_level(self):  
        original_level = self.logger.level  
        set_log_level(logging.WARNING)  
        self.assertEqual(self.logger.level, logging.WARNING)  
        set_log_level(logging.DEBUG)  
        self.assertEqual(self.logger.level, logging.DEBUG)  
        set_log_level(original_level)  # 恢复原始日志级别  

if __name__ == '__main__':  
    unittest.main()