import unittest
from unittest.mock import patch, MagicMock
from src.core.code_executor import CodeExecutor, PythonExecutor
import subprocess
from src.utils.security import SecurityManager 

class TestCodeExecutor(unittest.TestCase):
    def setUp(self):
        self.code_executor = CodeExecutor()

    def test_initialization(self):
        self.assertIsInstance(self.code_executor.security_manager, SecurityManager)
        self.assertIn("python", self.code_executor.language_executors)

    @patch('src.core.code_executor.SecurityManager.is_safe_code')
    def test_execute_code_unsafe(self, mock_is_safe_code):
        mock_is_safe_code.return_value = False
        result, error = self.code_executor.execute_code("print('Hello')", "python")
        self.assertEqual(result, "")
        self.assertIn("security concerns", error)

    @patch('src.core.code_executor.SecurityManager.is_safe_code')
    @patch('src.core.code_executor.PythonExecutor.execute')
    def test_execute_code_safe(self, mock_execute, mock_is_safe_code):
        mock_is_safe_code.return_value = True
        mock_execute.return_value = ("Hello\n", "")
        result, error = self.code_executor.execute_code("print('Hello')", "python")
        self.assertEqual(result, "Hello\n")
        self.assertEqual(error, "")

    def test_execute_code_unsupported_language(self):
        result, error = self.code_executor.execute_code("print('Hello')", "java")
        self.assertEqual(result, "")
        self.assertIn("not supported", error)

    @patch('src.core.code_executor.SecurityManager.is_safe_code')
    def test_validate_code(self, mock_is_safe_code):
        mock_is_safe_code.return_value = True
        self.assertTrue(self.code_executor.validate_code("print('Hello')"))
        
        mock_is_safe_code.return_value = False
        self.assertFalse(self.code_executor.validate_code("import os; os.system('rm -rf /')"))

    def test_get_supported_languages(self):
        languages = self.code_executor.get_supported_languages()
        self.assertIn("python", languages)

class TestPythonExecutor(unittest.TestCase):
    def setUp(self):
        self.python_executor = PythonExecutor()

    @patch('subprocess.run')
    def test_execute_success(self, mock_run):
        mock_run.return_value = MagicMock(stdout="Hello\n", stderr="")
        result, error = self.python_executor.execute("print('Hello')")
        self.assertEqual(result, "Hello\n")
        self.assertEqual(error, "")

    @patch('subprocess.run')
    def test_execute_error(self, mock_run):
        mock_run.side_effect = Exception("Test error")
        result, error = self.python_executor.execute("print('Hello')")
        self.assertEqual(result, "")
        self.assertIn("Test error", error)

    @patch('subprocess.run')
    def test_execute_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="python", timeout=5)
        result, error = self.python_executor.execute("while True: pass")
        self.assertEqual(result, "")
        self.assertIn("timed out", error)

if __name__ == '__main__':
    unittest.main()