import unittest
from src.core.code_executor import CodeExecutor


class TestCodeExecutor(unittest.TestCase):
    def setUp(self):
        self.executor = CodeExecutor()

    def test_execute_code(self):
        code = "print('Hello, World!')"
        result, error = self.executor.execute_code(code, "python")
        self.assertEqual(result.strip(), "Hello, World!")
        self.assertEqual(error, "")

    def test_validate_code(self):
        safe_code = "print('Safe code')"
        unsafe_code = "import os; os.system('rm -rf /')"
        self.assertTrue(self.executor.validate_code(safe_code))
        self.assertFalse(self.executor.validate_code(unsafe_code))

    def test_get_supported_languages(self):
        languages = self.executor.get_supported_languages()
        self.assertIn("python", languages)


if __name__ == "__main__":
    unittest.main()
