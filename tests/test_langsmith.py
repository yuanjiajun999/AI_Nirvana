import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from src.core.langsmith import LangSmith

class TestLangSmith(unittest.TestCase):
    def setUp(self):
        self.smith = LangSmith()

    def test_code_generation(self):
        prompt = "Generate a simple Python function to calculate the factorial of a number."
        code = self.smith.generate_code(prompt)
        self.assertIsNotNone(code)

    def test_refactoring(self):
        code = """
        def factorial(n):
            if n == 0:
                return 1
            else:
                return n * factorial(n - 1)
        """
        refactored_code = self.smith.refactor_code(code)
        self.assertNotEqual(code, refactored_code)

    def test_translation(self):
        text = "Hello, how are you?"
        translated_text = self.smith.translate_text(text, target_lang="fr")
        self.assertIsNotNone(translated_text)

if __name__ == '__main__':
    unittest.main()