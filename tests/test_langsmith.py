import unittest

from src.core.langsmith import LangSmith


class TestLangSmith(unittest.TestCase):
    def setUp(self):
        self.smith = LangSmith()

    def test_code_generation(self):

        prompt = (
            "Generate a simple Python function to calculate the factorial of anumber."
        )
        code = self.smith.generate_code(prompt)
        self.assertIsNotNone(code)
        self.assertIn("def", code)
        self.assertIn("factorial", code)

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
        self.assertIn("def", refactored_code)
        self.assertIn("factorial", refactored_code)

    def test_translation(self):
        text = "Hello, how are you?"
        translated_text = self.smith.translate_text(text, target_lang="fr")
        self.assertIsNotNone(translated_text)
        self.assertNotEqual(text, translated_text)

    def test_generate_code_with_complex_requirement(self):
        prompt = "Generate a Python function to implement quicksort algorithm"
        code = self.smith.generate_code(prompt)
        self.assertIn("def quicksort", code)
        self.assertIn("return", code)

    def test_refactor_code_with_bad_practices(self):
        bad_code = """
        def f(x):
            if x == 0:
                return 0
            else:
                return x + f(x - 1)
        """
        refactored_code = self.smith.refactor_code(bad_code)
        self.assertNotIn("if x == 0:", refactored_code)
        self.assertIn("def", refactored_code)

    def test_translate_text_with_idiomatic_expression(self):
        text = "It's raining cats and dogs"
        translated_text = self.smith.translate_text(text, target_lang="fr")
        self.assertNotIn("chats", translated_text.lower())
        self.assertNotIn("chiens", translated_text.lower())


if __name__ == "__main__":
    unittest.main()
