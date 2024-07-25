import unittest
from src.core.generative_ai import GenerativeAI


class TestGenerativeAI(unittest.TestCase):
    def setUp(self):
        self.gen_ai = GenerativeAI("path/to/model")

    def test_generate_text(self):
        prompt = "Once upon a time"
        generated_text = self.gen_ai.generate_text(prompt, max_length=50)
        self.assertIsInstance(generated_text, list)
        self.assertTrue(all(isinstance(text, str) for text in generated_text))
        self.assertTrue(all(text.startswith(prompt) for text in generated_text))


if __name__ == "__main__":
    unittest.main()
