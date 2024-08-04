import unittest
from unittest.mock import patch, MagicMock
from src.core.language_model import LanguageModel
from src.utils.error_handler import ModelError

class TestLanguageModel(unittest.TestCase):
    def setUp(self):
        self.model = LanguageModel()

    def test_generate_response(self):
        response = self.model.generate_response("Test prompt")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_get_available_models(self):
        models = self.model.get_available_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)

    def test_change_default_model(self):
        available_models = self.model.get_available_models()
        if available_models:
            new_model = available_models[0]
            self.model.change_default_model(new_model)
            self.assertEqual(self.model.default_model, new_model)

        with self.assertRaises(ModelError):
            self.model.change_default_model("non_existent_model")

    def test_get_model_info(self):
        info = self.model.get_model_info(self.model.default_model)
        self.assertIsInstance(info, dict)
        self.assertIn("id", info)
        self.assertIn("created", info)
        self.assertIn("owned_by", info)

    def test_analyze_sentiment(self):
        sentiment = self.model.analyze_sentiment("I love this product!")
        self.assertIsInstance(sentiment, dict)
        self.assertIn("positive", sentiment)
        self.assertIn("neutral", sentiment)
        self.assertIn("negative", sentiment)

    def test_summarize(self):
        summary = self.model.summarize("This is a long text that needs to be summarized. " * 10)
        self.assertIsInstance(summary, str)
        self.assertLess(len(summary), 500)  # Assuming max_length is 100 words

    def test_translate(self):
        translated = self.model.translate("Hello, world!", "French")
        self.assertIsInstance(translated, str)
        self.assertNotEqual(translated, "Hello, world!")

    def test_clear_context(self):
        try:
            self.model.clear_context()
        except Exception as e:
            self.fail(f"clear_context raised {type(e).__name__} unexpectedly!")

if __name__ == '__main__':
    unittest.main()