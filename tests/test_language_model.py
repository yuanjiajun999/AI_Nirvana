import unittest
from unittest.mock import patch
from src.core.language_model import LanguageModel


class TestLanguageModel(unittest.TestCase):
    def setUp(self):
        self.model = LanguageModel()

    @patch("src.core.language_model.OpenAI")
    def test_generate_response(self, mock_openai):
        mock_openai.return_value.chat.completions.create.return_value.choices[
            0
        ].message.content = "Test response"
        response = self.model.generate_response("Test prompt")
        self.assertEqual(response, "Test response")

    @patch("src.core.language_model.OpenAI")
    def test_get_available_models(self, mock_openai):
        mock_openai.return_value.models.list.return_value.data = [
            type("obj", (object,), {"id": "model1"}),
            type("obj", (object,), {"id": "model2"}),
        ]
        models = self.model.get_available_models()
        self.assertEqual(models, ["model1", "model2"])

    def test_change_default_model(self):
        with patch.object(self.model, "get_available_models", return_value=["gpt-4"]):
            self.model.change_default_model("gpt-4")
            self.assertEqual(self.model.default_model, "gpt-4")


if __name__ == "__main__":
    unittest.main()
