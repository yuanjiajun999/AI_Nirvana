import unittest
from unittest.mock import Mock, patch
from src.core.ai_assistant import AIAssistant


class TestAIAssistant(unittest.TestCase):
    def setUp(self):
        self.assistant = AIAssistant()

    @patch("src.core.ai_assistant.LanguageModel")
    def test_generate_response(self, mock_language_model):
        mock_language_model.return_value.generate_response.return_value = (
            "Test response"
        )
        response = self.assistant.generate_response("Test prompt")
        self.assertEqual(response, "Test response")

    def test_summarize(self):
        with patch.object(self.assistant, "generate_response") as mock_generate:
            mock_generate.return_value = "Summary"
            summary = self.assistant.summarize("Long text")
            self.assertEqual(summary, "Summary")

    def test_analyze_sentiment(self):
        with patch.object(self.assistant, "generate_response") as mock_generate:
            mock_generate.return_value = (
                '{"positive": 0.8, "neutral": 0.15, "negative": 0.05}'
            )
            sentiment = self.assistant.analyze_sentiment("Test text")
            self.assertEqual(
                sentiment, {"positive": 0.8, "neutral": 0.15, "negative": 0.05}
            )


if __name__ == "__main__":
    unittest.main()
