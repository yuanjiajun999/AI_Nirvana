import unittest
from unittest.mock import patch, Mock
from src.core.api_model import ConcreteAPIModel

class TestAPIModel(unittest.TestCase):
    def setUp(self):
        self.api_model = ConcreteAPIModel(
            api_key="test_key", api_url="https://test.com"
        )

    @patch("requests.post")
    def test_generate(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {"choices": [{"text": "Generated text"}]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.api_model.generate("Test prompt")
        self.assertEqual(result, "Generated text")

    @patch("requests.post")
    def test_generate_error(self, mock_post):
        mock_post.side_effect = Exception("API Error")

        result = self.api_model.generate("Test prompt")
        self.assertTrue(result.startswith("Error: Unable to get response"))

    def test_generate_response(self):
        response = self.api_model.generate_response("Test prompt")
        self.assertEqual(response, "This is a test response")

if __name__ == "__main__":
    unittest.main()