import unittest
from unittest.mock import patch

from src.core.langchain import LangChainAgent, get_response


class TestLangChain(unittest.TestCase):
    def setUp(self):
        self.agent = LangChainAgent()

    @patch("src.core.langchain.ChatOpenAI")
    def test_qa_task(self, mock_chat):

        mock_chat.return_value.invoke.return_value.content = (
            "Paris is the capital ofFrance."
        )
        result = self.agent.run_qa_task("What is the capital of France?")
        self.assertIn("Paris", result)

    @patch("src.core.langchain.ChatOpenAI")
    def test_summarization_task(self, mock_chat):
        mock_chat.return_value.invoke.return_value.content = "This is a summary."
        text = "This is a long text that needs to be summarized."
        result = self.agent.run_summarization_task(text)
        self.assertEqual(result, "This is a summary.")

    @patch("src.core.langchain.ChatOpenAI")
    def test_generation_task(self, mock_chat):
        mock_chat.return_value.invoke.return_value.content = "Once upon a time..."
        result = self.agent.run_generation_task("Write a short story")
        self.assertTrue(result.startswith("Once upon a time"))

    @patch("src.core.langchain.ChatOpenAI")
    def test_get_response(self, mock_chat):
        mock_chat.return_value.invoke.return_value.content = (
            "Hello, how can I help you?"
        )
        response = get_response("Hi")
        self.assertEqual(response, "Hello, how can I help you?")

    def test_chat_completion_caching(self):
        # Test that repeated calls with the same input return cached results
        response1 = get_response("Hello, World!")
        response2 = get_response("Hello, World!")
        self.assertEqual(response1, response2)


if __name__ == "__main__":
    unittest.main()
