import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from src.core.langchain import LangChainAgent, get_response

class TestLangChain(unittest.TestCase):
    def setUp(self):
        self.agent = LangChainAgent()

    def test_qa_task(self):
        query = "What is the capital of France?"
        result = self.agent.run_qa_task(query)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_generation_task(self):
        prompt = "Once upon a time, there was a..."
        generated_text = self.agent.run_generation_task(prompt)
        self.assertIsInstance(generated_text, str)
        self.assertTrue(len(generated_text) > 0)

    def test_chat_completion(self):
        response = get_response("Hello, World!")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_qa_task_with_get_response(self):
        response = get_response("What is the capital of France?")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        self.assertIn("paris", response.lower())

    def test_summarization_task_with_get_response(self):
        long_text = "This is a long text that needs to be summarized. It contains multiple sentences and ideas. The main point is about text summarization."
        response = get_response(f"Summarize this text: {long_text}")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        self.assertIn("summariz", response.lower())  # Check if the response mentions summarization
        self.assertLess(len(response.split()), len(long_text.split()) * 1.5 + 5)  # Allow 5 extra words

if __name__ == '__main__':
    unittest.main()