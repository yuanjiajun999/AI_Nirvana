import unittest

from src.core.langchain import LangChainAgent


class TestLangChain(unittest.TestCase):
    def setUp(self):
        self.agent = LangChainAgent()

    def test_qa_task(self):
        query = "What is the capital of France?"
        result = self.agent.run_qa_task(query)
        self.assertIsNotNone(result)

    def test_summarization_task(self):
        text = "This is a long text that needs to be summarized."
        summary = self.agent.run_summarization_task(text)
        self.assertIsNotNone(summary)

    def test_generation_task(self):
        prompt = "Once upon a time, there was a..."
        generated_text = self.agent.run_generation_task(prompt)
        self.assertIsNotNone(generated_text)