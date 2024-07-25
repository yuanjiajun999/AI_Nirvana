import unittest

from src.core.langchain import LangChainAgent
from src.core.langgraph import LangGraph
from src.core.langsmith import LangSmith


class TestLangChainAgent(unittest.TestCase):
    def setUp(self):
        self.agent = LangChainAgent()

    def test_qa_task(self):
        result = self.agent.run_qa_task("What is AI?")
        self.assertIn("AI", result)

    def test_summarization_task(self):

        result = self.agent.run_summarization_task(
            "AI is the simulation of humanintelligence in machines."
        )
        self.assertIn("human intelligence", result.lower())


class TestLangGraph(unittest.TestCase):
    def setUp(self):
        self.graph = LangGraph()

    def test_retrieve_knowledge(self):

        result = self.graph.retrieve_knowledge("Tell me about quantum computing.")
        self.assertIn("quantum computing", result["result"].lower())

    def test_reasoning(self):
        result = self.graph.reason("All humans are mortal.", "Socrates is mortal.")
        self.assertIn("yes", result["result"].lower())


class TestLangSmith(unittest.TestCase):
    def setUp(self):
        self.langsmith = LangSmith()

    def test_generate_code(self):
        result = self.langsmith.generate_code(
            "Create a Python function to add two numbers."
        )
        self.assertIn("def", result)

    def test_translate_text(self):
        result = self.langsmith.translate_text("Hello, world!", "Spanish")
        self.assertIn("Hola", result)


if __name__ == "__main__":
    unittest.main()
