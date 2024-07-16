import unittest

from src.core.langgraph import LangGraph


class TestLangGraph(unittest.TestCase):
    def setUp(self):
        self.graph = LangGraph()

    def test_knowledge_retrieval(self):
        query = "Who is the president of the United States?"
        result = self.graph.retrieve_knowledge(query)
        self.assertIsNotNone(result)

    def test_reasoning(self):
        premise = "All humans are mortal. Socrates is a human."
        conclusion = "Socrates is mortal."
        is_valid = self.graph.reason(premise, conclusion)
        self.assertTrue(is_valid)

    def test_commonsense_inference(self):
        context = "John went to the kitchen and opened the refrigerator."
        inference = self.graph.infer_commonsense(context)
        self.assertIsNotNone(inference)