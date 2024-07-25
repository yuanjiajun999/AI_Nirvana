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
        result = self.graph.reason(premise, conclusion)
        self.assertTrue(result["result"].lower().startswith("yes"))

    def test_commonsense_inference(self):
        context = "John went to the kitchen and opened the refrigerator."
        inference = self.graph.infer_commonsense(context)
        self.assertIsNotNone(inference)

    def test_reason_with_invalid_premise(self):
        invalid_premise = "This is not a valid logical premise."
        conclusion = "Therefore, the sky is green."
        result = self.graph.reason(invalid_premise, conclusion)
        self.assertFalse(result["result"].lower().startswith("yes"))

    def test_infer_commonsense_with_ambiguous_context(self):
        ambiguous_context = "The light was red."
        inference = self.graph.infer_commonsense(ambiguous_context)
        self.assertIn("traffic", inference.lower())


if __name__ == "__main__":
    unittest.main()
