import unittest
from src.core.knowledge_base import KnowledgeBase

class TestKnowledgeBase(unittest.TestCase):

    def setUp(self):
        self.kb = KnowledgeBase()

    def test_add_knowledge(self):
        self.kb.add_knowledge("AI", "Artificial Intelligence")
        self.assertIn("AI", self.kb.list_all_knowledge())

    def test_add_invalid_key(self):
        with self.assertRaises(ValueError):
            self.kb.add_knowledge("", "Invalid Key")

    def test_get_knowledge(self):
        self.kb.add_knowledge("AI", "Artificial Intelligence")
        self.assertEqual(self.kb.get_knowledge("AI"), "Artificial Intelligence")

    def test_get_non_existing_knowledge(self):
        with self.assertRaises(KeyError):
            self.kb.get_knowledge("NonExist")

    def test_update_knowledge(self):
        self.kb.add_knowledge("AI", "Artificial Intelligence")
        self.kb.update_knowledge("AI", "Advanced AI")
        self.assertEqual(self.kb.get_knowledge("AI"), "Advanced AI")

    def test_update_non_existing_knowledge(self):
        with self.assertRaises(KeyError):
            self.kb.update_knowledge("NonExist", "Value")

    def test_delete_knowledge(self):
        self.kb.add_knowledge("AI", "Artificial Intelligence")
        self.kb.delete_knowledge("AI")
        with self.assertRaises(KeyError):
            self.kb.get_knowledge("AI")

    def test_delete_non_existing_knowledge(self):
        with self.assertRaises(KeyError):
            self.kb.delete_knowledge("NonExist")

    def test_list_all_knowledge(self):
        self.kb.add_knowledge("AI", "Artificial Intelligence")
        self.kb.add_knowledge("ML", "Machine Learning")
        knowledge = self.kb.list_all_knowledge()
        self.assertEqual(len(knowledge), 2)
        self.assertIn("AI", knowledge)
        self.assertIn("ML", knowledge)

    def test_retrieve(self):
        self.kb.add_knowledge("AI", "Artificial Intelligence")
        self.kb.add_knowledge("ML", "Machine Learning")
        results = self.kb.retrieve("Intelligence")
        self.assertIn("Artificial Intelligence", results)

    def test_retrieve_no_match(self):
        self.kb.add_knowledge("AI", "Artificial Intelligence")
        results = self.kb.retrieve("Biology")
        self.assertEqual(len(results), 0)

if __name__ == "__main__":
    unittest.main()
