import unittest
from src.core.knowledge_base import KnowledgeBase


class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase()

    def test_add_and_get_knowledge(self):
        self.kb.add_knowledge("test_key", "test_value")
        self.assertEqual(self.kb.get_knowledge("test_key"), "test_value")

    def test_update_knowledge(self):
        self.kb.add_knowledge("test_key", "initial_value")
        self.kb.update_knowledge("test_key", "updated_value")
        self.assertEqual(self.kb.get_knowledge("test_key"), "updated_value")

    def test_delete_knowledge(self):
        self.kb.add_knowledge("test_key", "test_value")
        self.kb.delete_knowledge("test_key")
        with self.assertRaises(KeyError):
            self.kb.get_knowledge("test_key")

    def test_list_all_knowledge(self):
        self.kb.add_knowledge("key1", "value1")
        self.kb.add_knowledge("key2", "value2")
        all_knowledge = self.kb.list_all_knowledge()
        self.assertEqual(all_knowledge, {"key1": "value1", "key2": "value2"})


if __name__ == "__main__":
    unittest.main()
