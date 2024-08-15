import unittest
from unittest.mock import Mock, patch
from langchain_core.runnables import Runnable
from src.core.langgraph import LangGraph, APIConfig
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class MockRunnable(Runnable):
    def invoke(self, input):
        return "Mocked response"

    def batch(self, inputs):
        return ["Mocked response" for _ in inputs]

    def stream(self, input):
        yield "Mocked response"

    async def ainvoke(self, input):
        return "Mocked response"

    async def abatch(self, inputs):
        return ["Mocked response" for _ in inputs]

    async def astream(self, input):
        yield "Mocked response"

    def create_reasoning_chain(self):
        return Mock()

    def create_inference_chain(self):
        return Mock()

class TestLangGraph(unittest.TestCase):

    @patch('src.core.langgraph.create_react_agent')
    @patch('src.core.langgraph.AgentExecutor')
    @patch('src.core.langgraph.ChatOpenAI')
    @patch('src.core.langgraph.OpenAIEmbeddings')
    @patch('src.core.langgraph.FAISS')
    def setUp(self, mock_faiss, mock_embeddings, mock_chat_openai, mock_agent_executor, mock_create_react_agent):
        mock_runnable = MockRunnable()
        mock_chat_openai.return_value = mock_runnable
        mock_create_react_agent.return_value = Mock()
        mock_agent_executor.return_value = Mock()
        self.lang_graph = LangGraph()

    def test_initialization(self):
        self.assertIsNotNone(self.lang_graph.llm)
        self.assertIsNotNone(self.lang_graph.graph)
        self.assertIsNotNone(self.lang_graph.embeddings)
        self.assertIsNotNone(self.lang_graph.vector_store)
        self.assertIsNotNone(self.lang_graph.memory)

    @patch('src.core.langgraph.LangGraph._cached_run')
    def test_retrieve_knowledge(self, mock_cached_run):
        mock_cached_run.return_value = "Test knowledge"
        result = self.lang_graph.retrieve_knowledge("test query")
        self.assertEqual(result, "Test knowledge")
        mock_cached_run.assert_called_once_with("test query")

    @patch('src.core.langgraph.LangGraph.reasoning_chain')
    def test_reason(self, mock_reasoning_chain):
        mock_reasoning_chain.run.return_value = "Valid reasoning"
        result = self.lang_graph.reason("premise", "conclusion")
        self.assertEqual(result, "Valid reasoning")
        mock_reasoning_chain.run.assert_called_once_with(premise="premise", conclusion="conclusion")

    @patch('src.core.langgraph.LangGraph.inference_chain')
    def test_infer_commonsense(self, mock_inference_chain):
        mock_inference_chain.run.return_value = "Inference result"
        result = self.lang_graph.infer_commonsense("context")
        self.assertEqual(result, "Inference result")
        mock_inference_chain.run.assert_called_once_with(context="context")

    @patch('src.core.langgraph.LangGraph.entity_extraction_chain')
    def test_extract_entities(self, mock_entity_extraction_chain):
        mock_entity_extraction_chain.invoke.return_value = "Entity1, Entity2, Entity3"
        result = self.lang_graph.extract_entities("test text")
        self.assertEqual(result, ["Entity1", "Entity2", "Entity3"])
        mock_entity_extraction_chain.invoke.assert_called_once_with({"text": "test text"})

    def test_add_entity(self):
        self.lang_graph.add_entity("TestEntity", {"property": "value"})
        self.assertIn("TestEntity", self.lang_graph.graph.get_networkx_graph().nodes())
        self.assertEqual(self.lang_graph.get_entity_info("TestEntity"), {"property": "value"})

    def test_add_relationship(self):
        self.lang_graph.add_entity("Entity1", {})
        self.lang_graph.add_entity("Entity2", {})
        self.lang_graph.add_relationship("Entity1", "Entity2", "relatedTo")
        self.assertIn("Entity2", list(self.lang_graph.graph.get_networkx_graph().neighbors("Entity1")))

    def test_get_entity_info(self):
        self.lang_graph.add_entity("TestEntity", {"property": "value"})
        info = self.lang_graph.get_entity_info("TestEntity")
        self.assertEqual(info, {"property": "value"})

    def test_get_related_entities(self):
        self.lang_graph.add_entity("Entity1", {})
        self.lang_graph.add_entity("Entity2", {})
        self.lang_graph.add_relationship("Entity1", "Entity2", "relatedTo")
        related = self.lang_graph.get_related_entities("Entity1")
        self.assertEqual(related, ["Entity2"])

    @patch('src.core.langgraph.LangGraph.vector_store')
    def test_semantic_search(self, mock_vector_store):
        mock_vector_store.similarity_search_with_score.return_value = [("Result1", 0.8), ("Result2", 0.6)]
        result = self.lang_graph.semantic_search("query", k=2)
        self.assertEqual(result, [("Result1", 0.8), ("Result2", 0.6)])
        mock_vector_store.similarity_search_with_score.assert_called_once_with("query", k=2)

    @patch('src.core.langgraph.LangGraph.agent')
    def test_run_agent(self, mock_agent):
        mock_agent.run.return_value = "Agent response"
        result = self.lang_graph.run_agent("query")
        self.assertEqual(result, "Agent response")
        mock_agent.run.assert_called_once_with("query")

    def test_get_graph_summary(self):
        self.lang_graph.add_entity("Entity1", {})
        self.lang_graph.add_entity("Entity2", {})
        self.lang_graph.add_relationship("Entity1", "Entity2", "relatedTo")
        summary = self.lang_graph.get_graph_summary()
        self.assertEqual(summary["num_nodes"], 2)
        self.assertEqual(summary["num_edges"], 1)
        self.assertIn("density", summary)
        self.assertIn("connected_components", summary)

    @patch('networkx.write_graphml')
    @patch('networkx.write_gexf')
    def test_export_graph(self, mock_write_gexf, mock_write_graphml):
        result_graphml = self.lang_graph.export_graph("graphml")
        self.assertEqual(result_graphml, "Graph exported as graph.graphml")
        mock_write_graphml.assert_called_once()

        result_gexf = self.lang_graph.export_graph("gexf")
        self.assertEqual(result_gexf, "Graph exported as graph.gexf")
        mock_write_gexf.assert_called_once()

        result_unsupported = self.lang_graph.export_graph("unsupported")
        self.assertEqual(result_unsupported, "Unsupported format")

    def test_update_entity(self):
        self.lang_graph.add_entity("TestEntity", {"property": "value"})
        result = self.lang_graph.update_entity("TestEntity", {"new_property": "new_value"})
        self.assertEqual(result, "Entity 'TestEntity' updated successfully.")
        info = self.lang_graph.get_entity_info("TestEntity")
        self.assertEqual(info, {"property": "value", "new_property": "new_value"})

    def test_delete_entity(self):
        self.lang_graph.add_entity("TestEntity", {"property": "value"})
        result = self.lang_graph.delete_entity("TestEntity")
        self.assertEqual(result, "Entity 'TestEntity' deleted from the graph.")
        self.assertNotIn("TestEntity", self.lang_graph.get_all_entities())

    def test_get_all_entities(self):
        self.lang_graph.add_entity("Entity1", {})
        self.lang_graph.add_entity("Entity2", {})
        entities = self.lang_graph.get_all_entities()
        self.assertIn("Entity1", entities)
        self.assertIn("Entity2", entities)

    def test_get_all_relationships(self):
        self.lang_graph.add_entity("Entity1", {})
        self.lang_graph.add_entity("Entity2", {})
        self.lang_graph.add_relationship("Entity1", "Entity2", "relatedTo")
        relationships = self.lang_graph.get_all_relationships()
        self.assertIn(("Entity1", "Entity2", "relatedTo"), relationships)

if __name__ == '__main__':
    unittest.main()