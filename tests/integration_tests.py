import unittest
from src.core.langchain import LangChainAgent
from src.core.langgraph import LangGraph
from src.core.langsmith import LangSmith

class IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.langchain_agent = LangChainAgent()
        self.langgraph = LangGraph()
        self.langsmith = LangSmith()

    def test_langchain_and_langgraph(self):
        langchain_response = self.langchain_agent.run_qa_task("Who invented the telephone?")
        langgraph_response = self.langgraph.retrieve_knowledge("Who invented the telephone?")
        
        self.assertIn("Bell", langchain_response)
        self.assertIn("Bell", langgraph_response['result'])

    def test_langgraph_and_langsmith(self):
        langgraph_response = self.langgraph.reason("All birds can fly", "Penguins can fly")
        langsmith_response = self.langsmith.generate_code("Write a function to check if a bird can fly")
        
        self.assertTrue(any(phrase in langgraph_response['result'].lower() for phrase in ["no", "not true", "incorrect"]))
        self.assertIn("def", langsmith_response)

    def test_langchain_and_langsmith(self):
        langchain_response = self.langchain_agent.run_summarization_task("AI is a branch of computer science.")
        langsmith_response = self.langsmith.translate_text(langchain_response, "Spanish")
        
        self.assertIn("AI", langchain_response)
        self.assertTrue(any(phrase in langsmith_response.lower() for phrase in ["inteligencia artificial", "ia"]))

if __name__ == '__main__':
    unittest.main()