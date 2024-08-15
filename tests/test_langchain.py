import unittest
from unittest.mock import patch, MagicMock
from src.core.langchain import LangChainAgent, get_response, create_chat_model
from openai import OpenAIError
from langchain_core.runnables import RunnableSequence

class TestLangChain(unittest.TestCase):

    def setUp(self):
        self.mock_llm = MagicMock()
        self.agent = LangChainAgent(llm=self.mock_llm)

    def test_get_response(self):
        mock_summarize_chain = MagicMock()
        mock_summarize_chain.invoke.return_value = "这是摘要"
        mock_answer_chain = MagicMock()
        mock_answer_chain.invoke.return_value = "这是回答"
    
        result = get_response("总结这段文字", summarize_chain=mock_summarize_chain, answer_chain=mock_answer_chain)
        self.assertEqual(result, "这是回答")  # 因为 "总结" 不在关键词列表中
        mock_answer_chain.invoke.assert_called_once_with({"question": "总结这段文字"})
    
        result = get_response("Summarize this text", summarize_chain=mock_summarize_chain, answer_chain=mock_answer_chain)
        self.assertEqual(result, "这是摘要")
        mock_summarize_chain.invoke.assert_called_once_with({"question": "Summarize this text"})
    
        result = get_response("回答这个问题", summarize_chain=mock_summarize_chain, answer_chain=mock_answer_chain)
        self.assertEqual(result, "这是回答")
        self.assertEqual(mock_answer_chain.invoke.call_count, 2)

    def test_run_qa_task(self):
        self.mock_llm.return_value = "这是问答任务的回答"
        result = self.agent.run_qa_task("这是一个测试问题")
        self.assertEqual(result, "这是问答任务的回答")

    def test_run_summarization_task(self):
        self.mock_llm.return_value = "这是文本摘要"
        result = self.agent.run_summarization_task("这是一段需要摘要的长文本")
        self.assertEqual(result, "这是文本摘要")

    def test_run_generation_task(self):
        self.mock_llm.return_value = MagicMock(content="这是生成的文本")
        result = self.agent.run_generation_task("生成一些文本")
        self.assertEqual(result, "这是生成的文本")

    def test_analyze_sentiment(self):
        self.mock_llm.return_value = "积极"
        result = self.agent.analyze_sentiment("这是一个很棒的日子！")
        self.assertEqual(result, "积极")

    def test_extract_keywords(self):
        self.mock_llm.return_value = "人工智能, 机器学习, 深度学习"
        result = self.agent.extract_keywords("人工智能和机器学习是当前热门的技术领域，其中深度学习尤为重要。")
        self.assertEqual(result, "人工智能, 机器学习, 深度学习")

    def test_get_response_edge_cases(self):
        mock_summarize_chain = MagicMock()
        mock_answer_chain = MagicMock()
        mock_answer_chain.invoke.return_value = "这是回答"
        
        # 测试空字符串输入
        result = get_response("", summarize_chain=mock_summarize_chain, answer_chain=mock_answer_chain)
        self.assertEqual(result, "这是回答")
        
        # 测试非英文输入
        result = get_response("总结这段文字", summarize_chain=mock_summarize_chain, answer_chain=mock_answer_chain)
        self.assertEqual(result, "这是回答")

    def test_get_response_api_error(self):
        mock_summarize_chain = MagicMock()
        mock_summarize_chain.invoke.side_effect = OpenAIError("API Error")
        mock_answer_chain = MagicMock()
        
        result = get_response("Summarize this", summarize_chain=mock_summarize_chain, answer_chain=mock_answer_chain)
        self.assertEqual(result, "抱歉，处理您的请求时出现了错误。")

    @patch.object(LangChainAgent, '_run_chain')
    def test_run_qa_task_error(self, mock_run_chain):
        mock_run_chain.side_effect = OpenAIError("API Error")
        result = self.agent.run_qa_task("这是一个测试问题")
        self.assertEqual(result, "Sorry, an error occurred while processing your request.")

    def test_run_generation_task_string_response(self):
        self.agent.llm = MagicMock()
        self.agent.llm.return_value = "This is a string response"
        result = self.agent.run_generation_task("Generate some text")
        self.assertEqual(result, "This is a string response")

    def test_run_generation_task_object_response(self):
        self.agent.llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is an object response"
        self.agent.llm.return_value = mock_response
        result = self.agent.run_generation_task("Generate some text")
        self.assertEqual(result, "This is an object response")

    def test_run_generation_task_other_response(self):
        self.agent.llm = MagicMock()
        self.agent.llm.return_value = 12345  # 一个既不是字符串也不是带有content属性的对象
        result = self.agent.run_generation_task("Generate some text")
        self.assertEqual(result, "12345")

    def test_run_generation_task_exception(self):
        self.agent.llm = MagicMock(side_effect=Exception("Test exception"))
        result = self.agent.run_generation_task("Generate some text")
        self.assertEqual(result, "Sorry, unable to generate text.")

    @patch.object(LangChainAgent, '_run_chain')
    def test_run_summarization_task_error(self, mock_run_chain):
        mock_run_chain.side_effect = OpenAIError("API Error")
        result = self.agent.run_summarization_task("This is a test text to summarize")
        self.assertEqual(result, "Sorry, an error occurred while summarizing the text.")

    def test_run_chain_error(self):
        agent = LangChainAgent()
        mock_chain = MagicMock(spec=RunnableSequence)
        mock_chain.invoke.side_effect = OpenAIError("API Error")
        
        result = agent._run_chain(mock_chain, {"input": "test"})
        self.assertEqual(result, "Sorry, an error occurred while processing your request.")    
if __name__ == '__main__':
    unittest.main()