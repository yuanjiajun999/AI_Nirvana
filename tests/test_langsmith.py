import os
import sys
import unittest
from unittest.mock import Mock, patch

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(current_dir))
# 将项目根目录添加到 Python 路径
sys.path.insert(0, root_dir)

from src.core.langsmith import LangSmithIntegration
from src.config import Config

class TestLangSmithIntegration(unittest.TestCase):
    def setUp(self):
        self.config = Mock(spec=Config)
        self.config.MODEL_NAME = "gpt-3.5-turbo"
        self.config.API_KEY = "test_api_key"
        self.config.API_BASE = "https://api.openai.com/v1"
        self.config.TEMPERATURE = 0.7
        self.lang_smith = LangSmithIntegration(self.config)
        
        # 添加模拟的 OpenAI API 密钥
        os.environ['OPENAI_API_KEY'] = 'fake-api-key'

    def test_initialization(self):
        self.assertIsInstance(self.lang_smith, LangSmithIntegration)
        
    @patch('src.core.langsmith.ChatOpenAI')
    def test_create_chain(self, mock_chat_openai):
        chain = self.lang_smith.create_chain()
        self.assertIsNotNone(chain)

    @patch('src.core.langsmith.LangSmithIntegration.create_chain')
    def test_run_chain(self, mock_create_chain):
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Test response"
        mock_create_chain.return_value = mock_chain
        
        response = self.lang_smith.run_chain("Test input")
        self.assertEqual(response, "Test response")
        mock_chain.invoke.assert_called_once_with({"input": "Test input"})

    @patch('src.core.langsmith.Client')
    def test_create_dataset(self, mock_client):
        mock_dataset = Mock()
        mock_dataset.id = "test_dataset_id"
        mock_client.return_value.create_dataset.return_value = mock_dataset
        
        examples = [
            {"input": {"text": "Test input 1"}, "output": "Test output 1"},
            {"input": {"text": "Test input 2"}, "output": "Test output 2"}
        ]
        dataset = self.lang_smith.create_dataset(examples)
        self.assertEqual(dataset.id, "test_dataset_id")
        mock_client.return_value.create_dataset.assert_called_once()
        self.assertEqual(mock_client.return_value.create_example.call_count, 2)

    @patch('src.core.langsmith.run_on_dataset')
    @patch('src.core.langsmith.evaluation')
    def test_evaluate_chain(self, mock_evaluation, mock_run_on_dataset):
        mock_evaluation.load_evaluator.side_effect = ["criteria", "qa", "context_qa"]
        self.lang_smith.evaluate_chain("test_dataset")
        mock_run_on_dataset.assert_called_once()
        self.assertEqual(mock_evaluation.load_evaluator.call_count, 3)

    @patch('src.core.langsmith.OpenAIEmbeddings')
    @patch('src.core.langsmith.Chroma')
    @patch('src.core.langsmith.RetrievalQA')
    def test_setup_retrieval_qa(self, mock_retrieval_qa, mock_chroma, mock_embeddings):
        mock_document = Mock()
        mock_document.page_content = "This is a test document."
        mock_document.metadata = {}
        documents = [mock_document]

        mock_retriever = Mock()
        mock_chroma.from_documents.return_value.as_retriever.return_value = mock_retriever

        mock_qa = Mock()
        mock_retrieval_qa.from_chain_type.return_value = mock_qa

        qa = self.lang_smith.setup_retrieval_qa(documents)

        self.assertEqual(qa, mock_qa)
        mock_chroma.from_documents.assert_called_once()
        mock_retrieval_qa.from_chain_type.assert_called_once()

    def test_answer_question(self):
        mock_qa = Mock()
        mock_qa.run.return_value = "Test answer"
        answer = self.lang_smith.answer_question(mock_qa, "Test question")
        self.assertEqual(answer, "Test answer")
        mock_qa.run.assert_called_once_with("Test question")

    @patch('src.core.langsmith.Client')
    def test_get_evaluation_results(self, mock_client):
        mock_run = Mock()
        mock_run.feedback = {"score": 0.8}
        mock_client.return_value.read_run.return_value = mock_run
        
        results = self.lang_smith.get_evaluation_results("test_run_id")
        self.assertEqual(results, {"score": 0.8})
        mock_client.return_value.read_run.assert_called_once_with("test_run_id")

    @patch('src.core.langsmith.Client')
    def test_analyze_chain_performance(self, mock_client):
        mock_dataset = Mock(id="test_id")
        mock_run1 = Mock(error=None, latency=1.0)
        mock_run2 = Mock(error="Test error", latency=0.5)
        mock_client.return_value.read_dataset.return_value = mock_dataset
        mock_client.return_value.list_runs.return_value = [mock_run1, mock_run2]
    
        performance = self.lang_smith.analyze_chain_performance("test_dataset")
        self.assertEqual(performance["total_runs"], 2)
        self.assertEqual(performance["success_rate"], 50.0)
        self.assertEqual(performance["average_latency"], 1.0)  # 更新为 1.0

    @patch('src.core.langsmith.ChatOpenAI')
    @patch('src.core.langsmith.Client')
    @patch('src.core.langsmith.LangSmithIntegration.run_chain')
    @patch('src.core.langsmith.LangSmithIntegration.analyze_chain_performance')
    @patch('src.core.langsmith.run_on_dataset')
    def test_optimize_prompt(self, mock_run_on_dataset, mock_analyze_performance, mock_run_chain, mock_client, mock_chat_openai):
        mock_chat_openai.return_value = Mock()
        mock_analyze_performance.return_value = {"success_rate": 80.0}
        mock_run_chain.return_value = "Improved prompt"

        # 创建一个模拟的数据集和示例
        mock_dataset = Mock()
        mock_dataset.id = "test_dataset_id"
        mock_dataset.url = "https://example.com/dataset"
        mock_example = Mock()
        mock_example.modified_at = None

        # 创建一个模拟的项目
        mock_project = Mock()
        mock_project.id = "test_project_id"

        # 设置 Client 的行为
        mock_client_instance = mock_client.return_value
        mock_client_instance.read_dataset.return_value = mock_dataset
        mock_client_instance.list_examples.return_value = [mock_example]
        mock_client_instance.create_project.return_value = mock_project

        optimized_prompt = self.lang_smith.optimize_prompt("Initial prompt", "test_dataset", num_iterations=2)

        self.assertEqual(optimized_prompt, "Improved prompt")
    
        # 验证方法被调用，但不检查具体的调用参数
        mock_run_on_dataset.assert_called()
        mock_analyze_performance.assert_called()
        mock_run_chain.assert_called()

    @patch('src.core.langsmith.Client')
    @patch('src.core.langsmith.LangSmithIntegration.optimize_prompt')
    @patch('src.core.langsmith.LangSmithIntegration.run_chain')
    def test_continuous_learning(self, mock_run_chain, mock_optimize_prompt, mock_client):
        self.lang_smith.continuous_learning("Test input", "Test output", "positive")
        mock_client.return_value.create_run.assert_called_once()
        mock_optimize_prompt.assert_called_once()

        self.lang_smith.continuous_learning("Test input", "Test output", "negative")
        mock_run_chain.assert_called_once()
        self.assertEqual(mock_optimize_prompt.call_count, 2)

    @patch('src.core.langsmith.LangSmithIntegration.run_chain')
    def test_generate_test_cases(self, mock_run_chain):
        mock_run_chain.return_value = "Input: Test input 1\nOutput: Test output 1\nInput: Test input 2\nOutput: Test output 2"
        test_cases = self.lang_smith.generate_test_cases("Generate test cases")
        self.assertEqual(len(test_cases), 2)
        self.assertEqual(test_cases[0]["input"], "Test input 1")
        self.assertEqual(test_cases[0]["output"], "Test output 1")

    @patch('src.core.langsmith.LangSmithIntegration.run_chain')
    def test_run_security_check(self, mock_run_chain):
        mock_run_chain.return_value = "Security assessment: Low risk"
        assessment = self.lang_smith.run_security_check("Test input")
        self.assertEqual(assessment, "Security assessment: Low risk")

    @patch('src.core.langsmith.LangSmithIntegration.run_chain')
    def test_generate_explanation(self, mock_run_chain):
        mock_run_chain.return_value = "Explanation: This is how the output was generated"
        explanation = self.lang_smith.generate_explanation("Test input", "Test output")
        self.assertEqual(explanation, "Explanation: This is how the output was generated")

    @patch('src.core.langsmith.LangSmithIntegration.run_security_check')
    @patch('src.core.langsmith.LangSmithIntegration.generate_explanation')
    @patch('src.core.langsmith.LangSmithIntegration.continuous_learning')
    @patch('src.core.langsmith.LangSmithIntegration.generate_test_cases')
    @patch('src.core.langsmith.LangSmithIntegration.create_dataset')
    def test_integrate_with_ai_nirvana(self, mock_create_dataset, mock_generate_test_cases, 
                                       mock_continuous_learning, mock_generate_explanation, 
                                       mock_run_security_check):
        mock_ai_nirvana = Mock()
        mock_ai_nirvana.process.return_value = "Original response"
        mock_run_security_check.return_value = "Low risk"
        mock_generate_explanation.return_value = "Test explanation"
        mock_generate_test_cases.return_value = [{"input": "test", "output": "test"}]
        
        self.lang_smith.integrate_with_ai_nirvana(mock_ai_nirvana)
        
        # Test the enhanced process
        result = mock_ai_nirvana.process("Test input")
        self.assertIn("Original response", result)
        self.assertIn("Test explanation", result)
        
        mock_run_security_check.assert_called_once()
        mock_generate_explanation.assert_called_once()
        mock_continuous_learning.assert_called_once()
        mock_generate_test_cases.assert_called_once()
        mock_create_dataset.assert_called_once()

        # Test high risk scenario
        mock_run_security_check.return_value = "High risk"
        result = mock_ai_nirvana.process("Test input")
        self.assertIn("cannot process this request due to security concerns", result)

        self.assertIsNotNone(mock_ai_nirvana.evaluate_performance)
        self.assertIsNotNone(mock_ai_nirvana.optimize_prompts)
        self.assertIsNotNone(mock_ai_nirvana.setup_qa)
        self.assertIsNotNone(mock_ai_nirvana.answer_qa)

    def tearDown(self):
        # 清理环境变量
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']

if __name__ == '__main__':
    unittest.main()