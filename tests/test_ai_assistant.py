import unittest
from unittest.mock import Mock, patch
import json
from src.core.ai_assistant import AIAssistant
from src.utils.exceptions import InputValidationError, ModelError, AIAssistantException

class TestAIAssistant(unittest.TestCase):
    def setUp(self):
        self.assistant = AIAssistant()

    @patch('src.core.multimodal.MultimodalInterface.process')
    @patch('src.core.security.SecurityManager.is_safe')
    @patch('src.core.knowledge_base.KnowledgeBase.retrieve')
    @patch('src.core.reasoning.ReasoningEngine.reason')
    @patch('src.core.language_model.LanguageModel.generate_response')
    def test_process_input(self, mock_generate, mock_reason, mock_retrieve, mock_is_safe, mock_process):
        mock_process.return_value = "processed input"
        mock_is_safe.return_value = True
        mock_retrieve.return_value = "relevant knowledge"
        mock_reason.return_value = "reasoning result"
        mock_generate.return_value = "generated response"

        response = self.assistant.process_input("test input")
        self.assertEqual(response, "generated response")

        mock_is_safe.return_value = False
        with self.assertRaises(InputValidationError):
            self.assistant.process_input("unsafe input")

    @patch('src.core.security.SecurityManager.is_safe_code')
    @patch('src.core.language_model.LanguageModel.generate_response')
    def test_generate_response(self, mock_generate, mock_is_safe):
        mock_is_safe.return_value = True
        mock_generate.return_value = "generated response"

        response = self.assistant.generate_response("test prompt")
        self.assertEqual(response, "generated response")

        mock_is_safe.return_value = False
        with self.assertRaises(InputValidationError):
            self.assistant.generate_response("unsafe prompt")

    @patch('src.core.security.SecurityManager.is_safe_code')
    @patch('src.core.language_model.LanguageModel.generate_response')
    def test_summarize(self, mock_generate, mock_is_safe):
        mock_is_safe.return_value = True
        mock_generate.return_value = "summary"

        summary = self.assistant.summarize("test text")
        self.assertEqual(summary, "summary")

        mock_is_safe.return_value = False
        with self.assertRaises(InputValidationError):
            self.assistant.summarize("unsafe text")

    @patch('src.core.security.SecurityManager.is_safe_code')
    @patch('src.core.language_model.LanguageModel.generate_response')
    def test_analyze_sentiment(self, mock_generate, mock_is_safe):
        mock_is_safe.return_value = True
        mock_generate.return_value = '{"positive": 0.8, "neutral": 0.15, "negative": 0.05}'

        sentiment = self.assistant.analyze_sentiment("test text")
        self.assertEqual(sentiment, {"positive": 0.8, "neutral": 0.15, "negative": 0.05})

        mock_is_safe.return_value = False
        with self.assertRaises(InputValidationError):
            self.assistant.analyze_sentiment("unsafe text")

    def test_clear_context(self):
        self.assistant.context = [{"role": "user", "content": "test"}]
        self.assistant.clear_context()
        self.assertEqual(self.assistant.context, [])

    @patch('src.core.language_model.LanguageModel.change_default_model')
    def test_change_model(self, mock_change_model):
        self.assistant.change_model("new_model")
        mock_change_model.assert_called_once_with("new_model")

        mock_change_model.side_effect = Exception("Model change failed")
        with self.assertRaises(ModelError):
            self.assistant.change_model("failed_model")

    @patch('src.core.language_model.LanguageModel.get_available_models')
    def test_get_available_models(self, mock_get_models):
        mock_get_models.return_value = ["model1", "model2"]
        models = self.assistant.get_available_models()
        self.assertEqual(models, ["model1", "model2"])

        mock_get_models.side_effect = Exception("Failed to get models")
        with self.assertRaises(ModelError):
            self.assistant.get_available_models()

    @patch('src.core.security.SecurityManager.encrypt_sensitive_data')
    def test_encrypt_sensitive_data(self, mock_encrypt):
        mock_encrypt.return_value = "encrypted_data"
        encrypted = self.assistant.encrypt_sensitive_data("sensitive_data")
        self.assertEqual(encrypted, "encrypted_data")

        mock_encrypt.side_effect = Exception("Encryption failed")
        with self.assertRaises(AIAssistantException):
            self.assistant.encrypt_sensitive_data("sensitive_data")

    @patch('src.core.security.SecurityManager.decrypt_sensitive_data')
    def test_decrypt_sensitive_data(self, mock_decrypt):
        mock_decrypt.return_value = "decrypted_data"
        decrypted = self.assistant.decrypt_sensitive_data("encrypted_data")
        self.assertEqual(decrypted, "decrypted_data")

        mock_decrypt.side_effect = Exception("Decryption failed")
        with self.assertRaises(AIAssistantException):
            self.assistant.decrypt_sensitive_data("encrypted_data")

    @patch('src.core.security.SecurityManager.execute_in_sandbox')
    def test_execute_code(self, mock_execute):
        mock_execute.return_value = ("result", None)
        result, error = self.assistant.execute_code("print('Hello')", "python")
        self.assertEqual(result, "result")
        self.assertIsNone(error)

        mock_execute.side_effect = Exception("Code execution failed")
        with self.assertRaises(AIAssistantException):
            self.assistant.execute_code("print('Hello')", "python")

    @patch('src.core.language_model.LanguageModel.generate_response')
    def test_plan_task(self, mock_generate):
        mock_generate.return_value = "Task plan"
        plan = self.assistant.plan_task("Do something")
        self.assertEqual(plan, "Task plan")

        mock_generate.side_effect = Exception("Plan generation failed")
        with self.assertRaises(ModelError):
            self.assistant.plan_task("Do something")

    @patch('src.core.reasoning.ReasoningEngine.reinforcement_learning')
    def test_reinforcement_learning_action(self, mock_rl):
        mock_rl.return_value = "action"
        action = self.assistant.reinforcement_learning_action("state")
        self.assertEqual(action, "action")

        mock_rl.side_effect = Exception("RL failed")
        with self.assertRaises(ModelError):
            self.assistant.reinforcement_learning_action("state")

    @patch('src.core.reasoning.ReasoningEngine.active_learning')
    def test_active_learning_sample(self, mock_al):
        mock_al.return_value = "sample"
        sample = self.assistant.active_learning_sample(["data1", "data2"])
        self.assertEqual(sample, "sample")

        mock_al.side_effect = Exception("AL failed")
        with self.assertRaises(ModelError):
            self.assistant.active_learning_sample(["data1", "data2"])

if __name__ == '__main__':
    unittest.main()