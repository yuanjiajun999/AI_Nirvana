import unittest
from unittest.mock import Mock, patch
from src.core.enhanced_ai_assistant import EnhancedAIAssistant
from src.utils.exceptions import InputValidationError, ModelError
from unittest.mock import Mock, patch, call
from deep_translator import GoogleTranslator

class TestEnhancedAIAssistant(unittest.TestCase):

    def setUp(self):
        self.assistant = EnhancedAIAssistant()
        # Mock external dependencies
        self.assistant.ai_assistant = Mock()
        self.assistant.lang_chain_agent = Mock()
        self.assistant.translator = Mock()
        self.assistant.translator.translate = Mock(return_value="Translated text")
    
    def test_detect_language(self):
        with patch('langdetect.detect', return_value='en'):
            result = self.assistant.detect_language("Hello, world!")
            self.assertEqual(result, 'en')

    def test_translate(self):
        self.assistant.translator.translate = Mock(return_value="Bonjour, monde!")
        result = self.assistant.translate("Hello, world!", 'en', 'fr')
        self.assertEqual(result, "Bonjour, monde!")

    def test_process_input(self):
        self.assistant.detect_language = Mock(return_value='en')
        self.assistant.ai_assistant.process_input.return_value = "Processed response"
        result = self.assistant.process_input("Test input")
        self.assertEqual(result, "Processed response")

    def test_process_input_non_english(self):
        self.assistant.detect_language = Mock(return_value='fr')
        self.assistant.translate = Mock(side_effect=["Test input", "Réponse traitée"])
        self.assistant.ai_assistant.process_input.return_value = "Processed response"
        result = self.assistant.process_input("Entrée de test")
        self.assertEqual(result, "Réponse traitée")

    def test_generate_response(self):
        self.assistant.lang_chain_agent.run_generation_task.return_value = "Generated response"
        result = self.assistant.generate_response("Test prompt")
        self.assertEqual(result, "Generated response")

    def test_summarize(self):
        self.assistant.ai_assistant.summarize.return_value = "Summary"
        result = self.assistant.summarize("Long text to summarize")
        self.assertEqual(result, "Summary")

    def test_analyze_sentiment(self):
        self.assistant.ai_assistant.analyze_sentiment.return_value = {"positive": 0.8, "negative": 0.2}
        result = self.assistant.analyze_sentiment("Great day!")
        self.assertEqual(result, {"positive": 0.8, "negative": 0.2})

    def test_execute_code(self):
        self.assistant.ai_assistant.execute_code.return_value = ("Output", None)
        result = self.assistant.execute_code("print('Hello')", "python")
        self.assertEqual(result, ("Output", None))

    def test_plan_task(self):
        self.assistant.ai_assistant.plan_task.return_value = "Task plan"
        result = self.assistant.plan_task("Organize meeting")
        self.assertEqual(result, "Task plan")

    def test_extract_keywords(self):
        self.assistant.lang_chain_agent.extract_keywords.return_value = ["AI", "testing"]
        result = self.assistant.extract_keywords("AI testing is important")
        self.assertEqual(result, ["AI", "testing"])

    def test_change_model(self):
        self.assistant.change_model("new_model")
        self.assistant.ai_assistant.change_model.assert_called_once_with("new_model")
        self.assistant.lang_chain_agent.change_model.assert_called_once_with("new_model")

    def test_get_available_models(self):
        self.assistant.ai_assistant.get_available_models.return_value = ["model1", "model2"]
        self.assistant.lang_chain_agent.get_available_models.return_value = ["model2", "model3"]
        result = self.assistant.get_available_models()
        self.assertEqual(set(result), {"model1", "model2", "model3"})

    def test_encrypt_sensitive_data(self):
        self.assistant.ai_assistant.encrypt_sensitive_data.return_value = "encrypted_data"
        result = self.assistant.encrypt_sensitive_data("sensitive_info")
        self.assertEqual(result, "encrypted_data")

    def test_decrypt_sensitive_data(self):
        self.assistant.ai_assistant.decrypt_sensitive_data.return_value = "decrypted_data"
        result = self.assistant.decrypt_sensitive_data("encrypted_info")
        self.assertEqual(result, "decrypted_data")

    def test_reinforcement_learning_action(self):
        self.assistant.ai_assistant.reinforcement_learning_action.return_value = "action"
        result = self.assistant.reinforcement_learning_action("state")
        self.assertEqual(result, "action")

    def test_active_learning_sample(self):
        self.assistant.ai_assistant.active_learning_sample.return_value = "sample"
        result = self.assistant.active_learning_sample(["data1", "data2"])
        self.assertEqual(result, "sample")

    def test_clear_context(self):
        self.assistant.clear_context()
        self.assistant.ai_assistant.clear_context.assert_called_once()

    def test_get_dialogue_context(self):
        self.assistant.ai_assistant.context = [{"role": "user", "content": "Hello"}]
        result = self.assistant.get_dialogue_context()
        self.assertEqual(result, [{"role": "user", "content": "Hello"}])

    def test_error_handling(self):
        self.assistant.ai_assistant.process_input.side_effect = InputValidationError("Invalid input")
        with self.assertRaises(InputValidationError):
            self.assistant.process_input("Invalid input")

    def test_translate_different_languages(self):
        original_translator = self.assistant.translator
        try:
            # 创建一个模拟的 GoogleTranslator 对象
            mock_translator = Mock(spec=GoogleTranslator)
            self.assistant.translator = mock_translator

            # 调用 translate 方法
            self.assistant.translate("Hello", "en", "fr")

            # 验证 source 和 target 是否被正确设置
            self.assertEqual(mock_translator.source, "en")
            self.assertEqual(mock_translator.target, "fr")

            # 验证 translate 方法是否被调用
            mock_translator.translate.assert_called_once_with("Hello")

            # 再次调用 translate 方法，但这次 source 和 target 语言相同
            self.assistant.translate("Hello", "en", "en")

            # 验证在源语言和目标语言相同时，translate 方法没有被调用第二次
            mock_translator.translate.assert_called_once()
        finally:
            # 恢复原始的 translator
            self.assistant.translator = original_translator

    def test_generate_response_non_english(self):
        self.assistant.translate = Mock(side_effect=["Translated prompt", "Translated response"])
        self.assistant.lang_chain_agent.run_generation_task = Mock(return_value="Generated response")
        result = self.assistant.generate_response("Prompt", language="fr")
        self.assertEqual(result, "Translated response")
        self.assistant.translate.assert_has_calls([
            call("Prompt", "fr", "en"),
            call("Generated response", "en", "fr")
        ])

    def test_summarize_non_english(self):
        self.assistant.translate = Mock(side_effect=["Translated text", "Translated summary"])
        self.assistant.ai_assistant.summarize = Mock(return_value="Summary")
        result = self.assistant.summarize("Text", language="fr")
        self.assertEqual(result, "Translated summary")
        self.assistant.translate.assert_has_calls([
            call("Text", "fr", "en"),
            call("Summary", "en", "fr")
        ])

    def test_analyze_sentiment_non_english(self):
        self.assistant.translate = Mock(return_value="Translated text")
        self.assistant.ai_assistant.analyze_sentiment = Mock(return_value={"sentiment": "positive"})
        result = self.assistant.analyze_sentiment("Text", language="fr")
        self.assertEqual(result, {"sentiment": "positive"})
        self.assistant.translate.assert_called_once_with("Text", "fr", "en")

    def test_plan_task_non_english(self):
        self.assistant.translate = Mock(side_effect=["Translated description", "Translated plan"])
        self.assistant.ai_assistant.plan_task = Mock(return_value="Plan")
        result = self.assistant.plan_task("Task description", language="fr")
        self.assertEqual(result, "Translated plan")
        self.assistant.translate.assert_has_calls([
            call("Task description", "fr", "en"),
            call("Plan", "en", "fr")
        ])

    def test_extract_keywords_non_english(self):
        self.assistant.translate = Mock(side_effect=["Translated text", "Mot clé 1", "Mot clé 2"])
        self.assistant.lang_chain_agent.extract_keywords = Mock(return_value=["Keyword 1", "Keyword 2"])
        result = self.assistant.extract_keywords("Text", language="fr")
        self.assertEqual(result, ["Mot clé 1", "Mot clé 2"])
        self.assistant.translate.assert_has_calls([
            call("Text", "fr", "en"),
            call("Keyword 1", "en", "fr"),
            call("Keyword 2", "en", "fr")
        ])

if __name__ == '__main__':
    unittest.main()