import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.local_model import LocalModel
from src.core.api_model import APIModel, ConcreteAPIModel
from src.dialogue_manager import DialogueManager

class TestModels(unittest.TestCase):
    def test_local_model(self):
        model = LocalModel()
        response = model.generate_response("自我介绍")
        self.assertIsNotNone(response)
        self.assertIn("AI Nirvana智能助手", response)

        summary = model.summarize("这是一个很长的文本" * 20)
        self.assertTrue(len(summary) <= 103)  # 100 characters + "..."

        info = model.get_model_info()
        self.assertEqual(info["name"], "Local BERT Model")
        self.assertEqual(info["type"], "local")

    def test_api_model(self):
        model = ConcreteAPIModel(
            api_key="your_api_key", api_url="https://api.example.com/generate"
        )
        response = model.generate_response("测试提示")
        self.assertEqual(response, "This is a test response")

        # Test the base APIModel class
        api_model = APIModel("test_key", "https://test.com")
        with unittest.mock.patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = {"choices": [{"text": "API response"}]}
            mock_post.return_value.raise_for_status.return_value = None
            response = api_model.generate("Test prompt")
            self.assertEqual(response, "API response")

    def test_dialogue_manager(self):
        manager = DialogueManager(max_history=5)
        manager.add_to_history("How are you?", "I'm doing well, thank you for asking.")
        manager.add_to_history("What's the weather like?", "I'm sorry, I don't have real-time weather information.")
        
        context = manager.get_dialogue_context()
        self.assertIn("User: How are you?", context)
        self.assertIn("Assistant: I'm doing well, thank you for asking.", context)
        self.assertIn("User: What's the weather like?", context)
        
        clear_message = manager.clear_history()
        self.assertEqual(clear_message, "对话历史已清除。")
        self.assertEqual(manager.get_dialogue_context(), "")

if __name__ == "__main__":
    unittest.main()