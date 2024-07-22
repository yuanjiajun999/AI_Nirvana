import sys  
import os  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
import unittest  

from src.core.api_model import APIModel  
from src.core.local_model import LocalModel  
from src.dialogue_manager import DialogueManager  

class ConcreteAPIModel(APIModel):  
    def generate_response(self, prompt):  
        return "This is a test response"  
    
class TestModels(unittest.TestCase):  
    def test_local_model(self):  
        model = LocalModel()  
        response = model.generate_response("What is the capital of France?")  
        self.assertIsNotNone(response)  

    def test_api_model(self):  
        model = ConcreteAPIModel(api_key="your_api_key", api_url="https://api.example.com/generate")  
        response = model.generate_response("测试提示")  
        self.assertIsNotNone(response)  

    def test_dialogue_manager(self):  
        manager = DialogueManager(max_history=5)  
        manager.add_to_history("How are you?", "I'm doing well, thank you for asking.")  
        context = manager.get_dialogue_context()  
        self.assertIn("User: How are you?", context)  
        self.assertIn("Assistant: I'm doing well, thank you for asking.", context)  

if __name__ == '__main__':  
    unittest.main()  