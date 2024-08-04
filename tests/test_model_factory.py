from typing import Any, Dict
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

print(sys.path)  # This will print the Python path

# Rest of your import statements and test code

import unittest
from src.core.model_factory import ModelFactory
from src.core.model_interface import ModelInterface

class DummyModel(ModelInterface):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def load_model(self, model_path: str) -> None:
        pass

    def predict(self, input_data: Any) -> Any:
        return f"{self.param1}-{self.param2}"

    def train(self, training_data: Any, labels: Any) -> None:
        pass

    def evaluate(self, test_data: Any, test_labels: Any) -> Dict[str, float]:
        return {"accuracy": 0.9}

    def save_model(self, model_path: str) -> None:
        pass

    def preprocess_data(self, raw_data: Any) -> Any:
        return raw_data

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": "DummyModel", "version": "1.0"}

    def fine_tune(self, fine_tuning_data: Any, fine_tuning_labels: Any) -> None:
        pass

    def explain_prediction(self, input_data: Any, prediction: Any) -> str:
        return f"Prediction {prediction} was made for input {input_data}"

class TestModelFactory(unittest.TestCase):
    def setUp(self):
        ModelFactory._models = {}  # Reset registered models before each test

    def test_register_model(self):
        ModelFactory.register_model("dummy", DummyModel)
        self.assertIn("dummy", ModelFactory._models)

    def test_create_model(self):
        ModelFactory.register_model("dummy", DummyModel)
        model = ModelFactory.create_model("dummy", param1="test1", param2="test2")
        self.assertIsInstance(model, DummyModel)
        self.assertEqual(model.predict(None), "test1-test2")

    def test_get_available_models(self):
        ModelFactory.register_model("dummy1", DummyModel)
        ModelFactory.register_model("dummy2", DummyModel)
        available_models = ModelFactory.get_available_models()
        self.assertIn("dummy1", available_models)
        self.assertIn("dummy2", available_models)

    def test_create_unregistered_model(self):
        with self.assertRaises(ValueError):
            ModelFactory.create_model("unregistered_model")

if __name__ == '__main__':
    unittest.main()