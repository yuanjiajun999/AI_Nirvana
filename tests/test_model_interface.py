import unittest
from typing import Any, Dict, List
from src.core.model_interface import ModelInterface

class DummyModel(ModelInterface):
    def __init__(self):
        self.model = None
        self.model_path = ""

    def load_model(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = "DummyLoadedModel"  # Simple representation of a loaded model
        print(f"Model loaded from {model_path}")

    def predict(self, input_data: Any) -> Any:
        if not self.model:
            raise ValueError("Model not loaded. Call load_model first.")
        return [x * 2 for x in input_data] if isinstance(input_data, List) else input_data

    def train(self, training_data: Any, labels: Any) -> None:
        if not self.model:
            raise ValueError("Model not loaded. Call load_model first.")
        print(f"Training model with {len(training_data)} samples")

    def evaluate(self, test_data: Any, test_labels: Any) -> Dict[str, float]:
        if not self.model:
            raise ValueError("Model not loaded. Call load_model first.")
        return {"accuracy": 0.9, "precision": 0.85, "recall": 0.88}

    def save_model(self, model_path: str) -> None:
        if not self.model:
            raise ValueError("No model to save. Load or train a model first.")
        self.model_path = model_path
        print(f"Model saved to {model_path}")

    def preprocess_data(self, raw_data: Any) -> Any:
        return [x + 1 for x in raw_data] if isinstance(raw_data, List) else raw_data

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "DummyModel",
            "version": "1.1",
            "type": "Classification",
            "input_shape": [None, 10],
            "output_shape": [None, 1],
            "is_loaded": self.model is not None
        }

    def fine_tune(self, fine_tuning_data: Any, fine_tuning_labels: Any) -> None:
        if not self.model:
            raise ValueError("Model not loaded. Call load_model first.")
        print(f"Fine-tuning model with {len(fine_tuning_data)} samples")

    def explain_prediction(self, input_data: Any, prediction: Any) -> str:
        if not self.model:
            raise ValueError("Model not loaded. Call load_model first.")
        return f"Prediction {prediction} was made for input {input_data} based on feature importance"

class TestModelInterface(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()

    def test_load_model(self):
        self.model.load_model("dummy_path")
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.model_path, "dummy_path")

    def test_predict(self):
        self.model.load_model("dummy_path")
        input_data = [1, 2, 3]
        self.assertEqual(self.model.predict(input_data), [2, 4, 6])

    def test_predict_error(self):
        with self.assertRaises(ValueError):
            self.model.predict([1, 2, 3])

    def test_evaluate(self):
        self.model.load_model("dummy_path")
        result = self.model.evaluate(None, None)
        self.assertIn("accuracy", result)
        self.assertIn("precision", result)
        self.assertIn("recall", result)

    def test_preprocess_data(self):
        raw_data = [4, 5, 6]
        self.assertEqual(self.model.preprocess_data(raw_data), [5, 6, 7])

    def test_get_model_info(self):
        info = self.model.get_model_info()
        self.assertIn("name", info)
        self.assertIn("version", info)
        self.assertIn("type", info)
        self.assertIn("input_shape", info)
        self.assertIn("output_shape", info)
        self.assertIn("is_loaded", info)
        self.assertFalse(info["is_loaded"])

        self.model.load_model("dummy_path")
        info = self.model.get_model_info()
        self.assertTrue(info["is_loaded"])

    def test_explain_prediction(self):
        self.model.load_model("dummy_path")
        input_data = [7, 8, 9]
        prediction = "test"
        explanation = self.model.explain_prediction(input_data, prediction)
        self.assertIn("test", explanation)
        self.assertIn(str(input_data), explanation)
        self.assertIn("feature importance", explanation)

    def test_load_and_save_model(self):
        self.model.load_model("dummy_path")
        self.model.save_model("new_path")
        self.assertEqual(self.model.model_path, "new_path")

    def test_train_and_fine_tune(self):
        self.model.load_model("dummy_path")
        self.model.train([1, 2, 3], [0, 1, 1])
        self.model.fine_tune([4, 5], [1, 0])

    def test_operations_without_loading(self):
        with self.assertRaises(ValueError):
            self.model.predict([1, 2, 3])
        with self.assertRaises(ValueError):
            self.model.train([1, 2, 3], [0, 1, 1])
        with self.assertRaises(ValueError):
            self.model.evaluate(None, None)
        with self.assertRaises(ValueError):
            self.model.fine_tune([1, 2], [0, 1])
        with self.assertRaises(ValueError):
            self.model.explain_prediction([1], "test")
        with self.assertRaises(ValueError):
            self.model.save_model("path")

if __name__ == '__main__':
    unittest.main()