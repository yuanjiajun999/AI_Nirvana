import unittest

from src.core.model_interface import ModelInterface


class DummyModel(ModelInterface):
    def load_model(self, model_path: str) -> None:
        pass

    def predict(self, input_data: Any) -> Any:
        return input_data

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


class TestModelInterface(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()

    def test_predict(self):
        input_data = [1, 2, 3]
        self.assertEqual(self.model.predict(input_data), input_data)

    def test_evaluate(self):
        result = self.model.evaluate(None, None)
        self.assertIn("accuracy", result)
        self.assertEqual(result["accuracy"], 0.9)

    def test_preprocess_data(self):
        raw_data = [4, 5, 6]
        self.assertEqual(self.model.preprocess_data(raw_data), raw_data)

    def test_get_model_info(self):
        info = self.model.get_model_info()
        self.assertIn("name", info)
        self.assertIn("version", info)

    def test_explain_prediction(self):
        input_data = [7, 8, 9]
        prediction = "test"
        explanation = self.model.explain_prediction(input_data, prediction)
        self.assertIn("test", explanation)
        self.assertIn(str(input_data), explanation)


if __name__ == "__main__":
    unittest.main()
