from src.core import ModelFactory, ModelInterface
from typing import Any, Dict

class ExampleModel(ModelInterface):
    def __init__(self):
        self.model = None

    def load_model(self, model_path: str) -> None:
        print(f"Loading model from {model_path}")
        self.model = "Loaded Model"  # 这里应该是实际的模型加载代码

    def predict(self, input_data: Any) -> Any:
        return f"Prediction for {input_data}"

    def train(self, training_data: Any, labels: Any) -> None:
        print(f"Training model with {len(training_data)} samples")

    def evaluate(self, test_data: Any, test_labels: Any) -> Dict[str, float]:
        return {"accuracy": 0.95, "f1_score": 0.94}

    def save_model(self, model_path: str) -> None:
        print(f"Saving model to {model_path}")

    def preprocess_data(self, raw_data: Any) -> Any:
        return f"Preprocessed {raw_data}"

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": "ExampleModel", "version": "1.0"}

    def fine_tune(self, fine_tuning_data: Any, fine_tuning_labels: Any) -> None:
        print(f"Fine-tuning model with {len(fine_tuning_data)} samples")

    def explain_prediction(self, input_data: Any, prediction: Any) -> str:
        return f"Model predicted {prediction} for {input_data} because of XYZ reasons"

def main():
    # 注册模型
    ModelFactory.register_model("ExampleModel", ExampleModel)

    # 创建模型实例
    model = ModelFactory.create_model("ExampleModel")

    # 使用模型
    model.load_model("path/to/model")
    input_data = [1, 2, 3]
    preprocessed_data = model.preprocess_data(input_data)
    prediction = model.predict(preprocessed_data)
    explanation = model.explain_prediction(input_data, prediction)

    print(f"Prediction: {prediction}")
    print(f"Explanation: {explanation}")

    # 评估模型
    evaluation_result = model.evaluate([1, 2, 3, 4], [0, 1, 1, 0])
    print(f"Evaluation result: {evaluation_result}")

    # 获取模型信息
    model_info = model.get_model_info()
    print(f"Model info: {model_info}")

if __name__ == "__main__":
    main()