from abc import ABC, abstractmethod
from typing import Any, Dict, List

class ModelInterface(ABC):
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load a model from the specified path."""
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Make a prediction using the loaded model."""
        pass

    @abstractmethod
    def train(self, training_data: Any, labels: Any) -> None:
        """Train the model using the provided data and labels."""
        pass

    @abstractmethod
    def evaluate(self, test_data: Any, test_labels: Any) -> Dict[str, float]:
        """Evaluate the model's performance on test data."""
        pass

    @abstractmethod
    def save_model(self, model_path: str) -> None:
        """Save the current model to the specified path."""
        pass

    @abstractmethod
    def preprocess_data(self, raw_data: Any) -> Any:
        """Preprocess raw input data before prediction or training."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the current model."""
        pass

    @abstractmethod
    def fine_tune(self, fine_tuning_data: Any, fine_tuning_labels: Any) -> None:
        """Fine-tune the model on new data."""
        pass

    @abstractmethod
    def explain_prediction(self, input_data: Any, prediction: Any) -> str:
        """Provide an explanation for a given prediction."""
        pass