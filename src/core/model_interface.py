from abc import ABC, abstractmethod
from typing import Any, Dict

class ModelInterface(ABC):
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load a model from the specified path."""
        ...  # pragma: no cover

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Make predictions using the loaded model."""
        ...  # pragma: no cover

    @abstractmethod
    def train(self, training_data: Any, labels: Any) -> None:
        """Train the model using the provided data and labels."""
        ...  # pragma: no cover

    @abstractmethod
    def evaluate(self, test_data: Any, test_labels: Any) -> Dict[str, float]:
        """Evaluate the model performance."""
        ...  # pragma: no cover

    @abstractmethod
    def save_model(self, model_path: str) -> None:
        """Save the model to the specified path."""
        ...  # pragma: no cover

    @abstractmethod
    def preprocess_data(self, raw_data: Any) -> Any:
        """Preprocess the input data."""
        ...  # pragma: no cover

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        ...  # pragma: no cover

    @abstractmethod
    def fine_tune(self, fine_tuning_data: Any, fine_tuning_labels: Any) -> None:
        """Fine-tune the model with the provided data."""
        ...  # pragma: no cover

    @abstractmethod
    def explain_prediction(self, input_data: Any, prediction: Any) -> str:
        """Provide an explanation for a given prediction."""
        ...  # pragma: no cover