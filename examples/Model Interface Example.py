from src.core import ModelFactory, ModelInterface
from typing import Any, Dict, List
import numpy as np

class SimpleLinearRegressionModel(ModelInterface):
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.is_trained = False

    def load_model(self, model_path: str) -> None:
        try:
            with open(model_path, 'r') as f:
                params = f.read().split(',')
                self.intercept = float(params[0])
                self.coefficients = [float(coef) for coef in params[1:]]
            self.is_trained = True
            print(f"Model loaded from {model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        except ValueError:
            raise ValueError("Invalid model file format")

    def predict(self, input_data: List[float]) -> float:
        if not self.is_trained:
            raise ValueError("Model is not trained. Call load_model or train first.")
        return self.intercept + sum(x * coef for x, coef in zip(input_data, self.coefficients))

    def train(self, training_data: List[List[float]], labels: List[float]) -> None:
        X = np.array(training_data)
        y = np.array(labels)
        X = np.column_stack((np.ones(X.shape[0]), X))
        self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]
        self.is_trained = True
        print(f"Model trained on {len(training_data)} samples")

    def evaluate(self, test_data: List[List[float]], test_labels: List[float]) -> Dict[str, float]:
        predictions = [self.predict(x) for x in test_data]
        mse = np.mean((np.array(predictions) - np.array(test_labels)) ** 2)
        r2 = 1 - (np.sum((np.array(test_labels) - np.array(predictions)) ** 2) / 
                  np.sum((np.array(test_labels) - np.mean(test_labels)) ** 2))
        return {"mse": mse, "r2": r2}

    def save_model(self, model_path: str) -> None:
        if not self.is_trained:
            raise ValueError("Model is not trained. Train the model before saving.")
        try:
            with open(model_path, 'w') as f:
                f.write(f"{self.intercept},{','.join(map(str, self.coefficients))}")
            print(f"Model saved to {model_path}")
        except IOError:
            raise IOError(f"Error writing model to {model_path}")

    def preprocess_data(self, raw_data: List[float]) -> List[float]:
        return [float(x) for x in raw_data]  # Simple conversion to float

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "SimpleLinearRegression",
            "version": "1.0",
            "type": "Regression",
            "input_shape": [None, len(self.coefficients)] if self.is_trained else None,
            "is_trained": self.is_trained
        }

    def fine_tune(self, fine_tuning_data: List[List[float]], fine_tuning_labels: List[float]) -> None:
        self.train(fine_tuning_data, fine_tuning_labels)  # For this simple model, fine-tuning is the same as training

    def explain_prediction(self, input_data: List[float], prediction: float) -> str:
        if not self.is_trained:
            raise ValueError("Model is not trained. Cannot explain prediction.")
        feature_contributions = [f"{coef:.2f} * {x:.2f}" for coef, x in zip(self.coefficients, input_data)]
        explanation = f"Prediction {prediction:.2f} is calculated as:\n"
        explanation += f"{self.intercept:.2f} (intercept) + " + " + ".join(feature_contributions)
        return explanation

def main():
    # Register the model
    ModelFactory.register_model("SimpleLinearRegression", SimpleLinearRegressionModel)

    # Create a model instance
    model = ModelFactory.create_model("SimpleLinearRegression")

    # Generate some sample data
    np.random.seed(0)
    X = np.random.rand(100, 2)
    y = 3 + 2 * X[:, 0] + 1.5 * X[:, 1] + np.random.randn(100) * 0.1

    # Split the data into training and test sets
    X_train, X_test = X[:80].tolist(), X[80:].tolist()
    y_train, y_test = y[:80].tolist(), y[80:].tolist()

    # Train the model
    model.train(X_train, y_train)

    # Make a prediction
    sample_input = [0.5, 0.6]
    prediction = model.predict(sample_input)
    print(f"Prediction for {sample_input}: {prediction}")

    # Evaluate the model
    eval_results = model.evaluate(X_test, y_test)
    print(f"Evaluation results: {eval_results}")

    # Get model info
    model_info = model.get_model_info()
    print(f"Model info: {model_info}")

    # Explain a prediction
    explanation = model.explain_prediction(sample_input, prediction)
    print(f"Explanation:\n{explanation}")

    # Save the model
    model.save_model("simple_linear_model.txt")

    # Load the model
    new_model = ModelFactory.create_model("SimpleLinearRegression")
    new_model.load_model("simple_linear_model.txt")

    # Verify the loaded model
    new_prediction = new_model.predict(sample_input)
    print(f"Prediction with loaded model: {new_prediction}")

if __name__ == "__main__":
    main()