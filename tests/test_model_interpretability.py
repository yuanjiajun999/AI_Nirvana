import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.core.model_interpretability import ModelInterpreter


class TestModelInterpreter(unittest.TestCase):
    def setUp(self):
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 2, 100)
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        self.interpreter = ModelInterpreter(self.model, X)

    def test_create_explainer(self):
        self.interpreter.create_explainer()
        self.assertIsNotNone(self.interpreter.explainer)

    def test_get_shap_values(self):
        self.interpreter.create_explainer()
        shap_values = self.interpreter.get_shap_values()
        self.assertIsNotNone(shap_values)

    def test_plot_summary(self):
        self.interpreter.create_explainer()
        # This test just ensures the method runs without error
        try:
            self.interpreter.plot_summary()
        except Exception as e:
            self.fail(f"plot_summary() raised {type(e).__name__} unexpectedly!")


if __name__ == "__main__":
    unittest.main()
