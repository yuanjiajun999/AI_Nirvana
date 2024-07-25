import matplotlib.pyplot as plt
import numpy as np


class ModelInterpreter:
    def __init__(self, model, X):
        self.model = model
        self.X = X
        self.explainer = None

    def create_explainer(self):
        import shap

        self.explainer = shap.TreeExplainer(self.model)

    def get_shap_values(self):
        return self.explainer.shap_values(self.X)

    def plot_summary(self, class_index=None):
        import shap

        shap_values = self.get_shap_values()
        if class_index is not None:
            shap.summary_plot(shap_values[class_index], self.X)
        else:
            shap.summary_plot(shap_values, self.X)

    def plot_force(self, instance_index, class_index=None):
        import shap

        shap_values = self.get_shap_values()
        if class_index is not None:
            shap.force_plot(
                self.explainer.expected_value[class_index],
                shap_values[class_index][instance_index],
                self.X.iloc[instance_index],
            )
        else:
            shap.force_plot(
                self.explainer.expected_value,
                shap_values[instance_index],
                self.X.iloc[instance_index],
            )
