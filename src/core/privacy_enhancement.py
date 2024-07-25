import random


class PrivacyEnhancement:
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon

    def add_laplace_noise(self, value):
        """Add Laplace noise to a single value."""
        beta = 1 / self.epsilon
        noise = random.laplace(0, beta)
        return value + noise

    def apply_differential_privacy(self, data):
        """Apply differential privacy to a list of numerical data."""
        return [self.add_laplace_noise(value) for value in data]

    def set_privacy_budget(self, epsilon):
        """Set the privacy budget (epsilon)."""
        self.epsilon = epsilon
