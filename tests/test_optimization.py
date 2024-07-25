import unittest

import torch
import torch.nn as nn

from src.core.optimization import profile_model, prune_model, quantize_model

optimize_model


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class TestOptimization(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.input_data = torch.randn(1, 10)

    def test_profile_model(self):
        execution_time = profile_model(self.model, self.input_data)
        self.assertGreater(execution_time, 0)

    def test_quantize_model(self):
        quantized_model = quantize_model(self.model)
        self.assertIsInstance(quantized_model, nn.Module)

    def test_prune_model(self):
        pruned_model = prune_model(self.model)
        self.assertIsInstance(pruned_model, nn.Module)

    def test_optimize_model(self):
        optimized_model = optimize_model(self.model, self.input_data)
        self.assertIsInstance(optimized_model, nn.Module)


if __name__ == "__main__":
    unittest.main()
