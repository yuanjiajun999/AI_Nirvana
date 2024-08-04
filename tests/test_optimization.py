import unittest
import torch
import torch.nn as nn
import numpy as np

from src.core.optimization import profile_model, prune_model, quantize_model, optimize_model

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

class TestOptimization(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.input_data = torch.randn(1000, 100)

    def test_profile_model(self):
        execution_time = profile_model(self.model, self.input_data)
        self.assertGreater(execution_time, 0)

    def test_quantize_model(self):
        quantized_model = quantize_model(self.model)
        self.assertIsInstance(quantized_model, nn.Module)
        # Check if the model structure has changed (quantized)
        self.assertNotEqual(type(self.model.fc1), type(quantized_model.fc1))

    def test_prune_model(self):
        pruned_model = prune_model(self.model)
        self.assertIsInstance(pruned_model, nn.Module)
        # Check if weights have been pruned (set to zero)
        fc1_weight_zero_count = torch.sum(pruned_model.fc1.weight == 0).item()
        self.assertGreater(fc1_weight_zero_count, 0)

    def test_optimize_model(self):
        optimized_model = optimize_model(self.model, self.input_data)
        self.assertIsInstance(optimized_model, nn.Module)
        # Check if the optimized model is different from the original
        self.assertNotEqual(type(self.model.fc1), type(optimized_model.fc1))

if __name__ == "__main__":
    unittest.main()