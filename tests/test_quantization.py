import unittest
import torch
import torch.nn as nn
from src.core.quantization import prepare_model_for_quantization, quantize_and_evaluate


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class TestQuantization(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.test_input = torch.randn(1, 10)

    def test_prepare_model_for_quantization(self):
        quantized_model = prepare_model_for_quantization(self.model)
        self.assertIsInstance(quantized_model, torch.nn.Module)

    def test_quantize_and_evaluate(self):
        def dummy_eval_func(model):
            return 0.9  # Dummy accuracy

        quantized_model, accuracy = quantize_and_evaluate(self.model, dummy_eval_func)
        self.assertIsInstance(quantized_model, torch.nn.Module)
        self.assertIsInstance(accuracy, float)
        self.assertTrue(0 <= accuracy <= 1)


if __name__ == "__main__":
    unittest.main()
