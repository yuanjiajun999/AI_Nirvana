# tests/test_lora.py

import unittest
import torch
from src.core.lora import LoRALayer, LoRAModel

class TestLoRA(unittest.TestCase):
    def test_lora_layer(self):
        lora_layer = LoRALayer(10, 5, r=2)
        x = torch.randn(1, 10)
        output = lora_layer(x)
        self.assertEqual(output.shape, (1, 5))

    def test_lora_model(self):
        base_model = torch.nn.Linear(10, 5)
        lora_layers = [LoRALayer(5, 5, r=2)]
        model = LoRAModel(base_model, lora_layers)
        x = torch.randn(1, 10)
        output = model(x)
        self.assertEqual(output.shape, (1, 5))

if __name__ == '__main__':
    unittest.main()