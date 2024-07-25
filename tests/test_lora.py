import unittest

import torch

from src.core.lora import LoRALayer, LoRAModel


class TestLoRA(unittest.TestCase):
    def test_lora_layer(self):
        in_features, out_features = 10, 5
        lora_layer = LoRALayer(in_features, out_features, r=2)
        x = torch.randn(1, in_features)
        output = lora_layer(x)
        self.assertEqual(output.shape, (1, out_features))

    def test_lora_model(self):
        in_features, hidden_features, out_features = 10, 5, 2
        base_model = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_features),
        )
        lora_config = [
            (out_features, out_features, 2),  # Apply LoRA to the output layer
        ]
        model = LoRAModel(base_model, lora_config)
        x = torch.randn(1, in_features)
        output = model(x)
        self.assertEqual(output.shape, (1, out_features))


if __name__ == "__main__":
    unittest.main()
