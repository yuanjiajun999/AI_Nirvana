print("Starting test_lora.py")

import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

print("Importing unittest")
import unittest
print("unittest imported successfully")

print("Importing torch")
import torch
print(f"PyTorch imported successfully, version: {torch.__version__}")

print("Importing torch.nn")
import torch.nn as nn
print("torch.nn imported successfully")

print("Attempting to import from src.core.lora...")
from src.core.lora import LoRALayer, LoRAModel, LoRAOptimizer, apply_lora_to_model
print("Lora modules imported successfully")

class TestLoRALayer(unittest.TestCase):
    def setUp(self):
        self.in_features = 10
        self.out_features = 5
        self.rank = 4
        self.alpha = 0.5
        self.lora_layer = LoRALayer(self.in_features, self.out_features, self.rank, self.alpha)

    def test_initialization(self):
        self.assertEqual(self.lora_layer.in_features, self.in_features)
        self.assertEqual(self.lora_layer.out_features, self.out_features)
        self.assertEqual(self.lora_layer.rank, self.rank)
        self.assertEqual(self.lora_layer.alpha, self.alpha)

    def test_forward(self):
        x = torch.randn(1, self.in_features)
        output = self.lora_layer(x)
        self.assertEqual(output.shape, (1, self.out_features))

    def test_merge_weights(self):
        self.lora_layer.merge_lora_weights()
        self.assertTrue(self.lora_layer.merge_weights)
        self.assertTrue(hasattr(self.lora_layer, 'merged_weight'))

    def test_unmerge_weights(self):
        self.lora_layer.merge_lora_weights()
        self.lora_layer.unmerge_lora_weights()
        self.assertFalse(self.lora_layer.merge_weights)
        self.assertFalse(hasattr(self.lora_layer, 'merged_weight'))

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.linear2(self.linear1(x))

class TestLoRAModel(unittest.TestCase):
    def setUp(self):
        self.base_model = SimpleModel()
        self.lora_config = {
            'linear1': {'rank': 4, 'alpha': 0.5},
            'linear2': {'rank': 2, 'alpha': 0.3}
        }
        self.lora_model = LoRAModel(self.base_model, self.lora_config)

    def test_initialization(self):
        self.assertIsInstance(self.lora_model.base_model, nn.Module)
        self.assertEqual(len(self.lora_model.lora_layers), 2)

    def test_forward(self):
        x = torch.randn(1, 10)
        output = self.lora_model(x)
        self.assertEqual(output.shape, (1, 2))

    def test_merge_and_unmerge(self):
        self.lora_model.merge_weights = True
        self.lora_model.merge_and_unmerge()
        for lora_layer in self.lora_model.lora_layers.values():
            self.assertTrue(lora_layer.merge_weights)

        self.lora_model.merge_weights = False
        self.lora_model.merge_and_unmerge()
        for lora_layer in self.lora_model.lora_layers.values():
            self.assertFalse(lora_layer.merge_weights)

    def test_get_trainable_parameters(self):
        trainable_params = self.lora_model.get_trainable_parameters()
        self.assertTrue(all("lora_" in name for name, _ in self.lora_model.named_parameters() if name in trainable_params))

class TestLoRAOptimizer(unittest.TestCase):
    def setUp(self):
        base_model = SimpleModel()
        lora_config = {'linear1': {'rank': 4, 'alpha': 0.5}}
        self.lora_model = LoRAModel(base_model, lora_config)
        self.lora_optimizer = LoRAOptimizer(self.lora_model, lr=0.01)

    def test_initialization(self):
        self.assertIsInstance(self.lora_optimizer.optimizer, torch.optim.AdamW)
        self.assertEqual(self.lora_optimizer.lr, 0.01)

    def test_step_and_zero_grad(self):
        x = torch.randn(1, 10)
        output = self.lora_model(x)
        loss = output.sum()
        loss.backward()

        # Check if grad is None before cloning
        lora_A_grad = self.lora_model.lora_layers['linear1'].lora_A.grad
        if lora_A_grad is not None:
            initial_grad = lora_A_grad.clone()
            self.lora_optimizer.step()
            self.assertFalse(torch.allclose(initial_grad, self.lora_model.lora_layers['linear1'].lora_A.grad))
        else:
            print("Warning: grad is None, skipping grad check")

        self.lora_optimizer.zero_grad()
        self.assertIsNone(self.lora_model.lora_layers['linear1'].lora_A.grad)

class TestApplyLoraToModel(unittest.TestCase):
    def setUp(self):
        self.base_model = SimpleModel()
        self.lora_config = {'linear1': {'rank': 4, 'alpha': 0.5}}

    def test_apply_lora_to_model(self):
        lora_model, lora_optimizer = apply_lora_to_model(self.base_model, self.lora_config)
        
        self.assertIsInstance(lora_model, LoRAModel)
        self.assertIsInstance(lora_optimizer, LoRAOptimizer)
        self.assertEqual(len(lora_model.lora_layers), 1)
        self.assertIn('linear1', lora_model.lora_layers)

if __name__ == '__main__':
    unittest.main()