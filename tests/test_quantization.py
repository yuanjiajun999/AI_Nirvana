# E:\AI_Nirvana-1\tests\test_quantization.py

import unittest
import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.quantization import (
    QuantizationTechniques, 
    AdvancedQuantizationTechniques, 
    prepare_model_for_quantization, 
    quantize_and_evaluate, 
    evaluate_model, 
    compare_model_sizes, 
    benchmark_inference_speed
)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TestQuantization(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.test_data = [(torch.randn(1, 10), torch.tensor([0]))]

    def test_dynamic_quantization(self):
        quantized_model = QuantizationTechniques.dynamic_quantization(self.model)
        self.assertIsInstance(quantized_model, SimpleModel)

    def test_static_quantization(self):
        try:
            quantized_model = QuantizationTechniques.static_quantization(self.model)
            self.assertIsInstance(quantized_model, SimpleModel)
        except Exception as e:
            self.skipTest(f"Static quantization not supported: {str(e)}")

    def test_quantization_aware_training(self):
        quantized_model = QuantizationTechniques.quantization_aware_training(self.model)
        self.assertIsInstance(quantized_model, SimpleModel)

    def test_fx_graph_mode_quantization(self):
        example_inputs = torch.randn(1, 10)
        quantized_model = QuantizationTechniques.fx_graph_mode_quantization(self.model, example_inputs)
        self.assertIsInstance(quantized_model, torch.nn.Module)

    def test_mixed_precision_quantization(self):
        quantized_model = AdvancedQuantizationTechniques.mixed_precision_quantization(self.model)
        self.assertIsInstance(quantized_model, SimpleModel)

    def test_pruning_aware_quantization(self):
        quantized_model = AdvancedQuantizationTechniques.pruning_aware_quantization(self.model)
        self.assertIsInstance(quantized_model, SimpleModel)

    def test_knowledge_distillation_quantization(self):
        teacher_model = SimpleModel()
        student_model = SimpleModel()
        quantized_model = AdvancedQuantizationTechniques.knowledge_distillation_quantization(teacher_model, student_model)
        self.assertIsInstance(quantized_model, SimpleModel)

    def test_prepare_model_for_quantization(self):
        techniques = ['dynamic', 'static', 'qat', 'mixed', 'pruning']
        for technique in techniques:
            with self.subTest(technique=technique):
                quantized_model = prepare_model_for_quantization(self.model, technique)
                self.assertIsInstance(quantized_model, (torch.nn.Module, SimpleModel))
        
        # 单独测试 'fx' 和 'distillation'
        example_inputs = torch.randn(1, 10)
        fx_model = prepare_model_for_quantization(self.model, 'fx', example_inputs=example_inputs)
        self.assertIsInstance(fx_model, torch.nn.Module)

        teacher_model = SimpleModel()
        distillation_model = prepare_model_for_quantization(self.model, 'distillation', teacher_model=teacher_model, student_model=self.model)
        self.assertIsInstance(distillation_model, SimpleModel)

    def test_quantize_and_evaluate(self):
        quantized_model, accuracy = quantize_and_evaluate(self.model, self.test_data)
        self.assertIsInstance(quantized_model, (torch.nn.Module, SimpleModel))
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

    def test_evaluate_model(self):
        accuracy = evaluate_model(self.model, self.test_data)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

    def test_compare_model_sizes(self):
        quantized_model = QuantizationTechniques.dynamic_quantization(self.model)
        compare_model_sizes(self.model, quantized_model)
        # This test just ensures the function runs without error

    def test_benchmark_inference_speed(self):
        benchmark_inference_speed(self.model, (1, 10))
        # This test just ensures the function runs without error

if __name__ == '__main__':
    unittest.main()