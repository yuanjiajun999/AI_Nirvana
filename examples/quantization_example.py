# quantization_example.py

import torch
import torch.nn as nn
from src.core.quantization import (
    QuantizationTechniques, 
    AdvancedQuantizationTechniques, 
    prepare_model_for_quantization, 
    quantize_and_evaluate, 
    compare_model_sizes, 
    benchmark_inference_speed
)

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型和测试数据
model = SimpleModel()
test_data = [(torch.randn(1, 10), torch.tensor([0])) for _ in range(100)]

print("原始模型:")
benchmark_inference_speed(model, (1, 10))

# 1. 动态量化
dynamic_quantized_model = QuantizationTechniques.dynamic_quantization(model)
print("\n动态量化后的模型:")
benchmark_inference_speed(dynamic_quantized_model, (1, 10))
compare_model_sizes(model, dynamic_quantized_model)

# 2. 静态量化
static_quantized_model = QuantizationTechniques.static_quantization(model)
print("\n静态量化后的模型:")
benchmark_inference_speed(static_quantized_model, (1, 10))
compare_model_sizes(model, static_quantized_model)

# 3. 量化感知训练
qat_model = QuantizationTechniques.quantization_aware_training(model)
print("\n量化感知训练后的模型:")
benchmark_inference_speed(qat_model, (1, 10))
compare_model_sizes(model, qat_model)

# 4. FX图模式量化
example_inputs = torch.randn(1, 10)
fx_quantized_model = QuantizationTechniques.fx_graph_mode_quantization(model, example_inputs)
print("\nFX图模式量化后的模型:")
benchmark_inference_speed(fx_quantized_model, (1, 10))
compare_model_sizes(model, fx_quantized_model)

# 5. 混合精度量化
mixed_precision_model = AdvancedQuantizationTechniques.mixed_precision_quantization(model)
print("\n混合精度量化后的模型:")
benchmark_inference_speed(mixed_precision_model, (1, 10))
compare_model_sizes(model, mixed_precision_model)

# 6. 剪枝感知量化
pruned_quantized_model = AdvancedQuantizationTechniques.pruning_aware_quantization(model)
print("\n剪枝感知量化后的模型:")
benchmark_inference_speed(pruned_quantized_model, (1, 10))
compare_model_sizes(model, pruned_quantized_model)

# 7. 知识蒸馏量化
teacher_model = SimpleModel()
student_model = SimpleModel()
distilled_quantized_model = AdvancedQuantizationTechniques.knowledge_distillation_quantization(teacher_model, student_model)
print("\n知识蒸馏量化后的模型:")
benchmark_inference_speed(distilled_quantized_model, (1, 10))
compare_model_sizes(model, distilled_quantized_model)

# 使用统一接口进行量化和评估
for technique in ['dynamic', 'static', 'qat', 'fx', 'mixed', 'pruning', 'distillation']:
    print(f"\n使用 {technique} 技术进行量化:")
    if technique == 'fx':
        quantized_model, accuracy = quantize_and_evaluate(model, test_data, technique=technique, example_inputs=example_inputs)
    elif technique == 'distillation':
        quantized_model, accuracy = quantize_and_evaluate(model, test_data, technique=technique, teacher_model=teacher_model, student_model=student_model)
    else:
        quantized_model, accuracy = quantize_and_evaluate(model, test_data, technique=technique)
    print(f"准确率: {accuracy}")
    benchmark_inference_speed(quantized_model, (1, 10))
    compare_model_sizes(model, quantized_model)