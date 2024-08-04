# E:\AI_Nirvana-1\src\core\quantization.py

import torch
import torch.nn as nn
import torch.quantization as q
from torch.quantization import quantize_dynamic, prepare, convert, prepare_qat
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import torch.nn.functional as F
import os
import time

class QuantizationTechniques:
    @staticmethod
    def dynamic_quantization(model, dtype=torch.qint8):
        """动态量化"""
        return quantize_dynamic(
            model, 
            {nn.Linear, nn.LSTM, nn.GRU, nn.RNN}, 
            dtype=dtype
        )

    @staticmethod
    def static_quantization(model, qconfig='fbgemm'):
        """静态量化"""
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig(qconfig)
        model_prepared = prepare(model)
        # 这里应该插入校准代码
        model_quantized = convert(model_prepared)
        return model_quantized

    @staticmethod
    def quantization_aware_training(model, qconfig='fbgemm'):
        """量化感知训练"""
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig(qconfig)
        model_prepared = prepare_qat(model)
        # 这里应该插入训练代码
        model_quantized = convert(model_prepared)
        return model_quantized

    @staticmethod
    def fx_graph_mode_quantization(model, example_inputs, qconfig='fbgemm'):
        """FX图模式量化"""
        model.eval()
        qconfig = torch.quantization.get_default_qconfig(qconfig)
        model_prepared = prepare_fx(model, {'': qconfig}, example_inputs)
        # 这里应该插入校准代码
        model_quantized = convert_fx(model_prepared)
        return model_quantized

class AdvancedQuantizationTechniques:
    @staticmethod
    def mixed_precision_quantization(model, bits_weights=8, bits_activations=8):
        """混合精度量化"""
        def quantize_layer(layer):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                weight = layer.weight.data
                scale = weight.abs().max() / (2**(bits_weights-1) - 1)
                layer.weight.data = torch.fake_quantize_per_tensor_affine(
                    weight, scale.item(), 0, -(2**(bits_weights-1)), 2**(bits_weights-1)-1
                )
            return layer
        
        return model.apply(quantize_layer)

    @staticmethod
    def pruning_aware_quantization(model, pruning_rate=0.5):
        """剪枝感知量化"""
        for name, param in model.named_parameters():
            if 'weight' in name:
                mask = torch.zeros_like(param.data).float()
                mask.masked_fill_(abs(param.data) > param.data.std()*pruning_rate, 1)
                param.data.mul_(mask)
        return QuantizationTechniques.dynamic_quantization(model)

    @staticmethod
    def knowledge_distillation_quantization(teacher_model, student_model, temperature=2.0):
        """知识蒸馏量化"""
        def distillation_loss(y, teacher_scores, labels, T):
            p = F.log_softmax(y/T, dim=1)
            q = F.softmax(teacher_scores/T, dim=1)
            l_kl = F.kl_div(p, q, reduction='batchmean') * (T**2)
            l_ce = F.cross_entropy(y, labels)
            return l_kl + l_ce

        # 这里应该插入使用distillation_loss的训练代码
        return QuantizationTechniques.dynamic_quantization(student_model)

def prepare_model_for_quantization(model, technique='dynamic', **kwargs):
    model.eval()
    if technique == 'dynamic':
        return QuantizationTechniques.dynamic_quantization(model, **kwargs)
    elif technique == 'static':
        return QuantizationTechniques.static_quantization(model, **kwargs)
    elif technique == 'qat':
        return QuantizationTechniques.quantization_aware_training(model, **kwargs)
    elif technique == 'fx':
        return QuantizationTechniques.fx_graph_mode_quantization(model, **kwargs)
    elif technique == 'mixed':
        return AdvancedQuantizationTechniques.mixed_precision_quantization(model, **kwargs)
    elif technique == 'pruning':
        return AdvancedQuantizationTechniques.pruning_aware_quantization(model, **kwargs)
    elif technique == 'distillation':
        return AdvancedQuantizationTechniques.knowledge_distillation_quantization(**kwargs)
    else:
        raise ValueError(f"Unsupported quantization technique: {technique}")

def quantize_and_evaluate(model, test_data, technique='dynamic', **kwargs):
    quantized_model = prepare_model_for_quantization(model, technique, **kwargs)
    accuracy = evaluate_model(quantized_model, test_data)
    return quantized_model, accuracy

def evaluate_model(model, test_data):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_data:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def compare_model_sizes(original_model, quantized_model):
    def get_model_size(model):
        torch.save(model.state_dict(), "temp.p")
        size = os.path.getsize("temp.p")
        os.remove('temp.p')
        return size
    
    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)
    
    print(f"Original model size: {original_size/1e6:.2f} MB")
    print(f"Quantized model size: {quantized_size/1e6:.2f} MB")
    print(f"Compression ratio: {original_size/quantized_size:.2f}x")

def benchmark_inference_speed(model, input_shape, num_runs=100):
    input_tensor = torch.randn(input_shape)
    
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_tensor)
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / num_runs
    print(f"Average inference time: {avg_inference_time*1000:.2f} ms")