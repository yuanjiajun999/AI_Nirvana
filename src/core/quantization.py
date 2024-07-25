import torch
import torch.nn as nn
import torch.quantization as q


def prepare_model_for_quantization(model):
    model.eval()
    model = q.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )
    return model


def quantize_and_evaluate(model, test_data):
    quantized_model = prepare_model_for_quantization(model)
    accuracy = evaluate_model(quantized_model, test_data)
    return quantized_model, accuracy
