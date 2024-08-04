import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def profile_model(model: nn.Module, input_data: torch.Tensor) -> float:
    """
    Profile the execution time of a model.

    Args:
    model (nn.Module): The PyTorch model to profile.
    input_data (torch.Tensor): Input data for the model.

    Returns:
    float: Execution time in seconds.
    """
    start_time = time.time()
    with torch.no_grad():
        _ = model(input_data)
    end_time = time.time()
    return end_time - start_time

def quantize_model(model: nn.Module) -> nn.Module:
    """
    Quantize a PyTorch model to reduce its size and increase inference speed.

    Args:
    model (nn.Module): The PyTorch model to quantize.

    Returns:
    nn.Module: The quantized model.
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    return quantized_model

def prune_model(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """
    Prune a PyTorch model by setting a percentage of the smallest weights to zero.

    Args:
        model (nn.Module): The PyTorch model to prune.
        amount (float): The percentage of weights to prune (0.0 to 1.0).

    Returns:
        nn.Module: The pruned model.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model

def optimize_model(model: nn.Module, input_data: torch.Tensor) -> nn.Module:
    """
    Optimize a PyTorch model by quantizing and pruning.

    Args:
    model (nn.Module): The PyTorch model to optimize.
    input_data (torch.Tensor): Sample input data for profiling.

    Returns:
    nn.Module: The optimized model.
    """
    print("Original model execution time:", profile_model(model, input_data))

    quantized_model = quantize_model(model)
    print("Quantized model execution time:", profile_model(quantized_model, input_data))

    pruned_model = prune_model(quantized_model)
    print("Pruned model execution time:", profile_model(pruned_model, input_data))

    return pruned_model