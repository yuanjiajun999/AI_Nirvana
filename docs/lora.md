# LoRA (Low-Rank Adaptation) Module Documentation

## Overview

The LoRA (Low-Rank Adaptation) module is an implementation of the technique described in the paper "LoRA: Low-Rank Adaptation of Large Language Models" by Microsoft Research. It provides an efficient way to fine-tune large pre-trained models by adding pairs of rank decomposition matrices to existing weights, significantly reducing the number of trainable parameters.

## Main Components

### 1. LoRALayer

`LoRALayer` is the core component that implements the LoRA technique for a single layer.

#### Attributes:
- `in_features`: Number of input features
- `out_features`: Number of output features
- `rank`: Rank of the low-rank decomposition
- `alpha`: Scaling factor for LoRA
- `merge_weights`: Boolean flag to merge LoRA weights with original weights

#### Methods:
- `forward(x)`: Performs the forward pass, applying LoRA transformation
- `merge_lora_weights()`: Merges LoRA weights with original weights
- `unmerge_lora_weights()`: Unmerges LoRA weights from original weights

### 2. LoRAModel

`LoRAModel` wraps a base model and applies LoRA to specified layers.

#### Attributes:
- `base_model`: The original model to apply LoRA to
- `lora_config`: Configuration for LoRA layers
- `merge_weights`: Boolean flag to merge weights
- `lora_layers`: Dictionary of LoRA layers

#### Methods:
- `forward(*args, **kwargs)`: Performs the forward pass of the model
- `merge_and_unmerge()`: Merges or unmerges LoRA weights based on current state
- `get_trainable_parameters()`: Returns list of trainable LoRA parameters

### 3. LoRAOptimizer

`LoRAOptimizer` is a custom optimizer for LoRA parameters.

#### Attributes:
- `model`: The LoRA model to optimize
- `lr`: Learning rate
- `weight_decay`: Weight decay factor
- `optimizer_class`: The optimizer class to use (default: AdamW)

#### Methods:
- `step()`: Performs a single optimization step
- `zero_grad()`: Resets gradients to zero

### 4. apply_lora_to_model

This function is a utility to apply LoRA to a model and create an optimizer.

#### Parameters:
- `base_model`: The original model to apply LoRA to
- `lora_config`: Configuration for LoRA layers
- `lr`: Learning rate for the optimizer
- `weight_decay`: Weight decay for the optimizer
- `merge_weights`: Whether to merge LoRA weights initially

#### Returns:
A tuple containing the LoRA model and its optimizer

## Usage

1. Define your base model and LoRA configuration.
2. Use `apply_lora_to_model` to create a LoRA-enabled model and optimizer.
3. Train the model using the LoRA optimizer.
4. Before inference, call `merge_and_unmerge()` to merge LoRA weights.
5. After inference, call `merge_and_unmerge()` again if you want to continue training.

## Best Practices

- Choose appropriate rank values in the LoRA configuration to balance between model capacity and efficiency.
- The `alpha` parameter in the LoRA config acts as a scaling factor for LoRA updates. Adjust it based on your specific use case.
- Always merge weights before using the model for inference to get the full benefit of LoRA.
- If you need to switch between training and inference frequently, you can use the `merge_and_unmerge()` method to toggle between merged and unmerged states.

## Note

This implementation is designed to work with PyTorch models. Ensure that your base model is compatible with the LoRA implementation, especially in terms of layer naming and structure.