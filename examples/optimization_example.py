import torch
import torch.nn as nn
from src.core.optimization import (
    profile_model,
    quantize_model,
    prune_model,
    optimize_model,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def main():
    model = SimpleModel()
    input_data = torch.randn(100, 10)

    print("Original model:")
    print(model)
    print("Execution time:", profile_model(model, input_data))

    optimized_model = optimize_model(model, input_data)

    print("\nOptimized model:")
    print(optimized_model)
    print("Execution time:", profile_model(optimized_model, input_data))

    # 单独展示量化和剪枝效果
    quantized_model = quantize_model(model)
    print("\nQuantized model:")
    print(quantized_model)
    print("Execution time:", profile_model(quantized_model, input_data))

    pruned_model = prune_model(model)
    print("\nPruned model:")
    print(pruned_model)
    print("Execution time:", profile_model(pruned_model, input_data))


if __name__ == "__main__":
    main()
