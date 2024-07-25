import torch
import torch.nn as nn
from src.core.quantization import Quantizer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


def main():
    # 创建一个简单的模型
    model = SimpleModel()

    # 创建一些随机输入数据
    input_data = torch.randn(1, 10)

    # 初始化量化器
    quantizer = Quantizer()

    print("Original model:")
    print(model)
    print("Output:", model(input_data))

    # 量化模型
    quantized_model = quantizer.quantize(model)

    print("\nQuantized model:")
    print(quantized_model)
    print("Output:", quantized_model(input_data))

    # 评估量化效果
    original_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    quantized_size = sum(
        p.numel() for p in quantized_model.parameters() if p.requires_grad
    )

    print(f"\nOriginal model size: {original_size} parameters")
    print(f"Quantized model size: {quantized_size} parameters")
    print(f"Size reduction: {(1 - quantized_size/original_size)*100:.2f}%")


if __name__ == "__main__":
    main()
