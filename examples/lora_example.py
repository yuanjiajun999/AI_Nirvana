import torch
from src.core.lora import LoRALayer, LoRAModel


def main():
    # 创建一个简单的基础模型
    base_model = torch.nn.Sequential(
        torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 5)
    )

    # 创建 LoRA 层配置
    lora_config = [
        (20, 20, 4),  # 对应第一个 Linear 层
        (5, 5, 2),  # 对应第二个 Linear 层
    ]

    # 创建 LoRA 模型
    lora_model = LoRAModel(base_model, lora_config)

    # 生成一些随机输入数据
    x = torch.randn(1, 10)

    # 使用 LoRA 模型进行前向传播
    output = lora_model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

    # 展示 LoRA 层的参数
    for i, layer in enumerate(lora_model.lora_layers):
        print(f"\nLoRA Layer {i}:")
        print(f"A shape: {layer.lora_A.shape}")
        print(f"B shape: {layer.lora_B.shape}")


if __name__ == "__main__":
    main()
