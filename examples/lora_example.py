import torch
from src.core.lora import LoRALayer, LoRAModel

def main():
    # Create a simple base model
    base_model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )

    # Create LoRA layers
    lora_layers = [
        LoRALayer(20, 20, r=4),
        LoRALayer(5, 5, r=2)
    ]

    # Create LoRA model
    lora_model = LoRAModel(base_model, lora_layers)

    # Generate some random input
    x = torch.randn(1, 10)

    # Get output from the LoRA model
    output = lora_model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

if __name__ == "__main__":
    main()