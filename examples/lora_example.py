import torch
import torch.nn as nn
from src.core.lora import apply_lora_to_model

# Define a simple base model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)

# Create an instance of the base model
base_model = SimpleModel()

# Define LoRA configuration
lora_config = {
    'linear1': {'rank': 4, 'alpha': 0.5},
    'linear2': {'rank': 2, 'alpha': 0.3}
}

# Apply LoRA to the model
lora_model, lora_optimizer = apply_lora_to_model(base_model, lora_config)

# Simulate a training loop
for epoch in range(5):
    # Forward pass
    x = torch.randn(1, 10)
    output = lora_model(x)
    
    # Compute loss
    target = torch.randn(1, 5)
    loss = nn.functional.mse_loss(output, target)
    
    # Backward pass and optimization
    loss.backward()
    lora_optimizer.step()
    lora_optimizer.zero_grad()
    
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Merge LoRA weights for inference
lora_model.merge_and_unmerge()

# Use the model for inference
with torch.no_grad():
    test_input = torch.randn(1, 10)
    inference_output = lora_model(test_input)
    print("Inference output:", inference_output)

# Unmerge weights if you want to continue training
lora_model.merge_and_unmerge()