import torch
from torch import nn

class LoRALayer(nn.Module):
    """
    Implements a Low-Rank Adaptation (LoRA) layer.

    This layer adds a low-rank update to the original weight matrix,
    allowing for efficient fine-tuning of large pre-trained models.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        r (int): Rank of the update matrices. Default is 8.

    Attributes:
        lora_A (nn.Parameter): The first update matrix of shape (r, in_features).
        lora_B (nn.Parameter): The second update matrix of shape (out_features, r).
        scale (float): Scaling factor for the LoRA update.
        linear (nn.Linear): The original linear transformation.
    """

    def __init__(self, in_features, out_features, r=8):
        super().__init__()
        self.r = r
        self.in_features = in_features
        self.out_features = out_features
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, r))
        self.scale = 0.01
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Compute the forward pass of the LoRA layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return self.linear(x) + self.scale * (self.lora_B @ self.lora_A @ x.T).T

class LoRAModel(nn.Module):
    """
    A model that incorporates LoRA layers into a base model.

    This model wraps a base model and applies LoRA layers after it.

    Args:
        base_model (nn.Module): The original model to be adapted.
        lora_config (list): A list of tuples, each containing (in_features, out_features, r)
                            for each LoRA layer to be applied.

    Attributes:
        base_model (nn.Module): The original model.
        lora_layers (nn.ModuleList): The list of LoRA layers.
    """

    def __init__(self, base_model, lora_config):
        super().__init__()
        self.base_model = base_model
        self.lora_layers = nn.ModuleList([
            LoRALayer(in_features, out_features, r)
            for in_features, out_features, r in lora_config
        ])

    def forward(self, x):
        """
        Compute the forward pass of the LoRA model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after applying the base model and LoRA layers.
        """
        out = self.base_model(x)
        for lora_layer in self.lora_layers:
            residual = out
            out = lora_layer(out)
            out = out + residual  # Add residual connection
        return out