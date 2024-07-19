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
        A (nn.Parameter): The first update matrix.
        B (nn.Parameter): The second update matrix.
    """

    def __init__(self, in_features, out_features, r=8):
        super().__init__()
        self.r = r
        self.A = nn.Parameter(torch.randn(out_features, r))
        self.B = nn.Parameter(torch.randn(r, in_features))

    def forward(self, x):
        """
        Compute the forward pass of the LoRA layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after applying the LoRA update.
        """
        return self.A @ self.B @ x

class LoRAModel(nn.Module):
    """
    A model that incorporates LoRA layers into a base model.

    This model wraps a base model and applies LoRA layers after it.

    Args:
        base_model (nn.Module): The original model to be adapted.
        lora_layers (list): A list of LoRALayer instances to be applied.

    Attributes:
        base_model (nn.Module): The original model.
        lora_layers (nn.ModuleList): The list of LoRA layers.
    """

    def __init__(self, base_model, lora_layers):
        super().__init__()
        self.base_model = base_model
        self.lora_layers = nn.ModuleList(lora_layers)

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
            out = lora_layer(out)
        return out