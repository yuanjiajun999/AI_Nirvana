import torch
from torch import nn


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r=8):
        super().__init__()
        self.r = r
        self.A = nn.Parameter(torch.randn(out_features, r))
        self.B = nn.Parameter(torch.randn(r, in_features))

    def forward(self, x):
        return self.A @ self.B @ x

class LoRAModel(nn.Module):
    def __init__(self, base_model, lora_layers):
        super().__init__()
        self.base_model = base_model
        self.lora_layers = nn.ModuleList(lora_layers)

    def forward(self, x):
        out = self.base_model(x)
        for lora_layer in self.lora_layers:
            out = lora_layer(out)
        return out