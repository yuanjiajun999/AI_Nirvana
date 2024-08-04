import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

print("lora.py is being imported")

class LoRALayer(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.1,
        merge_weights: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.merge_weights = merge_weights

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
        self.scaling = self.alpha / self.rank

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merge_weights:
            return F.linear(x, self.merged_weight, self.bias)
        else:
            return (
                F.linear(x, self.weight, self.bias)
                + self.dropout(x) @ self.lora_A.t() @ self.lora_B.t() * self.scaling
            )

    def merge_lora_weights(self):
        if not self.merge_weights:
            self.merged_weight = self.weight + (self.lora_B @ self.lora_A) * self.scaling
            self.merge_weights = True

    def unmerge_lora_weights(self):
        if self.merge_weights:
            self.merge_weights = False
            if hasattr(self, 'merged_weight'):
                del self.merged_weight

class LoRAModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        lora_config: Dict[str, Dict],
        merge_weights: bool = False
    ):
        super().__init__()
        self.base_model = base_model
        self.lora_config = lora_config
        self.merge_weights = merge_weights
        self.lora_layers = nn.ModuleDict()

        self._add_lora_layers()

    def _add_lora_layers(self):
        for name, module in self.base_model.named_modules():
            if name in self.lora_config:
                config = self.lora_config[name]
                if isinstance(module, nn.Linear):
                    lora_layer = LoRALayer(
                        module.in_features,
                        module.out_features,
                        rank=config.get('rank', 4),
                        alpha=config.get('alpha', 1.0),
                        dropout=config.get('dropout', 0.1),
                        merge_weights=self.merge_weights
                    )
                    self.lora_layers[name] = lora_layer

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def merge_and_unmerge(self):
        for lora_layer in self.lora_layers.values():
            if self.merge_weights:
                lora_layer.merge_lora_weights()
            else:
                lora_layer.unmerge_lora_weights()

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if "lora_" in n]

class LoRAOptimizer:
    def __init__(
        self,
        model: LoRAModel,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        optimizer_class: Optional[type] = None
    ):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_class = optimizer_class or torch.optim.AdamW

        self.optimizer = self._create_optimizer()

    def _create_optimizer(self):
        trainable_params = self.model.get_trainable_parameters()
        return self.optimizer_class(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

def apply_lora_to_model(
    base_model: nn.Module,
    lora_config: Dict[str, Dict],
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    merge_weights: bool = False
) -> Tuple[LoRAModel, LoRAOptimizer]:
    lora_model = LoRAModel(base_model, lora_config, merge_weights)
    lora_optimizer = LoRAOptimizer(lora_model, lr, weight_decay)
    return lora_model, lora_optimizer