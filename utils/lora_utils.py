import torch
import torch.nn as nn
import math
import yaml
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LinearWithLoRA(nn.Module):
    def __init__(self, original_linear, rank=8, alpha=16.0):
        super().__init__()
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        self.lora = LoRALayer(original_linear.in_features, original_linear.out_features, rank, alpha)

    def forward(self, x):
        return self.original_linear(x) + self.lora(x)


def inject_lora(model, rank=8, alpha=16.0, target_layers=[nn.Linear]):
    for name, module in model.named_children():
        if isinstance(module, tuple(target_layers)):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            inject_lora(module, rank, alpha, target_layers)
    return model
