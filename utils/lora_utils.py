import torch
import torch.nn as nn
import math


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        # 初始化 LoRA 矩阵
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        # Kaiming 初始化 A，全 0 初始化 B 以确保初始状态下 LoRA 分量为 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LinearWithLoRA(nn.Module):
    def __init__(self, original_linear, rank=8, alpha=16.0):
        super().__init__()
        self.original_linear = original_linear
        # 冻结原始权重
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        self.lora = LoRALayer(
            original_linear.in_features,
            original_linear.out_features,
            rank,
            alpha
        )

    def forward(self, x):
        # 结果为原始线性层输出 + LoRA 旁路输出
        return self.original_linear(x) + self.lora(x)


def inject_lora(model, rank=8, alpha=16.0, target_layers=(nn.Linear,)):
    """
    递归地将模型中的指定层替换为带 LoRA 的版本
    """
    for name, module in model.named_children():
        if isinstance(module, target_layers):
            # 执行替换
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 递归处理子模块
            inject_lora(module, rank, alpha, target_layers)
    return model


def get_lora_params(model):
    """
    辅助函数：获取所有需要训练的 LoRA 参数
    """
    return [p for n, p in model.named_parameters() if "lora_" in n]