import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 层
    逻辑: forward = original_linear(x) + (x @ W_A @ W_B) * scaling
    """

    def __init__(self, in_features, out_features, rank=4, alpha=16.0):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 1. 低秩矩阵 A (降维) 和 B (升维)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # 2. 初始化 (LoRA 标准初始化: A 高斯分布, B 全零)
        # 这样初始状态下，LoRA 的输出为 0，不影响预训练模型
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 计算 LoRA 分支: x @ A @ B
        # 维度变化: (B, ..., in) -> (B, ..., r) -> (B, ..., out)
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LinearWithLoRA(nn.Module):
    """
    包装器: 将标准 Linear 层替换为 Linear + LoRA
    """

    def __init__(self, original_linear, rank=4, alpha=16.0):
        super(LinearWithLoRA, self).__init__()
        self.original_linear = original_linear
        # 冻结原始参数
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        # 新增 LoRA 层
        self.lora = LoRALayer(
            original_linear.in_features,
            original_linear.out_features,
            rank,
            alpha
        )

    def forward(self, x):
        return self.original_linear(x) + self.lora(x)


def inject_lora(model, rank=4, alpha=16.0, target_layers=[nn.Linear]):
    """
    辅助函数: 遍历模型，将指定的 Linear 层替换为 LoRA 层
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # 替换 Linear 层
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 递归处理子模块
            inject_lora(module, rank, alpha, target_layers)
    return model