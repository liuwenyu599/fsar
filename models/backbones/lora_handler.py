import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, lora_alpha=16):
        super().__init__()
        self.original_layer = original_layer
        # 冻结原始层参数
        for p in self.original_layer.parameters():
            p.requires_grad = False

        in_f, out_f = original_layer.in_features, original_layer.out_features

        # 🌟 优化：预对齐矩阵形状，避免 forward 中的转置操作
        # A 矩阵负责降维，B 矩阵负责升维
        self.lora_A = nn.Parameter(torch.zeros((in_f, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, out_f)))
        self.scaling = lora_alpha / rank

        # 初始化：A 采用 Kaiming，B 初始化为 0 保证初始输出为 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 🌟 优化：直接计算矩阵乘法，移除 .to(device) 同步操作
        # x: [B, T, in_features]
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return self.original_layer(x) + lora_out


def inject_lora_to_motionbert(model, rank=8):
    """递归替换 DSTformer 中的 Linear 层为 LoRA 版本"""
    for param in model.parameters():
        param.requires_grad = False

    model_dict = dict(model.named_modules())
    injected_count = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 针对官方 DSTformer 核心计算层的命名匹配
            if any(target in name for target in ["qkv", "mlp_s.fc1", "mlp_t.fc1", "mlp_s.fc2", "mlp_t.fc2"]):
                name_list = name.split('.')
                parent_name = ".".join(name_list[:-1])
                layer_name = name_list[-1]
                parent = model_dict[parent_name]

                # 执行替换
                setattr(parent, layer_name, LoRALinear(module, rank=rank))
                injected_count += 1

    # 仅开启 LoRA 参数的梯度
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True

    print(f"✅ 官方 DSTformer LoRA 注入成功: 替换了 {injected_count} 个关键层")
    return model