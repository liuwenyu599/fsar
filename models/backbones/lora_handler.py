import torch
import torch.nn as nn
import math


class ViewLoRALinear(nn.Module):
    def __init__(self, base_layer, rank=16, num_views=11):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.num_views = num_views

        # 🌟 为 11 个视角准备独立的低秩分解矩阵
        self.lora_A = nn.Parameter(torch.randn(num_views, self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(num_views, rank, self.out_features))

        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.view_modulator = True

    def forward(self, x, view_idx=None):
        # 1. 基础路径 (Frozen Backbone)
        result = self.base_layer(x)

        # 2. 视角感知路径
        if view_idx is not None:
            # x shape: [BF, J, C] (例如 [3000, 17, 256])
            # view_idx shape: [B] (例如 [50])
            BF = x.shape[0]
            B = view_idx.shape[0]

            # 🌟 核心：处理 DSTformer 内部展开的 Batch 维度
            if BF != B:
                frames = BF // B
                # 将 [B] 扩展并平铺为 [B*F]
                curr_view_idx = view_idx.unsqueeze(1).expand(-1, frames).reshape(-1)
            else:
                curr_view_idx = view_idx

            # 3. 提取当前 batch 对应的视角权重
            # A_v: [BF, in, r], B_v: [BF, r, out]
            A_v = self.lora_A[curr_view_idx]
            B_v = self.lora_B[curr_view_idx]

            # 4. 执行低秩矩阵乘法 (x @ A @ B)
            # [BF, J, in] @ [BF, in, r] -> [BF, J, r]
            # [BF, J, r] @ [BF, r, out] -> [BF, J, out]
            lora_path = torch.bmm(torch.bmm(x, A_v), B_v)

            return result + lora_path

        return result


# 🌟 必须包含这个函数，否则 train_structure.py 无法运行
def inject_view_lora(model, rank=16, num_views=11):
    """
    遍历模型，将指定的 Linear 层替换为 ViewLoRALinear
    """
    # 获取模型所在的设备 (cuda:0 或 cpu)
    device = next(model.parameters()).device
    replaced_count = 0

    # 深度优先遍历所有子模块
    for name, module in model.named_modules():
        # 目标层：DSTformer 的 QKV 投影和 MLP 的第一层
        if any(k in name for k in ["attn_s.qkv", "attn_t.qkv", "mlp_s.fc", "mlp_t.fc"]):
            # 获取父模块名称
            parent_name = ".".join(name.split(".")[:-1])
            layer_name = name.split(".")[-1]
            parent = dict(model.named_modules())[parent_name]

            # 获取原始线性层
            old_layer = getattr(parent, layer_name)

            # 🌟 创建视角感知的 LoRA 包装层并移动到正确设备
            new_layer = ViewLoRALinear(old_layer, rank=rank, num_views=num_views).to(device)

            # 执行手术式替换
            setattr(parent, layer_name, new_layer)
            replaced_count += 1

    print(f"🛡️ 视角感知 LoRA 注入成功: 替换了 {replaced_count} 个关键层 (Device: {device})")
    return model