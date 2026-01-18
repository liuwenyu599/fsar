import torch
import torch.nn as nn


class PlainLoRALinear(nn.Module):
    def __init__(self, base_layer, rank=16):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank

        # 标准 LoRA 参数 (初始化在 CPU)
        self.lora_A = nn.Parameter(torch.randn(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))

        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x, view_idx=None):  # 🌟 接收但忽略 view_idx，保持接口兼容
        result = self.base_layer(x)
        lora_path = x @ self.lora_A @ self.lora_B
        return result + lora_path


def inject_view_lora(model, rank=16):
    # 🌟 修复 Device 问题的关键：获取模型当前设备
    device = next(model.parameters()).device
    replaced_count = 0

    for name, module in model.named_modules():
        if any(k in name for k in ["attn_s.qkv", "attn_t.qkv", "mlp_s.fc"]):
            parent_name = ".".join(name.split(".")[:-1])
            layer_name = name.split(".")[-1]
            parent = dict(model.named_modules())[parent_name]

            old_layer = getattr(parent, layer_name)
            # 🌟 创建新层并立即同步到 GPU
            new_layer = PlainLoRALinear(old_layer, rank=rank).to(device)

            setattr(parent, layer_name, new_layer)
            replaced_count += 1

    print(f"🛡️ 对抗专用 LoRA 注入成功: 替换了 {replaced_count} 个关键层 (Device: {device})")
    return model