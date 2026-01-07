import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, lora_alpha=16):
        super().__init__()
        self.original_layer = original_layer
        for p in self.original_layer.parameters():
            p.requires_grad = False

        in_f, out_f = original_layer.in_features, original_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros((rank, in_f)))
        self.lora_B = nn.Parameter(torch.zeros((out_f, rank)))
        self.scaling = lora_alpha / rank

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 🔥 关键修复：确保 LoRA 参数与输入数据在同一设备
        # 使用 .to(x.device) 确保万无一失
        lora_out = (x @ self.lora_A.t().to(x.device) @ self.lora_B.t().to(x.device)) * self.scaling
        return self.original_layer(x) + lora_out


def inject_lora_to_motionbert(model, rank=8):
    for param in model.parameters():
        param.requires_grad = False

    model_dict = dict(model.named_modules())
    injected_count = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 针对官方 DSTformer 的命名
            if any(target in name for target in ["qkv", "mlp_s.fc1", "mlp_t.fc1", "mlp_s.fc2", "mlp_t.fc2"]):
                name_list = name.split('.')
                parent_name = ".".join(name_list[:-1])
                layer_name = name_list[-1]
                parent = model_dict[parent_name]

                setattr(parent, layer_name, LoRALinear(module, rank=rank))
                injected_count += 1

    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True

    print(f"✅ 官方 DSTformer LoRA 注入成功: 替换了 {injected_count} 个关键层")
    return model