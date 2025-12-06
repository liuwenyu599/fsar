import torch
import torch.nn as nn
import timm
from utils.lora_utils import inject_lora


class VisualBackbone(nn.Module):
    """
    视觉流: EfficientViT / ViT + LoRA
    """

    def __init__(self, model_name='vit_tiny_patch16_224', pretrained=False, lora_rank=4):
        super(VisualBackbone, self).__init__()

        # 1. 加载预训练 ViT (移除分类头 num_classes=0)
        print(f"Loading Visual Backbone: {model_name}...")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # 2. 冻结所有参数 (W0 Frozen)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 3. 注入 LoRA (仅在 Linear 层注入)
        # 注意: 实际 ViT 中 QKV 通常是 Linear 层，inject_lora 会自动处理
        print(f"Injecting LoRA (rank={lora_rank})...")
        inject_lora(self.backbone, rank=lora_rank)

        # 获取输出维度 (timm 模型通常有 num_features 属性)
        self.out_dim = self.backbone.num_features

    def forward(self, x):
        # x: (B*T, C, H, W) - 轮廓图序列
        # 输出: (B*T, D_vis)
        features = self.backbone(x)
        return features