import torch
import torch.nn as nn
import timm


class VisualBackbone(nn.Module):
    """
    Visual Backbone: ResNet-18
    适合: 想要更强的特征提取能力，且显存允许的情况。
    """

    def __init__(self, model_name='resnet18', pretrained=False, lora_rank=None):
        super(VisualBackbone, self).__init__()

        print(f"Loading Visual Backbone: {model_name} (Pretrained={pretrained})...")

        # 加载 ResNet-18
        # num_classes=0: 移除分类头，直接输出特征
        # global_pool='avg': 输出 (B, 512)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        # ResNet-18 输出维度通常是 512
        self.out_dim = self.backbone.num_features
        print(f"Visual Backbone Output Dim: {self.out_dim}")

    def forward(self, x):
        # x: (B*T, 3, 224, 224)
        # out: (B*T, 512)
        return self.backbone(x)