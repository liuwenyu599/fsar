import torch
import torch.nn as nn
from torchvision import models
import sys
import os

# --- 路径修复逻辑：确保直接运行此文件时能找到项目根目录 ---
try:
    # 正常项目调用
    from models.backbones.visual_resnet import *
except (ModuleNotFoundError, ImportError):
    # 直接运行此脚本进行测试时
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class VisualBackbone(nn.Module):
    """
    基于 ResNet-18 的步态特征提取器
    [核心修改]：移除了最后两层（avgpool 和 fc），输出 4D 特征图以适配 HPM。
    """

    def __init__(self, lora_rank=16, pretrained=False):
        super(VisualBackbone, self).__init__()

        # 1. 加载 ResNet-18 基础架构
        # 若 PyTorch 版本较高，建议使用 weights=models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(pretrained=pretrained)

        # 2. 提取卷积层（Layer0 到 Layer4）
        # 移除 avgpool 和 fc 层，保留空间结构 [B, 512, 7, 7]
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

        # ResNet18 layer4 的输出通道数
        self.out_dim = 512

        # 3. LoRA rank 占位逻辑（如果后续需要微调）
        if lora_rank > 0:
            self.lora_rank = lora_rank
            # 这里可以根据需要添加 LoRA 权重注入逻辑

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像 [B, 3, 224, 224]
        Returns:
            f_map: 空间特征图 [B, 512, 7, 7]
        """
        # 输入检查：如果输入是单通道轮廓图，则自动扩展为 3 通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        f_map = self.features(x)
        return f_map


# --- 单元测试代码 ---
if __name__ == "__main__":
    # 模拟输入：1个 batch, 3通道, 224x224 轮廓图
    mock_input = torch.randn(1, 3, 224, 224)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VisualBackbone(pretrained=False).to(device)
    model.eval()

    print("--- 🚀 视觉骨干网络 (HPM 适配版) 测试 ---")
    with torch.no_grad():
        output = model(mock_input.to(device))

    print(f"输入形状: {mock_input.shape}")
    print(f"输出特征图形状: {output.shape}")

    # 验证维度是否正确
    expected_shape = (1, 512, 7, 7)
    if output.shape == expected_shape:
        print("✅ 维度验证成功！特征图已保留，可以适配 HPM。")
    else:
        print(f"❌ 维度验证失败！预期 {expected_shape}, 实际 {output.shape}")