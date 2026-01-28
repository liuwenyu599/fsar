import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.checkpoint import checkpoint


class HPM(nn.Module):
    """
    水平金字塔映射 (Horizontal Pyramid Mapping)
    将特征图沿高度方向切分为不同粒度的条带
    """

    def __init__(self, bin_list=[1, 2, 4, 8, 16]):
        super(HPM, self).__init__()
        self.bin_list = bin_list

    def forward(self, x):
        n, c, h, w = x.size()
        features = []
        for b in self.bin_list:
            # 沿 H 方向做自适应最大池化
            z = F.adaptive_max_pool2d(x, (b, 1))  # [N, C, b, 1]
            features.append(z.view(n, -1))  # [N, C*b]
        return torch.cat(features, dim=-1)


class VisualBackbone(nn.Module):
    """
    定制化 ResNet-18
    保持 Layer4 输出为 14x14 高分辨率特征图
    """

    def __init__(self, pretrained=True):
        super(VisualBackbone, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 强制修改 Layer4 步长为 1，防止分辨率损失
        for m in self.layer4.modules():
            if isinstance(m, nn.Conv2d):
                if m.stride == (2, 2): m.stride = (1, 1)
            if isinstance(m, nn.Sequential):  # 处理 downsample 模块
                for sub_m in m.modules():
                    if isinstance(sub_m, nn.Conv2d) and sub_m.stride == (2, 2):
                        sub_m.stride = (1, 1)

    def forward(self, x):
        if x.shape[1] == 1:  # 适配单通道剪影
            x = x.repeat(1, 3, 1, 1)

        # 训练时使用梯度检查点以节省显存
        if self.training:
            x = checkpoint(self.layer0, x, use_reentrant=False)
            x = checkpoint(self.layer1, x, use_reentrant=False)
            x = checkpoint(self.layer2, x, use_reentrant=False)
            x = checkpoint(self.layer3, x, use_reentrant=False)
            x = checkpoint(self.layer4, x, use_reentrant=False)
        else:
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x


class PureGaitResNet(nn.Module):
    """
    时序增强版纯视觉模型
    创新点：分组时序卷积 + 双路混合池化 (Max-Mean Pooling)
    """

    def __init__(self, common_dim=512):
        super().__init__()
        self.vis_backbone = VisualBackbone(pretrained=True)
        self.hpm = HPM(bin_list=[1, 2, 4, 8, 16])

        # 计算 HPM 后的总维度: 512 * (1+2+4+8+16) = 15872
        spatial_dim = 512 * sum([1, 2, 4, 8, 16])

        # 1. 🌟 轻量化时序卷积
        # groups=31 意味着每个 HPM 条带内部独立进行时序建模，符合生物力学逻辑
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(spatial_dim, spatial_dim, kernel_size=3, padding=1, groups=31),
            nn.BatchNorm1d(spatial_dim),
            nn.GELU()
        )

        # 2. 🌟 投影头：输入维度翻倍 (因为 Mean+Max 拼接)
        self.proj_vis = nn.Sequential(
            nn.Linear(spatial_dim * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, common_dim)
        )

    def forward(self, x_vis, batch_size=4):
        # x_vis 形状: [B, T, C, H, W]
        B, T, C, H, W = x_vis.shape
        x_vis_flat = x_vis.view(B * T, C, H, W)

        # --- A. 空间特征提取 ---
        features_vis = []
        for i in range(0, B * T, batch_size):
            x_batch = x_vis_flat[i:i + batch_size]
            f_map = self.vis_backbone(x_batch)  # [batch, 512, 14, 14]
            f_hpm = self.hpm(f_map)  # [batch, 15872]
            features_vis.append(f_hpm)

        # 重新整合时序维度: [B, T, 15872]
        f_vis_seq = torch.cat(features_vis, dim=0).view(B, T, -1)

        # --- B. 🌟 时序演化建模 ---
        # Conv1d 期望输入: [N, C, L] -> [B, Spatial_Dim, T]
        f_vis_seq = f_vis_seq.transpose(1, 2)
        f_vis_seq = self.temporal_conv(f_vis_seq)
        f_vis_seq = f_vis_seq.transpose(1, 2)  # [B, T, 15872]

        # --- C. 🌟 双路池化 (Cat-Pooling) ---
        # Max 捕捉步态瞬时张力，Mean 捕捉整体体型统计特征
        f_max = torch.max(f_vis_seq, dim=1)[0]
        f_mean = torch.mean(f_vis_seq, dim=1)

        # 拼接后的维度: 15872 * 2
        f_pool = torch.cat([f_max, f_mean], dim=-1)

        # --- D. 映射与 L2 归一化 ---
        f_vis = self.proj_vis(f_pool)
        return F.normalize(f_vis, p=2, dim=-1)


# --- 单元测试 ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 模拟输入: [Batch=2, T=16, C=3, H=224, W=224]
    mock_input = torch.randn(2, 16, 3, 224, 224).to(device)
    model = PureGaitResNet(common_dim=512).to(device)

    with torch.no_grad():
        output = model(mock_input)

    print(f"✅ 模型构建成功！")
    print(f"输入形状: {mock_input.shape}")
    print(f"输出特征维度: {output.shape}")  # 应为 [2, 512]