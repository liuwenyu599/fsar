import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewAwarePPM(nn.Module):
    def __init__(self, feature_dim=512, num_views=11):
        super().__init__()
        # 🌟 核心：视角先验嵌入
        self.view_embedding = nn.Embedding(num_views, 128)

        # 🌟 核心：视角门控网络，根据视角动态调节 512 个通道的权重
        self.view_gate = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.Sigmoid()
        )
        self.feature_dim = feature_dim

    def forward(self, x, phase_w, view_idx):
        """
        x: [Batch, T, D]
        view_idx: [Batch] - 采样器返回的视角索引
        """
        # 1. 生成视角特定的特征掩码
        v_emb = self.view_embedding(view_idx)  # [B, 128]
        v_mask = self.view_gate(v_emb)  # [B, 512]

        # 2. 视角加权：在 180° 时自动抑制不稳定的关节点通道
        x = x * v_mask.unsqueeze(1)

        # 3. 基础相位加权聚合
        phase_w = F.softmax(phase_w, dim=1).unsqueeze(-1)
        return torch.sum(x * phase_w, dim=1)