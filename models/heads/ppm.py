import torch
import torch.nn as nn


class PPMStructuredAggregator(nn.Module):
    """
    PPM (Physics-Prior Modeling) 模块
    作用: 根据 GPE 生成的相位权重，对时序特征进行加权聚合。
    """

    def __init__(self, feature_dim):
        super(PPMStructuredAggregator, self).__init__()
        # 聚合后的特征精炼层 (Optional)
        self.refiner = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )

    def forward(self, motion_features, phase_weights):
        """
        Args:
            motion_features: (B, T, D) - 骨架流特征序列
            phase_weights: (B, T) - GPE 产生的权重 (支撑期权重高, 摆动期权重低)
        Returns:
            F_struct: (B, D) - 聚合后的鲁棒特征
        """
        # 1. 扩展权重维度: (B, T) -> (B, T, 1) -> (B, T, D)
        weights_expanded = phase_weights.unsqueeze(-1)

        # 2. 加权求和
        # 广播机制: weights 会自动复制到 D 维度
        weighted_features = motion_features * weights_expanded
        sum_features = weighted_features.sum(dim=1)  # (B, D)

        # 3. 归一化 (除以权重的和，相当于加权平均)
        sum_weights = phase_weights.sum(dim=1).unsqueeze(-1) + 1e-6
        F_struct_raw = sum_features / sum_weights

        # 4. 精炼
        F_struct = self.refiner(F_struct_raw)

        return F_struct