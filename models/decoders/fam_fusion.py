import torch
import torch.nn as nn


class FAMFusion(nn.Module):
    """
    FAM (Feature Alignment & Fusion Module)
    策略: 拼接 + MLP 融合
    """

    def __init__(self, common_dim):
        super(FAMFusion, self).__init__()

        self.fusion_net = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim * 2),
            nn.BatchNorm1d(common_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(common_dim * 2, common_dim)  # 最终输出维度
        )

    def forward(self, f_vis, f_struct):
        # f_vis: (B, D)
        # f_struct: (B, D)

        # 1. 特征拼接
        concat_features = torch.cat((f_vis, f_struct), dim=-1)  # (B, 2D)

        # 2. 深度融合
        f_final = self.fusion_net(concat_features)

        return f_final