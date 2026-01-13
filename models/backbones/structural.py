import torch
import torch.nn as nn
from models.backbones.dstformer import DSTformer  # 确保文件名一致


class StructuralBackbone(nn.Module):
    def __init__(self, embed_dim=256, dim_rep=512, depth=5, num_heads=8):
        super(StructuralBackbone, self).__init__()

        # 严格按照官方 YAML 和类定义参数实例化
        self.encoder = DSTformer(
            dim_in=3,  # (x, y, c)
            dim_out=0,  # 设置为0以使 head 为 Identity
            dim_feat=embed_dim,  # 256
            dim_rep=dim_rep,  # 512
            depth=depth,  # 5
            num_heads=num_heads,  # 8
            mlp_ratio=4,
            num_joints=17,
            maxlen=243,
            att_fuse=True  # 官方双流融合逻辑
        )
        self.out_dim = dim_rep

    def forward(self, x):
        # x: (B, T, 51) -> (B, F, J, C) 其中 F=T, J=17, C=3
        B, T, _ = x.shape
        x = x.view(B, T, 17, 3)
        # print(f"输入 DSTformer 之前的形状: {x.shape}")  # 预期: [B, 60, 17, 3]
        # 调用官方获取特征的方法，返回 [B, F, J, 512]
        x = self.encoder.get_representation(x)
        # print(f"DSTformer 提取特征后的形状: {x.shape}")  # 预期: [B, 60, 17, 512]
        # 在关节维度 (J) 做平均池化，保留时间维度用于后续 PPM 聚合
        # 输出 [B, T, 512]
        x_mean = x.mean(dim=2)
        # print(f"关节点聚合后的形状: {x_mean.shape}")  # 预期: [B, 60, 512]
        return x_mean