import torch
import torch.nn as nn
from models.backbones.dstformer import DSTformer  # 确保文件名一致


class StructuralBackbone(nn.Module):
    def __init__(self, embed_dim=256, dim_rep=512, depth=5, num_heads=8):
        super(StructuralBackbone, self).__init__()

        # 实例化 DSTformer 编码器
        self.encoder = DSTformer(
            dim_in=3,  # 输入坐标 (x, y, c)
            dim_out=0,  # Identity head
            dim_feat=embed_dim,  # 256
            dim_rep=dim_rep,  # 512
            depth=depth,  # 5
            num_heads=num_heads,  # 8
            mlp_ratio=4,
            num_joints=17,
            maxlen=243,
            att_fuse=True
        )
        self.out_dim = dim_rep

    def forward(self, x, view_idx=None):
        """
        x: (B, T, 51) -> 输入原始骨骼点序列
        view_idx: (B) -> 🌟 强视角感知参数，透传至内部 LoRA 层
        """
        B, T, _ = x.shape
        # 重塑形状以符合 DSTformer 要求: [B, T, 17, 3]
        x = x.view(B, T, 17, 3)

        # 🌟 核心改进：将 view_idx 传递给 encoder
        # 这要求你的 dstformer.py 中的 get_representation 方法也必须支持 view_idx 参数
        x = self.encoder.get_representation(x, view_idx=view_idx)

        # DSTformer 输出形状预期: [B, T, 17, 512]
        # 在关节维度 (J=17) 做平均池化，保留时间维度用于后续聚合
        x_mean = x.mean(dim=2)  # 输出: [B, T, 512]

        return x_mean