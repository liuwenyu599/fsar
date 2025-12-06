import torch
import torch.nn as nn


class StructuralBackbone(nn.Module):
    """
    结构流: MotionBERT Encoder (简化版实现)
    实际项目中应加载 MotionBERT 预训练权重
    """

    def __init__(self, input_dim=68, embed_dim=512, depth=4, num_heads=8):
        super(StructuralBackbone, self).__init__()

        # 1. 骨架嵌入层 (Input -> Feature)
        self.embedding = nn.Linear(input_dim, embed_dim)

        # 2. 位置编码 (简化为可学习参数)
        self.pos_embed = nn.Parameter(torch.zeros(1, 100, embed_dim))  # 假设最大长度100

        # 3. Transformer Encoder (模拟 MotionBERT 核心)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.out_dim = embed_dim

    def forward(self, x):
        # x: (B, T, Input_Dim) - 3D 骨架序列
        B, T, _ = x.shape

        # Embedding
        x = self.embedding(x)

        # Add Positional Encoding (Broadcasting)
        x = x + self.pos_embed[:, :T, :]

        # Transformer Forward
        output = self.encoder(x)

        return output  # (B, T, embed_dim)