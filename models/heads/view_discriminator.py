import torch.nn as nn
from utils.grl import grad_reverse

class ViewDiscriminator(nn.Module):
    def __init__(self, feature_dim=512, num_views=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_views)
        )

    def forward(self, x, alpha=1.0):
        # 1. 梯度逆转：让 Backbone 尽量提取不出视角信息
        x = grad_reverse(x, alpha)
        # 2. 判别视角
        return self.net(x)