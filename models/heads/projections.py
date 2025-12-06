import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    非线性投影头 (MLP)
    作用: 将特征映射到公共维度空间 (Common Dimension)
    """

    def __init__(self, input_dim, common_dim=512, hidden_dim=None):
        super(ProjectionHead, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, common_dim)
        )

    def forward(self, x):
        return self.mlp(x)