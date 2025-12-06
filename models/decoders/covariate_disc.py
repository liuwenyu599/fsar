import torch.nn as nn


class CovariateDiscriminator(nn.Module):
    """
    协变量判别器 (D_cov)
    用于对抗训练: 试图从视觉特征中预测协变量(如衣着)
    """

    def __init__(self, feature_dim, num_covariates):
        super(CovariateDiscriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, num_covariates)  # 输出 logits
        )

    def forward(self, x):
        return self.net(x)