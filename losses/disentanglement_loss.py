import torch
import torch.nn as nn
import torch.nn.functional as F


class DisentanglementLoss(nn.Module):
    """
    对抗解耦损失 (Adversarial Disentanglement Loss)

    包含两部分对抗逻辑:
    1. 判别器损失 (Train Discriminator): 希望 D_cov 能准确预测协变量 (如识别出穿了大衣)。
    2. 提取器损失 (Train Extractor/Backbone): 希望 D_cov 无法预测协变量 (输出均匀分布/最大化熵)。
    """

    def __init__(self, num_covariates):
        super(DisentanglementLoss, self).__init__()
        self.num_covariates = num_covariates
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, disc_logits, labels, for_extractor=False):
        """
        Args:
            disc_logits: (B, Num_Covariates) - 协变量判别器的输出 logits
            labels: (B,) - 真实的协变量标签 (例如 0:Normal, 1:Bag, 2:Coat)
            for_extractor: bool
                - False: 计算判别器的 Loss (使其变强)
                - True:  计算提取器的 Loss (使其能骗过判别器)
        Returns:
            loss: 标量
        """
        if not for_extractor:
            # --- 模式 A: 训练判别器 (Discriminator Step) ---
            # 目标: 最小化分类误差 (判别器越准越好)
            # 就像训练一个普通的分类器
            loss = self.criterion(disc_logits, labels)
            return loss

        else:
            # --- 模式 B: 训练提取器 (Extractor/Adversarial Step) ---
            # 目标: 最大化判别器的困惑度 (让判别器瞎猜)
            # 理想状态: 判别器输出的概率分布接近均匀分布 (Uniform Distribution)
            # 即 P(y|x) = 1 / Num_Classes

            # 方法: 最小化输出分布与均匀分布之间的 KL 散度
            # 这等价于最大化输出分布的信息熵 (Entropy)

            # 1. 计算 Log Softmax
            log_probs = F.log_softmax(disc_logits, dim=1)

            # 2. 定义均匀分布概率 (常数)
            uniform_prob = 1.0 / self.num_covariates

            # 3. 计算 "Uniform Entropy Loss"
            # 我们希望 -sum(p_uniform * log_prob) 越小越好
            # 这会迫使 log_prob 趋向于 log(1/N)，即概率趋向于均匀
            loss = -torch.mean(torch.sum(uniform_prob * log_probs, dim=1))

            return loss