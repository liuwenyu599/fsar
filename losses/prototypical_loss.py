import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalLoss(nn.Module):
    def __init__(self):
        super(PrototypicalLoss, self).__init__()

    def forward(self, support_features, query_features, labels_query, n_way, k_shot):
        """
        Args:
            support_features: (N_way * K_shot, D)
            query_features: (N_way * Q_query, D)
            labels_query: (N_way * Q_query) 真实的类别索引
        """
        # 获取特征维度 D (例如 512)
        feat_dim = support_features.size(1)

        # ---------------------------------------------------------
        # 1. 计算原型 (Prototypes)
        # ---------------------------------------------------------
        # 重塑为 (N_way, K_shot, D)
        support_reshaped = support_features.reshape(n_way, k_shot, feat_dim)

        # dim=1 指的是 K_shot 那个维度，求均值后得到 (N_way, D)
        # 这就是每个类别的“代表”
        prototypes = support_reshaped.mean(dim=1)

        # ---------------------------------------------------------
        # 2. 计算欧氏距离的平方 (防爆核心)
        # ---------------------------------------------------------
        # 为了防止 nan，我们不直接使用 torch.cdist(p=2.0)
        # 也不要在距离里开根号。直接计算 (a-b)^2 的展开式：
        # 使用广播机制计算: (Query - Prototype)^2

        # query_features: (N_q, D) -> (N_q, 1, D)
        # prototypes:     (N_way, D) -> (1, N_way, D)
        query_ext = query_features.unsqueeze(1)
        proto_ext = prototypes.unsqueeze(0)

        # 计算平方差之和: (N_query, N_way)
        # 🌟 重点：计算 (x-y)^2 而不是 sqrt((x-y)^2)，这在反向传播时极其稳定
        dists = torch.sum((query_ext - proto_ext) ** 2, dim=2)

        # ---------------------------------------------------------
        # 3. 计算 Loss (Logits = -Distances)
        # ---------------------------------------------------------
        # 距离越小，概率越大。所以取负值作为 Softmax 的输入。
        logits = -dists

        # F.cross_entropy 会自动处理 LogSoftmax
        loss = F.cross_entropy(logits, labels_query)

        # ----------------------------------------
        # 4. 计算准确率 (监控用)
        # ----------------------------------------
        with torch.no_grad():
            _, predictions = logits.max(dim=1)
            acc = (predictions == labels_query).float().mean()

        return loss, acc