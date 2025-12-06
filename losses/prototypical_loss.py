import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalLoss(nn.Module):
    """
    原型损失 (Prototypical Loss)
    论文: Prototypical Networks for Few-shot Learning (NeurIPS 2017)

    核心逻辑:
    1. 基于 Support Set 计算每个类别的原型 (Prototype)。
    2. 计算 Query Set 到所有原型的欧氏距离。
    3. 通过 Softmax 将距离转化为概率，最小化正确类别的负对数概率。
    """

    def __init__(self):
        super(PrototypicalLoss, self).__init__()

    def forward(self, support_features, query_features, labels_query,n_way, k_shot):
        """
        Args:
            support_features: (N_way * K_shot, D) - 支撑集特征，假设已按类别排序
            query_features: (N_way * Q_query, D) - 查询集特征
            n_way: 类别数量 N
            k_shot: 每个类别的样本数量 K
        Returns:
            loss: 标量损失值
            acc: 当前 Batch 的识别准确率 (用于监控)
        """
        # 获取特征维度 D
        dim = support_features.size(1)

        # ----------------------------------------
        # 1. 计算原型 (Prototypes)
        # ----------------------------------------
        # 将支撑集特征重塑为 (N_way, K_shot, D)
        # 假设输入顺序: [Class0_0...Class0_K, Class1_0...Class1_K, ...]
        if support_features.shape[0] != n_way * k_shot:
            raise ValueError(f"Support features size {support_features.shape[0]} does not match N*K ({n_way}*{k_shot})")

        support_features = support_features.reshape(n_way, k_shot, dim)

        # 对 K_shot 维度求平均 -> 得到 N 个原型 (N_way, D)
        prototypes = support_features.mean(dim=1)

        # ----------------------------------------
        # 2. 计算距离矩阵 (Distances)
        # ----------------------------------------
        # 计算 Query (N*Q, D) 到 Prototypes (N, D) 的距离
        # torch.cdist 默认计算 p=2 的欧氏距离
        # 输出 dists shape: (Total_Queries, N_way)
        dists = torch.cdist(query_features, prototypes, p=2.0)

        # 原型网络论文中使用距离的平方作为 Logits
        # Logits = -Distance^2
        logits = -(dists ** 2)

        # ----------------------------------------
        # 3. 生成真实标签 (Targets) & 计算损失
        # ----------------------------------------
        # 计算每个类别的查询样本数 Q
        n_query = query_features.size(0)
        q_query_per_class = n_query // n_way
        target_labels = labels_query
        # CrossEntropyLoss 内部会自动做 LogSoftmax
        loss = F.cross_entropy(logits, target_labels)

        # ----------------------------------------
        # 4. 计算准确率 (监控用)
        # ----------------------------------------
        # 找到概率最大的类别索引
        _, predictions = logits.max(dim=1)
        acc = (predictions == target_labels).float().mean()

        return loss, acc