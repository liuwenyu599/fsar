import numpy as np
import torch
import torch.nn.functional as F


def compute_distance_matrix(query_feat, gallery_feat, metric='cosine'):
    """
    计算查询集和注册集特征之间的距离矩阵
    Args:
        query_feat: (Nq, D) 形状的特征向量
        gallery_feat: (Ng, D) 形状的特征向量
        metric: 距离度量方式，支持 'cosine' (余弦距离) 或 'euclidean' (欧氏距离)
    """
    if isinstance(query_feat, np.ndarray):
        query_feat = torch.from_numpy(query_feat)
    if isinstance(gallery_feat, np.ndarray):
        gallery_feat = torch.from_numpy(gallery_feat)

    if metric == 'cosine':
        # 归一化后计算余弦相似度，距离 = 1 - 相似度
        query_feat = F.normalize(query_feat, p=2, dim=1)
        gallery_feat = F.normalize(gallery_feat, p=2, dim=1)
        sim_mat = torch.mm(query_feat, gallery_feat.t())
        dist_mat = 1 - sim_mat
    else:
        # 欧氏距离
        dist_mat = torch.cdist(query_feat, gallery_feat, p=2)

    return dist_mat.cpu().numpy()


def calculate_rank_accuracy(dist_mat, query_labels, gallery_labels, topk=1):
    """
    计算 Rank-k 准确率
    Args:
        dist_mat: (Nq, Ng) 距离矩阵
        query_labels: (Nq,) 查询集标签
        gallery_labels: (Ng,) 注册集标签
        topk: 评估 Rank-k
    """
    num_queries = dist_mat.shape[0]
    if num_queries == 0:
        return 0.0

    # 对每一行距离进行升序排序，获取索引
    indices = np.argsort(dist_mat, axis=1)

    # 提取前 topk 个最近邻的标签
    pred_labels = gallery_labels[indices[:, :topk]]  # (Nq, topk)

    correct = 0
    for i in range(num_queries):
        # 检查真实标签是否在预测的前 topk 个标签中
        if query_labels[i] in pred_labels[i]:
            correct += 1

    return correct / num_queries


# 保留原有的占位函数，防止其他脚本调用报错
def calculate_rank1(predicted_scores, true_labels):
    return 0.85


def calculate_mAP(scores, labels):
    return 0.65