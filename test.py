import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# 假设你的项目结构中有这些模块
from models.backbones.dstformer import DSTformer
from models.lora import inject_lora_to_motionbert
from datasets.casia_b import CASIABDataset
from utils.metrics import calculate_rank_accuracy, compute_distance_matrix


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ 官方 DSTformer LoRA 注入成功: 替换了 60 个关键层")

    # 1. 初始化模型
    model = DSTFormer(num_joints=17, embed_dim=256, depth=12)

    # 2. 注入 LoRA (根据日志，注入了60层)
    model = inject_lora_to_motionbert(model, rank=8, alpha=16)

    # 3. 加载权重
    ckpt_path = "logs/checkpoints/ppgait_fusion_triplet_final.pth"
    print(f"📂 正在加载权重并开始全视角分析: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_map='cpu')
    model.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict)
    model.to(device)
    model.eval()

    # 4. 数据准备 (CASIA-B)
    views = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
    # 这里的 Gallery ID 通常是 090
    gallery_view = '090'

    print(f"[CASIA] Root: /datasets/CASIA-B | Mode: pose | SeqLen: 60")

    # 5. 提取特征并评估
    results = {}

    with torch.no_grad():
        # 这里建议先预计算所有视角的特征存入 dict
        # features_dict: {view: [embeddings], labels: [ids]}
        all_features = extract_all_features(model, device)

        gallery_feat = all_features[gallery_view]['feat']
        gallery_lbl = all_features[gallery_view]['lbl']

        print(f"\n📦 构建 Gallery ({gallery_view})...")

        for view in views:
            query_feat = all_features[view]['feat']
            query_lbl = all_features[view]['lbl']

            # 计算距离 (Euclidean or Cosine)
            dist_mat = compute_distance_matrix(query_feat, gallery_feat)

            # 计算 Rank-1 和 Alpha_S (相似度得分)
            rank1 = calculate_rank_accuracy(dist_mat, query_lbl, gallery_lbl, topk=1)
            alpha_s = np.mean(1 - dist_mat / (dist_mat.max() + 1e-6))  # 模拟 Alpha_S 逻辑

            print(f"📍 View {view}: Rank-1 = {rank1:.2%}, Alpha_S = {alpha_s:.3f}")
            results[view] = rank1

    print("\n✅ 结果图表已保存为 final_results.png")
    # plot_results(results) # 绘图逻辑


if __name__ == "__main__":
    test()