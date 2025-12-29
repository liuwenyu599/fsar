import sys
import os

# 将项目根目录加入到搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from tabulate import tabulate  # 如果没有请 pip install tabulate，或者用下面的简易打印

# ===== 模型与数据加载导入 =====
from models.backbones.structural import StructuralBackbone
from models.heads.ppm import PPMStructuredAggregator
from models.architecture_res import PAGFSLModel
from train_fusion import UncertaintyFusionGate
from dataloader.multloader import CASIABMultiDataset as StructDataset
from dataloader.silu_resnet_loader import CASIASiluDataset as VisDataset


# --------------------------------------------------
# 核心推理逻辑
# --------------------------------------------------
@torch.no_grad()
def extract_fused_embedding(pose_seq, silu_seq, models, device):
    s_model, s_ppm, v_back, v_hpm, v_proj, gate = models

    if len(pose_seq.shape) == 3:
        pose_seq = pose_seq.view(pose_seq.shape[0], -1)
    p_in = pose_seq.unsqueeze(0).to(device)
    f_s = s_ppm(s_model(p_in), torch.ones(1, p_in.shape[1]).to(device))
    f_s = F.normalize(f_s, dim=-1)

    v_in = silu_seq.to(device)
    if v_in.shape[1] == 1: v_in = v_in.repeat(1, 3, 1, 1)

    f_v_maps = v_back(v_in)
    f_v_hpm = v_hpm(f_v_maps)
    f_v_pooled = f_v_hpm.max(dim=0, keepdim=True)[0]
    f_v = F.normalize(v_proj(f_v_pooled), dim=-1)

    f_fused, _ = gate(f_s, f_v)
    return f_fused


# --------------------------------------------------
# 视角对比评估主函数
# --------------------------------------------------
def main_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n--- 📊 PPGait: Multi-View Accuracy Comparison (5-Way 5-Shot) ---")

    # 1. 加载权重
    struct_ckpt = torch.load("logs/checkpoints/ppgait_struct_best.pth", map_location=device, weights_only=False)
    vis_ckpt = torch.load("logs/checkpoints/ppgait_vis_ema_best.pth", map_location=device, weights_only=False)
    fusion_ckpt = torch.load("logs/checkpoints/ppgait_fusion_final.pth", map_location=device, weights_only=False)

    # 2. 初始化模型
    s_model = StructuralBackbone(input_dim=struct_ckpt["config"]["input_dim"]).to(device)
    s_ppm = PPMStructuredAggregator(feature_dim=512).to(device)
    s_model.load_state_dict(struct_ckpt["backbone_state_dict"])
    s_ppm.load_state_dict(struct_ckpt["ppm_state_dict"])

    vis_all = PAGFSLModel(common_dim=512).to(device)
    vis_all.load_state_dict(vis_ckpt["state_dict"])

    fusion_gate = UncertaintyFusionGate(512).to(device)
    fusion_gate.load_state_dict(fusion_ckpt["gate_state_dict"])

    models = (s_model, s_ppm, vis_all.vis_backbone, vis_all.hpm, vis_all.proj_vis, fusion_gate)
    for m in models: m.eval()

    # 3. 准备数据
    s_ds = StructDataset("/datasets/CASIA-B", mode="pose")
    v_ds = VisDataset("/datasets/CASIA-B/silu", target_len=8)
    common_ids = sorted(list(set(s_ds.all_subject_ids) & set(v_ds.all_subject_ids)))

    # 4. 配置视角测试
    target_views = ["090", "180", "054"]
    episodes_per_view = 100  # 每个视角跑 100 次以获得统计显著性
    n_way, k_shot, q_query = 5, 5, 10

    view_results = {}

    for view in target_views:
        print(f"\n>>> Evaluating View: {view}")
        view_accs = []

        for ep in tqdm(range(episodes_per_view), desc=f"View {view}"):
            sampled_ids = random.sample(common_ids, n_way)
            prototypes, queries, query_gt = [], [], []

            for label_idx, sid in enumerate(sampled_ids):
                s_paths = [p for p in s_ds.all_sequences[sid] if view in p]
                v_dict = v_ds.all_sequences[sid]
                v_paths = [v_dict[k][view] for k in v_dict if view in v_dict[k]]

                max_samples = min(len(s_paths), len(v_paths))
                if max_samples < (k_shot + 1): continue  # 样本不足跳过

                indices = random.sample(range(max_samples), min(max_samples, k_shot + q_query))

                # Support -> Prototypes
                sup_feats = [extract_fused_embedding(s_ds.load_pose(s_paths[i]),
                                                     v_ds.load_sequence(v_paths[i]), models, device) for i in
                             indices[:k_shot]]
                prototypes.append(torch.cat(sup_feats).mean(dim=0, keepdim=True))

                # Query
                for i in indices[k_shot:]:
                    queries.append(extract_fused_embedding(s_ds.load_pose(s_paths[i]),
                                                           v_ds.load_sequence(v_paths[i]), models, device))
                    query_gt.append(label_idx)

            if not queries: continue

            proto_tensor = torch.cat(prototypes)
            query_tensor = torch.cat(queries)
            dists = torch.cdist(query_tensor, proto_tensor)
            preds = torch.argmin(dists, dim=1)
            acc = (preds.cpu() == torch.tensor(query_gt)).float().mean()
            view_accs.append(acc.item())

        view_results[view] = (np.mean(view_accs), np.std(view_accs))

    # 5. 打印对比表
    print("\n" + "=" * 50)
    print("📈 PPGait View-Wise Performance Comparison")
    print("=" * 50)
    table_data = []
    for view, (mean, std) in view_results.items():
        table_data.append([view, f"{mean * 100:.2f}%", f"±{std * 100:.2f}%"])

    # 计算 Overall
    all_means = [m for m, s in view_results.values()]
    table_data.append(["-------", "-------", "-------"])
    table_data.append(["OVERALL", f"{np.mean(all_means) * 100:.2f}%", "-"])

    # 简单的表格打印逻辑
    headers = ["Viewing Angle", "Mean Accuracy", "Std Dev"]
    print(f"{headers[0]:<15} | {headers[1]:<15} | {headers[2]:<10}")
    print("-" * 50)
    for row in table_data:
        print(f"{row[0]:<15} | {row[1]:<15} | {row[2]:<10}")
    print("=" * 50)


if __name__ == "__main__":
    main_eval()