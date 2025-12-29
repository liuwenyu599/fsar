import sys
import os

# 1. 自动处理路径：确保能找到根目录下的 models 和 dataloader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm

# ===== 模型与数据加载导入 =====
from models.backbones.structural import StructuralBackbone
from models.heads.ppm import PPMStructuredAggregator
from models.architecture_res import PAGFSLModel
from train_fusion import UncertaintyFusionGate
from dataloader.multloader import CASIABMultiDataset as StructDataset
from dataloader.silu_resnet_loader import CASIASiluDataset as VisDataset


# --------------------------------------------------
# 核心逻辑：一次性提取所有模式特征 (包含安全钳位逻辑 🔥)
# --------------------------------------------------
@torch.no_grad()
def extract_ablation_features(pose_seq, silu_seq, models, device):
    """
    提取多种模式特征用于对比：Pose-Only, Vis-Only, Naive, Adaptive (Safe)
    """
    s_model, s_ppm, v_back, v_hpm, v_proj, gate = models

    # --- 1. 结构特征 (Pose-Only) ---
    if len(pose_seq.shape) == 3:
        pose_seq = pose_seq.view(pose_seq.shape[0], -1)
    p_in = pose_seq.unsqueeze(0).to(device)
    f_s = s_ppm(s_model(p_in), torch.ones(1, p_in.shape[1]).to(device))
    f_s = F.normalize(f_s, dim=-1)  # [1, 512]

    # --- 2. 视觉特征 (Vis-Only) ---
    v_in = silu_seq.to(device)
    if v_in.shape[1] == 1:
        v_in = v_in.repeat(1, 3, 1, 1)
    f_v_map = v_back(v_in)
    f_v_hpm = v_hpm(f_v_map)
    f_v_pooled = f_v_hpm.max(dim=0, keepdim=True)[0]
    f_v = F.normalize(v_proj(f_v_pooled), dim=-1)  # [1, 512]

    # --- 3. 动态融合 (Adaptive with Bounded Constraint 🔥) ---
    # 获取原始门控输出
    _, alpha_raw_tensor = gate(f_s, f_v)
    alpha_raw = alpha_raw_tensor.item()

    # 执行安全钳位：限制在 [0.3, 0.7]，防止门控在极端视角下带偏模型
    alpha_safe = np.clip(alpha_raw, 0.3, 0.7)

    # 使用安全权重重新融合
    f_adaptive = alpha_safe * f_s + (1 - alpha_safe) * f_v
    f_adaptive = F.normalize(f_adaptive, dim=-1)

    # --- 4. Naive 融合 (简单平均作为 Baseline) ---
    f_naive = F.normalize((f_s + f_v) / 2.0, dim=-1)

    return f_s, f_v, f_naive, f_adaptive, alpha_safe


# --------------------------------------------------
# 批量评估函数
# --------------------------------------------------
def main_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n--- 🔬 PPGait: Bounded Ablation Study (5-Way 5-Shot) ---")
    print("Strategy: Alpha clamped to [0.3, 0.7] to prevent modality collapse.\n")

    # 1. 加载所有权重
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

    gate = UncertaintyFusionGate(512).to(device)
    gate.load_state_dict(fusion_ckpt["gate_state_dict"])

    models = (s_model, s_ppm, vis_all.vis_backbone, vis_all.hpm, vis_all.proj_vis, gate)
    for m in models:
        m.eval()

    # 3. 准备数据
    s_ds = StructDataset("/datasets/CASIA-B", mode="pose")
    v_ds = VisDataset("/datasets/CASIA-B/silu", target_len=8)
    common_ids = sorted(list(set(s_ds.all_subject_ids) & set(v_ds.all_subject_ids)))

    # 4. 评估配置
    target_views = ["090", "180", "054"]
    episodes_per_view = 100
    n_way, k_shot, q_query = 5, 5, 5

    view_results = {v: {"pose": [], "vis": [], "naive": [], "adaptive": [], "alpha": []} for v in target_views}

    for view in target_views:
        for ep in tqdm(range(episodes_per_view), desc=f"View {view}"):
            sampled_ids = random.sample(common_ids, n_way)

            protos = {"pose": [], "vis": [], "naive": [], "adaptive": []}
            queries = {"pose": [], "vis": [], "naive": [], "adaptive": []}
            query_gt = []

            for label_idx, sid in enumerate(sampled_ids):
                s_paths = [p for p in s_ds.all_sequences[sid] if view in p]
                v_dict = v_ds.all_sequences[sid]
                v_paths = [v_dict[k][view] for k in v_dict if view in v_dict[k]]

                max_samples = min(len(s_paths), len(v_paths))
                indices = random.sample(range(max_samples), min(max_samples, k_shot + q_query))

                # --- Support 阶段 ---
                s_feats, v_feats, n_feats, a_feats = [], [], [], []
                for i in indices[:k_shot]:
                    fs, fv, fn, fa, _ = extract_ablation_features(s_ds.load_pose(s_paths[i]),
                                                                  v_ds.load_sequence(v_paths[i]), models, device)
                    s_feats.append(fs);
                    v_feats.append(fv);
                    n_feats.append(fn);
                    a_feats.append(fa)

                protos["pose"].append(torch.cat(s_feats).mean(0, keepdim=True))
                protos["vis"].append(torch.cat(v_feats).mean(0, keepdim=True))
                protos["naive"].append(torch.cat(n_feats).mean(0, keepdim=True))
                protos["adaptive"].append(torch.cat(a_feats).mean(0, keepdim=True))

                # --- Query 阶段 ---
                for i in indices[k_shot:]:
                    fs, fv, fn, fa, al = extract_ablation_features(s_ds.load_pose(s_paths[i]),
                                                                   v_ds.load_sequence(v_paths[i]), models, device)
                    queries["pose"].append(fs);
                    queries["vis"].append(fv)
                    queries["naive"].append(fn);
                    queries["adaptive"].append(fa)
                    query_gt.append(label_idx)
                    view_results[view]["alpha"].append(al)

            # --- 计算 Acc ---
            for mode in ["pose", "vis", "naive", "adaptive"]:
                dist = torch.cdist(torch.cat(queries[mode]), torch.cat(protos[mode]))
                acc = (torch.argmin(dist, 1).cpu() == torch.tensor(query_gt)).float().mean().item()
                view_results[view][mode].append(acc)

    # 5. 打印对比表
    print("\n" + "=" * 90)
    print(
        f"{'View':<8} | {'Pose-Only':<12} | {'Vis-Only':<12} | {'Naive-Fuse':<12} | {'PPGait (Safe)':<15} | {'Avg Alpha'}")
    print("-" * 90)

    all_metrics = {"pose": [], "vis": [], "naive": [], "adaptive": []}
    for v in target_views:
        m = {mode: np.mean(view_results[v][mode]) * 100 for mode in ["pose", "vis", "naive", "adaptive"]}
        alpha_val = np.mean(view_results[v]["alpha"])
        print(
            f"{v:<8} | {m['pose']:>10.2f}% | {m['vis']:>10.2f}% | {m['naive']:>10.2f}% | {m['adaptive']:>13.2f}% | {alpha_val:.3f}")
        for mode in all_metrics:
            all_metrics[mode].append(m[mode])

    print("-" * 90)
    print(
        f"{'OVERALL':<8} | {np.mean(all_metrics['pose']):>10.2f}% | {np.mean(all_metrics['vis']):>10.2f}% | {np.mean(all_metrics['naive']):>10.2f}% | {np.mean(all_metrics['adaptive']):>13.2f}% | -")
    print("=" * 90)


if __name__ == "__main__":
    main_ablation()