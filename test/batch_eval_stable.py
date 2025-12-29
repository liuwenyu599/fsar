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
# 基础特征提取：只拿原始 Embedding 和 Gate 的建议 Alpha
# --------------------------------------------------
@torch.no_grad()
def get_raw_features(pose_seq, silu_seq, models, device):
    s_model, s_ppm, v_back, v_hpm, v_proj, gate = models

    # 提取结构流原始特征 (f_s)
    if len(pose_seq.shape) == 3:
        pose_seq = pose_seq.view(pose_seq.shape[0], -1)
    p_in = pose_seq.unsqueeze(0).to(device)
    f_s = F.normalize(s_ppm(s_model(p_in), torch.ones(1, p_in.shape[1]).to(device)), dim=-1)

    # 提取视觉流原始特征 (f_v)
    v_in = silu_seq.to(device)
    if v_in.shape[1] == 1:
        v_in = v_in.repeat(1, 3, 1, 1)
    f_v_pooled = v_hpm(v_back(v_in)).max(dim=0, keepdim=True)[0]
    f_v = F.normalize(v_proj(f_v_pooled), dim=-1)

    # 获取 Gate 建议的原始 Alpha
    _, alpha_raw = gate(f_s, f_v)

    return f_s, f_v, alpha_raw.item()


# --------------------------------------------------
# 稳定评估主函数
# --------------------------------------------------
def main_stable_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n--- 📊 PPGait: Stable Episode-Level Evaluation (5-Way 5-Shot) ---")
    print("Logic: Using Consensus Alpha per Episode to maintain Metric Space Stability.\n")

    # 1. 加载权重 (PyTorch 2.6 兼容)
    struct_ckpt = torch.load("logs/checkpoints/ppgait_struct_best.pth", map_location=device, weights_only=False)
    vis_ckpt = torch.load("logs/checkpoints/ppgait_vis_ema_best.pth", map_location=device, weights_only=False)
    fusion_ckpt = torch.load("logs/checkpoints/ppgait_fusion_final.pth", map_location=device, weights_only=False)

    # 2. 初始化并配置模型
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

    # 4. 配置测试
    target_views = ["090", "180", "054"]
    episodes_per_view = 100
    n_way, k_shot, q_query = 5, 5, 5

    view_final_results = {}

    for view in target_views:
        view_accs = []
        episode_alphas_log = []

        for ep in tqdm(range(episodes_per_view), desc=f"View {view}"):
            sampled_ids = random.sample(common_ids, n_way)

            episode_samples = []  # 暂存本 Episode 所有的原始特征
            suggested_alphas = []  # 暂存 Gate 建议的权重

            # 第一步：收集本 Episode 所有样本的原始数据
            for label, sid in enumerate(sampled_ids):
                s_paths = [p for p in s_ds.all_sequences[sid] if view in p]
                v_dict = v_ds.all_sequences[sid]
                v_paths = [v_dict[k][view] for k in v_dict if view in v_dict[k]]

                max_s = min(len(s_paths), len(v_paths))
                indices = random.sample(range(max_s), min(max_s, k_shot + q_query))

                for i, idx in enumerate(indices):
                    fs, fv, al = get_raw_features(s_ds.load_pose(s_paths[idx]),
                                                  v_ds.load_sequence(v_paths[idx]),
                                                  models, device)
                    episode_samples.append({
                        'f_s': fs, 'f_v': fv, 'label': label, 'is_sup': i < k_shot
                    })
                    suggested_alphas.append(al)

            # 第二步：计算本 Episode 的“共识 Alpha” 🔥
            # 使用中位数过滤异常决策，并限制在安全区间 [0.3, 0.7]
            consensus_alpha = np.clip(np.median(suggested_alphas), 0.3, 0.7)
            episode_alphas_log.append(consensus_alpha)

            # 第三步：使用共识 Alpha 进行统一空间融合
            protos = []
            queries = []
            query_labels = []

            for label in range(n_way):
                # 融合 Support 形成类原型
                sup_list = [F.normalize(consensus_alpha * d['f_s'] + (1 - consensus_alpha) * d['f_v'], dim=-1)
                            for d in episode_samples if d['label'] == label and d['is_sup']]
                protos.append(torch.cat(sup_list).mean(0, keepdim=True))

                # 融合 Query
                que_list = [F.normalize(consensus_alpha * d['f_s'] + (1 - consensus_alpha) * d['f_v'], dim=-1)
                            for d in episode_samples if d['label'] == label and not d['is_sup']]
                for q in que_list:
                    queries.append(q)
                    query_labels.append(label)

            # 第四步：在稳定的度量空间内计算距离
            proto_tensor = torch.cat(protos)
            query_tensor = torch.cat(queries)
            dists = torch.cdist(query_tensor, proto_tensor)
            preds = torch.argmin(dists, dim=1)

            acc = (preds.cpu() == torch.tensor(query_labels)).float().mean().item()
            view_accs.append(acc)

        view_final_results[view] = (np.mean(view_accs), np.std(view_accs), np.mean(episode_alphas_log))

    # 5. 打印最终结果表
    print("\n" + "=" * 70)
    print(f"{'Viewing Angle':<15} | {'Stable Accuracy':<18} | {'Consensus Alpha'}")
    print("-" * 70)
    all_accs = []
    for v in target_views:
        acc, std, alpha = view_final_results[v]
        print(f"{v:<15} | {acc * 100:>12.2f}% ±{std * 100:<5.2f} | {alpha:.3f}")
        all_accs.append(acc)
    print("-" * 70)
    print(f"{'OVERALL':<15} | {np.mean(all_accs) * 100:>12.2f}%            | -")
    print("=" * 70)


if __name__ == "__main__":
    main_stable_eval()