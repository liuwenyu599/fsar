import sys
import os

# 1. 自动处理路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from tqdm import tqdm

# ===== 模型与数据加载导入 =====
from models.backbones.structural import StructuralBackbone
from models.heads.ppm import PPMStructuredAggregator
from models.architecture_res import PAGFSLModel
from train_fusion import UncertaintyFusionGate
from dataloader.multloader import CASIABMultiDataset as StructDataset
from dataloader.silu_resnet_loader import CASIASiluDataset as VisDataset


@torch.no_grad()
def extract_feature(pose_path, silu_path, s_ds, v_ds, models, device):
    s_model, s_ppm, v_back, v_hpm, v_proj, gate = models

    # 1. 结构流特征
    pose_seq = s_ds.load_pose(pose_path)
    if len(pose_seq.shape) == 3:
        pose_seq = pose_seq.view(pose_seq.shape[0], -1)
    p_in = pose_seq.unsqueeze(0).to(device)
    f_s = F.normalize(s_ppm(s_model(p_in), torch.ones(1, p_in.shape[1]).to(device)), dim=-1)

    # 2. 视觉流特征
    silu_seq = v_ds.load_sequence(silu_path)
    v_in = silu_seq.to(device)
    if v_in.shape[1] == 1:
        v_in = v_in.repeat(1, 3, 1, 1)
    f_v_pooled = v_hpm(v_back(v_in)).max(dim=0, keepdim=True)[0]
    f_v = F.normalize(v_proj(f_v_pooled), dim=-1)

    # 3. 动态融合 (Bounded Fusion)
    _, alpha_raw = gate(f_s, f_v)
    alpha_s = torch.clamp(alpha_raw, 0.3, 0.7)
    f_final = F.normalize(alpha_s * f_s + (1 - alpha_s) * f_v, dim=-1)

    return f_final.cpu().numpy()


def run_full_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n--- 🚀 PPGait: Full-Set Rank-1 Evaluation (CASIA-B Custom Path) ---")

    # 1. 加载模型
    struct_ckpt = torch.load("logs/checkpoints/ppgait_struct_best.pth", map_location=device, weights_only=False)
    vis_ckpt = torch.load("logs/checkpoints/ppgait_vis_ema_best.pth", map_location=device, weights_only=False)
    fusion_ckpt = torch.load("logs/checkpoints/ppgait_fusion_final.pth", map_location=device, weights_only=False)

    s_model = StructuralBackbone(input_dim=struct_ckpt["config"]["input_dim"]).to(device)
    s_ppm = PPMStructuredAggregator(feature_dim=512).to(device)
    s_model.load_state_dict(struct_ckpt["backbone_state_dict"])
    s_ppm.load_state_dict(struct_ckpt["ppm_state_dict"])
    vis_all = PAGFSLModel(common_dim=512).to(device)
    vis_all.load_state_dict(vis_ckpt["state_dict"])
    gate = UncertaintyFusionGate(512).to(device)
    gate.load_state_dict(fusion_ckpt["gate_state_dict"])
    models = (s_model, s_ppm, vis_all.vis_backbone, vis_all.hpm, vis_all.proj_vis, gate)
    for m in models: m.eval()

    # 2. 准备数据
    s_ds = StructDataset("/datasets/CASIA-B", mode="pose")
    v_ds = VisDataset("/datasets/CASIA-B/silu", target_len=8)
    all_common_ids = set(s_ds.all_subject_ids) & set(v_ds.all_subject_ids)
    test_ids = sorted([sid for sid in all_common_ids if 75 <= int(sid) <= 124])

    gallery_feats, gallery_labels = [], []
    probe_results = collections.defaultdict(list)

    print(f"Processing {len(test_ids)} subjects based on directory depth...")

    for sid in tqdm(test_ids):
        s_paths = s_ds.all_sequences[sid]
        v_dict = v_ds.all_sequences[sid]

        for s_p in s_paths:
            # 🔥 针对你的路径结构：.../001/bg-01/000/000.pkl
            norm_p = s_p.replace('\\', '/')
            parts = norm_p.split('/')

            try:
                seq_type = parts[-3].lower()  # e.g., bg-01
                view = parts[-2]  # e.g., 000
            except IndexError:
                continue

            # 匹配视觉流
            if seq_type not in v_dict or view not in v_dict[seq_type]:
                continue
            v_p = v_dict[seq_type][view]

            feat = extract_feature(s_p, v_p, s_ds, v_ds, models, device)

            # 划分
            if any(tag in seq_type for tag in ['nm-01', 'nm-02', 'nm-03', 'nm-04']):
                gallery_feats.append(feat)
                gallery_labels.append(sid)
            else:
                condition = seq_type.split('-')[0].upper()
                if condition in ['NM', 'BG', 'CL']:
                    probe_results[condition].append({'feat': feat, 'label': sid})

    # 3. 结果计算
    if not gallery_feats:
        print("❌ Gallery 依然为空，请检查逻辑。")
        return

    g_feats = np.concatenate(gallery_feats, axis=0)
    g_labels = np.array(gallery_labels)

    print("\n" + "=" * 50)
    print(f"{'Condition':<12} | {'Rank-1 Accuracy':<15}")
    print("-" * 50)
    for cond in ['NM', 'BG', 'CL']:
        p_data = probe_results[cond]
        if not p_data: continue

        # 批量余弦相似度计算
        p_feats = np.concatenate([p['feat'] for p in p_data], axis=0)
        sim_matrix = np.dot(p_feats, g_feats.T)
        preds = g_labels[np.argmax(sim_matrix, axis=1)]
        gt = np.array([p['label'] for p in p_data])

        acc = np.mean(preds == gt)
        print(f"{cond:<12} | {acc * 100:>13.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    run_full_eval()