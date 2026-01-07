import sys
import os
# 将项目根目录加入到搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from tqdm import tqdm

# 1. 自动处理路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入你的架构组件
from models.backbones.structural import StructuralBackbone
from models.backbones.lora_handler import inject_lora_to_motionbert
from models.heads.ppm import PPMStructuredAggregator
from models.architecture_res import PAGFSLModel
from dataloader.multloader import CASIABMultiDataset as StructDataset
from dataloader.silu_resnet_loader import CASIASiluDataset as VisDataset


# ---------------------------------------------------------
# 1. 融合门控类 (需与 train_fusion_lora.py 一致)
# ---------------------------------------------------------
class UncertaintyFusionGate(nn.Module):
    def __init__(self, feature_dim=512):
        super(UncertaintyFusionGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, f_s, f_v):
        combined = torch.cat([f_s, f_v], dim=-1)
        alpha = torch.sigmoid(self.gate(combined))
        f_fused = alpha * f_s + (1 - alpha) * f_v
        return f_fused, alpha


# ---------------------------------------------------------
# 2. 核心特征提取函数 (支持 Alpha 手动干预)
# ---------------------------------------------------------
@torch.no_grad()
def extract_feature_diagnostic(pose_path, silu_path, s_ds, v_ds, models, device, condition, force_alpha=None):
    s_model, s_ppm, v_back, v_hpm, v_proj, gate = models

    # A. 结构流特征 (LoRA 增强)
    pose_seq = s_ds.load_pose(pose_path)
    if len(pose_seq.shape) == 3: pose_seq = pose_seq.view(pose_seq.shape[0], -1)
    p_in = pose_seq.unsqueeze(0).to(device)
    f_s = F.normalize(s_ppm(s_model(p_in), torch.ones(1, p_in.shape[1]).to(device)), dim=-1)

    # B. 视觉流特征 (Frozen ResNet)
    silu_seq = v_ds.load_sequence(silu_path)
    v_in = silu_seq.to(device)
    if v_in.shape[1] == 1: v_in = v_in.repeat(1, 3, 1, 1)
    f_v = F.normalize(v_proj(v_hpm(v_back(v_in)).max(dim=0, keepdim=True)[0]), dim=-1)

    # C. 融合逻辑
    if condition == 'CL' and force_alpha is not None:
        # 🔥 诊断核心：强制使用设定的 Alpha (例如 0.9)
        f_final = force_alpha * f_s + (1 - force_alpha) * f_v
    else:
        # 使用原始训练好的 Gate 权重
        f_final, _ = gate(f_s, f_v)

    return F.normalize(f_final, dim=-1).cpu().numpy()


# ---------------------------------------------------------
# 3. 诊断主逻辑
# ---------------------------------------------------------
def run_diagnostic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print("🧪 PPGait: CL Potential Diagnostic Experiment")
    print("=" * 60)

    # 1. 加载 Branch A 的权重 (即你之前 68% 那版)
    # 注意：确保这里加载的是 ppgait_fusion_lora_final.pth
    ckpt_path = 'logs/checkpoints/ppgait_fusion_lora_final.pth'
    fusion_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    struct_base_ckpt = torch.load('logs/checkpoints/ppgait_struct_best.pth', map_location=device, weights_only=False)
    vis_base_ckpt = torch.load('logs/checkpoints/ppgait_vis_ema_best.pth', map_location=device, weights_only=False)

    # 2. 构建模型
    s_model = StructuralBackbone(input_dim=struct_base_ckpt['config']['input_dim']).to(device)
    s_model.load_state_dict(struct_base_ckpt['backbone_state_dict'])
    s_model = inject_lora_to_motionbert(s_model, rank=8)
    s_model.load_state_dict(fusion_ckpt['struct_lora_state_dict'], strict=False)

    s_ppm = PPMStructuredAggregator(feature_dim=512).to(device)
    s_ppm.load_state_dict(struct_base_ckpt['ppm_state_dict'])

    vis_all = PAGFSLModel(common_dim=512).to(device)
    vis_all.load_state_dict(vis_base_ckpt['state_dict'])

    gate = UncertaintyFusionGate(512).to(device)
    gate.load_state_dict(fusion_ckpt['gate_state_dict'])

    models = (s_model, s_ppm, vis_all.vis_backbone, vis_all.hpm, vis_all.proj_vis, gate)
    for m in models: m.eval()

    # 3. 准备测试数据
    s_ds = StructDataset("/datasets/CASIA-B", mode="pose")
    v_ds = VisDataset("/datasets/CASIA-B/silu", target_len=8)
    test_ids = sorted([sid for sid in (set(s_ds.all_subject_ids) & set(v_ds.all_subject_ids)) if 75 <= int(sid) <= 124])

    # 4. 跑两组 CL 评估
    for mode in ['Original', 'Forced Alpha (0.9)']:
        print(f"\n🚀 Evaluating CL in [{mode}] Mode...")
        gallery_feats, gallery_labels = [], []
        probe_cl_feats, probe_cl_labels = [], []

        fa = 0.9 if 'Forced' in mode else None

        for sid in tqdm(test_ids):
            s_paths = s_ds.all_sequences[sid]
            v_dict = v_ds.all_sequences[sid]
            for s_p in s_paths:
                parts = s_p.replace('\\', '/').split('/')
                try:
                    seq_type, view = parts[-3].lower(), parts[-2]
                except:
                    continue
                if seq_type not in v_dict or view not in v_dict[seq_type]: continue

                # 提取特征
                feat = extract_feature_diagnostic(s_p, v_dict[seq_type][view], s_ds, v_ds, models, device,
                                                  seq_type.split('-')[0].upper(), force_alpha=fa)

                if any(tag in seq_type for tag in ['nm-01', 'nm-02', 'nm-03', 'nm-04']):
                    gallery_feats.append(feat)
                    gallery_labels.append(sid)
                elif 'cl' in seq_type:
                    probe_cl_feats.append(feat)
                    probe_cl_labels.append(sid)

        # 计算 Rank-1
        g_f = np.concatenate(gallery_feats, axis=0)
        p_f = np.concatenate(probe_cl_feats, axis=0)
        sim = np.dot(p_f, g_f.T)
        acc = np.mean(np.array(gallery_labels)[np.argmax(sim, axis=1)] == np.array(probe_cl_labels))
        print(f"📊 {mode} CL Rank-1: {acc * 100:.2f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_diagnostic()