import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from tqdm import tqdm

# 1. 自动处理路径：确保能找到根目录下的 models 和 dataloader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入你的架构组件
from models.backbones.structural import StructuralBackbone
from models.backbones.lora_handler import inject_lora_to_motionbert
from models.heads.ppm import PPMStructuredAggregator
from models.architecture_res import PAGFSLModel
from dataloader.multloader import CASIABMultiDataset as StructDataset
from dataloader.silu_resnet_loader import CASIASiluDataset as VisDataset


# ---------------------------------------------------------
# 1. 融合门控类定义 (必须与训练时完全一致)
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
# 2. 特征提取核心函数 (带 L2 归一化)
# ---------------------------------------------------------
@torch.no_grad()
def extract_feature(pose_path, silu_path, s_ds, v_ds, models, device):
    s_model, s_ppm, v_back, v_hpm, v_proj, gate = models

    # A. 结构流特征 (LoRA 已经在初始化阶段注入并在 forward 中生效)
    pose_seq = s_ds.load_pose(pose_path)
    if len(pose_seq.shape) == 3:
        pose_seq = pose_seq.view(pose_seq.shape[0], -1)
    p_in = pose_seq.unsqueeze(0).to(device)
    f_s_raw = s_ppm(s_model(p_in), torch.ones(1, p_in.shape[1]).to(device))
    f_s = F.normalize(f_s_raw, p=2, dim=-1)

    # B. 视觉流特征
    silu_seq = v_ds.load_sequence(silu_path)
    v_in = silu_seq.to(device)
    if v_in.shape[1] == 1:
        v_in = v_in.repeat(1, 3, 1, 1)

    f_v_hpm = v_hpm(v_back(v_in))
    f_v_pooled = f_v_hpm.max(dim=0, keepdim=True)[0]
    f_v_raw = v_proj(f_v_pooled)
    f_v = F.normalize(f_v_raw, p=2, dim=-1)

    # C. 动态融合与最终归一化
    f_fused, _ = gate(f_s, f_v)
    return F.normalize(f_fused, p=2, dim=-1).cpu().numpy()


# ---------------------------------------------------------
# 3. 全量评估主逻辑
# ---------------------------------------------------------
def run_lora_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print("🚀 PPGait: Full-Set Rank-1 Evaluation (LoRA-Enhanced Mode)")
    print("=" * 60)

    # 1. 加载所有检查点 (显式禁用 weights_only 以加载自定义类)
    print("📦 Loading checkpoints...")
    try:
        fusion_ckpt = torch.load('logs/checkpoints/ppgait_fusion_lora_final.pth', map_location=device,
                                 weights_only=False)
        struct_base_ckpt = torch.load('logs/checkpoints/ppgait_struct_best.pth', map_location=device,
                                      weights_only=False)
        vis_base_ckpt = torch.load('logs/checkpoints/ppgait_vis_ema_best.pth', map_location=device, weights_only=False)
    except FileNotFoundError as e:
        print(f"❌ Error: 权重文件缺失！请检查 logs/checkpoints/: {e}")
        return

    # 2. 构建模型结构
    # A. 结构流：基础权重 -> LoRA 注入 -> 补丁加载
    s_model = StructuralBackbone(input_dim=struct_base_ckpt['config']['input_dim']).to(device)
    s_model.load_state_dict(struct_base_ckpt['backbone_state_dict'])

    # --- 核心：确保 Rank 与训练时一致 ---
    s_model = inject_lora_to_motionbert(s_model, rank=8)
    s_model.load_state_dict(fusion_ckpt['struct_lora_state_dict'], strict=False)
    print("   - LoRA weights patched into MotionBERT.")

    s_ppm = PPMStructuredAggregator(feature_dim=512).to(device)
    s_ppm.load_state_dict(struct_base_ckpt['ppm_state_dict'])

    # B. 视觉流
    vis_all = PAGFSLModel(common_dim=512).to(device)
    vis_all.load_state_dict(vis_base_ckpt['state_dict'])

    # C. 门控
    gate = UncertaintyFusionGate(512).to(device)
    gate.load_state_dict(fusion_ckpt['gate_state_dict'])

    models = (s_model, s_ppm, vis_all.vis_backbone, vis_all.hpm, vis_all.proj_vis, gate)
    for m in models: m.eval()

    # 3. 数据处理 (075-124 号受试者)
    s_ds = StructDataset('/datasets/CASIA-B', mode='pose')
    v_ds = VisDataset('/datasets/CASIA-B/silu', target_len=8)

    # 获取测试交集 ID
    all_ids = set(s_ds.all_subject_ids) & set(v_ds.all_subject_ids)
    test_ids = sorted([sid for sid in all_ids if 75 <= int(sid) <= 124])

    gallery_feats, gallery_labels = [], []
    probe_results = collections.defaultdict(list)

    print(f"🔍 Processing {len(test_ids)} subjects...")
    for sid in tqdm(test_ids):
        s_paths = s_ds.all_sequences[sid]
        v_dict = v_ds.all_sequences[sid]

        for s_p in s_paths:
            # 适配目录深度: /datasets/CASIA-B/pose/CASIA-B_HRNet/001/bg-01/000/000.pkl
            norm_p = s_p.replace('\\', '/')
            parts = norm_p.split('/')
            try:
                seq_type = parts[-3].lower()  # e.g., 'bg-01'
                view = parts[-2]  # e.g., '000'
            except IndexError:
                continue

            if seq_type not in v_dict or view not in v_dict[seq_type]:
                continue

            feat = extract_feature(s_p, v_dict[seq_type][view], s_ds, v_ds, models, device)

            # 标准协议划分: NM 01-04 为 Gallery
            if any(tag in seq_type for tag in ['nm-01', 'nm-02', 'nm-03', 'nm-04']):
                gallery_feats.append(feat)
                gallery_labels.append(sid)
            else:
                condition = seq_type.split('-')[0].upper()  # NM, BG, CL
                if condition in ['NM', 'BG', 'CL']:
                    probe_results[condition].append({'feat': feat, 'label': sid})

    # 4. 矩阵运算计算 Rank-1 Accuracy
    if not gallery_feats:
        print("❌ Error: Gallery 为空！请检查路径解析索引。")
        return

    g_feats = np.concatenate(gallery_feats, axis=0)
    g_labels = np.array(gallery_labels)

    print("\n" + "=" * 50)
    print(f"{'Condition':<12} | {'Rank-1 Accuracy':<15}")
    print("-" * 50)

    for cond in ['NM', 'BG', 'CL']:
        p_data = probe_results[cond]
        if not p_data:
            print(f"{cond:<12} | No Data Found")
            continue

        p_feats = np.concatenate([p['feat'] for p in p_data], axis=0)
        p_labels_gt = np.array([p['label'] for p in p_data])

        # 余弦相似度计算 (归一化向量的点积)
        sim_matrix = np.dot(p_feats, g_feats.T)

        # 寻找最大概率的 ID
        preds_idx = np.argmax(sim_matrix, axis=1)
        preds_labels = g_labels[preds_idx]

        acc = np.mean(preds_labels == p_labels_gt)
        print(f"{cond:<12} | {acc * 100:>13.2f}%")

    print("=" * 50)


if __name__ == "__main__":
    run_lora_eval()