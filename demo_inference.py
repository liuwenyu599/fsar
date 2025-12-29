import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# ===== 模型导入 =====
from models.backbones.structural import StructuralBackbone
from models.heads.ppm import PPMStructuredAggregator
from models.architecture_res import PAGFSLModel
from train_fusion import UncertaintyFusionGate

# ===== 数据加载（复用你已有的）=====
from dataloader.multloader import CASIABMultiDataset as StructDataset
from dataloader.silu_resnet_loader import CASIASiluDataset as VisDataset


# --------------------------------------------------
# 工具函数：时间切段
# --------------------------------------------------
def temporal_segments(x, num_segments=8):
    """
    x: Tensor [T, ...]
    return: list of Tensor segments
    """
    T = x.shape[0]
    seg_len = T // num_segments
    segments = []
    for i in range(num_segments):
        start = i * seg_len
        end = T if i == num_segments - 1 else (i + 1) * seg_len
        segments.append(x[start:end])
    return segments


# --------------------------------------------------
# 主推理函数
# --------------------------------------------------
def infer_single_sequence(
    pose_seq,
    silu_seq,
    struct_model,
    struct_ppm,
    vis_backbone,
    vis_hpm,
    vis_proj,
    fusion_gate,
    device,
    num_segments=8
):
    """
    pose_seq: [T, 17, 3] 原生数据
    silu_seq: [T, 1, H, W]
    """
    # 1. 预处理：先将骨骼特征展平 [T, 17, 3] -> [T, 51] 🔥
    if len(pose_seq.shape) == 3:
        pose_seq = pose_seq.view(pose_seq.shape[0], -1)

    pose_segs = temporal_segments(pose_seq, num_segments)
    silu_segs = temporal_segments(silu_seq, num_segments)

    fused_feats = []
    alpha_curve = []

    with torch.no_grad():
        for p_seg, v_seg in zip(pose_segs, silu_segs):

            # --- 结构流 ---
            p_seg = p_seg.unsqueeze(0).to(device) # 现在确认为 [1, t, 51]
            mask = torch.ones(1, p_seg.shape[1]).to(device)

            f_s = struct_model(p_seg)      # 进入模型，解包 B, T, _ 不再报错
            f_s = struct_ppm(f_s, mask)
            f_s = F.normalize(f_s, dim=-1)

            # --- 视觉流 ---
            v_seg = v_seg.to(device)
            # 处理单通道/三通道兼容性
            if v_seg.shape[1] == 1:
                v_seg = v_seg.repeat(1, 3, 1, 1)

            f_v_map = vis_backbone(v_seg)  # [t, 512, 7, 7]
            f_v_hpm = vis_hpm(f_v_map)     # [t, 7680]
            f_v = f_v_hpm.max(dim=0, keepdim=True)[0] # [1, 7680]
            f_v = F.normalize(vis_proj(f_v), dim=-1)  # [1, 512]

            # --- 融合 ---
            f_fused, alpha = fusion_gate(f_s, f_v)

            fused_feats.append(f_fused)
            alpha_curve.append(alpha.item())

    fused_feats = torch.cat(fused_feats, dim=0)
    final_embedding = F.normalize(fused_feats.mean(dim=0, keepdim=True), dim=-1)

    return final_embedding, alpha_curve


# --------------------------------------------------
# Alpha 可视化
# --------------------------------------------------
def plot_alpha_curve(alpha_curve, save_path="alpha_curve.png"):
    # 增加这一行，强制 Matplotlib 不使用 X 窗口（解决服务器卡住问题）
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = np.arange(len(alpha_curve))
    alpha_curve = np.array(alpha_curve)

    plt.figure(figsize=(8, 4))
    plt.plot(x, alpha_curve, marker='o', color='#3498db', label="Structural Weight (α)")
    plt.plot(x, 1 - alpha_curve, marker='x', color='#e74c3c', label="Visual Weight (1−α)")

    plt.xlabel("Temporal Segment")
    plt.ylabel("Fusion Weight")
    plt.title("PPGait Dynamic Modality Trust")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path)
    # plt.show()  # 🔥 注释掉这一行，不要在服务器上弹出窗口
    plt.close()  # 释放内存

    print(f"📊 Alpha curve saved to {os.getcwd()}/{save_path}")

# --------------------------------------------------
# Demo 主入口
# --------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n🎬 PPGait — Single Video Inference Demo\n")

    # ===== 加载权重 =====
    struct_ckpt = torch.load("logs/checkpoints/ppgait_struct_best.pth", map_location=device, weights_only=False)
    vis_ckpt = torch.load("logs/checkpoints/ppgait_vis_ema_best.pth", map_location=device, weights_only=False)
    fusion_ckpt = torch.load("logs/checkpoints/ppgait_fusion_final.pth", map_location=device, weights_only=False)

    # ===== 构建模型 =====
    struct_model = StructuralBackbone(
        input_dim=struct_ckpt["config"]["input_dim"]
    ).to(device)
    struct_ppm = PPMStructuredAggregator(feature_dim=512).to(device)

    struct_model.load_state_dict(struct_ckpt["backbone_state_dict"])
    struct_ppm.load_state_dict(struct_ckpt["ppm_state_dict"])

    vis_all = PAGFSLModel(common_dim=512).to(device)
    vis_all.load_state_dict(vis_ckpt["state_dict"])
    vis_backbone, vis_hpm, vis_proj = vis_all.vis_backbone, vis_all.hpm, vis_all.proj_vis

    fusion_gate = UncertaintyFusionGate(feature_dim=512).to(device)
    fusion_gate.load_state_dict(fusion_ckpt["gate_state_dict"])

    for m in [struct_model, struct_ppm, vis_backbone, vis_hpm, vis_proj, fusion_gate]:
        m.eval()

    # ===== 构造 Few-Shot Support =====
    struct_ds = StructDataset("/datasets/CASIA-B", mode="pose")
    vis_ds = VisDataset("/datasets/CASIA-B/silu", target_len=8)

    common_ids = list(set(struct_ds.all_subject_ids) & set(vis_ds.all_subject_ids))
    sampled_ids = random.sample(common_ids, 5)
    view = "090"

    support_embeddings = []

    print(f">>> Support IDs: {sampled_ids}")

    for sid in sampled_ids:
        s_path = [p for p in struct_ds.all_sequences[sid] if view in p][0]
        v_path = [vis_ds.all_sequences[sid][k][view]
                  for k in vis_ds.all_sequences[sid]
                  if view in vis_ds.all_sequences[sid][k]][0]

        pose = struct_ds.load_pose(s_path)
        silu = vis_ds.load_sequence(v_path)

        emb, _ = infer_single_sequence(
            pose, silu,
            struct_model, struct_ppm,
            vis_backbone, vis_hpm, vis_proj,
            fusion_gate, device
        )
        support_embeddings.append(emb)

    support_embeddings = torch.cat(support_embeddings, dim=0)

    # ===== Query（模拟“输入一个视频”）=====
    query_id = random.choice(sampled_ids)
    print(f">>> Query ID (unknown to model): {query_id}")

    q_s_path = [p for p in struct_ds.all_sequences[query_id] if view in p][1]
    q_v_path = [vis_ds.all_sequences[query_id][k][view]
                for k in vis_ds.all_sequences[query_id]
                if view in vis_ds.all_sequences[query_id][k]][1]

    pose_q = struct_ds.load_pose(q_s_path)
    silu_q = vis_ds.load_sequence(q_v_path)

    query_emb, alpha_curve = infer_single_sequence(
        pose_q, silu_q,
        struct_model, struct_ppm,
        vis_backbone, vis_hpm, vis_proj,
        fusion_gate, device
    )

    # ===== 识别 =====
    dists = torch.cdist(query_emb, support_embeddings)
    pred = torch.argmin(dists, dim=1).item()
    pred_id = sampled_ids[pred]

    print(f"\n🎯 Predicted ID: {pred_id}")
    print(f"🧠 Alpha Curve: {np.round(alpha_curve, 3)}")

    plot_alpha_curve(alpha_curve)


if __name__ == "__main__":
    main()
