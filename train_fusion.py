import torch
import torch.nn as nn
torch.cuda.empty_cache() # 强制清理之前的残留
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import random
import datetime

# 导入之前的架构
from models.backbones.structural import StructuralBackbone
from models.heads.ppm import PPMStructuredAggregator
from models.architecture_res import PAGFSLModel
from losses.prototypical_loss import PrototypicalLoss
from dataloader.multloader import CASIABMultiDataset as StructDataset
from dataloader.silu_resnet_loader import CASIASiluDataset as VisDataset


# ---------------------------------------------------------
# 1. 核心模块：保持“白板”状态的融合门控
# ---------------------------------------------------------
class UncertaintyFusionGate(nn.Module):
    def __init__(self, feature_dim=512):
        super(UncertaintyFusionGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # 保持公平初始化：Sigmoid(0) = 0.5
        nn.init.constant_(self.gate[2].weight, 0)
        nn.init.constant_(self.gate[2].bias, 0)

    def forward(self, f_struct, f_vis):
        combined = torch.cat([f_struct, f_vis], dim=-1)
        alpha = torch.sigmoid(self.gate(combined))
        # 融合公式
        f_fused = alpha * f_struct + (1 - alpha) * f_vis
        return f_fused, alpha


# ---------------------------------------------------------
# 2. 熵损失计算 (用于鼓励模型探索)
# ---------------------------------------------------------
def compute_entropy_loss(alpha, epsilon=1e-6):
    """
    计算二元分布的熵：H(alpha) = -[a*log(a) + (1-a)*log(1-a)]
    """
    entropy = - (alpha * torch.log(alpha + epsilon) + (1 - alpha) * torch.log(1 - alpha + epsilon))
    return entropy.mean()


# ---------------------------------------------------------
# 3. 融合训练逻辑
# ---------------------------------------------------------
def train_fusion():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- 🏁 PPGait: Fusion Training with Entropy Regularization ---")

    # A. 加载权重 (PyTorch 2.6 兼容)
    struct_ckpt = torch.load('logs/checkpoints/ppgait_struct_best.pth', weights_only=False)
    vis_ckpt = torch.load('logs/checkpoints/ppgait_vis_ema_best.pth', weights_only=False)

    # B. 初始化单流模型并冻结
    struct_model = StructuralBackbone(input_dim=struct_ckpt['config']['input_dim']).to(device)
    struct_ppm = PPMStructuredAggregator(feature_dim=512).to(device)
    struct_model.load_state_dict(struct_ckpt['backbone_state_dict'])
    struct_ppm.load_state_dict(struct_ckpt['ppm_state_dict'])

    vis_all_model = PAGFSLModel(common_dim=512).to(device)
    vis_all_model.load_state_dict(vis_ckpt['state_dict'])

    vis_backbone = vis_all_model.vis_backbone
    vis_hpm = vis_all_model.hpm
    vis_proj = vis_all_model.proj_vis

    # 深度冻结
    for p in [struct_model, struct_ppm, vis_backbone, vis_hpm, vis_proj]:
        p.eval()
        for param in p.parameters(): param.requires_grad = False
    print(">>> Backbones Frozen. Starting adaptive exploration...")

    # C. 数据对齐准备
    s_dataset = StructDataset('/datasets/CASIA-B', mode='pose')
    v_dataset = VisDataset('/datasets/CASIA-B/silu', target_len=8)
    n_way, k_shot, q_query = 5, 5, 4
    target_views = ['090', '180', '054']

    # D. 初始化门控
    fusion_gate = UncertaintyFusionGate(feature_dim=512).to(device)
    optimizer = optim.AdamW(fusion_gate.parameters(), lr=0.0005)
    criterion = PrototypicalLoss().to(device)

    # E. 训练循环
    total_steps = 1000
    for step in range(1, total_steps + 1):
        # 1. 同步采样
        episode_view = random.choice(target_views)
        common_ids = list(set(s_dataset.all_subject_ids) & set(v_dataset.all_subject_ids))
        sampled_ids = random.sample(common_ids, n_way)
        X_s_list, X_v_list = [], []
        needed = k_shot + q_query

        for sid in sampled_ids:
            s_paths = [p for p in s_dataset.all_sequences[sid] if episode_view in p]
            v_seqs = v_dataset.all_sequences[sid]
            v_paths = [v_seqs[c][episode_view] for c in v_seqs if episode_view in v_seqs[c]]
            for _ in range(needed):
                X_s_list.append(s_dataset.load_pose(random.choice(s_paths)))
                X_v_list.append(v_dataset.load_sequence(random.choice(v_paths)))

        X_s = torch.stack(X_s_list).view(n_way * needed, 60, -1).to(device)
        X_v = torch.stack(X_v_list).to(device)

        # 2. 提取特征
        with torch.no_grad():
            # A. 结构流特征 (计算开销小，保持原样)
            f_s_raw = struct_ppm(struct_model(X_s), torch.ones(X_s.shape[0], 60).to(device))

            # B. 视觉流特征 (🔥🔥🔥 显存优化版 🔥🔥🔥)
            B_v, T_v, C_v, H_v, W_v = X_v.shape
            x_v_flat = X_v.view(-1, C_v, H_v, W_v)

            # --- 关键：将 B*T 个样本切分成小块处理 ---
            micro_batch_size = 16  # 每次只送 16 张图进 ResNet
            f_v_hpm_list = []

            # 开启混合精度 autocast
            with torch.amp.autocast('cuda'):
                for i in range(0, x_v_flat.size(0), micro_batch_size):
                    x_mini = x_v_flat[i: i + micro_batch_size]

                    # 1. 提取卷积图
                    f_maps = vis_backbone(x_mini)
                    # 2. HPM 空间解耦 (这是显存消耗大户)
                    f_hpm = vis_hpm(f_maps)

                    # 转回 float32 并从显存缓存中分离，防止梯度图堆积
                    f_v_hpm_list.append(f_hpm.float().cpu())  # 暂时存入内存

            # 拼接并移回 GPU
            f_v_hpm_all = torch.cat(f_v_hpm_list, dim=0).to(device)

            # (3) 时序最大聚合
            f_v_seq = f_v_hpm_all.view(B_v, T_v, -1)
            f_v_pooled = f_v_seq.max(dim=1)[0]

            # (4) 投影到公共空间 [B, 7680] -> [B, 512]
            f_v_raw = vis_proj(f_v_pooled)

        # 3. 归一化
        f_s = F.normalize(f_s_raw, p=2, dim=-1)
        f_v = F.normalize(f_v_raw, p=2, dim=-1)

        # 4. 融合与计算 Loss
        f_fused, alpha_vals = fusion_gate(f_s, f_v)

        # (a) 基本原型损失
        f_reshaped = f_fused.view(n_way, k_shot + q_query, -1)
        f_supp = f_reshaped[:, :k_shot].contiguous().view(-1, 512)
        f_query = f_reshaped[:, k_shot:].contiguous().view(-1, 512)
        q_labels = torch.arange(n_way).repeat_interleave(q_query).to(device)
        loss_proto, acc = criterion(f_supp, f_query, q_labels, n_way, k_shot)

        # (b) 熵正则项 (物理约束 🔥)
        # 前期设置较大的 lambda 鼓励探索，后期逐渐减小让其自由收敛
        lambda_ent = 0.1 * (1 - step / total_steps)
        loss_ent = compute_entropy_loss(alpha_vals)

        # 我们的目标是最大化熵，所以是减去熵
        total_loss = loss_proto - lambda_ent * loss_ent

        # 5. 优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 20 == 0:
            avg_alpha = alpha_vals.mean().item()
            print(f"[Step {step}] ProtoLoss: {loss_proto.item():.4f} | Acc: {acc.item():.4f} | "
                  f"Avg Alpha: {avg_alpha:.4f} | Ent_Weight: {lambda_ent:.4f}")

    # 保存
    save_path = 'logs/checkpoints/ppgait_fusion_final.pth'
    os.makedirs('logs/checkpoints', exist_ok=True)
    torch.save({'gate_state_dict': fusion_gate.state_dict(), 'best_acc': acc.item()}, save_path)
    print(f"✅ PPGait Fusion Complete. Saved to {save_path}")


if __name__ == "__main__":
    train_fusion()