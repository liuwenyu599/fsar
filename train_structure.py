import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import datetime
import random
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

# 导入项目组件
from models.backbones.structural import StructuralBackbone
from models.backbones.lora_handler import inject_view_lora  # 确保是 PlainLoRA 版本
from models.heads.view_discriminator import ViewDiscriminator
from losses.prototypical_loss import PrototypicalLoss
from dataloader.multloader import CASIABMultiDataset, FewShotSampler

# 🌟 显存与计算优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# --- 1. 结构流聚合器 ---
class StructuralPPM(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x, phase_w):
        phase_w = F.softmax(phase_w, dim=1).unsqueeze(-1)
        return torch.sum(x * phase_w, dim=1)


# --- 2. 采样器 (与之前保持一致) ---
class FewShotSamplerV2(FewShotSampler):
    def __init__(self, dataset, n_way, k_shot, q_query, active_views=None):
        self.dataset, self.n_way, self.k_shot, self.q_query = dataset, n_way, k_shot, q_query
        self.target_views = active_views if active_views is not None else \
            ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
        self.view_to_idx = {v: i for i, v in enumerate(self.target_views)}
        self.view_to_ids = {v: [] for v in self.target_views}
        for sid, paths in dataset.all_sequences.items():
            for v in self.target_views:
                if any(v in p for p in paths): self.view_to_ids[v].append(sid)
        self.active_views = [v for v in self.target_views if len(self.view_to_ids[v]) >= n_way]
        all_ids = sorted(dataset.all_subject_ids)
        random.seed(42)
        shuffled_ids = all_ids[:]
        random.shuffle(shuffled_ids)
        split = int(0.8 * len(shuffled_ids))
        self.train_ids_all, self.test_ids_all = shuffled_ids[:split], shuffled_ids[split:]

    def get_episode(self, mode="train"):
        episode_view = random.choice(self.active_views)
        pool = self.train_ids_all if mode == "train" else self.test_ids_all
        candidates = [sid for sid in pool if sid in self.view_to_ids.get(episode_view, [])]
        if len(candidates) < self.n_way: candidates = pool
        sampled_ids = random.sample(candidates, self.n_way)
        X, Y = [], []
        for cls_idx, sid in enumerate(sampled_ids):
            all_paths = self.dataset.all_sequences.get(sid, [])
            valid_paths = [p for p in all_paths if episode_view in p] or all_paths
            selected = random.sample(valid_paths * 5, self.k_shot + self.q_query)
            for pkl_path in selected:
                X.append(self.dataset.load_pose(pkl_path))
                Y.append(cls_idx)
        v_idx = self.view_to_idx.get(episode_view, 0)
        v_idx_tensor = torch.full((len(X),), v_idx, dtype=torch.long)
        X = torch.stack(X).view(len(X), 60, -1)
        return X, torch.tensor(Y), torch.ones(X.shape[0], X.shape[1]), v_idx_tensor


# --- 3. 训练主程序 ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 50 + "\n🛡️ Stage 2: View-Adversarial(Anti-Collapse)\n" + "=" * 50)

    dataset = CASIABMultiDataset('/datasets/CASIA-B', mode='pose', seq_len=60)
    train_sampler = FewShotSamplerV2(dataset, n_way=5, k_shot=5, q_query=5)

    model = StructuralBackbone().to(device)
    model = inject_view_lora(model, rank=16)
    model.to(device)

    backbone_dim = 512
    ppm = StructuralPPM(feature_dim=backbone_dim).to(device)
    discriminator = ViewDiscriminator(feature_dim=backbone_dim, num_views=11).to(device)

    # 🌟 优化器：分层学习率
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 1e-4},
        {'params': ppm.parameters(), 'lr': 1e-4},
        {'params': discriminator.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000)
    scaler = GradScaler()
    criterion_id = PrototypicalLoss().to(device)
    criterion_view = nn.CrossEntropyLoss().to(device)

    writer = SummaryWriter(
        os.path.join("logs", "tensorboard", "adv_" + datetime.datetime.now().strftime("%m%d-%H%M")))

    # 🌟 对抗配置
    warmup_steps = 800  # 前 800 步专注 ID 学习，不进行对抗
    temperature = 16.0  # 强制特征尺度，防止 Softmax 饱和导致的 Loss 锁死

    for step in range(1, 3001):
        X, Y, W, v_idx = train_sampler.get_episode(mode='train')
        X, Y, W, v_idx = X.to(device), Y.to(device), W.to(device), v_idx.to(device)

        # 🌟 计算动态 Alpha (含 Warm-up 逻辑)
        if step <= warmup_steps:
            alpha = 0.0
        else:
            # 在 warmup 结束后，alpha 在 1000 步内从 0 线性增加到 1.0
            alpha = min(1.0, (step - warmup_steps) / 1000.0)

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            # 1. 提取特征并强制特征尺度
            # F.normalize 后的向量长度为 1，乘以 temperature 确保度量空间有足够的“压力”
            feat_seq = F.normalize(model(X, view_idx=None), p=2, dim=-1) * temperature
            f_final = ppm(feat_seq, W)

            # 2. 身份识别 (ID) 分支
            n, k, q = train_sampler.n_way, train_sampler.k_shot, train_sampler.q_query
            f_r = f_final.view(n, k + q, -1)
            q_labels = torch.arange(n).repeat_interleave(q).to(device)
            loss_id, acc = criterion_id(f_r[:, :k].reshape(-1, backbone_dim),
                                        f_r[:, k:].reshape(-1, backbone_dim), q_labels, n, k)

            # 3. 对抗 (View) 分支
            view_pred = discriminator(f_final, alpha=alpha)
            loss_view = criterion_view(view_pred, v_idx)

            # 🌟 总损失：alpha 控制对抗强度的介入
            total_loss = loss_id + alpha * loss_view

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % 20 == 0:
            print(
                f"[Step {step}/3000] L_ID: {loss_id.item():.3f} | L_View: {loss_view.item():.3f} | Acc: {acc.item():.4f} | Alpha: {alpha:.3f}")
            writer.add_scalar('Train/Loss_ID', loss_id.item(), step)
            writer.add_scalar('Train/Loss_View', loss_view.item(), step)
            writer.add_scalar('Train/Acc', acc.item(), step)

        if step % 500 == 0:
            torch.save({
                'struct_backbone': model.state_dict(),
                'ppm_state_dict': ppm.state_dict(),
                'best_acc': acc.item()
            }, f'logs/checkpoints/ppgait_adv_0118_s{step}.pth')

    writer.close()


if __name__ == '__main__':
    train()