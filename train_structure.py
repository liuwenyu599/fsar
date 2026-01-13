import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import datetime
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

# 导入项目组件
from models.backbones.structural import StructuralBackbone
from models.backbones.lora_handler import inject_lora_to_motionbert
from losses.prototypical_loss import PrototypicalLoss
from dataloader.multloader import CASIABMultiDataset, FewShotSampler

# 🌟 显存优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# --- 1. 视角感知聚合器 ---
class ViewAwarePPM(nn.Module):
    def __init__(self, feature_dim=512, num_views=11):
        super().__init__()
        self.feature_dim = feature_dim
        # 视角嵌入层：为 11 个视角学习专属的特征先验
        self.view_embedding = nn.Embedding(num_views, 128)
        # 视角门控：根据视角动态生成特征掩码，抑制不稳定的关节点通道
        self.view_gate = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x, phase_w, view_idx):
        """
        x: [B, T, D], phase_w: [B, T], view_idx: [B]
        """
        B, T, D = x.shape
        # 获取视角权重掩码
        v_emb = self.view_embedding(view_idx)  # [B, 128]
        v_mask = self.view_gate(v_emb)  # [B, D]

        # 视角特征调制
        x = x * v_mask.unsqueeze(1)

        # 时间维度加权聚合
        phase_w = F.softmax(phase_w, dim=1).unsqueeze(-1)
        return torch.sum(x * phase_w, dim=1)


# --- 2. 修正版采样器 (支持返回 v_idx) ---
class FewShotSamplerV2(FewShotSampler):
    def __init__(self, dataset, n_way, k_shot, q_query, active_views=None):
        self.dataset, self.n_way, self.k_shot, self.q_query = dataset, n_way, k_shot, q_query
        # 定义全视角列表
        self.target_views = active_views if active_views is not None else \
            ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']

        self.view_to_idx = {v: i for i, v in enumerate(self.target_views)}
        self.view_to_ids = {v: [] for v in self.target_views}

        for sid, paths in dataset.all_sequences.items():
            for v in self.target_views:
                if any(v in p for p in paths): self.view_to_ids[v].append(sid)

        self.active_views = [v for v in self.target_views if len(self.view_to_ids[v]) >= n_way]
        print(f"📊 Sampler Initialized: {len(self.active_views)} active views.")

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
        view_stats = {v: 0 for v in self.target_views}

        for cls_idx, sid in enumerate(sampled_ids):
            all_paths = self.dataset.all_sequences.get(sid, [])
            valid_paths = [p for p in all_paths if episode_view in p]
            if not valid_paths: valid_paths = all_paths
            selected = random.sample(valid_paths * 5, self.k_shot + self.q_query)
            for pkl_path in selected:
                for v in self.target_views:
                    if v in pkl_path: view_stats[v] += 1
                X.append(self.dataset.load_pose(pkl_path))
                Y.append(cls_idx)

        # 核心：生成视角索引 Tensor
        v_idx = self.view_to_idx.get(episode_view, 0)
        v_idx_tensor = torch.full((len(X),), v_idx, dtype=torch.long)

        X = torch.stack(X)
        X = X.view(X.shape[0], X.shape[1], -1)
        return X, torch.tensor(Y), torch.ones(X.shape[0], X.shape[1]), v_idx_tensor, view_stats


# --- 3. 训练主程序 ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 50 + "\n🚀 Stage 2: View-Aware Structural Training\n" + "=" * 50)

    # A. 数据与模型准备
    dataset = CASIABMultiDataset('/datasets/CASIA-B', mode='pose', seq_len=60)
    train_sampler = FewShotSamplerV2(dataset, n_way=5, k_shot=5, q_query=5)

    model = StructuralBackbone().to(device)
    # 注入 LoRA (Rank 16 增强多视角适应性)
    model = inject_lora_to_motionbert(model, rank=16)
    model.to(device)

    with torch.no_grad():
        backbone_dim = model(torch.zeros(1, 60, 51).to(device)).shape[-1]
    ppm = ViewAwarePPM(feature_dim=backbone_dim).to(device)

    # B. 优化器与调度器
    trainable_params = [p for n, p in model.named_parameters() if "lora_" in n] + list(ppm.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    # 余弦退火：平滑减小学习率，帮助 Loss 跌破 1.6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000)
    scaler = GradScaler()
    criterion = PrototypicalLoss().to(device)
    writer = SummaryWriter(os.path.join("logs", "tensorboard", datetime.datetime.now().strftime("%m%d-%H%M")))

    best_acc = 0.0
    for step in range(1, 3001):
        # 🌟 修复 ValueError: 解包 5 个值
        X, _, W, v_idx, _ = train_sampler.get_episode(mode='train')
        X, W, v_idx = X.to(device), W.to(device), v_idx.to(device)

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            feat_seq = F.normalize(model(X), p=2, dim=-1)
            # 🌟 传入视角索引进行条件化聚合
            f_final = ppm(feat_seq, W, v_idx)

            n, k, q = train_sampler.n_way, train_sampler.k_shot, train_sampler.q_query
            f_r = f_final.view(n, k + q, -1)
            q_labels = torch.arange(n).repeat_interleave(q).to(device)
            loss, acc = criterion(f_r[:, :k].reshape(-1, backbone_dim),
                                  f_r[:, k:].reshape(-1, backbone_dim), q_labels, n, k)

        scaler.scale(loss).backward()

        # 🌟 稳健性处理：梯度净化
        scaler.unscale_(optimizer)
        for param in trainable_params:
            if param.grad is not None:
                param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % 20 == 0:
            print(
                f"[Step {step}/3000] Loss: {loss.item():.4f} | Acc: {acc.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            writer.add_scalar('Train/Loss', loss.item(), step)
            writer.add_scalar('Train/Acc', acc.item(), step)
            writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], step)

        if step % 100 == 0:
            torch.save({
                'struct_lora_state_dict': {k: v for k, v in model.state_dict().items() if 'lora_' in k},
                'ppm_state_dict': ppm.state_dict(),
                'step': step,
                'best_acc': acc.item()
            }, 'logs/checkpoints/ppgait_struct_best.pth')

    writer.close()
    print("✅ Training Finished.")


if __name__ == '__main__':
    train()