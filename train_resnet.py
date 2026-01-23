import os
import torch
import torch.optim as optim
import yaml
import numpy as np
from timm.utils import ModelEmaV2

# 🌟 必须在 import plt 之前，解决 SSH 服务器无图形界面报错
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch.optim.lr_scheduler as lr_scheduler
from models.architecture_res import PAGFSLModel
from losses.prototypical_loss import PrototypicalLoss
from dataloader.silu_resnet_loader import CASIASiluDataset, FewShotSampler

# 视角名称映射字典 (CASIA-B 标准)
VIEW_MAP = {
    '000': 0, '018': 1, '036': 2, '054': 3, '072': 4,
    '090': 5, '108': 6, '126': 7, '144': 8, '162': 9, '180': 10
}


def train():
    print("\n" + "=" * 60 + "\n🚀 PPGait Stage 1: Silhouette Training (Physical-Aligned)\n" + "=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    common_dim, n_way, k_shot, q_query = 512, 5, 5, 15

    # 1. 数据准备
    dataset = CASIASiluDataset('/datasets/CASIA-B/silu', target_len=8)
    sampler = FewShotSampler(dataset, n_way, k_shot, q_query)

    # 2. 模型初始化
    model = PAGFSLModel(common_dim=common_dim).to(device)
    model.train()
    model_ema = ModelEmaV2(model, decay=0.99, device=device)

    # 3. 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    criterion_proto = PrototypicalLoss().to(device)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # --- 启动前预检 ---
    print("🛡️  Running Pre-flight Check...")
    batch_data = sampler.get_episode(mode='train')
    v_raw = batch_data[3]
    if isinstance(v_raw, dict):
        v_name = list(v_raw.keys())[0]
        if v_name not in VIEW_MAP:
            print(f"⚠️ 警告: 采样器视角 {v_name} 不在映射表中，请检查！")
    print("✅ Pre-flight Check Passed. Starting Loop.")

    best_acc = 0.0
    history = {'loss': [], 'acc': []}

    for epoch in range(1, 101):
        # A. 数据解包与视角映射
        batch_data = sampler.get_episode(mode='train')
        X_vis = batch_data[0].to(device)
        phase_W = batch_data[2].to(device)

        # 🌟 映射字典处理
        v_data = batch_data[3]
        v_name = list(v_data.keys())[0] if isinstance(v_data, dict) else '090'
        v_idx = torch.full((X_vis.shape[0],), VIEW_MAP.get(v_name, 5), dtype=torch.long).to(device)

        # 结构流 Dummy 输入 (练视觉流时保持为0)
        X_struct_zero = torch.zeros(X_vis.shape[0], X_vis.shape[1], 51).to(device)

        # B. 训练前向
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            # 🌟 修正：传递 view_idx
            _, f_vis, _ = model.forward_feature(
                X_vis, X_struct_zero, phase_W,
                batch_size=32, view_idx=v_idx
            )

            # 原型网络计算
            f_re = f_vis.view(n_way, k_shot + q_query, -1)
            f_s = f_re[:, :k_shot].contiguous().view(n_way * k_shot, -1)
            f_q = f_re[:, k_shot:].contiguous().view(n_way * q_query, -1)
            labels = torch.arange(n_way).repeat_interleave(q_query).to(device)

            loss, acc = criterion_proto(f_s, f_q, labels, n_way, k_shot)

        # C. 更新权重
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        model_ema.update(model)

        # D. 保存与打印
        history['loss'].append(loss.item())
        history['acc'].append(acc.item())

        if acc.item() > best_acc:
            best_acc = acc.item()
            torch.save(model_ema.module.state_dict(), 'logs/checkpoints/ppgait_vis_best.pth')
            print(f"   >>> 💾 Best Updated: {best_acc:.4f} (View: {v_name})")

        if epoch % 5 == 0:
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f} | Batch-Acc: {acc.item():.4f}")

    # E. 结果绘图
    _plot_save(history)
    print(f"✅ Finished. Best Accuracy: {best_acc:.4f}")


def _plot_save(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1);
    plt.plot(history['loss']);
    plt.title('Training Loss')
    plt.subplot(1, 2, 2);
    plt.plot(history['acc'], color='red');
    plt.title('Batch Accuracy')
    plt.savefig('logs/checkpoints/vis_train_summary.png')
    plt.close()


if __name__ == "__main__":
    train()