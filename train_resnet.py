import os
import torch
import torch.optim as optim
import yaml
import argparse
from timm.utils import ModelEmaV2
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler  # 引入学习率调度器

# 确保 models/architecture_res.py 存在且 PAGFSLModel 类定义正确
from models.architecture_res import PAGFSLModel
from losses.prototypical_loss import PrototypicalLoss
from dataloader.silu_resnet_loader import CASIASiluDataset, FewShotSampler


def train(config_path='configs/config.yaml'):
    # 🔥🔥🔥 保持原有的打印信息不变 🔥🔥🔥
    print("--- 🚀 PAG-FSL Training Start (Visual Stream Stabilized) ---")

    # ---------------------------
    # 1. 配置加载
    # ---------------------------
    if os.path.exists(config_path):
        config = yaml.safe_load(open(config_path))
    else:
        config = {
            'system': {'common_dim': 512, 'device': 'cuda'},
            'train': {'lr': 1e-3, 'epochs': 200},  # 默认使用 200 步
            'few_shot': {'n_way': 5, 'k_shot': 5, 'q_query': 15, 'batch_size': 64}
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------------------
    # 2. 数据集 & 采样器
    # ---------------------------
    dataset = CASIASiluDataset('/datasets/CASIA-B/silu')
    n_way, k_shot, q_query = 5, 5, 15
    sampler = FewShotSampler(dataset, n_way, k_shot, q_query)

    # ---------------------------
    # 3. 初始化模型 (Main Model)
    # ---------------------------
    model = PAGFSLModel(common_dim=config['system']['common_dim']).to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    # ---------------------------
    # 4. 初始化 EMA 模型 (Shadow Model)
    # ---------------------------
    print("--- Initializing Model EMA (decay=0.99) ---")
    model_ema = ModelEmaV2(model, decay=0.99, device=device)

    # ---------------------------
    # 5. 优化器 & 调度器 (关键修正 🔥)
    # ---------------------------
    epochs = 150  # 确保 scheduler T_max 与 epochs 一致

    # 🔥 使用 AdamW，并加入 Weight Decay 正则化 🔥
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion_proto = PrototypicalLoss().to(device)
    scaler = torch.cuda.amp.GradScaler()

    # 🔥 引入 CosineAnnealingLR 调度器 🔥
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ---------------------------
    # 6. 训练循环 (含数据收集)
    # ---------------------------
    batch_size_vit = 128
    loss_ema_decay = 0.9
    ema_loss_val = None

    # 🔥 数据收集列表 🔥 (确保定义在循环外部)
    epochs_list = []
    loss_list = []
    acc_list = []
    ema_loss_list = []
    ema_acc_list = []

    print(f"Start training: {n_way}-way {k_shot}-shot, Total Epochs: {epochs}...")

    for epoch in range(1, epochs + 1):
        # A. 获取数据
        X_vis, _, _ = sampler.get_episode(mode='train')

        # B. 维度变换
        B, T, C, H, W = X_vis.shape
        total_BT = B * T
        X_flat = X_vis.reshape(total_BT, C, H, W).to(device)

        # C. 主模型特征提取 (Main Model)
        feats = []
        for i in range(0, total_BT, batch_size_vit):
            chunk = X_flat[i:i + batch_size_vit]
            with torch.cuda.amp.autocast():
                f = model.vis_backbone(chunk)
                f_proj = model.proj_vis(f)
            feats.append(f_proj)
        F_flat = torch.cat(feats, dim=0).float()

        # 主模型时序聚合
        f_final = F_flat.reshape(B, T, -1).max(dim=1)[0]

        # D. 标签准备
        f_final_reshaped = f_final.view(n_way, k_shot + q_query, -1)
        f_supp = f_final_reshaped[:, :k_shot, :].contiguous().view(n_way * k_shot, -1)
        f_query = f_final_reshaped[:, k_shot:, :].contiguous().view(n_way * q_query, -1)
        labels_query = torch.arange(n_way).repeat_interleave(q_query).to(device)

        # E. 主模型 Loss & Acc
        loss, acc = criterion_proto(f_supp, f_query, labels_query, n_way, k_shot)

        # --- 计算 Loss EMA ---
        if ema_loss_val is None:
            ema_loss_val = loss.item()
        else:
            ema_loss_val = loss_ema_decay * ema_loss_val + (1 - loss_ema_decay) * loss.item()

        # F. 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # G. 更新 Model EMA 权重
        model_ema.update(model)

        # H. 更新 Scheduler 权重 🔥
        scheduler.step()

        # --- 验证 Model EMA 的性能 ---
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                feats_ema = []
                for i in range(0, total_BT, batch_size_vit):
                    chunk = X_flat[i:i + batch_size_vit]
                    f_e = model_ema.module.vis_backbone(chunk)
                    p_e = model_ema.module.proj_vis(f_e)
                    feats_ema.append(p_e)

                F_flat_ema = torch.cat(feats_ema, dim=0).float()
                f_final_ema = F_flat_ema.reshape(B, T, -1).max(dim=1)[0]

                f_reshaped_e = f_final_ema.view(n_way, k_shot + q_query, -1)
                f_supp_e = f_reshaped_e[:, :k_shot, :].contiguous().view(n_way * k_shot, -1)
                f_query_e = f_reshaped_e[:, k_shot:, :].contiguous().view(n_way * q_query, -1)

                _, acc_ema = criterion_proto(f_supp_e, f_query_e, labels_query, n_way, k_shot)

        # 🔥 数据收集 🔥 (确保每个 epoch 都执行)
        epochs_list.append(epoch)
        loss_list.append(loss.item())
        acc_list.append(acc.item())
        ema_loss_list.append(ema_loss_val)
        acc_ema_float = acc_ema.item()
        ema_acc_list.append(acc_ema_float)

        # 打印日志 (注意：日志格式保持不变)
        if epoch % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[Epoch {epoch}/{epochs}] "
                  f"Loss: {loss.item():.4f} (EMA: {ema_loss_val:.4f}) | "
                  f"Acc: {acc.item():.4f} (EMA-Acc: {acc_ema_float:.4f}) | LR: {current_lr:.6f}")

    # ---------------------------
    # 7. 保存模型 (保存两个版本)
    # ---------------------------
    save_dir = 'logs/checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, 'pag_fsl_vis_latest.pth'))
    torch.save(model_ema.module.state_dict(), os.path.join(save_dir, 'pag_fsl_vis_ema.pth'))

    print("✅ Training finished. Saved 'latest' and 'ema' checkpoints.")

    # ---------------------------
    # 8. 可视化结果 🔥
    # ---------------------------
    plt.figure(figsize=(12, 5))

    # 子图 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, loss_list, label='Batch Loss', color='blue', alpha=0.6)
    plt.plot(epochs_list, ema_loss_list, label='EMA Loss', color='red', linewidth=2)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, acc_list, label='Batch Acc', color='blue', alpha=0.6)
    plt.plot(epochs_list, ema_acc_list, label='EMA Acc', color='red', linewidth=2)
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = os.path.join(save_dir, 'training_metrics_vis_stream.png')
    plt.savefig(plot_filename)
    print(f"✅ Training metrics saved to {plot_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    train(args.config)