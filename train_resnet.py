import os
import torch
import torch.optim as optim
import yaml
import argparse
from timm.utils import ModelEmaV2
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

# 确保模型、损失函数和加载器路径正确
from models.architecture_res import PAGFSLModel
from losses.prototypical_loss import PrototypicalLoss
from dataloader.silu_resnet_loader import CASIASiluDataset, FewShotSampler


def train(config_path='configs/config.yaml'):
    print("--- 🚀 PAG-FSL Training Start (Visual Stream Stabilized) ---")

    # 1. 配置加载
    if os.path.exists(config_path):
        config = yaml.safe_load(open(config_path))
    else:
        config = {
            'system': {'common_dim': 512, 'device': 'cuda'},
            'train': {'lr': 1e-3, 'epochs': 150},
            'few_shot': {'n_way': 5, 'k_shot': 5, 'q_query': 15}
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 数据集 & 采样器 (这里的 FewShotSampler 必须是带视角锁死的版本)
    dataset = CASIASiluDataset('/datasets/CASIA-B/silu', target_len=8)
    n_way, k_shot, q_query = 5, 5, 15
    sampler = FewShotSampler(dataset, n_way, k_shot, q_query)

    # 3. 初始化模型
    model = PAGFSLModel(common_dim=config['system']['common_dim']).to(device)
    model.train()

    # 4. 初始化 Model EMA
    print("--- Initializing Model EMA (decay=0.99) ---")
    model_ema = ModelEmaV2(model, decay=0.99, device=device)

    # 5. 优化器、Loss 和 调度器
    epochs = config['train']['epochs']
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion_proto = PrototypicalLoss().to(device)

    # 使用新版 AMP Scaler
    scaler = torch.amp.GradScaler('cuda')
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 数据记录
    epochs_list, loss_list, acc_list, ema_acc_list = [], [], [], []
    loss_ema_decay = 0.9
    ema_loss_val = None

    print(f"Start training: {n_way}-way {k_shot}-shot, Total Epochs: {epochs}...")

    for epoch in range(1, epochs + 1):
        # A. 获取数据 (适配 4 个返回值)
        X_vis, _, _, view_stats = sampler.get_episode(mode='train')

        B, T, C, H, W = X_vis.shape
        X_flat = X_vis.reshape(B * T, C, H, W).to(device)

        # B. 混合精度前向计算
        with torch.amp.autocast('cuda'):
            # 视觉特征提取
            f_vis = model.vis_backbone(X_flat)
            f_proj = model.proj_vis(f_vis)

            # 时序聚合 (B*T -> B, T, Dim -> B, Dim)
            f_final = f_proj.reshape(B, T, -1).max(dim=1)[0]

            # 准备原型网络输入
            f_reshaped = f_final.view(n_way, k_shot + q_query, -1)
            f_supp = f_reshaped[:, :k_shot].contiguous().view(n_way * k_shot, -1)
            f_query = f_reshaped[:, k_shot:].contiguous().view(n_way * q_query, -1)
            labels_query = torch.arange(n_way).repeat_interleave(q_query).to(device)

            loss, acc = criterion_proto(f_supp, f_query, labels_query, n_way, k_shot)

        # C. 反向传播与权重更新 (顺序修正 🔥)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 权重更新后再更新学习率
        scheduler.step()

        # 更新 Model EMA
        model_ema.update(model)

        # D. EMA 性能评估 (No Grad)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                f_e = model_ema.module.vis_backbone(X_flat)
                p_e = model_ema.module.proj_vis(f_e)
                f_final_ema = p_e.reshape(B, T, -1).max(dim=1)[0]

                f_reshaped_e = f_final_ema.view(n_way, k_shot + q_query, -1)
                f_supp_e = f_reshaped_e[:, :k_shot].contiguous().view(n_way * k_shot, -1)
                f_query_e = f_reshaped_e[:, k_shot:].contiguous().view(n_way * q_query, -1)

                _, acc_ema = criterion_proto(f_supp_e, f_query_e, labels_query, n_way, k_shot)

        # E. 日志统计
        if ema_loss_val is None:
            ema_loss_val = loss.item()
        else:
            ema_loss_val = loss_ema_decay * ema_loss_val + (1 - loss_ema_decay) * loss.item()

        epochs_list.append(epoch)
        loss_list.append(loss.item())
        acc_list.append(acc.item())
        ema_acc_list.append(acc_ema.item())

        if epoch % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            view_str = ", ".join([f"{k}:{v}" for k, v in view_stats.items() if v > 0])
            print(f"[Epoch {epoch}/{epochs}] Loss: {loss.item():.4f} | Acc: {acc.item():.4f} "
                  f"| EMA-Acc: {acc_ema.item():.4f} | LR: {current_lr:.6f} | Views: [{view_str}]")

    # 7. 保存模型
    save_dir = 'logs/checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'pag_fsl_vis_latest.pth'))
    torch.save(model_ema.module.state_dict(), os.path.join(save_dir, 'pag_fsl_vis_ema.pth'))
    print("✅ Training finished. Saved models.")

    # 8. 可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, loss_list, label='Train Loss')
    plt.title('Loss Curve');
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, acc_list, label='Batch Acc', alpha=0.3)
    plt.plot(epochs_list, ema_acc_list, label='EMA Acc', color='red')
    plt.title('Accuracy Curve');
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_metrics_vis_stream.png'))


if __name__ == "__main__":
    train()