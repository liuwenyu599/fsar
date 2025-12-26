import os
import torch
import torch.optim as optim
import yaml
import argparse
import datetime  # 引入时间戳
from timm.utils import ModelEmaV2
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

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

    # 2. 数据集 & 采样器
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

    scaler = torch.amp.GradScaler('cuda')
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 数据记录
    epochs_list, loss_list, acc_list, ema_acc_list = [], [], [], []
    loss_ema_decay = 0.9
    ema_loss_val = None
    best_ema_acc = 0.0  # 🔥 用于追踪最优模型

    print(f"Start training: {n_way}-way {k_shot}-shot, Total Epochs: {epochs}...")

    for epoch in range(1, epochs + 1):
        # A. 获取数据
        X_vis, _, _, view_stats = sampler.get_episode(mode='train')
        B, T, C, H, W = X_vis.shape
        X_flat = X_vis.reshape(B * T, C, H, W).to(device)

        # B. 混合精度前向计算
        with torch.amp.autocast('cuda'):
            f_vis = model.vis_backbone(X_flat)
            f_proj = model.proj_vis(f_vis)
            f_final = f_proj.reshape(B, T, -1).max(dim=1)[0]

            f_reshaped = f_final.view(n_way, k_shot + q_query, -1)
            f_supp = f_reshaped[:, :k_shot].contiguous().view(n_way * k_shot, -1)
            f_query = f_reshaped[:, k_shot:].contiguous().view(n_way * q_query, -1)
            labels_query = torch.arange(n_way).repeat_interleave(q_query).to(device)

            loss, acc = criterion_proto(f_supp, f_query, labels_query, n_way, k_shot)

        # C. 反向传播与权重更新 (顺序修正)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        model_ema.update(model)

        # D. EMA 性能评估
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
        curr_ema_acc = acc_ema.item()
        if ema_loss_val is None:
            ema_loss_val = loss.item()
        else:
            ema_loss_val = loss_ema_decay * ema_loss_val + (1 - loss_ema_decay) * loss.item()

        epochs_list.append(epoch)
        loss_list.append(loss.item())
        acc_list.append(acc.item())
        ema_acc_list.append(curr_ema_acc)

        # 🔥 F. 加强版保存逻辑：保存最优 EMA 模型 🔥
        save_dir = 'logs/checkpoints'
        os.makedirs(save_dir, exist_ok=True)

        if curr_ema_acc > best_ema_acc:
            best_ema_acc = curr_ema_acc
            checkpoint = {
                'model_type': 'visual_stream',
                'best_acc_ema': best_ema_acc,
                'epoch': epoch,
                # --- 🔧 存 Few-Shot 设定 ---
                'episode_config': {
                    'n_way': n_way,
                    'k_shot': k_shot,
                    'q_query': q_query,
                },
                # --- 🔧 存视角策略 ---
                'view_config': {
                    'target_views': sampler.target_views,
                    'active_views': sampler.active_views,
                    'view_lock': True
                },
                'config': {
                    'common_dim': config['system']['common_dim'],
                },
                'state_dict': model_ema.module.state_dict(),  # 保存平滑后的 EMA 权重
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            torch.save(checkpoint, os.path.join(save_dir, 'ppgait_vis_ema_best.pth'))
            print(f"   >>> 💾 [Scientific Checkpoint] Best EMA Acc Updated: {best_ema_acc:.4f}")

        if epoch % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            view_str = ", ".join([f"{k}:{v}" for k, v in view_stats.items() if v > 0])
            print(f"[Epoch {epoch}/{epochs}] Loss: {loss.item():.4f} | Acc: {acc.item():.4f} "
                  f"| EMA-Acc: {curr_ema_acc:.4f} | LR: {current_lr:.6f} | Views: [{view_str}]")

    # 7. 保存最后的模型作为备份
    torch.save(model.state_dict(), os.path.join(save_dir, 'pag_fsl_vis_latest.pth'))
    print(f"✅ Training finished. Best EMA Acc: {best_ema_acc:.4f}")

    # 8. 可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, loss_list, label='Train Loss');
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