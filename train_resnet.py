import os
import torch
import torch.optim as optim
import yaml
import argparse
import datetime
from timm.utils import ModelEmaV2
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

# 确保路径正确
from models.architecture_res import PAGFSLModel
from losses.prototypical_loss import PrototypicalLoss
from dataloader.silu_resnet_loader import CASIASiluDataset, FewShotSampler


def train(config_path='configs/config.yaml'):
    print("--- 🚀 PPGait Visual Stream Training (HPM + Memory Optimized) ---")

    # 1. 配置加载
    if os.path.exists(config_path):
        config = yaml.safe_load(open(config_path))
    else:
        config = {
            'system': {'common_dim': 512, 'device': 'cuda'},
            'train': {'lr': 1e-4, 'epochs': 150},  # HPM 建议使用稍低的学习率
            'few_shot': {'n_way': 5, 'k_shot': 5, 'q_query': 15}
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 数据集 & 采样器
    dataset = CASIASiluDataset('/datasets/CASIA-B/silu', target_len=8)
    n_way, k_shot, q_query = config['few_shot']['n_way'], config['few_shot']['k_shot'], config['few_shot']['q_query']
    sampler = FewShotSampler(dataset, n_way, k_shot, q_query)

    # 3. 初始化模型 (PAGFSLModel 内部已集成 HPM 和取消池化的 VisualBackbone)
    model = PAGFSLModel(common_dim=config['system']['common_dim']).to(device)
    model.train()

    # 4. 初始化 Model EMA
    print("--- Initializing Model EMA (decay=0.99) ---")
    model_ema = ModelEmaV2(model, decay=0.99, device=device)

    # 5. 优化器、Loss 和 调度器
    epochs = config['train']['epochs']
    # 针对 HPM 高维特征，建议稍微增加 weight_decay
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=5e-4)
    criterion_proto = PrototypicalLoss().to(device)

    scaler = torch.amp.GradScaler('cuda')
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 6. 推理参数设置
    # 🔥 核心修正：分批大小，防止 HPM 膨胀导致爆显存
    batch_size_vit = 32

    # 数据记录
    epochs_list, loss_list, acc_list, ema_acc_list = [], [], [], []
    loss_ema_decay = 0.9
    ema_loss_val = None
    best_ema_acc = 0.0

    print(f"Start training: {n_way}-way {k_shot}-shot, Total Epochs: {epochs}...")

    for epoch in range(1, epochs + 1):
        # A. 获取数据
        # X_vis: (B, T, C, H, W)
        X_vis, _, _, view_stats = sampler.get_episode(mode='train')
        B, T = X_vis.shape[0], X_vis.shape[1]

        # 构造模拟的结构流输入 (由于此时只练视觉流，我们传零向量给结构分支)
        X_struct_dummy = torch.zeros(B, T, 51).to(device)
        phase_w_dummy = torch.ones(B, T).to(device)

        # B. 训练前向传播 (利用 architecture_res.py 中的 forward_feature)
        optimizer.zero_grad()

        # 使用混合精度
        with torch.amp.autocast('cuda'):
            # 🔥 我们只关心视觉流特征 f_vis
            _, f_vis, _ = model.forward_feature(
                X_vis.to(device),
                X_struct_dummy,
                phase_w_dummy,
                batch_size=batch_size_vit
            )

            # 准备原型网络输入 (f_vis 已经是 [B, 512])
            f_reshaped = f_vis.view(n_way, k_shot + q_query, -1)
            f_supp = f_reshaped[:, :k_shot].contiguous().view(n_way * k_shot, -1)
            f_query = f_reshaped[:, k_shot:].contiguous().view(n_way * q_query, -1)
            labels_query = torch.arange(n_way).repeat_interleave(q_query).to(device)

            loss, acc = criterion_proto(f_supp, f_query, labels_query, n_way, k_shot)

        # C. 反向传播与更新
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        model_ema.update(model)

        # D. EMA 性能评估 (同样采用分批推理，避免显存溢出)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                # 注意：model_ema.module 调用 forward_feature
                _, f_vis_ema, _ = model_ema.module.forward_feature(
                    X_vis.to(device),
                    X_struct_dummy,
                    phase_w_dummy,
                    batch_size=batch_size_vit
                )

                f_reshaped_e = f_vis_ema.view(n_way, k_shot + q_query, -1)
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

        # F. 最优模型保存
        save_dir = 'logs/checkpoints'
        os.makedirs(save_dir, exist_ok=True)

        if curr_ema_acc > best_ema_acc:
            best_ema_acc = curr_ema_acc
            checkpoint = {
                'model_type': 'visual_stream_hpm',
                'best_acc_ema': best_ema_acc,
                'epoch': epoch,
                'episode_config': {'n_way': n_way, 'k_shot': k_shot, 'q_query': q_query},
                'view_config': {'target_views': sampler.target_views, 'view_lock': True},
                'state_dict': model_ema.module.state_dict(),
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            torch.save(checkpoint, os.path.join(save_dir, 'ppgait_vis_ema_best.pth'))
            print(f"   >>> 💾 [Best Updated] EMA Acc: {best_ema_acc:.4f}")

        if epoch % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            view_str = ", ".join([f"{k}:{v}" for k, v in view_stats.items() if v > 0])
            print(f"[Epoch {epoch}/{epochs}] Loss: {loss.item():.4f} | Acc: {acc.item():.4f} "
                  f"| EMA-Acc: {curr_ema_acc:.4f} | LR: {current_lr:.6f} | View: {view_str}")

    # 7. 最终保存
    torch.save(model.state_dict(), os.path.join(save_dir, 'ppgait_vis_latest.pth'))
    print(f"✅ Training finished. Best EMA Acc: {best_ema_acc:.4f}")

    # 8. 绘制曲线
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
    plt.savefig(os.path.join(save_dir, 'training_metrics_vis_stream_hpm.png'))


if __name__ == "__main__":
    train()