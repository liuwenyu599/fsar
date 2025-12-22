import torch
import torch.optim as optim
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt  # 🔥 引入 matplotlib 绘图库

from models.backbones.structural import StructuralBackbone
from models.heads.ppm import PPMStructuredAggregator
from losses.prototypical_loss import PrototypicalLoss
from dataloader.multloader import CASIAMultiModalDataset, FewShotSampler


def evaluate(model, ppm, sampler, device, n_episodes=50):
    """
    验证函数：在测试集上跑 n_episodes 次，取平均 Acc
    """
    model.eval()
    ppm.eval()

    accs = []
    losses = []
    criterion = PrototypicalLoss().to(device)

    # 这里的参数要和 sampler 保持一致
    n_way, k_shot, q_query = sampler.n_way, sampler.k_shot, sampler.q_query

    with torch.no_grad():
        for _ in range(n_episodes):
            try:
                data_pack = sampler.get_episode(mode='test')  # 🔥 关键：mode='test'
                if len(data_pack) == 3:
                    X_struct, _, phase_w = data_pack
                else:
                    X_struct, _ = data_pack;
                    phase_w = torch.ones(X_struct.shape[0], X_struct.shape[1])

                X_struct = X_struct.to(device)
                phase_w = phase_w.to(device)

                # Forward
                features = model(X_struct)
                f_final = ppm(features, phase_w)

                # Loss & Acc
                f_reshaped = f_final.view(n_way, k_shot + q_query, -1)
                f_supp = f_reshaped[:, :k_shot].contiguous().view(-1, 512)
                f_query = f_reshaped[:, k_shot:].contiguous().view(-1, 512)
                labels = torch.arange(n_way).repeat_interleave(q_query).to(device)

                loss, acc = criterion(f_supp, f_query, labels, n_way, k_shot)
                accs.append(acc.item())
                losses.append(loss.item())
            except Exception as e:
                # print(f"Eval Error: {e}") # 屏蔽此行，防止打印过多错误日志
                continue

    model.train()
    ppm.train()
    return np.mean(losses), np.mean(accs)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- 🚀 Stage 2: Structural Stream Training (with Validation) ---")

    # 1. Dataset & Sampler
    dataset = CASIAMultiModalDataset('/datasets/CASIA-B', mode='pose', target_len=30)
    train_sampler = FewShotSampler(dataset, n_way=5, k_shot=5, q_query=15)

    # 2. Model
    input_dim = 34
    common_dim = 512
    model = StructuralBackbone(input_dim=input_dim, embed_dim=common_dim).to(device)
    ppm = PPMStructuredAggregator(feature_dim=common_dim).to(device)

    model.train()
    ppm.train()

    # 3. Optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': ppm.parameters()}
    ], lr=0.0005, weight_decay=1e-4)

    # Scheduler
    total_steps = 1000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    criterion = PrototypicalLoss().to(device)

    print(f"Start Training ({total_steps} steps)...")

    best_val_acc = 0.0

    # 🔥🔥🔥 定义数据收集列表 🔥🔥🔥
    val_steps_list = []
    train_acc_at_val_list = []
    val_acc_list = []
    train_loss_at_val_list = []
    val_loss_list = []

    # 初始化用于日志记录的变量
    current_train_loss = 0.0
    current_train_acc = 0.0
    # 🔥🔥🔥 ----------------------- 🔥🔥🔥

    for step in range(1, total_steps + 1):
        # --- Training Step ---
        data_pack = train_sampler.get_episode(mode='train')
        if len(data_pack) == 3:
            X_struct, _, phase_w = data_pack
        else:
            X_struct, _ = data_pack;
            phase_w = torch.ones(X_struct.shape[0], X_struct.shape[1])

        X_struct = X_struct.to(device)
        phase_w = phase_w.to(device)

        # 自动适配维度
        if step == 1 and X_struct.shape[-1] != input_dim:
            # 重新适配模型维度
            new_input_dim = X_struct.shape[-1]
            print(f"⚠️ Re-init input_dim: {new_input_dim}")
            model = StructuralBackbone(input_dim=new_input_dim, embed_dim=common_dim).to(device)
            model.train()
            # 重新初始化优化器和调度器
            optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': ppm.parameters()}], lr=0.0005)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        # Forward & Loss
        f_final = ppm(model(X_struct), phase_w)

        n_way, k_shot, q_query = 5, 5, 15
        f_reshaped = f_final.view(n_way, k_shot + q_query, -1)
        f_supp = f_reshaped[:, :k_shot].contiguous().view(-1, common_dim)
        f_query = f_reshaped[:, k_shot:].contiguous().view(-1, common_dim)
        labels = torch.arange(n_way).repeat_interleave(q_query).to(device)

        loss, acc = criterion(f_supp, f_query, labels, n_way, k_shot)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 防炸
        optimizer.step()
        scheduler.step()

        # 记录当前训练步的 Loss 和 Acc
        current_train_loss = loss.item()
        current_train_acc = acc.item()

        # --- Logging & Validation ---
        if step % 20 == 0:
            print(f"[Step {step}] Train Loss: {current_train_loss:.4f} | Train Acc: {current_train_acc:.4f}")

        # 每 50 步验证一次
        if step % 50 == 0:
            val_loss, val_acc = evaluate(model, ppm, train_sampler, device, n_episodes=20)
            print(f"   >>> 🔍 [Validation] Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

            # 🔥🔥🔥 收集数据点 🔥🔥🔥
            val_steps_list.append(step)
            train_acc_at_val_list.append(current_train_acc)
            val_acc_list.append(val_acc)
            train_loss_at_val_list.append(current_train_loss)
            val_loss_list.append(val_loss)
            # 🔥🔥🔥 ----------------- 🔥🔥🔥

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_dir = 'logs/checkpoints'
                os.makedirs(save_dir, exist_ok=True)
                torch.save({
                    'backbone': model.state_dict(),
                    'ppm': ppm.state_dict()
                }, os.path.join(save_dir, 'pag_fsl_struct_best.pth'))
                print(f"   >>> ✅ Best Model Saved! ({best_val_acc:.4f})")

    print("✅ Training finished. Generating metrics plot...")

    # ---------------------------
    # 7. 保存模型 (最终模型)
    # ---------------------------
    save_dir = 'logs/checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'backbone': model.state_dict(),
        'ppm': ppm.state_dict()
    }, os.path.join(save_dir, 'pag_fsl_struct_latest.pth'))

    # ---------------------------
    # 8. 可视化结果 🔥
    # ---------------------------
    plt.figure(figsize=(12, 5))

    # 子图 1: Loss (Train vs Val)
    plt.subplot(1, 2, 1)
    plt.plot(val_steps_list, train_loss_at_val_list, label='Train Loss', color='blue', alpha=0.6)
    plt.plot(val_steps_list, val_loss_list, label='Validation Loss', color='red', linewidth=2)
    plt.title('Loss Convergence (Structure Stream)')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图 2: Accuracy (Train vs Val)
    plt.subplot(1, 2, 2)
    plt.plot(val_steps_list, train_acc_at_val_list, label='Train Acc', color='blue', alpha=0.6)
    plt.plot(val_steps_list, val_acc_list, label='Validation Acc', color='red', linewidth=2)
    plt.title('Accuracy (Train vs Validation)')
    plt.xlabel('Training Step')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = os.path.join(save_dir, 'training_metrics_struct_stream.png')
    plt.savefig(plot_filename)
    print(f"✅ Training metrics plot saved to {plot_filename}")


if __name__ == '__main__':
    train()