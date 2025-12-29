import torch
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 导入你之前的模块
from models.backbones.structural import StructuralBackbone
from models.heads.ppm import PPMStructuredAggregator
from losses.prototypical_loss import PrototypicalLoss
from dataloader.multloader import CASIABMultiDataset, FewShotSampler


# ---------------------------------------------------------
# 1. 验证函数
# ---------------------------------------------------------
def evaluate(model, ppm, sampler, device, n_episodes=20):
    model.eval()
    ppm.eval()
    accs, losses = [], []
    criterion = PrototypicalLoss().to(device)
    total_view_stats = {v: 0 for v in sampler.target_views}

    with torch.no_grad():
        for i in range(n_episodes):
            try:
                X_struct, labels, phase_w, view_stats = sampler.get_episode(mode='test')
                X_struct, phase_w = X_struct.to(device), phase_w.to(device)

                for k, v in view_stats.items():
                    if k in total_view_stats: total_view_stats[k] += v

                f_final = ppm(model(X_struct), phase_w)
                n_way, k_shot, q_query = sampler.n_way, sampler.k_shot, sampler.q_query

                f_reshaped = f_final.view(n_way, k_shot + q_query, -1)
                f_supp = f_reshaped[:, :k_shot].contiguous().view(-1, 512)
                f_query = f_reshaped[:, k_shot:].contiguous().view(-1, 512)
                q_labels = torch.arange(n_way).repeat_interleave(q_query).to(device)

                loss, acc = criterion(f_supp, f_query, q_labels, n_way, k_shot)
                accs.append(acc.item())
                losses.append(loss.item())
            except Exception as e:
                if i == 0: print(f"  ⚠️ Eval Episode Error: {e}")
                continue

    model.train()
    ppm.train()
    return np.mean(losses) if losses else 0.0, np.mean(accs) if accs else 0.0, total_view_stats


# ---------------------------------------------------------
# 2. 增强型保存函数 (存为 .pth)
# ---------------------------------------------------------
def save_best_checkpoint(model, ppm, sampler, acc, step, input_dim, common_dim, save_dir):
    """
    科研加强版保存逻辑：不仅存权重，还存实验的“灵魂”（采样配置）
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, 'ppgait_struct_best.pth')

    checkpoint = {
        'model_type': 'structural_stream',
        'best_acc': acc,
        'step': step,
        # --- 🔧 建议 1：存 Few-Shot 设定 ---
        'episode_config': {
            'n_way': sampler.n_way,
            'k_shot': sampler.k_shot,
            'q_query': sampler.q_query,
        },
        # --- 🔧 建议 2：存视角策略 ---
        'view_config': {
            'target_views': sampler.target_views,
            'active_views': sampler.active_views,
            'view_lock': True  # 标记你的核心创新点
        },
        'config': {
            'input_dim': input_dim,
            'common_dim': common_dim,
        },
        'backbone_state_dict': model.state_dict(),
        'ppm_state_dict': ppm.state_dict(),
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    torch.save(checkpoint, file_path)
    print(f"   >>> 💾 [Scientific Checkpoint Saved] Acc: {acc:.4f} at Step {step}")


# ---------------------------------------------------------
# 3. 主训练循环
# ---------------------------------------------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- 🚀 Stage 2: Structural Stream Training (Episodic View-Locking) ---")

    # A. 数据准备
    dataset = CASIABMultiDataset('/datasets/CASIA-B', mode='pose', seq_len=60)
    train_sampler = FewShotSampler(dataset, n_way=5, k_shot=5, q_query=4)

    # B. 模型初始化 (自动探测维度)
    X_init, _, _, _ = train_sampler.get_episode(mode='train')
    input_dim = X_init.shape[-1]  # 自动识别是 51 还是其他
    common_dim = 512

    model = StructuralBackbone(input_dim=input_dim, embed_dim=common_dim).to(device)
    ppm = PPMStructuredAggregator(feature_dim=common_dim).to(device)

    # 冻结 MotionBERT，只练 Adapter
    for param in model.parameters(): param.requires_grad = False
    print(f">>> Backbone Frozen. Training Adapter (PPM) only. Input Dim: {input_dim}")

    optimizer = torch.optim.AdamW(ppm.parameters(), lr=0.0005, weight_decay=1e-4)
    total_steps = 1000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = PrototypicalLoss().to(device)

    # 训练监控
    val_steps, val_accs = [], []
    best_val_acc = 0.0
    save_dir = 'logs/checkpoints'

    for step in range(1, total_steps + 1):
        # 采样 (已锁定 Episode 视角)
        X_struct, _, phase_w, view_stats = train_sampler.get_episode(mode='train')
        X_struct, phase_w = X_struct.to(device), phase_w.to(device)

        # 前向计算
        f_final = ppm(model(X_struct), phase_w)

        n_way, k_shot, q_query = train_sampler.n_way, train_sampler.k_shot, train_sampler.q_query
        f_reshaped = f_final.view(n_way, k_shot + q_query, -1)
        f_supp = f_reshaped[:, :k_shot].contiguous().view(-1, common_dim)
        f_query = f_reshaped[:, k_shot:].contiguous().view(-1, common_dim)
        q_labels = torch.arange(n_way).repeat_interleave(q_query).to(device)

        loss, acc = criterion(f_supp, f_query, q_labels, n_way, k_shot)

        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 日志打印
        if step % 20 == 0:
            view_str = ", ".join([f"{k}:{v}" for k, v in view_stats.items() if v > 0])
            print(f"[Step {step}] Loss: {loss.item():.4f} | Acc: {acc.item():.4f} | Views: [{view_str}]")

        # 验证与保存
        if step % 50 == 0:
            v_loss, v_acc, v_views = evaluate(model, ppm, train_sampler, device)
            v_view_str = ", ".join([f"{k}:{v}" for k, v in v_views.items() if v > 0])
            print(f"   >>> 🔍 [Val] Acc: {v_acc:.4f} | Loss: {v_loss:.4f} | Views: [{v_view_str}]")

            val_steps.append(step)
            val_accs.append(v_acc)

            if v_acc > best_val_acc:
                best_val_acc = v_acc
                save_best_checkpoint(model, ppm, train_sampler,v_acc, step, input_dim, common_dim, save_dir)

    # 绘制简易曲线
    plt.figure();
    plt.plot(val_steps, val_accs);
    plt.title('Val Acc');
    plt.savefig(os.path.join(save_dir, 'val_curve.png'))
    print(f"✅ Training Finished. Best Acc: {best_val_acc:.4f}")


if __name__ == '__main__':
    train()