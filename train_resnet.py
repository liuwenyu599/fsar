import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from timm.utils import ModelEmaV2
import matplotlib.pyplot as plt

# 🌟 环境配置：防止显存碎裂化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 核心组件导入
from models.backbones.visual_resnet import PureGaitResNet
from dataloader.silu_resnet_loader import CASIASiluDataset, FewShotSampler


def compute_strict_proto_loss(f, n_way, k_shot, q_query):
    """
    🌟 核心损失函数：多视角原型损失 + 类内一致性约束
    """
    # 1. 拆分 Support 和 Query
    # f 形状: [N*(K+Q), 512]
    f_support = f[:n_way * k_shot].view(n_way, k_shot, -1)
    f_query = f[n_way * k_shot:].view(n_way, q_query, -1)

    # 2. 计算多视角混合原型 (Multi-View Prototypes)
    # [n_way, 512]
    prototypes = f_support.mean(dim=1)

    # 3. 计算 Query 到原型的距离 (分类损失)
    f_query_flat = f_query.view(n_way * q_query, -1)
    # 采用平方欧氏距离，对 Hard Case 更敏感
    dists = torch.cdist(f_query_flat, prototypes)

    query_labels = torch.arange(n_way).repeat_interleave(q_query).to(f.device)
    loss_proto = F.cross_entropy(-dists, query_labels)

    # 4. 🌟 类内一致性约束 (Alignment Loss)
    # 强制让 Support Set 里的不同视角特征向原型靠拢
    loss_const = 0
    for i in range(n_way):
        # 计算第 i 类 support 样本到其原型的 L2 距离均值
        diff = f_support[i] - prototypes[i].unsqueeze(0)
        loss_const += torch.mean(torch.norm(diff, p=2, dim=1))
    loss_const /= n_way

    # 计算准确率
    preds = torch.argmin(dists, dim=1)
    acc = (preds == query_labels).float().mean()

    return loss_proto, loss_const, acc


def train():
    print("\n" + "=" * 60)
    print("🚀 Stage 1: Strict Multi-View Alignment Training")
    print("=" * 60 + "\n")

    # --- 1. 环境与路径 ---
    os.makedirs('logs/checkpoints', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 超参数配置
    # 🌟 n_way 改为 8，增加任务复杂度，防止模型“记答案”
    common_dim, n_way, k_shot, q_query = 512, 5, 5, 10
    target_len = 16

    # --- 2. 数据准备 ---
    dataset = CASIASiluDataset('/datasets/CASIA-B/silu', target_len=target_len)
    # 使用我们重构的“多视角混合”采样器
    sampler = FewShotSampler(dataset, n_way, k_shot, q_query)

    # --- 3. 模型初始化 ---
    model = PureGaitResNet(common_dim=common_dim).to(device)
    model.train()
    # 引入 EMA 提高测试集的鲁棒性
    model_ema = ModelEmaV2(model, decay=0.99, device=device)

    # 使用 AdamW 配合余弦退火
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-3)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    history = {'total_loss': [], 'proto_loss': [], 'const_loss': [], 'acc': []}
    best_acc = 0.0

    for epoch in range(1, 101):
        # A. 采样：获取多视角 Episode
        X_vis, Y, _, _ = sampler.get_episode(mode='train')
        X_vis = X_vis.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            # B. 前向传播 (返回 L2 归一化后的特征)
            f_vis = model(X_vis, batch_size=2)

            # C. 增强版损失计算
            loss_proto, loss_const, acc = compute_strict_proto_loss(f_vis, n_way, k_shot, q_query)

            # 🌟 联合 Loss：分类为主，对齐为辅
            # $$L_{total} = L_{proto} + 0.1 \times L_{const}$$
            loss_total = loss_proto + 0.1 * loss_const

        # D. 反向传播
        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        model_ema.update(model)

        # E. 记录
        history['total_loss'].append(loss_total.item())
        history['proto_loss'].append(loss_proto.item())
        history['const_loss'].append(loss_const.item())
        history['acc'].append(acc.item())

        if acc.item() > best_acc:
            best_acc = acc.item()
            torch.save(model_ema.module.state_dict(), 'logs/checkpoints/baseline_vis_best.pth')
            print(f"   >>> 💾 Epoch {epoch} | Best Acc: {best_acc:.4f} | Loss: {loss_total.item():.4f}")

        if epoch % 10 == 0:
            print(
                f"--- [Epoch {epoch}] Loss: {loss_total.item():.4f} | Alignment: {loss_const.item():.4f} | Acc: {acc.item():.4f} ---")

    # --- 4. 可视化 ---
    plot_training_results(history)
    print(f"✅ 训练圆满完成。最终最佳训练 Episode Acc: {best_acc:.4f}")


def plot_training_results(history):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['proto_loss'], label='Proto Loss')
    plt.title('Classification Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['const_loss'], color='green', label='Const Loss')
    plt.title('Alignment (Intra-class)')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['acc'], color='orange', label='Train Acc')
    plt.title('Episode Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('logs/checkpoints/alignment_report.png')


if __name__ == "__main__":
    train()