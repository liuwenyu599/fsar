import os
import torch
import torch.optim as optim
import yaml
import argparse
from timm.utils import ModelEmaV2  # 🔥 引入 Model EMA 工具

# 确保 models/architecture.py 存在且 PAGFSLModel 类定义正确
from models.architecture_res import PAGFSLModel
from losses.prototypical_loss import PrototypicalLoss
from dataloader.silu_resnet_loader import CASIASiluDataset, FewShotSampler

def train(config_path='configs/config.yaml'):
    print("--- 🚀 PAG-FSL Training Start (SimpleCNN + Model EMA) ---")

    # ---------------------------
    # 1. 配置加载
    # ---------------------------
    if os.path.exists(config_path):
        config = yaml.safe_load(open(config_path))
    else:
        config = {
            'system': {'common_dim': 512, 'device': 'cuda'},
            'train': {'lr': 1e-3, 'epochs': 100},
            'few_shot': {'n_way': 5, 'k_shot': 5, 'q_query': 15, 'batch_size': 64}
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------------------
    # 2. 数据集 & 采样器
    # ---------------------------
    dataset = CASIASiluDataset('/datasets/CASIA-B/silu')
    # 5-way 5-shot
    n_way, k_shot, q_query = 5, 5, 15
    sampler = FewShotSampler(dataset, n_way, k_shot, q_query)

    # ---------------------------
    # 3. 初始化模型 (Main Model)
    # ---------------------------
    model = PAGFSLModel(common_dim=config['system']['common_dim']).to(device)
    model.train()

    # 全量解冻 (SimpleCNN 必需)
    for p in model.parameters():
        p.requires_grad = True

    # ---------------------------
    # 4. 初始化 EMA 模型 (Shadow Model)
    # ---------------------------
    # decay=0.999 是标准值，意味着当前权重只占 0.1%，历史权重占 99.9%
    # 这能极大平滑梯度的抖动
    print("--- Initializing Model EMA (decay=0.999) ---")
    model_ema = ModelEmaV2(model, decay=0.999, device=device)

    # ---------------------------
    # 5. 优化器 (强制 1e-3)
    # ---------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion_proto = PrototypicalLoss().to(device)
    scaler = torch.cuda.amp.GradScaler()

    # ---------------------------
    # 6. 训练循环
    # ---------------------------
    epochs = 100
    batch_size_vit = 128

    # 定义 Loss 平滑系数
    loss_ema_decay = 0.9
    ema_loss_val = None  # 初始化平滑 Loss

    print(f"Start training: {n_way}-way {k_shot}-shot...")

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

        # --- 🔥 计算 Loss EMA (用于日志平滑) ---
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

        # --- 🔥 验证 Model EMA 的性能 (额外跑一次前向，只为了看精度) ---
        # 注意：这里我们用 EMA 模型再算一遍 Acc，看看是不是更稳
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # 1. 提取特征 (使用 EMA 模型)
                # ModelEmaV2 包装了 module，所以要用 model_ema.module 来调用
                feats_ema = []
                for i in range(0, total_BT, batch_size_vit):
                    chunk = X_flat[i:i + batch_size_vit]
                    # 调用 shadow model
                    f_e = model_ema.module.vis_backbone(chunk)
                    p_e = model_ema.module.proj_vis(f_e)
                    feats_ema.append(p_e)

                # 2. 聚合与计算
                F_flat_ema = torch.cat(feats_ema, dim=0).float()
                f_final_ema = F_flat_ema.reshape(B, T, -1).max(dim=1)[0]

                f_reshaped_e = f_final_ema.view(n_way, k_shot + q_query, -1)
                f_supp_e = f_reshaped_e[:, :k_shot, :].contiguous().view(n_way * k_shot, -1)
                f_query_e = f_reshaped_e[:, k_shot:, :].contiguous().view(n_way * q_query, -1)

                # 计算 EMA 模型的 Acc (不需要算 loss，只看 acc)
                _, acc_ema = criterion_proto(f_supp_e, f_query_e, labels_query, n_way, k_shot)

        # 打印日志 (显示 Loss_EMA 和 Acc_EMA)
        if epoch % 5 == 0:
            print(f"[Epoch {epoch}/{epochs}] "
                  f"Loss: {loss.item():.4f} (EMA: {ema_loss_val:.4f}) | "
                  f"Acc: {acc:.4f} (EMA-Acc: {acc_ema:.4f})")
    # ---------------------------
    # 7. 保存模型 (保存两个版本)
    # ---------------------------
    save_dir = 'logs/checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # 保存普通模型 (最新权重)
    torch.save(model.state_dict(), os.path.join(save_dir, 'pag_fsl_vis_latest.pth'))

    # 🔥 保存 EMA 模型 (平滑权重，通常泛化更好)
    # 注意要取 .module，因为 ModelEmaV2 包装了一层
    torch.save(model_ema.module.state_dict(), os.path.join(save_dir, 'pag_fsl_vis_ema.pth'))

    print("✅ Training finished. Saved 'latest' and 'ema' checkpoints.")


if __name__ == "__main__":
    train()