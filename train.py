# train_pagfsl.py
import torch
import torch.optim as optim
import yaml
import os
import argparse

# --- 项目模块 ---
from models.architecture import PAGFSLModel
from models.decoders.covariate_disc import CovariateDiscriminator
from losses.prototypical_loss import PrototypicalLoss
from losses.alignment_loss import AlignmentLoss
from losses.disentanglement_loss import DisentanglementLoss
from dataloader.casia_b_loader import FewShotSampler, CASBDataset


def train(config_path='configs/config.yaml'):
    print("--- 🚀 PAG-FSL Training Start ---")

    # ---------------------------
    # 1. 加载配置
    # ---------------------------
    if os.path.exists(config_path):
        config = yaml.safe_load(open(config_path))
        print(f"Loaded config: {config_path}")
    else:
        print("⚠ Config not found, using defaults.")
        config = {
            'system': {'common_dim': 512, 'num_covariates': 4},
            'train': {'lr': 1e-4, 'epochs': 50},
            'few_shot': {'n_way': 5, 'k_shot': 5, 'q_query': 15, 'batch_size': 64},
            'weights': {'lambda_align': 0.5, 'lambda_disc': 0.01}
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------
    # 2. 数据集
    # ---------------------------
    default_root = '/datasets/CASIA-B'
    dataset = CASBDataset(default_root)

    # ---------------------------
    # 3. 初始化模型
    # ---------------------------
    model = PAGFSLModel(common_dim=config['system']['common_dim']).to(device)
    discriminator = CovariateDiscriminator(
        input_dim=config['system']['common_dim'],
        num_covariates=config['system']['num_covariates']
    ).to(device)

    model.train()
    discriminator.train()

    # ---------------------------
    # 4. 优化器 & 损失
    # ---------------------------
    optimizer_model = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['train']['lr']
    )
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=config['train']['lr'])

    criterion_proto = PrototypicalLoss().to(device)
    criterion_align = AlignmentLoss().to(device)
    criterion_disc = DisentanglementLoss(
        num_covariates=config['system']['num_covariates']
    ).to(device)

    # ---------------------------
    # 5. 训练循环
    # ---------------------------
    epochs = config['train']['epochs']
    n_way = config['few_shot']['n_way']
    k_shot = config['few_shot']['k_shot']
    q_query = config['few_shot']['q_query']
    batch_size_vit = config['few_shot'].get('batch_size', 64)
    ema_alpha = 0.9  # EMA 平滑系数
    ema_proto_loss = None
    for epoch in range(1, epochs + 1):
        # --- A. Few-shot episode sampling ---
        sampler = FewShotSampler(dataset, n_way, k_shot, q_query)
        X_vis, X_struct, labels, cov_labels, phase_weights = sampler.get_episode(mode='train')

        # --- B. 特征融合 ---
        f_final, f_vis, f_struct = model.forward_feature(
            X_vis.to(device),
            X_struct.to(device),
            phase_weights.to(device),
            batch_size=batch_size_vit
        )

        # --- C. 支撑集 / 查询集划分 ---
        total_samples = n_way * (k_shot + q_query)
        if f_final.size(0) != total_samples:
            raise ValueError(f"f_final shape {f_final.size(0)} != total episode samples {total_samples}")

        support_idx = torch.arange(n_way * k_shot)
        query_idx = torch.arange(n_way * k_shot, total_samples)

        f_final_supp = f_final[support_idx].to(device)
        f_final_query = f_final[query_idx].to(device)
        labels_query = labels[query_idx].to(device)

        # --- D. 损失计算 ---
        loss_proto, acc_proto = criterion_proto(f_final_supp, f_final_query, labels_query, n_way, k_shot)
        # loss_align = criterion_align(f_vis, f_struct)
        # disc_logits = discriminator(f_vis)
        # loss_disc_extractor = criterion_disc(disc_logits, cov_labels.to(device), for_extractor=True)
        #
        # loss_total = loss_proto + config['weights']['lambda_align'] * loss_align + \
        #              config['weights']['lambda_disc'] * loss_disc_extractor

        # --- EMA 更新 ---
        if ema_proto_loss is None:
            ema_proto_loss = loss_proto.item()
        else:
            ema_proto_loss = ema_alpha * ema_proto_loss + (1 - ema_alpha) * loss_proto.item()
        # --- E. 更新模型 ---
        optimizer_model.zero_grad()
        # loss_total.backward()
        # optimizer_model.step()

        # --- F. 更新判别器 ---
        # disc_logits_detach = discriminator(f_vis.detach())
        # loss_disc_d = criterion_disc(disc_logits_detach, cov_labels.to(device), for_extractor=False)

        # optimizer_disc.zero_grad()
        # loss_disc_d.backward()
        loss_proto.backward()
        optimizer_disc.step()

        # --- G. 日志 ---
        if epoch % 5 == 0:
            # print(f"[Epoch {epoch}/{epochs}] Total: {loss_total.item():.4f} | "
            #       f"Proto: {loss_proto.item():.4f} (Acc: {acc_proto:.4f}) | "
            #       f"Align: {loss_align.item():.4f} | Disc(E): {loss_disc_extractor.item():.4f} | Disc(D): {loss_disc_d.item():.4f}"
            #       f"(EMA: {ema_proto_loss:.4f}) "
            #       )
            print(f"[Epoch {epoch}/{epochs}] Proto: {loss_proto.item():.4f} "
                  f"(Acc: {acc_proto:.4f}, EMA: {ema_proto_loss:.4f})")
    # ---------------------------
    # 6. 保存模型
    # ---------------------------
    save_dir = 'logs/checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'pag_fsl_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training finished. Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    train(args.config)
