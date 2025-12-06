# train_proto_lora.py
import torch
import torch.optim as optim
import yaml
import os
import argparse

from models.architecture import PAGFSLModel
from losses.prototypical_loss import PrototypicalLoss
from dataloader.casia_silu_dataset import CASIASiluDataset, FewShotSampler


def train(config_path='configs/config.yaml'):
    print("--- 🚀 PAG-FSL Training Start ---")

    # ---------------------------
    # 1. 配置加载
    # ---------------------------
    if os.path.exists(config_path):
        config = yaml.safe_load(open(config_path))
        print(f"Loaded config: {config_path}")
    else:
        print("⚠ Config not found, using defaults.")
        config = {
            'system': {'common_dim': 512},
            'train': {'lr': 5e-5, 'epochs': 50},  # lr 调低
            'few_shot': {'n_way': 5, 'k_shot': 5, 'q_query': 15, 'batch_size': 64}
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------
    # 2. 数据集
    # ---------------------------
    dataset = CASBDataset('/datasets/CASIA-B')

    # ---------------------------
    # 3. 初始化模型
    # ---------------------------
    model = PAGFSLModel(common_dim=config['system']['common_dim']).to(device)

    # 冻结 backbone 参数，只训练 LoRA
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False

    model.train()

    # ---------------------------
    # 4. 优化器 & 损失
    # ---------------------------
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['train']['lr']
    )
    criterion_proto = PrototypicalLoss().to(device)

    # ---------------------------
    # 5. 训练循环
    # ---------------------------
    epochs = config['train']['epochs']
    n_way = config['few_shot']['n_way']
    k_shot = config['few_shot']['k_shot']
    q_query = config['few_shot']['q_query']
    batch_size_vit = config['few_shot'].get('batch_size', 64)

    ema_alpha = 0.9
    ema_proto_loss = None

    for epoch in range(1, epochs + 1):
        sampler = FewShotSampler(dataset, n_way, k_shot, q_query)
        X_vis, X_struct, labels, _, phase_weights = sampler.get_episode(mode='train')

        f_final, _, _ = model.forward_feature(
            X_vis.to(device),
            X_struct.to(device),
            phase_weights.to(device),
            batch_size=batch_size_vit
        )

        total_samples = n_way * (k_shot + q_query)
        support_idx = torch.arange(n_way * k_shot)
        query_idx = torch.arange(n_way * k_shot, total_samples)

        f_supp = f_final[support_idx].to(device)
        f_query = f_final[query_idx].to(device)
        labels_query = labels[query_idx].to(device)

        loss_proto, acc_proto = criterion_proto(f_supp, f_query, labels_query, n_way, k_shot)

        # EMA 更新
        if ema_proto_loss is None:
            ema_proto_loss = loss_proto.item()
        else:
            ema_proto_loss = ema_alpha * ema_proto_loss + (1 - ema_alpha) * loss_proto.item()

        optimizer.zero_grad()
        loss_proto.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"[Epoch {epoch}/{epochs}] Proto: {loss_proto.item():.4f} "
                  f"(Acc: {acc_proto:.4f}, EMA: {ema_proto_loss:.4f})")

    # ---------------------------
    # 6. 保存模型
    # ---------------------------
    save_dir = 'logs/checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'pag_fsl_lora.pth')
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training finished. Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    train(args.config)
