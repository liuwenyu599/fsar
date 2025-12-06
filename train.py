import torch
import torch.optim as optim
import yaml
import os
import argparse

# --- 导入项目模块 ---
# 确保您的 models/__init__.py 等文件已创建，Python 能识别包
from models.architecture import PAGFSLModel
from models.decoders.covariate_disc import CovariateDiscriminator
from losses.prototypical_loss import PrototypicalLoss
from losses.alignment_loss import AlignmentLoss
from losses.disentanglement_loss import DisentanglementLoss
from dataloader.casia_b_loader import few_shot_sampler


def train(config_path='configs/config.yaml'):
    print("--- 🚀 PAG-FSL Training Start ---")

    # 1. 加载配置 & 设备设定
    if os.path.exists(config_path):
        config = yaml.safe_load(open(config_path))
    else:
        # 如果配置文件还没生成，使用默认字典
        print("Warning: Config file not found, using defaults.")
        config = {
            'system': {'common_dim': 512, 'num_covariates': 4},
            'train': {'lr': 0.0001, 'epochs': 50},
            'few_shot': {'n_way': 5, 'k_shot': 1, 'q_query': 15},
            'weights': {'lambda_align': 0.5, 'lambda_disc': 0.01}
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 初始化模型
    # 主模型: 双流 + PPM + FAM
    model = PAGFSLModel(
        common_dim=config['system']['common_dim']
    ).to(device)

    # 辅助模型: 协变量判别器 (D_cov)
    discriminator = CovariateDiscriminator(
        input_dim=config['system']['common_dim'],
        num_covariates=config['system']['num_covariates']
    ).to(device)

    model.train()
    discriminator.train()

    # 3. 优化器 (Optimizers)
    # LoRA 参数和 Projection Head 参数都在 model.parameters() 里
    optimizer_model = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['train']['lr'])
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=config['train']['lr'])

    # 4. 损失函数 (Loss Functions)
    criterion_proto = PrototypicalLoss().to(device)
    criterion_align = AlignmentLoss().to(device)
    criterion_disc = DisentanglementLoss().to(device)

    # 5. 训练循环 (Training Loop)
    epochs = config['train']['epochs']
    print(f">>> Starting Training for {epochs} Epochs...")

    for epoch in range(1, epochs + 1):
        # --- A. 数据采样 (Simulated Episode) ---
        # 获取一个 N-way K-shot 任务的数据
        # X_vis: (Batch, Time, D)
        X_vis, X_struct, labels, cov_labels, phase_weights = few_shot_sampler(
            config['few_shot']['n_way'],
            config['few_shot']['k_shot'],
            config['few_shot']['q_query']
        )

        # 转移到 GPU
        X_vis, X_struct = X_vis.to(device), X_struct.to(device)
        labels, cov_labels = labels.to(device), cov_labels.to(device)
        phase_weights = phase_weights.to(device)

        # --- B. 前向传播 (Forward Pass) ---
        # F_final: 用于分类 (融合后)
        # f_vis: 用于解耦/对齐 (视觉流投影后)
        # f_struct: 用于对齐 (结构流投影后)
        f_final, f_vis, f_struct = model.forward_feature(X_vis, X_struct, phase_weights)

        # --- C. 数据切分 (Support / Query) ---
        n_way = config['few_shot']['n_way']
        k_shot = config['few_shot']['k_shot']
        n_support = n_way * k_shot

        # 切分用于 ProtoLoss 的特征
        f_final_supp = f_final[:n_support]
        f_final_query = f_final[n_support:]
        labels_query = labels[n_support:]

        # --- D. 损失计算 (Loss Calculation - Stage II) ---

        # 1. 主任务: 原型损失 (L_proto)
        loss_proto = criterion_proto(f_final_supp, f_final_query, labels_query, n_way, k_shot)

        # 2. 辅助任务: 对齐损失 (L_align) - 知识蒸馏
        # 强制视觉流(学生) 靠近 结构流(老师)
        loss_align = criterion_align(f_vis, f_struct)

        # 3. 辅助任务: 解耦损失 (L_disc - Extractor部分)
        # 目标: 让判别器猜错 (最大化熵 / 最小化准确率)
        disc_logits = discriminator(f_vis)
        loss_disc_extractor = criterion_disc(disc_logits, cov_labels, for_extractor=True)

        # 总损失 (加权求和)
        loss_total = loss_proto + \
                     config['weights']['lambda_align'] * loss_align + \
                     config['weights']['lambda_disc'] * loss_disc_extractor

        # --- E. 反向传播 (Update Model) ---
        optimizer_model.zero_grad()
        # retain_graph=True 是因为 f_vis 还要用来算判别器的梯度
        loss_total.backward(retain_graph=True)
        optimizer_model.step()

        # --- F. 反向传播 (Update Discriminator) ---
        # 训练判别器: 目标是猜对
        # 必须 detach()，防止梯度回传给骨干网络 (我们只更新判别器参数)
        disc_logits_detach = discriminator(f_vis.detach())
        loss_disc_d = criterion_disc(disc_logits_detach, cov_labels, for_extractor=False)

        optimizer_disc.zero_grad()
        loss_disc_d.backward()
        optimizer_disc.step()

        # --- 日志打印 ---
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}] "
                  f"Total: {loss_total.item():.4f} | "
                  f"Proto: {loss_proto.item():.4f} | "
                  f"Align: {loss_align.item():.4f} | "
                  f"Disc(E): {loss_disc_extractor.item():.4f}")

    # 6. 保存模型
    save_dir = 'logs/checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'pag_fsl_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training finished. Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()

    train(args.config)