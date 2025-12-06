import torch
import yaml
import os
import argparse

# --- 导入项目模块 ---
from models.architecture import PAGFSLModel
from dataloader.casia_b_loader import few_shot_sampler
from utils.metrics import calculate_rank1


def test(config_path='configs/config.yaml', checkpoint_path='logs/checkpoints/pag_fsl_model.pth'):
    print("--- 🔍 PAG-FSL Evaluation Start ---")

    # 1. 加载配置 & 设备
    if os.path.exists(config_path):
        config = yaml.safe_load(open(config_path))
    else:
        print("Warning: Config file not found, using defaults.")
        config = {'system': {'common_dim': 512}, 'few_shot': {'n_way': 5, 'k_shot': 1, 'q_query': 15}}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 初始化模型
    model = PAGFSLModel(
        common_dim=config['system']['common_dim']
    ).to(device)

    # 3. 加载权重 (Checkpoint)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"⚠️ Warning: Checkpoint {checkpoint_path} not found! Testing with random weights.")

    # 切换到评估模式 (关闭 Dropout, BatchNorm 统计停止更新)
    model.eval()

    # 4. 评估循环 (Evaluation Loop)
    print("Evaluating on Novel Identities (Simulated Episode)...")

    # 禁用梯度计算，节省显存并加速
    with torch.no_grad():
        # --- A. 数据采样 ---
        # 模拟获取一个测试 Episode 的数据
        X_vis, X_struct, labels, _, phase_weights = few_shot_sampler(
            config['few_shot']['n_way'],
            config['few_shot']['k_shot'],
            config['few_shot']['q_query']
        )

        X_vis, X_struct = X_vis.to(device), X_struct.to(device)
        phase_weights = phase_weights.to(device)

        # --- B. 前向推理 ---
        # 我们只需要 F_final 来做最终识别
        f_final, _, _ = model.forward_feature(X_vis, X_struct, phase_weights)

        # --- C. 计算指标 ---
        # 模拟计算 Rank-1 精度
        # 在真实代码中，这里需要计算 Probe (Query) 和 Gallery (Support) 的距离矩阵
        acc = calculate_rank1(f_final, labels)

    print(f"🏆 Final Rank-1 Accuracy: {acc:.2f}% (Simulated Result)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='logs/checkpoints/pag_fsl_model.pth',
                        help='Path to checkpoint')
    args = parser.parse_args()

    test(args.config, args.checkpoint)