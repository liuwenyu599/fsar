import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pickle
import cv2

# 1. 环境与路径配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 🌟 确保导入的是包含 TemporalConv 和 Cat-Pooling 的最新模型
from models.backbones.visual_resnet import PureGaitResNet
from dataloader.silu_resnet_loader import CASIASiluDataset


class BaselineEvaluator:
    def __init__(self, model_path, data_root, device='cuda'):
        self.device = torch.device(device)

        # 初始化模型 (内部会自动构建新版架构)
        self.model = PureGaitResNet(common_dim=512).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 权重路径错误: {model_path}")

        print(f"📦 正在加载时序增强版权重: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # 移除 'module.' 前缀以兼容不同保存格式
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}

        # 🌟 这里的 strict=True 会严格校验架构是否对齐
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.eval()

        # 保持与训练一致的预处理 (重心对齐 + 16帧)
        self.dataset = CASIASiluDataset(data_root, target_len=16)
        self.views = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
        self.test_subjects = self.dataset.subjects[74:]  # CASIA-B 测试集划分

    def extract_all_features(self):
        """ 提取包含时序动态信息的特征向量 """
        feature_bank = {}
        print(f"--- 🚀 正在提取测试集特征 (时序建模模式) ---")

        for sub in tqdm(self.test_subjects):
            feature_bank[sub] = {}
            for view in self.dataset.index[sub]:
                feature_bank[sub][view] = {}
                for pkl_path in self.dataset.index[sub][view]:
                    parts = pkl_path.split('/')
                    cond = parts[-3].lower()

                    with torch.no_grad():
                        # 加载对齐后的 16 帧数据
                        X_vis = self.dataset.load_frames(pkl_path).unsqueeze(0).to(self.device)

                        # 🌟 核心：此时 forward 会通过 Temporal Conv 提取帧间关联
                        f_vis = self.model(X_vis, batch_size=4)

                        feat = f_vis.cpu().numpy().flatten()
                        # L2 归一化，用于后续计算余弦相似度
                        feat = feat / (np.linalg.norm(feat) + 1e-9)
                        feature_bank[sub][view][cond] = feat
        return feature_bank

    def run_eval(self, bank, g_cond='nm-01', p_cond='nm-05'):
        """ 计算跨视角 Rank-1 准确率矩阵 """
        matrix = np.zeros((11, 11))
        for g_idx, g_view in enumerate(self.views):
            # 准备 Gallery
            g_feats, g_labels = [], []
            for sub in self.test_subjects:
                if g_view in bank[sub] and g_cond in bank[sub][g_view]:
                    g_feats.append(bank[sub][g_view][g_cond])
                    g_labels.append(sub)

            if not g_feats: continue
            g_feats = np.array(g_feats)

            # 准备 Probe
            for p_idx, p_view in enumerate(self.views):
                if g_view == p_view: continue

                correct, total = 0, 0
                for sub in self.test_subjects:
                    if p_view in bank[sub] and p_cond in bank[sub][p_view]:
                        p_feat = bank[sub][p_view][p_cond]
                        # 矩阵点积即余弦相似度
                        similarities = np.dot(g_feats, p_feat)
                        if g_labels[np.argmax(similarities)] == sub:
                            correct += 1
                        total += 1
                matrix[p_idx, g_idx] = correct / total if total > 0 else 0
        return matrix

    def print_matrix(self, matrix, title):
        print(f"\n{f' {title} ':=>80}")
        header = "       " + " ".join([f"{v:>6}" for v in self.views])
        print(header)
        print("-" * 80)
        for i, row in enumerate(matrix):
            row_str = f"{self.views[i]:>6} |"
            for val in row:
                row_str += f"{val * 100:>6.1f}" if val > 0 else "   -  "
            print(row_str)

        # 🌟 统计平均精度
        mean_acc = matrix[matrix > 0].mean()
        print("-" * 80)
        print(f"✨ NM-05 跨视角平均精度 (Mean Acc): {mean_acc * 100:.2f}%")
        print("=" * 80)


if __name__ == "__main__":
    evaluator = BaselineEvaluator(
        model_path='logs/checkpoints/baseline_vis_best.pth',  # 确保这是新练的权重
        data_root='/datasets/CASIA-B/silu'
    )
    bank = evaluator.extract_all_features()
    nm_matrix = evaluator.run_eval(bank, g_cond='nm-01', p_cond='nm-05')
    evaluator.print_matrix(nm_matrix, "Temporal-Aware Baseline: NM-05 Rank-1")