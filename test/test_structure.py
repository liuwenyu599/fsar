import sys
import os

# 1. 显存与环境配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random

# 导入项目组件
from models.backbones.structural import StructuralBackbone
from models.backbones.lora_handler import inject_lora_to_motionbert
from models.heads.ppm import ViewAwarePPM
from losses.prototypical_loss import PrototypicalLoss
from dataloader.multloader import CASIABMultiDataset, FewShotSampler


# 🌟 采样器定义：与训练保持高度一致，返回 v_idx
class FewShotSamplerV2(FewShotSampler):
    def __init__(self, dataset, n_way, k_shot, q_query, active_views=None):
        self.dataset, self.n_way, self.k_shot, self.q_query = dataset, n_way, k_shot, q_query
        self.target_views = active_views if active_views is not None else \
            ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']

        self.view_to_idx = {v: i for i, v in enumerate(self.target_views)}
        self.view_to_ids = {v: [] for v in self.target_views}

        for sid, paths in dataset.all_sequences.items():
            for v in self.target_views:
                if any(v in p for p in paths): self.view_to_ids[v].append(sid)

        self.active_views = [v for v in self.target_views if len(self.view_to_ids[v]) >= n_way]
        print(f"📊 Sampler Initialized: {len(self.active_views)} active views.")

        all_ids = sorted(dataset.all_subject_ids)
        random.seed(42)
        shuffled_ids = all_ids[:]
        random.shuffle(shuffled_ids)
        split = int(0.8 * len(shuffled_ids))
        self.train_ids_all, self.test_ids_all = shuffled_ids[:split], shuffled_ids[split:]

    def get_episode(self, mode="train"):
        episode_view = random.choice(self.active_views)
        pool = self.train_ids_all if mode == "train" else self.test_ids_all
        candidates = [sid for sid in pool if sid in self.view_to_ids.get(episode_view, [])]
        if len(candidates) < self.n_way: candidates = pool

        sampled_ids = random.sample(candidates, self.n_way)
        X, Y = [], []
        view_stats = {v: 0 for v in self.target_views}

        for cls_idx, sid in enumerate(sampled_ids):
            all_paths = self.dataset.all_sequences.get(sid, [])
            valid_paths = [p for p in all_paths if episode_view in p]
            if not valid_paths: valid_paths = all_paths
            selected = random.sample(valid_paths * 5, self.k_shot + self.q_query)
            for pkl_path in selected:
                for v in self.target_views:
                    if v in pkl_path: view_stats[v] += 1
                X.append(self.dataset.load_pose(pkl_path))
                Y.append(cls_idx)

        # 🌟 核心修复：生成并返回视角索引
        v_idx = self.view_to_idx.get(episode_view, 0)
        v_idx_tensor = torch.full((len(X),), v_idx, dtype=torch.long)

        X = torch.stack(X)
        X = X.view(X.shape[0], X.shape[1], -1)
        return X, torch.tensor(Y), torch.ones(X.shape[0], X.shape[1]), v_idx_tensor, view_stats


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 50 + "\n🔍 Stage 2: Structural Stream Final Test\n" + "=" * 50)

    dataset = CASIABMultiDataset('/datasets/CASIA-B', mode='pose', seq_len=60)
    test_sampler = FewShotSamplerV2(dataset, n_way=5, k_shot=5, q_query=5)

    model = StructuralBackbone()
    model = inject_lora_to_motionbert(model, rank=16)
    model.to(device)

    with torch.no_grad():
        backbone_dim = model(torch.zeros(1, 60, 51).to(device)).shape[-1]

    # 🌟 使用视角感知聚合器
    ppm = ViewAwarePPM(feature_dim=backbone_dim).to(device)

    checkpoint_path = 'logs/checkpoints/ppgait_struct_best.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['struct_lora_state_dict'], strict=False)
        ppm.load_state_dict(checkpoint['ppm_state_dict'])
        print(f"✅ 成功加载最佳模型 (训练最高 Acc: {checkpoint.get('best_acc', 'unknown')})")
    else:
        print(f"❌ 警告: 未找到 {checkpoint_path}")

    model.eval()
    ppm.eval()
    criterion = PrototypicalLoss().to(device)

    n_test_episodes = 100
    all_accs = []
    view_performance = {v: [] for v in test_sampler.active_views}

    print(f"🚀 开始测试 {n_test_episodes} 个 Episode (测试集人物)...")
    with torch.no_grad():
        for i in tqdm(range(n_test_episodes)):
            # 🌟 核心修复：解包 5 个值，获取 v_idx
            X, _, W, v_idx, view_stats = test_sampler.get_episode(mode='test')
            X, W, v_idx = X.to(device), W.to(device), v_idx.to(device)

            feat_seq = F.normalize(model(X), p=2, dim=-1)

            # 🌟 核心修复：调用 ppm 时传入 v_idx
            f_final = ppm(feat_seq, W, v_idx)

            n, k, q = test_sampler.n_way, test_sampler.k_shot, test_sampler.q_query
            f_r = f_final.view(n, k + q, -1)
            q_labels = torch.arange(n).repeat_interleave(q).to(device)

            _, acc = criterion(f_r[:, :k].reshape(-1, backbone_dim),
                               f_r[:, k:].reshape(-1, backbone_dim), q_labels, n, k)

            all_accs.append(acc.item())

            current_view = [v for v, count in view_stats.items() if count > 0][0]
            view_performance[current_view].append(acc.item())

    print("\n" + "=" * 30)
    print(f"🏆 Final Test Acc (Overall): {np.mean(all_accs) * 100:.2f}%")
    print(f"📏 标准差: {np.std(all_accs) * 100:.2f}%")
    print("=" * 30)
    for v, accs in view_performance.items():
        if accs:
            print(f"  - View {v}: {np.mean(accs) * 100:.2f}% ({len(accs)} episodes)")


if __name__ == '__main__':
    test()