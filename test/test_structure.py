import sys
import os

# 1. 环境配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random

# 环境与路径配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones.structural import StructuralBackbone
from models.backbones.lora_handler import inject_view_lora  # 确保是 PlainLoRA 版本
from losses.prototypical_loss import PrototypicalLoss
from dataloader.multloader import CASIABMultiDataset, FewShotSampler


# --- 1. 与训练一致的聚合器 ---
class StructuralPPM(torch.nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x, phase_w):
        phase_w = F.softmax(phase_w, dim=1).unsqueeze(-1)
        return torch.sum(x * phase_w, dim=1)


# --- 2. 视角感知采样器 ---
class FewShotSamplerV2(FewShotSampler):
    def __init__(self, dataset, n_way, k_shot, q_query, active_views=None):
        self.dataset, self.n_way, self.k_shot, self.q_query = dataset, n_way, k_shot, q_query
        self.target_views = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
        self.view_to_idx = {v: i for i, v in enumerate(self.target_views)}
        self.view_to_ids = {v: [] for v in self.target_views}
        for sid, paths in dataset.all_sequences.items():
            for v in self.target_views:
                if any(v in p for p in paths): self.view_to_ids[v].append(sid)
        self.active_views = [v for v in self.target_views if len(self.view_to_ids[v]) >= n_way]

        all_ids = sorted(dataset.all_subject_ids)
        random.seed(42)
        shuffled_ids = all_ids[:]
        random.shuffle(shuffled_ids)
        split = int(0.8 * len(shuffled_ids))
        self.test_ids_all = shuffled_ids[split:]  # 仅使用测试集

    def get_episode(self):
        episode_view = random.choice(self.active_views)
        candidates = [sid for sid in self.test_ids_all if sid in self.view_to_ids.get(episode_view, [])]
        if len(candidates) < self.n_way: candidates = self.test_ids_all
        sampled_ids = random.sample(candidates, self.n_way)
        X, Y = [], []
        for cls_idx, sid in enumerate(sampled_ids):
            all_paths = self.dataset.all_sequences.get(sid, [])
            valid_paths = [p for p in all_paths if episode_view in p] or all_paths
            selected = random.sample(valid_paths * 5, self.k_shot + self.q_query)
            for pkl_path in selected:
                X.append(self.dataset.load_pose(pkl_path))
                Y.append(cls_idx)
        v_idx = self.view_to_idx.get(episode_view, 0)
        return torch.stack(X).view(len(X), 60, -1), torch.tensor(Y), torch.ones(len(X), 60), episode_view


# --- 3. 测试主程序 ---
def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 50 + "\n🏁 Stage 2: View-Adversarial Performance Test\n" + "=" * 50)

    dataset = CASIABMultiDataset('/datasets/CASIA-B', mode='pose', seq_len=60)
    test_sampler = FewShotSamplerV2(dataset, n_way=5, k_shot=5, q_query=5)

    # 1. 初始化 Backbone 并注入 PlainLoRA
    model = StructuralBackbone().to(device)
    model = inject_view_lora(model, rank=16)
    model.to(device)

    # 获取维度并初始化 PPM
    with torch.no_grad():
        backbone_dim = model(torch.zeros(1, 60, 51).to(device), view_idx=None).shape[-1]
    ppm = StructuralPPM(feature_dim=backbone_dim).to(device)

    # 2. 加载对抗训练权重
    checkpoint_path = 'logs/checkpoints/ppgait_struct_adversarial.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['struct_backbone'])
        ppm.load_state_dict(checkpoint['ppm_state_dict'])
        print(f"✅ 成功加载对抗模型 (Best Acc recorded: {checkpoint.get('best_acc', 'N/A')})")
    else:
        print(f"❌ 错误: 未找到权重文件 {checkpoint_path}")
        return

    model.eval()
    ppm.eval()
    criterion = PrototypicalLoss().to(device)

    n_test_episodes = 200  # 增加测试 Episode 以降低偶然性
    all_accs = []
    view_performance = {v: [] for v in test_sampler.target_views}

    print(f"🚀 开始跨视角测试 {n_test_episodes} 个 Episode...")
    with torch.no_grad():
        for _ in tqdm(range(n_test_episodes)):
            X, _, W, v_name = test_sampler.get_episode()
            X, W = X.to(device), W.to(device)

            # 🌟 核心：不传入 view_idx，测试 Backbone 的纯净提取能力
            feat_seq = F.normalize(model(X, view_idx=None), p=2, dim=-1)
            f_final = ppm(feat_seq, W)

            n, k, q = test_sampler.n_way, test_sampler.k_shot, test_sampler.q_query
            f_r = f_final.view(n, k + q, -1)
            q_labels = torch.arange(n).repeat_interleave(q).to(device)

            _, acc = criterion(f_r[:, :k].reshape(-1, backbone_dim),
                               f_r[:, k:].reshape(-1, backbone_dim), q_labels, n, k)

            all_accs.append(acc.item())
            view_performance[v_name].append(acc.item())

    print("\n" + "=" * 30)
    print(f"🏆 Final Adversarial Test Acc: {np.mean(all_accs) * 100:.2f}%")
    print(f"📏 标准差: {np.std(all_accs) * 100:.2f}%")
    print("=" * 30)
    for v in sorted(view_performance.keys()):
        accs = view_performance[v]
        if accs:
            print(f"  - View {v}: {np.mean(accs) * 100:.2f}% ({len(accs)} episodes)")


if __name__ == '__main__':
    test()