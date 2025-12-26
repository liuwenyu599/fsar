import os
import torch
import random
import numpy as np
import cv2
import pickle
from torch.utils.data import Dataset


class CASIASiluDataset:
    def __init__(self, silu_root, target_len=8):
        self.silu_root = silu_root
        self.target_len = target_len
        print(f"[CASIA-SILU] Loading from: {silu_root}")

        # 扫描数据并建立结构化字典
        self.all_sequences = self._scan_silu()
        self.all_subject_ids = sorted(list(self.all_sequences.keys()))
        print(f"[CASIA-SILU] Found {len(self.all_subject_ids)} subjects.")

    def _scan_silu(self):
        data = {}
        if not os.path.exists(self.silu_root):
            return data

        for sid in sorted(os.listdir(self.silu_root)):
            sid_dir = os.path.join(self.silu_root, sid)
            if not os.path.isdir(sid_dir): continue
            data[sid] = {}

            for cond in sorted(os.listdir(sid_dir)):
                cond_dir = os.path.join(sid_dir, cond)
                if not os.path.isdir(cond_dir): continue
                data[sid][cond] = {}

                for item in sorted(os.listdir(cond_dir)):
                    item_path = os.path.join(cond_dir, item)
                    if item.endswith('.pkl'):
                        view_name = item.replace('.pkl', '')
                        data[sid][cond][view_name] = item_path
                    elif os.path.isdir(item_path):
                        view_name = item
                        pkls = [p for p in os.listdir(item_path) if p.endswith('.pkl')]
                        if pkls:
                            data[sid][cond][view_name] = os.path.join(item_path, pkls[0])
        return data

    def load_sequence(self, pkl_path):
        """将轮廓序列转化为 (T, 3, 224, 224) 适配 ResNet"""
        try:
            with open(pkl_path, 'rb') as f:
                seq_data = pickle.load(f)
            if isinstance(seq_data, list): seq_data = np.array(seq_data)

            if seq_data.shape[0] > self.target_len:
                indices = np.linspace(0, seq_data.shape[0] - 1, self.target_len, dtype=int)
                seq_data = seq_data[indices]

            imgs = []
            for i in range(len(seq_data)):
                img = seq_data[i]
                if img is not None and img.size > 0:
                    img = cv2.resize(img, (224, 224))
                else:
                    img = np.zeros((224, 224), dtype=np.float32)

                img = img.astype(np.float32) / 255.0
                img = np.stack([img, img, img], axis=0)
                imgs.append(torch.from_numpy(img))

            while len(imgs) < self.target_len:
                imgs.append(torch.zeros(3, 224, 224))
            return torch.stack(imgs)
        except:
            return torch.zeros(self.target_len, 3, 224, 224)


class FewShotSampler:
    def __init__(self, dataset, n_way, k_shot, q_query):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        self.target_views = ['090', '180', '054']
        self.view_to_ids = {v: [] for v in self.target_views}

        for sid, cond_dict in dataset.all_sequences.items():
            for cond, view_dict in cond_dict.items():
                for v in self.target_views:
                    if v in view_dict:
                        if sid not in self.view_to_ids[v]:
                            self.view_to_ids[v].append(sid)

        self.active_views = []
        print("--- 🔍 Visual Sampler Health Check ---")
        for v in self.target_views:
            count = len(self.view_to_ids[v])
            if count >= self.n_way:
                self.active_views.append(v)
                print(f"✅ View {v}: {count} subjects")
            else:
                print(f"❌ View {v}: {count} subjects (Skipped)")

        ids = dataset.all_subject_ids
        random.seed(42)
        shuffled = ids[:]
        random.shuffle(shuffled)
        split = int(0.8 * len(shuffled))
        self.train_ids = shuffled[:split]
        self.test_ids = shuffled[split:]

    def get_episode(self, mode="train"):
        episode_view = random.choice(self.active_views if self.active_views else ['all'])
        pool = self.train_ids if mode == "train" else self.test_ids
        candidates = [sid for sid in pool if sid in self.view_to_ids.get(episode_view, [])]

        if len(candidates) < self.n_way:
            candidates = self.view_to_ids.get(episode_view, self.dataset.all_subject_ids)

        sampled_ids = random.sample(candidates, self.n_way)
        X, Y = [], []
        view_stats = {v: 0 for v in self.target_views}

        for cls_idx, sid in enumerate(sampled_ids):
            all_seq_paths = []
            if sid in self.dataset.all_sequences:
                for cond in self.dataset.all_sequences[sid]:
                    if episode_view in self.dataset.all_sequences[sid][cond]:
                        all_seq_paths.append(self.dataset.all_sequences[sid][cond][episode_view])

            needed = self.k_shot + self.q_query
            if not all_seq_paths:
                selected_paths = [None] * needed
            else:
                selected_paths = random.sample(all_seq_paths * (needed // len(all_seq_paths) + 1), needed)

            for pkl_path in selected_paths:
                view_stats[episode_view] = view_stats.get(episode_view, 0) + 1
                if pkl_path:
                    X.append(self.dataset.load_sequence(pkl_path))
                else:
                    X.append(torch.zeros(self.dataset.target_len, 3, 224, 224))
                Y.append(cls_idx)

        X = torch.stack(X)
        return X, torch.tensor(Y), torch.ones(X.shape[0], X.shape[1]), view_stats


# --- 🔥 Main 测试代码 🔥 ---
if __name__ == "__main__":
    # 1. 设置路径 (请根据你的实际路径修改)
    SILU_PATH = "/datasets/CASIA-B/silu"

    # 2. 检查路径是否存在
    if not os.path.exists(SILU_PATH):
        print(f"❌ 错误: 路径 {SILU_PATH} 不存在。请修改 SILU_PATH 变量。")
    else:
        # 3. 初始化数据集
        ds = CASIASiluDataset(silu_root=SILU_PATH, target_len=8)

        # 4. 初始化采样器 (5-way 1-shot 1-query 用于快速测试)
        n_way, k_shot, q_query = 5, 1, 1
        sampler = FewShotSampler(ds, n_way=n_way, k_shot=k_shot, q_query=q_query)

        print("\n--- 🚀 正在模拟获取一个训练 Episode ---")
        X, Y, W, V = sampler.get_episode(mode="train")

        # 5. 打印结果检查
        print(f"✅ 图像张量形状 [B, T, C, H, W]: {X.shape}")
        # 预期输出: [10, 8, 3, 224, 224] (因为 5-way * (1+1) = 10)

        print(f"✅ 标签张量 [B]: {Y}")
        print(f"✅ 视角锁定统计: {V}")

        # 6. 检查像素值范围
        print(f"✅ 像素值范围: [{X.min():.2f}, {X.max():.2f}] (预期 [0.00, 1.00])")

        # 7. 模拟多次采样，检查视角锁死逻辑是否生效
        print("\n--- 🚀 视角锁死逻辑稳定性测试 ---")
        for i in range(3):
            _, _, _, v_stats = sampler.get_episode()
            active_v = [k for k, val in v_stats.items() if val > 0]
            print(f"Episode {i + 1} 锁定的视角为: {active_v}")