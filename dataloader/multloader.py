import torch
import os
import random
import numpy as np
import pickle
from torch.utils.data import Dataset


class CASIABMultiDataset(Dataset):
    """
    CASIA-B 多模态数据集加载器 (骨骼流专用)
    """

    def __init__(self, root, mode="pose", seq_len=60):
        self.root = root
        self.mode = mode
        self.seq_len = seq_len
        self.pose_root = os.path.join(root, "pose", "CASIA-B_HRNet")

        print(f"[CASIA] Root: {root} | Mode: {mode} | SeqLen: {seq_len}")
        self.samples, self.all_subject_ids, self.all_sequences = self._collect_pose_samples()
        print(f"[CASIA] Found {len(self.all_subject_ids)} subjects and {len(self.samples)} total sequences.")

    def _collect_pose_samples(self):
        samples = []
        subject_ids = []
        all_sequences = {}  # 字典结构: { '001': [path1, path2...], '002': [...] }

        if not os.path.exists(self.pose_root):
            print(f"❌ Error: {self.pose_root} does not exist!")
            return [], [], {}

        pids = sorted(os.listdir(self.pose_root))
        for pid in pids:
            pid_dir = os.path.join(self.pose_root, pid)
            if not os.path.isdir(pid_dir): continue

            subject_ids.append(pid)
            all_sequences[pid] = []

            for cond in sorted(os.listdir(pid_dir)):
                cond_dir = os.path.join(pid_dir, cond)
                if not os.path.isdir(cond_dir): continue
                for view in sorted(os.listdir(cond_dir)):
                    view_dir = os.path.join(cond_dir, view)
                    if not os.path.isdir(view_dir): continue
                    for fname in sorted(os.listdir(view_dir)):
                        if fname.endswith(".pkl"):
                            full_path = os.path.join(view_dir, fname)
                            samples.append(full_path)
                            all_sequences[pid].append(full_path)
        return samples, subject_ids, all_sequences

    def load_pose(self, pkl_path):
        """核心：物理语义归一化 (H36M 映射 + Root-Center + 1/128 Scale)"""
        try:
            with open(pkl_path, "rb") as f:
                raw_data = pickle.load(f)

            pose = raw_data.get("keypoints", raw_data.get("data", raw_data)) if isinstance(raw_data, dict) else raw_data
            pose = np.asarray(pose, dtype=np.float32)

            if pose.ndim == 2:
                pose = pose.reshape(pose.shape[0], 17, -1)

            T_orig = pose.shape[0]
            final_pose = np.zeros((T_orig, 17, 3), dtype=np.float32)
            coords = pose[:, :, :2]

            # --- 物理语义映射: COCO-17 -> H36M-17 ---
            l_hip, r_hip, l_sho, r_sho = coords[:, 11], coords[:, 12], coords[:, 5], coords[:, 6]
            pelvis, neck = (l_hip + r_hip) / 2.0, (l_sho + r_sho) / 2.0

            final_pose[:, 0, :2] = pelvis  # 0: Pelvis
            final_pose[:, 1, :2], final_pose[:, 4, :2] = r_hip, l_hip  # 1:R-Hip, 4:L-Hip
            final_pose[:, 2, :2], final_pose[:, 5, :2] = coords[:, 14], coords[:, 13]  # 2:R-Kne, 5:L-Kne
            final_pose[:, 3, :2], final_pose[:, 6, :2] = coords[:, 16], coords[:, 15]  # 3:R-Ank, 6:L-Ank
            final_pose[:, 7, :2], final_pose[:, 8, :2] = (pelvis + neck) / 2.0, neck  # 7:Spine, 8:Neck
            final_pose[:, 9, :2] = coords[:, 0]  # 9: Nose
            final_pose[:, 10, :2] = coords[:, 0] + (coords[:, 0] - neck)  # 10: Head
            final_pose[:, 11, :2], final_pose[:, 14, :2] = l_sho, r_sho  # 11:L-Sho, 14:R-Sho
            final_pose[:, 12, :2], final_pose[:, 15, :2] = coords[:, 7], coords[:, 8]  # 12:L-Elb, 15:R-Elb
            final_pose[:, 13, :2], final_pose[:, 16, :2] = coords[:, 9], coords[:, 10]  # 13:L-Wri, 16:R-Wri
            final_pose[:, :, 2] = 1.0

            # --- Root-Centering & Scaling ---
            # 减去骨盆(0号点)，缩放128倍。确保骨盆始终在 (0,0,1)
            final_pose[:, :, :2] = (final_pose[:, :, :2] - final_pose[:, 0:1, :2]) / 128.0

            if T_orig >= self.seq_len:
                final_pose = final_pose[:self.seq_len]
            else:
                final_pose = np.tile(final_pose, ((self.seq_len // T_orig) + 1, 1, 1))[:self.seq_len]

            return torch.from_numpy(final_pose)
        except Exception:
            return torch.zeros((self.seq_len, 17, 3))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.load_pose(self.samples[idx])


class FewShotSampler:
    def __init__(self, dataset, n_way, k_shot, q_query):
        self.dataset, self.n_way, self.k_shot, self.q_query = dataset, n_way, k_shot, q_query

        # 1. 建立视角索引 & 存为成员变量以修复 AttributeError
        self.target_views = ['090', '180', '054']
        self.view_to_ids = {v: [] for v in self.target_views}

        for sid, paths in dataset.all_sequences.items():
            for v in self.target_views:
                if any(v in p for p in paths):
                    self.view_to_ids[v].append(sid)

        # 2. 筛选健康视角
        self.active_views = []
        print("--- 🔍 Sampler View Health Check ---")
        for v in self.target_views:
            count = len(self.view_to_ids[v])
            if count >= self.n_way:
                self.active_views.append(v)
                print(f"✅ View {v}: {count} subjects (Active)")
            else:
                print(f"❌ View {v}: {count} subjects (Skipped)")

        if not self.active_views:
            self.active_views = ['all']
            print("⚠️ Warning: No specific views are healthy. Falling back to all-view mode.")

        # 3. 划分训练/测试集
        all_ids = sorted(dataset.all_subject_ids)
        random.seed(42)
        shuffled_ids = all_ids[:]
        random.shuffle(shuffled_ids)

        split = int(0.8 * len(shuffled_ids))
        self.train_ids_all = shuffled_ids[:split]
        self.test_ids_all = shuffled_ids[split:]

    def get_episode(self, mode="train"):
        # 1. 选定本轮视角 (Episode-level Lock)
        episode_view = random.choice(self.active_views)
        pool = self.train_ids_all if mode == "train" else self.test_ids_all

        # 2. 确定候选人 (包含测试集借人逻辑防 nan)
        candidates = [sid for sid in pool if sid in self.view_to_ids.get(episode_view, [])]
        if len(candidates) < self.n_way:
            candidates = self.view_to_ids.get(episode_view, self.dataset.all_subject_ids)

        sampled_ids = random.sample(candidates, self.n_way)

        X, Y = [], []
        view_stats = {v: 0 for v in self.target_views}

        for cls_idx, sid in enumerate(sampled_ids):
            all_paths = self.dataset.all_sequences.get(sid, [])
            # 只取锁定视角的数据
            if episode_view != 'all':
                valid_paths = [p for p in all_paths if episode_view in p]
            else:
                valid_paths = all_paths

            if not valid_paths: valid_paths = all_paths

            needed = self.k_shot + self.q_query
            selected = random.sample(valid_paths * (needed // len(valid_paths) + 1), needed)

            for pkl_path in selected:
                # 统计各视角分布 (用于日志显示)
                for v in self.target_views:
                    if v in pkl_path: view_stats[v] += 1
                X.append(self.dataset.load_pose(pkl_path))
                Y.append(cls_idx)

        X = torch.stack(X)
        X = X.view(X.shape[0], X.shape[1], -1)  # 展平为 [Batch, T, 51]

        return X, torch.tensor(Y), torch.ones(X.shape[0], X.shape[1]), view_stats

if __name__ == "__main__":
    # 快速验证脚本
    ds = CASIABMultiDataset("/datasets/CASIA-B", mode="pose")
    sampler = FewShotSampler(ds, n_way=5, k_shot=1, q_query=1)
    X, Y, W, V = sampler.get_episode()
    print("X Shape (Should be [10, 60, 51]):", X.shape)
    print("Locked View Stats:", V)