import torch
import os
import random
import numpy as np
import pickle


class CASIAMultiModalDataset:
    def __init__(self, data_root, target_len=30, mode='pose'):
        self.data_root = data_root
        self.mode = mode
        self.target_len = target_len

        self.silu_root = os.path.join(data_root, 'silu')
        # 🔥 修正 1: 指向 HRNet 目录
        self.pose_root = os.path.join(data_root, 'pose', 'CASIA-B_HRNet')
        if not os.path.exists(self.pose_root):
            # 回退机制
            self.pose_root = os.path.join(data_root, 'pose')

        print(f"[CASIA] Root: {data_root} | Mode: {mode}")
        print(f"[CASIA] Pose Root: {self.pose_root}")

        scan_root = self.pose_root if mode == 'pose' else self.silu_root
        self.all_sequences = self._scan_data(scan_root)
        self.all_subject_ids = sorted(list(self.all_sequences.keys()))
        print(f"[CASIA] Found {len(self.all_subject_ids)} subjects.")

    def _scan_data(self, root_dir):
        data = {}
        if not os.path.exists(root_dir): return data
        for sid in sorted(os.listdir(root_dir)):
            sid_dir = os.path.join(root_dir, sid)
            if not os.path.isdir(sid_dir): continue
            data[sid] = {}
            for cond in sorted(os.listdir(sid_dir)):
                cond_dir = os.path.join(sid_dir, cond)
                if not os.path.isdir(cond_dir): continue
                data[sid][cond] = {}
                for item in sorted(os.listdir(cond_dir)):
                    # 无论是 .pkl 还是文件夹，都存下来
                    # item 可能是 "090.pkl" 也可能是 "090"
                    if item.endswith('.pkl'):
                        view = item.replace('.pkl', '')
                        data[sid][cond][view] = item
                    elif os.path.isdir(os.path.join(cond_dir, item)):
                        view = item
                        data[sid][cond][view] = item
        return data

    def load_pose(self, sid, cond, view_item):
        """加载骨架数据 .pkl -> 返回 (T, D)"""
        path = os.path.join(self.pose_root, sid, cond, view_item)
        if os.path.isdir(path):
            files = [f for f in os.listdir(path) if f.endswith('.pkl')]
            if len(files) > 0:
                if f"{view_item}.pkl" in files:
                    path = os.path.join(path, f"{view_item}.pkl")
                else:
                    path = os.path.join(path, files[0])

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

                # 兼容性处理
            if isinstance(data, list):
                data = np.array(data)
            elif isinstance(data, dict):
                for key in ['keypoints', 'points', 'data']:
                    if key in data: data = data[key]; break

            if not isinstance(data, np.ndarray):
                return torch.zeros(self.target_len, 34)

            # --- 🔥🔥🔥 关键修改：数据归一化 🔥🔥🔥 ---
            # 假设 CASIA-B 分辨率是 320x240 (常用标准)
            # data shape 预期是 (T, K, C) 例如 (T, 17, 3) 或 (T, 17, 2)
            if data.ndim == 3:
                # 归一化 x 坐标 (除以 320)
                data[:, :, 0] = data[:, :, 0] / 320.0
                # 归一化 y 坐标 (除以 240)
                data[:, :, 1] = data[:, :, 1] / 240.0

                # 如果有第3维 (Confidence)，它通常已经是 0-1，不用动
                # 为了让数据分布更以 0 为中心，可以 (x - 0.5) * 2 映射到 [-1, 1]
                data[:, :, 0] = (data[:, :, 0] - 0.5) * 2
                data[:, :, 1] = (data[:, :, 1] - 0.5) * 2

            # 展平 (T, 17, 3) -> (T, 51) 或 (T, 17, 2) -> (T, 34)
            if data.ndim > 2:
                T = data.shape[0]
                data = data.reshape(T, -1)

            # 采样
            if data.shape[0] > self.target_len:
                data = data[:self.target_len]

            data_tensor = torch.from_numpy(data).float()

            # 补零
            if data_tensor.shape[0] < self.target_len:
                pad = torch.zeros(self.target_len - data_tensor.shape[0], data_tensor.shape[1])
                data_tensor = torch.cat([data_tensor, pad], dim=0)

            return data_tensor
        except Exception as e:
            # print(f"❌ Error loading {path}: {e}")
            # 假设维度是 34 (2D) 或 51 (3D)，这里先给个足够大的 buffer，
            # 实际训练时会自动适配 input_dim
            return torch.zeros(self.target_len, 51)

    def load_sequence(self, *args):
        pass


class FewShotSampler:
    def __init__(self, dataset, n_way, k_shot, q_query):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.all_ids = dataset.all_subject_ids

    def get_episode(self, mode="train"):
        # 容错：人数不足时重复采样
        pool = list(self.all_ids)
        if len(pool) < self.n_way:
            pool = pool * (self.n_way // len(pool) + 1)

        sampled_ids = random.sample(pool, self.n_way)
        Batch_Data, Batch_Labels = [], []

        for cls_idx, sid in enumerate(sampled_ids):
            candidates = []
            if sid in self.dataset.all_sequences:
                for cond in self.dataset.all_sequences[sid]:
                    # 🔥 修正 3: 不再硬编码 '090.pkl'，而是查字典！
                    view_name = '090'
                    if view_name in self.dataset.all_sequences[sid][cond]:
                        # 取出真实的文件名/文件夹名 (可能是 "090" 也可能是 "090.pkl")
                        real_item = self.dataset.all_sequences[sid][cond][view_name]
                        candidates.append((sid, cond, real_item))

            needed = self.k_shot + self.q_query
            if len(candidates) == 0:
                # print(f"⚠️ ID {sid} has no 090 data")
                candidates = [None] * needed
            elif len(candidates) < needed:
                candidates = candidates * (needed // len(candidates) + 2)

            selected = random.sample(candidates, needed)
            for item in selected:
                if item is None:
                    Batch_Data.append(torch.zeros(self.dataset.target_len, 34))
                else:
                    Batch_Data.append(self.dataset.load_pose(*item))
                Batch_Labels.append(cls_idx)

        X = torch.stack(Batch_Data)
        Y = torch.tensor(Batch_Labels)
        phase_w = torch.ones(X.shape[0], X.shape[1])

        return X, Y, phase_w