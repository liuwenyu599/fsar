import torch
import os
import random
import numpy as np
import cv2
import pickle


class CASIASiluDataset:
    def __init__(self, silu_root, target_len=8):
        self.silu_root = silu_root
        self.target_len = target_len
        print(f"[CASIA-SILU] Loading from: {silu_root}")
        print("[CASIA-SILU] Mode: .pkl files (Pre-processed sequences)")

        self.all_sequences = self._scan_silu()
        self.all_subject_ids = sorted(list(self.all_sequences.keys()))
        print(f"[CASIA-SILU] Found {len(self.all_subject_ids)} subjects.")

    def _scan_silu(self):
        data = {}
        if not os.path.exists(self.silu_root):
            print("❌ silu root 不存在！")
            return data

        # 遍历 Subject (例如 001, 002...)
        for sid in sorted(os.listdir(self.silu_root)):
            sid_dir = os.path.join(self.silu_root, sid)
            if not os.path.isdir(sid_dir):
                continue
            data[sid] = {}

            # 遍历 Condition (例如 nm-01, bg-01...)
            for cond in sorted(os.listdir(sid_dir)):
                cond_dir = os.path.join(sid_dir, cond)
                if not os.path.isdir(cond_dir):
                    continue
                data[sid][cond] = {}

                # 遍历 View (例如 090.pkl 或 090文件夹)
                # 兼容逻辑：既支持 view.pkl 也支持 view/xxx.pkl
                for item in sorted(os.listdir(cond_dir)):
                    item_path = os.path.join(cond_dir, item)

                    # 情况 A: item 是一个 .pkl 文件 (例如 090.pkl)
                    if os.path.isfile(item_path) and item.endswith('.pkl'):
                        view_name = item.replace('.pkl', '')
                        # 存储 pkl 的绝对路径
                        data[sid][cond][view_name] = item_path

                    # 情况 B: item 是一个文件夹 (例如 090/)，里面可能包含 pkl
                    elif os.path.isdir(item_path):
                        view_name = item
                        # 看看里面有没有 .pkl
                        pkls = [p for p in os.listdir(item_path) if p.endswith('.pkl')]
                        if len(pkls) > 0:
                            # 假设文件夹里有一个主要的 pkl，或者取第一个
                            data[sid][cond][view_name] = os.path.join(item_path, pkls[0])

        # 修复点：return 必须在所有循环结束后，且变量名修正为 data
        return data

    def load_sequence(self, pkl_path):
        """
        读取 .pkl 文件并转换为 ViT 需要的 (T, 3, 224, 224)
        Args:
            pkl_path (str): .pkl 文件的绝对路径
        """
        try:
            with open(pkl_path, 'rb') as f:
                # 假设 pkl 加载出来是 (T, H, W) 的 numpy 数组
                seq_data = pickle.load(f)

            # 兼容性处理：有些 pkl 存的是 list
            if isinstance(seq_data, list):
                seq_data = np.array(seq_data)

            # 维度检查与修正
            # 如果是 (H, W) (单帧)，加一个维度 -> (1, H, W)
            if len(seq_data.shape) == 2:
                seq_data = seq_data[np.newaxis, ...]

            # 截取或填充逻辑
            # 如果原始长度 > 目标长度，直接切片 (或者可以改为均匀采样)
            if seq_data.shape[0] > self.target_len:
                seq_data = seq_data[:self.target_len]

            imgs = []
            for i in range(len(seq_data)):
                img = seq_data[i]  # (H, W)

                # Resize 到 224x224 (ViT 输入要求)
                # 如果图片非空则 resize，否则生成全黑图
                if img is not None and img.size > 0:
                    img = cv2.resize(img, (224, 224))
                else:
                    img = np.zeros((224, 224), dtype=np.float32)

                # 归一化 (0~255 -> 0.0~1.0) 并转 float32
                img = img.astype(np.float32) / 255.0

                # 扩展为 3 通道 (H, W) -> (3, H, W)
                img = np.stack([img, img, img], axis=0)

                imgs.append(torch.from_numpy(img))

            # 如果长度不够，补零帧 (Padding)
            while len(imgs) < self.target_len:
                imgs.append(torch.zeros(3, 224, 224))

            return torch.stack(imgs)  # (T, 3, 224, 224)

        except Exception as e:
            print(f"Error loading {pkl_path}: {e}")
            # 出错返回全0张量，避免训练崩溃
            return torch.zeros(self.target_len, 3, 224, 224)


class FewShotSampler:
    def __init__(self, dataset, n_way, k_shot, q_query):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        ids = dataset.all_subject_ids
        # 简单划分训练测试集 (8:2)
        split = int(0.8 * len(ids))
        # 边界检查：防止 split 为 0
        self.train_ids = ids[:split] if split > 0 else ids
        self.test_ids = ids[split:] if split > 0 else ids

    def get_episode(self, mode="train"):
        pool = self.train_ids if mode == "train" else self.test_ids

        # 容错：如果该模式下没有 ID，使用全集防止报错
        if len(pool) < self.n_way:
            pool = self.dataset.all_subject_ids

        sampled_ids = random.sample(pool, self.n_way)

        X = []
        Y = []

        for cls_idx, sid in enumerate(sampled_ids):
            all_seq_paths = []
            # 扁平化获取该 ID 下所有 Cond 和 View 的数据路径
            if sid in self.dataset.all_sequences:
                for cond in self.dataset.all_sequences[sid]:
                    for view in self.dataset.all_sequences[sid][cond]:
                        path = self.dataset.all_sequences[sid][cond][view]
                        all_seq_paths.append(path)

            needed = self.k_shot + self.q_query

            # 样本不足时的处理
            if len(all_seq_paths) == 0:
                print(f"⚠ Warning: Subject {sid} has no sequences!")
                # 塞 None 进去，load_sequence 会处理成全黑图
                selected_paths = [None] * needed
            elif len(all_seq_paths) < needed:
                # 重复采样以补足数量
                all_seq_paths = all_seq_paths * (needed // len(all_seq_paths) + 2)
                selected_paths = random.sample(all_seq_paths, needed)
            else:
                selected_paths = random.sample(all_seq_paths, needed)

            for pkl_path in selected_paths:
                if pkl_path is None:
                    # Dummy 数据
                    imgs = torch.zeros(self.dataset.target_len, 3, 224, 224)
                else:
                    # 调用修改后的加载逻辑
                    imgs = self.dataset.load_sequence(pkl_path)

                X.append(imgs)
                Y.append(cls_idx)

        X = torch.stack(X)  # (B, T, 3, 224, 224)
        Y = torch.tensor(Y)
        phase_w = torch.ones(X.shape[0], X.shape[1])  # 占位权重

        return X, Y, phase_w