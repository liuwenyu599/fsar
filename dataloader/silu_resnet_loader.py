import os
import torch
import cv2
import numpy as np
import pickle
import random
from torch.utils.data import Dataset
from torchvision import transforms


def align_silhouette(img, target_size=(224, 224)):
    """
    步态剪影重心对齐标准化 (Centroid Alignment)
    """
    # 找到所有非零像素
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        return np.zeros(target_size, dtype=np.uint8)

    # 1. 提取人体边界
    y_min, y_max = np.min(ys), np.max(ys)
    x_min, x_max = np.min(xs), np.max(xs)

    # 2. 计算横向重心 (Centroid) - 比取中点更稳健
    x_center = int(np.mean(xs))

    # 3. 裁剪人体
    body = img[y_min:y_max, x_min:x_max]

    # 4. 保持比例缩放
    h_target, w_target = target_size
    ratio = h_target / (y_max - y_min + 1e-6)
    new_w = int((x_max - x_min) * ratio)
    new_w = min(new_w, w_target)

    body_resized = cv2.resize(body, (new_w, h_target))

    # 5. 画布居中对齐
    aligned_img = np.zeros(target_size, dtype=np.uint8)
    start_x = w_target // 2 - new_w // 2
    aligned_img[:, start_x:start_x + new_w] = body_resized

    return aligned_img


class CASIASiluDataset(Dataset):
    def __init__(self, data_root, target_len=16, img_size=(224, 224)):
        self.data_root = data_root
        self.target_len = target_len
        self.img_size = img_size
        self.subjects = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

        # 索引格式: {sub_id: {view: [pkl_paths]}}
        self.index = self._build_index()

        # 标准化：针对 ResNet 预训练权重
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.RandomHorizontalFlip(p=0.5),  # 模拟镜像视角增强
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _build_index(self):
        index = {}
        print(f"--- 🔍 正在建立数据集索引: {self.data_root} ---")
        for sub in self.subjects:
            index[sub] = {}
            sub_path = os.path.join(self.data_root, sub)
            for cond in os.listdir(sub_path):
                # 我们暂时不区分 cond，或者在后续逻辑中筛选 NM
                cond_path = os.path.join(sub_path, cond)
                if not os.path.isdir(cond_path): continue
                for view in os.listdir(cond_path):
                    view_path = os.path.join(cond_path, view)
                    if not os.path.isdir(view_path): continue
                    if view not in index[sub]: index[sub][view] = []
                    for pkl in os.listdir(view_path):
                        if pkl.endswith('.pkl'):
                            index[sub][view].append(os.path.join(view_path, pkl))
        return index

    def load_frames(self, pkl_path):
        """
        核心加载函数：包含读取、采样、重心对齐
        """
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        except:
            # 容错处理
            return torch.zeros(self.target_len, 3, *self.img_size)

        # 采样 T 帧
        if len(data) >= self.target_len:
            start = random.randint(0, len(data) - self.target_len)
            frames = data[start:start + self.target_len]
        else:
            # 帧数不足循环补齐
            indices = np.arange(len(data))
            pad = np.random.choice(indices, self.target_len - len(data))
            frames = np.concatenate([data, data[pad]], axis=0)

        processed = []
        for frame in frames:
            # 执行对齐！
            aligned = align_silhouette(frame, target_size=self.img_size)
            # 扩展为 3 通道 Tensor
            processed.append(self.transform(aligned))

        return torch.stack(processed)  # [T, 3, H, W]


class FewShotSampler:
    """
    强化版多视角采样器 (Multi-View Prototype Sampler)
    核心逻辑：
    1. Support Set 包含同一个人在不同视角下的样本，强制 Prototype 融合多视角信息。
    2. Query Set 采样自该人未在 Support 中出现的视角，强制跨视角推理。
    """

    def __init__(self, dataset, n_way=5, k_shot=5, q_query=10):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.all_views = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']

    def get_episode(self, mode='train'):
        # 选取 N-Way 个受试者
        train_subs = self.dataset.subjects[:74]
        selected_subs = random.sample(train_subs, self.n_way)

        all_x, all_y = [], []

        for label, sub in enumerate(selected_subs):
            # 获取该受试者拥有的所有视角
            available_views = list(self.dataset.index[sub].keys())

            # 🌟 1. 构造 Multi-View Support Set
            # 随机选 K 个不同视角作为 Support
            s_views = random.sample(available_views, min(self.k_shot, len(available_views)))
            # 如果视角不够 K 个，则允许部分重复
            if len(s_views) < self.k_shot:
                s_views += random.choices(available_views, k=self.k_shot - len(s_views))

            for v in s_views:
                p = random.choice(self.dataset.index[sub][v])
                all_x.append(self.dataset.load_frames(p))
                all_y.append(label)

            # 🌟 2. 构造 Cross-View Query Set
            # 排除掉 Support 用过的视角，从剩余视角里选
            remaining_views = [v for v in available_views if v not in s_views]
            if not remaining_views:  # 极端情况：如果没剩下视角了，就全量选
                remaining_views = available_views

            q_view = random.choice(remaining_views)
            q_pool = self.dataset.index[sub][q_view]

            # 采样 Query 样本
            q_paths = random.choices(q_pool, k=self.q_query)
            for p in q_paths:
                all_x.append(self.dataset.load_frames(p))
                all_y.append(label)

        X = torch.stack(all_x)  # [N*(K+Q), T, 3, H, W]
        Y = torch.tensor(all_y)

        return X, Y, None, {}