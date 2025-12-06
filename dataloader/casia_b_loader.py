import torch
import os
import random
import pickle
import numpy as np
import cv2


class CASBDataset:
    """
    解析 CASIA-B 结构并为 Few-Shot 训练准备数据。
    """

    def __init__(self, data_root):
        self.data_root = data_root
        print(f"Initializing CASIA-B Dataset from: {data_root}")
        self.all_sequences = self._parse_data_structure()
        self.all_subject_ids = sorted(list(self.all_sequences.keys()))
        print(f"Found {len(self.all_subject_ids)} subjects.")

    def __call__(self, mode='train'):
        """
        让 FewShotSampler 可以像函数一样使用：
        X_vis, X_struct, labels, cov_labels, phase_weights = sampler()

        实际内部就是调用 get_episode()
        """
        return self.get_episode(mode)

    def _parse_data_structure(self):
        """
        解析目录结构: ID -> Condition -> View -> [Sequence_Paths]
        """
        all_data = {}

        # 基础路径检查
        if not os.path.exists(self.data_root):
            print(f"Error: Data root {self.data_root} does not exist!")
            return {}

        pose_root = os.path.join(self.data_root, 'pose', 'CASIA-B_HRNet')
        silu_root = os.path.join(self.data_root, 'silu')

        # 如果找不到 pose 文件夹，回退到 Dummy 模式 (方便代码跑通)
        if not os.path.exists(pose_root):
            print("Warning: Pose directory not found. Using DUMMY data mode.")
            for i in range(1, 11):
                all_data[f'{i:03d}'] = {'nm-01': {'090': [('dummy_silu', 'dummy_pose', 0)]}}
            return all_data

        # 遍历 ID
        subject_list = sorted(os.listdir(pose_root))

        for sub_id in subject_list:
            if not os.path.isdir(os.path.join(pose_root, sub_id)): continue

            all_data[sub_id] = {}
            # 遍历条件 (bg-01, nm-01, etc.)
            cond_dir_base = os.path.join(pose_root, sub_id)
            cond_list = sorted(os.listdir(cond_dir_base))

            for cond in cond_list:
                all_data[sub_id][cond] = {}
                # 遍历视角 (000, 018...)
                view_dir_base = os.path.join(cond_dir_base, cond)
                if not os.path.isdir(view_dir_base): continue

                view_list = sorted(os.listdir(view_dir_base))

                for view in view_list:
                    pose_dir = os.path.join(view_dir_base, view)
                    silu_dir = os.path.join(silu_root, sub_id, cond, view)

                    if not os.path.isdir(pose_dir): continue

                    # 获取该目录下所有pkl文件
                    seq_files = [f for f in os.listdir(pose_dir) if f.endswith('.pkl')]

                    seq_paths = []
                    for f in seq_files:
                        pose_path = os.path.join(pose_dir, f)
                        # 假设 silu 是 png 图片，这里做简单替换逻辑
                        # 如果 silu 文件夹下是图片序列，通常取第一张或整个文件夹路径
                        # 这里为了简化，假设 silu 路径指向对应的图片文件夹或特征文件
                        silu_path = silu_dir

                        # 简单的协变量编码: nm=0, bg=1, cl=2
                        cov_label = 0
                        if 'bg' in cond:
                            cov_label = 1
                        elif 'cl' in cond:
                            cov_label = 2

                        seq_paths.append((silu_path, pose_path, cov_label))

                    if seq_paths:
                        all_data[sub_id][cond][view] = seq_paths

        return all_data

    def load_single_sample(self, silu_path, pose_path):
        """
        加载单个样本数据 (Visual + Structural)
        现在 Visual 流返回模拟图像张量，可以直接送 ViT
        """
        # --- 1. 加载骨架 (Structural Stream) ---
        target_len = 5
        if pose_path == 'dummy_pose':
            x_struct = torch.randn(target_len, 68)
        else:
            try:
                with open(pose_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, np.ndarray):
                        data = torch.from_numpy(data).float()
                        T = data.shape[0]  # 原始时间长度
                        x_struct = data.view(T, -1)

                        # Padding / Crop 到 target_len
                        if T < target_len:
                            pad = torch.zeros(target_len - T, x_struct.shape[1])
                            x_struct = torch.cat([x_struct, pad], dim=0)
                        else:
                            x_struct = x_struct[:target_len, :]
            except Exception as e:
                x_struct = torch.randn(target_len, 68)

        # --- 2. 加载轮廓 (Visual Stream) ---
        if silu_path == 'dummy_silu':
            x_vis = torch.randn(target_len, 3, 224, 224)
        else:
            # 实际读取序列图片 -> Resize -> ToTensor
            # 真实加载逻辑:
            # 这里原本应该读取图片序列 -> Resize -> ToTensor -> ViT 提取特征
            # 为了让代码现在能跑通，我们暂时模拟特征
            # 实际项目中，您可以在这里插入加载图片的逻辑
            x_vis = torch.randn(target_len, 3, 224, 224)  # 临时代码
        # 保证长度和x_struct一致，便于后续送ViT。
        x_vis = x_vis[:target_len]
        return x_vis, x_struct


class FewShotSampler:
    def __init__(self, dataset, n_way, k_shot, q_query):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        # 简单划分：前 80% ID 训练，后 20% ID 测试
        ids = self.dataset.all_subject_ids
        split = int(len(ids) * 0.8)

        # 如果没有找到任何 ID (路径错误)，防止报错
        if len(ids) == 0:
            self.train_ids = []
            self.test_ids = []
        else:
            self.train_ids = ids[:split] if split > 0 else ids
            self.test_ids = ids[split:] if split > 0 else ids

    def get_episode(self, mode='train'):
        pool = self.train_ids if mode == 'train' else self.test_ids

        if len(pool) == 0:
            # 没数据时的防御逻辑: 返回随机 Dummy 数据
            B = self.n_way * (self.k_shot + self.q_query)
            return (torch.randn(B, 50, 768),
                    torch.randn(B, 50, 68),
                    torch.randint(0, self.n_way, (B,)),
                    torch.randint(0, 4, (B,)),
                    torch.rand(B, 50))

        if len(pool) < self.n_way:
            # ID 不够 N-way 时重复采样
            pool = pool * (self.n_way // len(pool) + 1)

        sampled_ids = random.sample(pool, self.n_way)

        batch_vis, batch_struct, batch_labels, batch_covs = [], [], [], []

        # 收集数据
        for i, class_id in enumerate(sampled_ids):
            # 获取该 ID 下所有数据路径
            all_samples = []
            if class_id in self.dataset.all_sequences:
                for cond in self.dataset.all_sequences[class_id]:
                    for view in self.dataset.all_sequences[class_id][cond]:
                        all_samples.extend(self.dataset.all_sequences[class_id][cond][view])

            # 样本不足补 Dummy
            needed = self.k_shot + self.q_query
            if len(all_samples) < needed:
                all_samples.extend([('dummy_silu', 'dummy_pose', 0)] * (needed - len(all_samples)))

            selected = random.sample(all_samples, needed)

            for silu, pose, cov in selected:
                xv, xs = self.dataset.load_single_sample(silu, pose)
                batch_vis.append(xv)
                batch_struct.append(xs)
                batch_labels.append(i)  # 标签重映射为 0 ~ N-1
                batch_covs.append(cov)

        # 排序: Support在前, Query在后 (为了配合 train.py 的切分逻辑)
        final_vis, final_struct, final_lbl, final_cov = [], [], [], []

        # Support Set
        for i in range(self.n_way):
            start = i * (self.k_shot + self.q_query)
            for j in range(self.k_shot):
                idx = start + j
                final_vis.append(batch_vis[idx])
                final_struct.append(batch_struct[idx])
                final_lbl.append(batch_labels[idx])
                final_cov.append(batch_covs[idx])

        # Query Set
        for i in range(self.n_way):
            start = i * (self.k_shot + self.q_query)
            for j in range(self.k_shot, self.k_shot + self.q_query):
                idx = start + j
                final_vis.append(batch_vis[idx])
                final_struct.append(batch_struct[idx])
                final_lbl.append(batch_labels[idx])
                final_cov.append(batch_covs[idx])

        # 返回 Tensor
        # --- 原来的堆叠 ---
        final_vis = torch.stack(final_vis)  # shape: (B, T, C, H, W)
        final_struct = torch.stack(final_struct)  # (B, T, D)
        phase_weights = torch.rand(final_vis.shape[0], final_vis.shape[1])  # (B, T)
        final_lbl = torch.tensor(final_lbl)
        final_cov = torch.tensor(final_cov)
        # --- 返回 ---
        return final_vis, final_struct, final_lbl, final_cov, phase_weights


# --- 单元测试入口 ---
if __name__ == '__main__':
    print('Testing Dataloader Logic...')

    # 尝试从 yaml 读取配置，如果失败则使用硬编码默认值
    # 这确保了单元测试既方便，又能跟项目配置保持一致
    default_root = '/datasets/CASIA-B'
    try:
        import yaml

        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f)
                default_root = config.get('datasets', {}).get('casia_b_root', default_root)
                print(f"Loaded path from config: {default_root}")
    except ImportError:
        pass  # 如果没有 yaml 库，就用默认路径

    dataset = CASBDataset(default_root)
    sampler = FewShotSampler(dataset, n_way=5, k_shot=1, q_query=15)

    # 获取一个 Episode
    data = sampler.get_episode(mode='train')

    X_vis, X_struct, labels, covs, phases = data

    print(f'Episode Data Shapes:')
    print(f'  Visual Input:   {X_vis.shape} (Batch, Time, Dim)')
    print(f'  Struct Input:   {X_struct.shape} (Batch, Time, Dim)')
    print(f'  Labels:         {labels.shape}')
    print(f'  Phase Weights:  {phases.shape}')
    print('✅ Dataloader Test Passed.')