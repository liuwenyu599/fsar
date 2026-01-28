import sys
import os

# 将项目根目录加入到搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

import random
import numpy as np
import pickle  # 用于读取 HRNet 提取后存储为 .pkl 格式的骨骼点数据。
from torch.utils.data import Dataset


# 这个类负责将硬盘上的原始文件转换成模型能理解的张量
class CASIABMultiDataset(Dataset):
    """
    CASIA-B 多模态数据集加载器 (骨骼流专用)
    """

    # 输入源是pose,每一个 gait 序列在送进模型之前，被统一成长度为 60 帧的时间序列。
    def __init__(self, root, mode="pose", seq_len=60):
        self.root = root
        self.mode = mode
        self.seq_len = seq_len
        self.pose_root = os.path.join(root, "pose", "CASIA-B_HRNet")

        print(f"[CASIA] Root: {root} | Mode: {mode} | SeqLen: {seq_len}")
        self.samples, self.all_subject_ids, self.all_sequences = self._collect_pose_samples()
        print(f"[CASIA] Found {len(self.all_subject_ids)} subjects and {len(self.samples)} total sequences.")

    # 样本搜集逻辑,收集所有.pkl骨骼文件路径，构建数据结构
    def _collect_pose_samples(self):
        samples = []  # 所有.pkl文件的完整路径列表
        subject_ids = []  # 所有行人ID（如001, 002等）
        all_sequences = {}  # 字典结构key,value: { '001': [path1, path2...], '002': [...] }

        if not os.path.exists(self.pose_root):
            print(f"❌ Error: {self.pose_root} does not exist!")
            return [], [], {}
        # 列出 self.pose_root 目录下的所有文件和子目录名称,排序，存储在pids列表
        pids = sorted(os.listdir(self.pose_root))
        # 检查每个pid是否是一个目录标签
        for pid in pids:
            pid_dir = os.path.join(self.pose_root, pid)  # 构建完整路径
            # continue 会跳出当前循环的剩余代码，进入下一个循环迭代。
            if not os.path.isdir(pid_dir): continue  # 跳过非目录文件

            subject_ids.append(pid)  # 记录标签pid
            all_sequences[pid] = []  # 为pid同一个pid的所有序列都会被收集到all_sequences[pid]中。

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
        # 完成后：
        # all_sequences = {'001': ['/path/to/001_nm-01_090_0001.pkl', ...]}
        return samples, subject_ids, all_sequences

    # 🌟 新增：在赋值前检查关键点索引是否匹配
    def _check_mapping_consistency(self, coords):
        """
        验证输入的关键点数量是否满足 H36M 映射逻辑（最高索引为 16，故至少需要 17 个点）
        """
        if coords.shape[1] < 17:
            raise IndexError(f"关键点数量不匹配: 映射逻辑需要 17 个点，但输入数据仅有 {coords.shape[1]} 个点。")

    def load_pose(self, pkl_path):
        '''该函数会打开 .pkl 文件，读取原始坐标
        进行 H36M 映射、去绝对位移（Root-Centering）、缩放（Scaling）以及时序对齐（统一为 60 帧）
        最后返回一个形状为 [60, 17, 3] 的 PyTorch 张量。'''
        try:
            with open(pkl_path, "rb") as f:  # 打开pkl
                raw_data = pickle.load(f)
            # 如果 raw_data 是字典，先尝试 "keypoints" 键
            # 如果不存在，尝试 "data" 键
            # 如果还是没有，使用整个字典
            # 如果不是字典，直接使用 raw_data
            pose = raw_data.get("keypoints", raw_data.get("data", raw_data)) if isinstance(raw_data, dict) else raw_data
            pose = np.asarray(pose, dtype=np.float32)  # 转换为NumPy数组,[T,维度]
            # 确定形状，如果是二维就变成三维，17是由姿态估计算法（如HRNet）决定的，通常使用的是COCO格式的17个关键点。
            if pose.ndim == 2:
                print(f"原始是2D数组，形状: {pose.shape}")
                pose = pose.reshape(pose.shape[0], 17, -1)
                print(f"   重塑后形状: {pose.shape}")
            # else:
            #     print(f"原始是{pose.ndim}D数组，形状: {pose.shape}")

            T_orig = pose.shape[0]  # 如果没有重塑，那就是帧数T
            # 创建空数组用于存放转换后的H36M格式数据
            final_pose = np.zeros((T_orig, 17, 3), dtype=np.float32)
            # 提取原始坐标的前两维（x, y），忽略第三维（可能是置信度）
            coords = pose[:, :, :2]

            # --- 物理语义映射: COCO-17 -> H36M-17 ---
            # 🌟 赋值前执行函数检查映射关系是否匹配
            self._check_mapping_consistency(coords)

            l_hip, r_hip, l_sho, r_sho = coords[:, 11], coords[:, 12], coords[:, 5], coords[:, 6]
            pelvis, neck = (l_hip + r_hip) / 2.0, (l_sho + r_sho) / 2.0
            # 批量赋值（每个赋值都是对时间维度的所有帧操作）
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
            final_pose[:, :, 2] = 1.0  # 所有关键点的第三维设为1.0
            #array[start:stop:step, start:stop:step, start:stop:step]
            #     ↑                 ↑                 ↑
            #     第一维             第二维             第三维
            #在NumPy中，使用切片 a:b 会保持维度，使用整数索引 n 会降维。
                #使用整数索引（降维）
                #single_element = arr[:, 0, :]  # 取第二维的索引0
                #print("使用0的形状:", single_element.shape)  # (2, 4) - 第二维消失了！
                # 使用切片索引（保持维度）
                #slice_element = arr[:, 0:1, :]  # 取第二维的索引0，但保持维度
                #print("使用0:1的形状:", slice_element.shape)  # (2, 1, 4) - 第二维保留，长度为1
            # --- Root-Centering & Scaling ---
            # 减去骨盆(0号点)，缩放128倍。确保骨盆final_pose[:, 0:1, :]始终在 (0,0,1),从所有关键点的x、y坐标中减去骨盆（索引0）的坐标。
            # 这样做的目的是将骨盆移动到原点(0,0)，所有其他关键点的坐标变为相对于骨盆的偏移量。使模型更关注姿态本身。
            final_pose[:, :, :2] = (final_pose[:, :, :2] - final_pose[:, 0:1, :2]) / 128.0
            #统一序列长度。如果原始序列长度T_orig大于等于目标长度seq_len（60），则直接截取前seq_len帧；如果不足，则通过复制（tile）的方式将序列重复多次，直到至少达到seq_len长度，然后同样截取前seq_len帧。
            if T_orig >= self.seq_len:
                final_pose = final_pose[:self.seq_len]
            else:
                final_pose = np.tile(final_pose, ((self.seq_len // T_orig) + 1, 1, 1))[:self.seq_len]

            return torch.from_numpy(final_pose)
        except Exception as e:
            # 捕获检查失败或读取失败的异常，打印路径并返回零张量
            print(f"⚠️ Error loading {pkl_path}: {e}")
            return torch.zeros((self.seq_len, 17, 3))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.load_pose(self.samples[idx])

class FewShotSampler:
    def __init__(self, dataset, n_way, k_shot, q_query, active_views=None):
        # 初始化采样器，存储 N-way（类别数）、K-shot（每个类别的支撑样本数）和 Q-query（查询样本数）
        self.dataset, self.n_way, self.k_shot, self.q_query = dataset, n_way, k_shot, q_query

        # 1. 建立视角列表与映射表 (用于视角感知逻辑)
        # 如果未指定视角，则默认使用 CASIA-B 的 11 个标准步态视角
        self.target_views = active_views if active_views is not None else \
            ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']

        # 🌟 建立视角字符串到整数索引的映射（例如 '090' -> 索引 5），用于模型内部的 Embedding 层
        self.view_to_idx = {v: i for i, v in enumerate(self.target_views)}
        # 初始化一个字典，用于记录每个视角下有哪些行人 ID（Subject ID）
        self.view_to_ids = {v: [] for v in self.target_views}

        # 遍历数据集中的所有行人序列，包括行人ID和路径
        for sid, paths in dataset.all_sequences.items():
            for v in self.target_views:
                # 如果该行人在某个视角下有数据，则将其 ID 加入该视角的候选池
                if any(v in p for p in paths):
                    self.view_to_ids[v].append(sid)

        # 2. 筛选健康视角 (Health Check)
        self.active_views = []
        print("--- 🔍 Sampler View Health Check ---")
        for v in self.target_views:
            count = len(self.view_to_ids[v])
            # 只有当某个视角下的行人数量大于等于 N-way 时，该视角才“健康”，可以进行采样
            if count >= self.n_way:
                self.active_views.append(v)
                print(f"✅ View {v}: {count} subjects (Active)")
            else:
                # 行人数量不足 N-way 的视角将被跳过，防止采样时报错
                print(f"❌ View {v}: {count} subjects (Skipped)")

        # 如果没有健康的视角，默认使用所有数据
        if not self.active_views:
            self.active_views = ['all']
            print(f"没有健康视角，使用所有数据")

        # 3. 划分数据集 (按行人 ID 划分，确保训练集和测试集的人完全不重叠)
        all_ids = sorted(dataset.all_subject_ids)
        random.seed(42)  # 固定随机种子，确保划分结果可复现
        shuffled_ids = all_ids[:]#Python 会在内存中开辟一块新的空间，把 all_ids 里的内容完整地复制一份放进去。
        random.shuffle(shuffled_ids)
        split = int(0.8 * len(shuffled_ids)) # 80% 用于训练，20% 用于测试
        self.train_ids_all = shuffled_ids[:split]
        self.test_ids_all = shuffled_ids[split:]

    def get_episode(self, mode="train"):
        """
        生成一个少样本任务 Episode
        X 形状: [Batch, T, 51], Y 形状: [Batch], v_idx_tensor 形状: [Batch]
        """
        # 1. 选定本轮 Episode 的视角 (确保同一个 Episode 内的样本视角一致，降低识别难度)
        episode_view = random.choice(self.active_views)
        # 根据模式选择训练集池或测试集池
        pool = self.train_ids_all if mode == "train" else self.test_ids_all

        # 2. 确定候选人：筛选出在该视角下有数据的行人id清单candidates['001', '005', '012', ...]
        candidates = [sid for sid in pool if sid in self.view_to_ids.get(episode_view, [])]
        # 如果该视角下的人数不足 N-way，则回退到视角池子或原始池子
        if len(candidates) < self.n_way:
            candidates = self.view_to_ids.get(episode_view, [])
        if len(candidates) < self.n_way:
            candidates = pool

        # 从候选人中随机抽取 N 个类别 (N-way)
        sampled_ids = random.sample(candidates, self.n_way)

        X, Y = [], []
        view_stats = {v: 0 for v in self.target_views} # 统计采样结果的视角分布情况,设置视角为key，值为0

        # 遍历选中的每一个类别（行人）
        for cls_idx, sid in enumerate(sampled_ids):
            all_paths = self.dataset.all_sequences.get(sid, [])
            # 仅保留该行人在 episode_view 视角下的路径
            valid_paths = [p for p in all_paths if episode_view in p] if episode_view != 'all' else all_paths
            if not valid_paths: valid_paths = all_paths

            # 每个类需要采样 K 个支撑集样本和 Q 个查询集样本
            needed = self.k_shot + self.q_query
            # 如果样本不足，则通过重复采样补齐
            selected = random.sample(valid_paths * (needed // len(valid_paths) + 1), needed)
            # selected是针对当前行人（sid）随机采样出的 K + Q 个动作序列的路径集合。
            for pkl_path in selected:
                for v in self.target_views:
                    if v in pkl_path: view_stats[v] += 1
                # 使用 dataset 类的 load_pose 函数加载归一化后的骨骼张量 [T, 17, 3]
                X.append(self.dataset.load_pose(pkl_path))
                Y.append(cls_idx) # 记录标签 (0 到 N-way-1)
        #self.view_to_idx：这是一个字典，存储了所有预设视角与其对应整数索引的映射关系（例如 {'000': 0, '018': 1, '036': 2, ...}）
        #episode_view比如是036，那么v_idx就是2
        # 生成视角索引 Tensor，告知模型当前样本属于哪个视角索引，形状为(len(X),)一维向量，长度等于当前批次的样本总数。填充为v_idx
        # 找到当前 episode_view 在映射表中的索引，并扩展到每个样本
        v_idx = self.view_to_idx.get(episode_view, 0)
        v_idx_tensor = torch.full((len(X),), v_idx, dtype=torch.long)

        # 将列表堆叠成 PyTorch 张量
        X = torch.stack(X)
        # 将骨骼点坐标展平：[Batch, T, 17, 3] -> [Batch, T, 51] (17*3=51)
        X = X.view(X.shape[0], X.shape[1], -1)

        # 🌟 返回 5 个值：数据 X, 标签 Y, 权重掩码 W (全1), 视角索引 v_idx_tensor, 视角统计 view_stats
        return X, torch.tensor(Y), torch.ones(X.shape[0], X.shape[1]), v_idx_tensor, view_stats


if __name__ == "__main__":
    # 快速验证脚本
    ds = CASIABMultiDataset("/datasets/CASIA-B", mode="pose")
    sampler = FewShotSampler(ds, n_way=5, k_shot=1, q_query=1)

    # 🌟 修正：接收 5 个返回值以匹配 get_episode 的定义
    X, Y, W, v_idx, V = sampler.get_episode("train")

    print("X Shape (Should be [10, 60, 51]):", X.shape)
    print("View Index Tensor:", v_idx)
    print("Locked View Stats:", V)