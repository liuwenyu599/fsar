import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from tqdm import tqdm

# 1. 导入项目组件
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.backbones.structural import StructuralBackbone
from models.backbones.lora_handler import inject_lora_to_motionbert
from dataloader.multloader import CASIABMultiDataset
from utils.metrics import calculate_rank_accuracy, compute_distance_matrix


# ---------------------------------------------------------
# 1. 核心预处理：确保输入对齐
# ---------------------------------------------------------
def unified_align_pose(pose_data):
    """确保输入形状为 [60, 51] 并转为 Tensor"""
    if not isinstance(pose_data, torch.Tensor):
        pose_data = torch.from_numpy(pose_data)
    flat = pose_data.reshape(-1)
    new_data = torch.zeros(3060)
    # 填充或裁剪到 3060 (60帧 * 17点 * 3维)
    new_data[:min(flat.shape[0], 3060)] = flat[:min(flat.shape[0], 3060)]
    return new_data.view(60, 51).float()


def extract_view_features(model, dataset, device, target_view, target_conds, batch_size=32):
    """
    提取特定视角和条件下的步态特征。
    """
    model.eval()
    all_feats = []
    all_lbls = []
    all_conds = []

    # 筛选路径
    valid_paths = []
    for sid in dataset.all_subject_ids:
        for path in dataset.all_sequences.get(sid, []):
            if target_view in path:
                for cond_key in target_conds:
                    if cond_key in path:
                        c_type = 'nm' if 'nm' in cond_key else ('bg' if 'bg' in cond_key else 'cl')
                        valid_paths.append((path, int(sid), c_type))
                        break

    if not valid_paths:
        return None, None, None

    print(f"🔍 提取 View {target_view} | 样本数: {len(valid_paths)}")

    with torch.no_grad():
        for i in range(0, len(valid_paths), batch_size):
            batch_info = valid_paths[i: i + batch_size]
            batch_poses = []

            for path, _, _ in batch_info:
                raw_pose = dataset.load_pose(path)  # 获取原始骨架数据
                pose = unified_align_pose(raw_pose)  # 物理对齐
                batch_poses.append(pose)

            x = torch.stack(batch_poses).to(device)  # [B, 60, 51]

            # 前向传播：[B, 60, 51] -> [B, 60, 512] (假设 backbone 输出已做空间聚合)
            feat_seq = model(x)

            # 时序聚合：对时间维度取平均 [B, 512]
            feat_identity = feat_seq.mean(dim=1)

            # L2 归一化，用于余弦距离计算
            feat_identity = F.normalize(feat_identity, p=2, dim=-1)

            all_feats.append(feat_identity.cpu().numpy())
            all_lbls.extend([info[1] for info in batch_info])
            all_conds.extend([info[2] for info in batch_info])

    return np.concatenate(all_feats, axis=0), np.array(all_lbls), all_conds


# ---------------------------------------------------------
# 2. 迁移与测试主程序
# ---------------------------------------------------------
def run_migration_test():
    # A. 环境准备
    config_path = "configs/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = config['datasets']['casia_b_root']

    # B. 初始化 StructuralBackbone
    print("\n🛠️  Step 1: 初始化 StructuralBackbone (DSTformer)...")
    model = StructuralBackbone()  # 基于 17 个关节点的结构流骨干

    # C. 加载 MotionBERT-Lite 官方权重 (主干)
    lite_path = "logs/checkpoints/motionbert_pretrain_lite.pth"
    if os.path.exists(lite_path):
        print(f"🔗 Step 2: 加载官方预训练主干: {lite_path}")
        state_dict = torch.load(lite_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"❌ 错误: 找不到预训练文件 {lite_path}，无法执行迁移测试！")
        return

    # D. 注入 LoRA
    print("💉 Step 3: 注入 LoRA (Rank=8)...")
    model = inject_lora_to_motionbert(model, rank=8)  #

    # E. 加载你训练好的 LoRA 插件
    ckpt_path = 'logs/checkpoints/ppgait_fusion_triplet_final.pth'
    if os.path.exists(ckpt_path):
        print(f"📂 Step 4: 加载你微调出的 LoRA 权重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        # 从融合模型的 checkpoint 中提取结构流 LoRA 部分
        lora_state = ckpt.get('struct_lora_state_dict', ckpt)

        # 加载权重
        msg = model.load_state_dict(lora_state, strict=False)

        # 验证匹配情况
        match_count = len(lora_state.keys()) - len(msg.unexpected_keys)
        print(f"✅ LoRA 加载成功，匹配并激活了 {match_count} 个参数键值")
    else:
        print(f"⚠️  警告: 找不到微调权重 {ckpt_path}，将仅使用预训练主干进行测试。")

    model.to(device)
    dataset = CASIABMultiDataset(data_root, seq_len=60)  #

    # F. 定义 CASIA-B 测试协议
    views = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
    gallery_view = '090'  # 步态识别标准注册视角
    gallery_conds = ['nm-01', 'nm-02', 'nm-03', 'nm-04']
    probe_conds = ['nm-05', 'nm-06', 'bg-01', 'bg-02', 'cl-01', 'cl-02']

    # G. 提取 Gallery
    print(f"\n📦 [Gallery] 正在提取视角 {gallery_view} 的身份原型...")
    g_feat, g_lbl, _ = extract_view_features(model, dataset, device, gallery_view, gallery_conds)

    if g_feat is None:
        print("❌ 错误: Gallery 提取失败，请检查数据路径。")
        return

    # H. 提取 Probe 并比对
    print("\n📦 [Probe] 开始全视角跨视角评估...")
    results = {}
    for v in views:
        q_feat, q_lbl, q_conds = extract_view_features(model, dataset, device, v, probe_conds)
        if q_feat is None: continue

        # 计算余弦距离矩阵
        dist_mat = compute_distance_matrix(q_feat, g_feat, metric='cosine')

        view_res = {}
        for c_type in ['nm', 'bg', 'cl']:
            mask = [i for i, c in enumerate(q_conds) if c == c_type]
            if not mask: continue

            # 计算 Rank-1 准确率
            acc = calculate_rank_accuracy(dist_mat[mask], q_lbl[mask], g_lbl, topk=1)
            view_res[c_type] = acc

        results[v] = view_res
        print(
            f"📍 View {v} | NM: {view_res.get('nm', 0):.1%} | BG: {view_res.get('bg', 0):.1%} | CL: {view_res.get('cl', 0):.1%}")

    # I. 结果汇总
    print("\n" + "=" * 50)
    print("📊 迁移测试最终汇总 (除 090 视角外):")
    for c in ['nm', 'bg', 'cl']:
        acc_list = [res[c] for v, res in results.items() if v != gallery_view and c in res]
        if acc_list:
            print(f"⭐ Average {c.upper()}: {np.mean(acc_list):.2%}")
    print("=" * 50)


if __name__ == "__main__":
    run_migration_test()