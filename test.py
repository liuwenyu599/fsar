import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from tqdm import tqdm

# 1. 导入项目内部组件
from dataloader.multloader import CASIABMultiDataset
from models.backbones.structural import StructuralBackbone
from utils.lora_utils import inject_lora
from utils.metrics import calculate_rank_accuracy, compute_distance_matrix


# ---------------------------------------------------------
# 1. 特征提取函数
# ---------------------------------------------------------
def extract_view_features(model, dataset, device, target_view, target_conds, batch_size=32):
    """
    提取特定视角和条件下的步态特征。
    """
    model.eval()
    all_feats = []
    all_lbls = []
    all_conds = []

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

    with torch.no_grad():
        for i in range(0, len(valid_paths), batch_size):
            batch_info = valid_paths[i: i + batch_size]
            batch_poses = []

            for path, _, _ in batch_info:
                # load_pose 返回 [T, 17, 3] -> 展平为 [T, 51]
                pose = dataset.load_pose(path).view(dataset.seq_len, -1)
                batch_poses.append(pose)

            x = torch.stack(batch_poses).to(device)  # [B, T, 51]
            feat_seq = model(x)  # [B, T, 512]

            # 时序聚合 (Temporal Mean Pooling) 并 L2 归一化
            feat_identity = feat_seq.mean(dim=1)  # [B, 512]
            feat_identity = F.normalize(feat_identity, p=2, dim=-1)

            all_feats.append(feat_identity.cpu().numpy())
            all_lbls.extend([info[1] for info in batch_info])
            all_conds.extend([info[2] for info in batch_info])

    return np.concatenate(all_feats, axis=0), np.array(all_lbls), all_conds


# ---------------------------------------------------------
# 2. 核心评估流程
# ---------------------------------------------------------
def run_evaluation():
    # A. 加载配置
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 错误: 找不到配置文件 {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['system']['device'] if torch.cuda.is_available() else "cpu")
    data_root = config['datasets']['casia_b_root']

    # B. 初始化主干网络
    print("🛠️ 初始化 StructuralBackbone...")
    model = StructuralBackbone(embed_dim=256, dim_rep=512)

    # C. 加载官方预训练权重 (解决 3.29% 极低准确率的关键)
    # 请确保该文件存在，否则模型主干是随机权重的，识别率会非常差
    pretrained_backbone_path = "pretrained_models/motionbert_pretrain_lite.pth"
    if os.path.exists(pretrained_backbone_path):
        print(f"🔗 加载主干预训练权重: {pretrained_backbone_path}")
        base_ckpt = torch.load(pretrained_backbone_path, map_location='cpu')
        base_state = base_ckpt.get('model', base_ckpt.get('state_dict', base_ckpt))
        model.load_state_dict(base_state, strict=False)
    else:
        print(f"⚠️ 警告: 未找到主干预训练文件 {pretrained_backbone_path}，当前使用的是随机初始化主干！")

    # D. 注入 LoRA
    print("💉 注入 LoRA 层 (Rank=8)...")
    model = inject_lora(model, rank=8, alpha=16.0)

    # E. 加载 LoRA 权重
    ckpt_path = "logs/checkpoints/ppgait_fusion_triplet_final.pth"
    if os.path.exists(ckpt_path):
        print(f"📂 加载训练好的 LoRA 插件: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        # 兼容不同的保存格式
        lora_state = checkpoint.get('model', checkpoint.get('struct_lora_state_dict', checkpoint))

        # 自动对齐前缀 (移除 module. 等)
        new_lora_state = {}
        for k, v in lora_state.items():
            if k.startswith('module.'):
                new_lora_state[k[7:]] = v
            else:
                new_lora_state[k] = v

        # 加载权重
        msg = model.load_state_dict(new_lora_state, strict=False)

        # --- 修正后的统计逻辑 ---
        # 成功匹配数 = 传入的总键数 - 冗余(不匹配)的键数
        match_count = len(new_lora_state.keys()) - len(msg.unexpected_keys)
        print(f"✅ LoRA 加载完成，成功匹配并更新了 {match_count} 个参数键值")

        if len(msg.missing_keys) > 0:
            # 这里的 missing_keys 通常是主干网络的参数名，这是正常的，因为我们只加载了 LoRA 的 120 个参数
            print(f"ℹ️  模型中有 {len(msg.missing_keys)} 个参数保持原样 (通常为主干权重)")
    else:
        print(f"❌ 错误: 未找到 LoRA 权重文件 {ckpt_path}")
        return

    model.to(device)
    dataset = CASIABMultiDataset(data_root, seq_len=60)

    # F. 标准测试协议 (CASIA-B)
    views = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
    gallery_view = '090'
    gallery_conds = ['nm-01', 'nm-02', 'nm-03', 'nm-04']
    probe_conds = ['nm-05', 'nm-06', 'bg-01', 'bg-02', 'cl-01', 'cl-02']

    # G. 提取 Gallery 特征
    print(f"\n📦 Step 1: 提取 Gallery 特征 (视角 {gallery_view})...")
    g_feat, g_lbl, _ = extract_view_features(model, dataset, device, gallery_view, gallery_conds)

    if g_feat is None:
        print("❌ 错误: 无法获取 Gallery 特征，请检查数据集路径。")
        return

    # H. 全视角评估
    print("\n📦 Step 2: 开始全视角 Probe 评估...")
    results = {}

    for v in views:
        q_feat, q_lbl, q_conds = extract_view_features(model, dataset, device, v, probe_conds)
        if q_feat is None:
            continue

        # 计算距离矩阵 (余弦距离)
        dist_mat = compute_distance_matrix(q_feat, g_feat, metric='cosine')

        view_res = {}
        for c_type in ['nm', 'bg', 'cl']:
            mask = [i for i, c in enumerate(q_conds) if c == c_type]
            if not mask: continue

            # 统计 Rank-1 准确率
            acc = calculate_rank_accuracy(dist_mat[mask], q_lbl[mask], g_lbl, topk=1)
            view_res[c_type] = acc

        results[v] = view_res
        nm_acc = view_res.get('nm', 0)
        bg_acc = view_res.get('bg', 0)
        cl_acc = view_res.get('cl', 0)
        print(f"📍 View {v} | NM: {nm_acc:.1%} | BG: {bg_acc:.1%} | CL: {cl_acc:.1%}")

    # I. 最终汇总
    print("\n" + "=" * 50)
    print("📊 最终平均准确率 (除 090 视角外):")
    for c in ['nm', 'bg', 'cl']:
        acc_list = [res[c] for v, res in results.items() if v != gallery_view and c in res]
        if acc_list:
            print(f"⭐ Average {c.upper()}: {np.mean(acc_list):.2%}")
    print("=" * 50)


if __name__ == "__main__":
    run_evaluation()