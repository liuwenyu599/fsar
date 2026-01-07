import sys
import os

# 显存碎片优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from tqdm import tqdm

from models.backbones.structural import StructuralBackbone
from models.backbones.lora_handler import inject_lora_to_motionbert
from models.heads.ppm import PPMStructuredAggregator
from dataloader.multloader import CASIABMultiDataset as StructDataset


# ---------------------------------------------------------
# 1. 严格的权重加载逻辑 (已验证 module. -> encoder.)
# ---------------------------------------------------------
def load_official_motionbert_weights(model, path):
    print(f"📦 正在执行 1:1 物理对齐注入: {path}")
    if not os.path.exists(path):
        print(f"❌ 错误：权重文件不存在 {path}")
        return model

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    # 定位 model_pos 字典
    state_dict = checkpoint.get('model_pos', checkpoint.get('model', checkpoint))

    new_state_dict = {}
    model_keys = model.state_dict().keys()

    match_count = 0
    for k, v in state_dict.items():
        # 将官方的 'module.' 替换为你的包装名 'encoder.'
        nk = k.replace('module.', 'encoder.')

        if nk in model_keys:
            new_state_dict[nk] = v
            match_count += 1
        else:
            # 备选路径：处理不带前缀的情况
            nk_alt = 'encoder.' + k if not k.startswith('encoder.') else k
            if nk_alt in model_keys:
                new_state_dict[nk_alt] = v
                match_count += 1

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ [SUCCESS] 成功匹配 Keys: {match_count} | 缺失: {len(msg.missing_keys)}")

    if match_count > 250:
        print("🎉 [CRITICAL] 官方架构已 1:1 注入，物理动作先验加载完毕！")
    return model


# ---------------------------------------------------------
# 2. 训练主程序
# ---------------------------------------------------------
def train_struct_boost():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 60 + "\n🚀 STAGE 1: MOTIONBERT DUAL-GPU IDENTITY BOOST\n" + "=" * 60)

    # A. 实例化模型并加载预训练权重
    model = StructuralBackbone(embed_dim=256, dim_rep=512, depth=5, num_heads=8)
    model = load_official_motionbert_weights(model, 'logs/checkpoints/latest_epoch.bin')

    # B. 注入 LoRA (在并行分发前注入)
    model = inject_lora_to_motionbert(model, rank=8)

    # C. 启用双卡并行 (DataParallel)
    if torch.cuda.device_count() > 1:
        print(f"📡 检测到 {torch.cuda.device_count()} 个 GPU，启用并行模式加速...")
        model = nn.DataParallel(model)

    model = model.to(device)

    # D. 辅助组件 (PPM 与 分类器)
    ppm = PPMStructuredAggregator(feature_dim=512).to(device)
    classifier = nn.Linear(512, 74).to(device)  # 训练集 74 人

    # E. 优化器设置
    # 只针对 LoRA、PPM 和 Classifier 进行更新
    params = [p for p in model.parameters() if p.requires_grad] + \
             list(ppm.parameters()) + list(classifier.parameters())

    optimizer = optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    criterion_ce = nn.CrossEntropyLoss()

    # F. 数据加载 (Batch Size = 16人 * 4序列 = 64)
    ds = StructDataset('/datasets/CASIA-B', mode='pose')
    train_ids = sorted([sid for sid in ds.all_subject_ids if int(sid) <= 74])
    id_map = {sid: i for i, sid in enumerate(train_ids)}

    # G. 训练循环
    epochs = 90
    for epoch in range(1, epochs + 1):
        model.train();
        ppm.train();
        classifier.train()
        torch.cuda.empty_cache()  # 每轮清空碎片

        pbar = tqdm(range(100), desc=f"Epoch {epoch}/{epochs}")
        for _ in pbar:
            # 随机采样 batch
            sids = random.choices(train_ids, k=16)
            X, Y = [], []
            for sid in sids:
                # 每人采样 4 个序列进行对比学习
                seqs = random.sample(ds.all_sequences[sid], 4)
                for p in seqs:
                    X.append(ds.load_pose(p))
                    Y.append(id_map[sid])

            inputs = torch.stack(X).to(device)  # [64, 60, 51]
            if len(inputs.shape) == 4:
                inputs = inputs.flatten(2)  # [64, 60, 51]
            targets = torch.tensor(Y).to(device)

            # 前向传播
            # 1. 骨骼流提取 (DataParallel 自动分发)
            struct_features = model(inputs)  # [64, 60, 512]

            # 2. 时序聚合 (PPM 目前在主卡 cuda:0)
            agg_feats = ppm(struct_features, torch.ones(struct_features.shape[0], 60).to(device))
            agg_feats = F.normalize(agg_feats, p=2, dim=-1)

            # 3. 身份分类
            logits = classifier(agg_feats)
            loss = criterion_ce(logits, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # H. 安全保存 (处理 DataParallel 封装)
        if epoch % 30 == 0 or epoch == 1:
            raw_model = model.module if hasattr(model, 'module') else model
            save_path = f'logs/checkpoints/struct_boost_aligned_e{epoch}.pth'
            torch.save({
                'struct_lora_state_dict': {k: v for k, v in raw_model.state_dict().items() if 'lora_' in k},
                'ppm_state_dict': ppm.state_dict(),
                'epoch': epoch
            }, save_path)
            print(f"💾 权重已存档: {save_path}")


if __name__ == "__main__":
    train_struct_boost()