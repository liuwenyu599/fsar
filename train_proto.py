# train_proto_lora.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.optim as optim
import yaml
import argparse

from models.architecture import PAGFSLModel
from losses.prototypical_loss import PrototypicalLoss
from dataloader.casia_silu_dataset import CASIASiluDataset, FewShotSampler
from utils.lora_utils import inject_lora, LinearWithLoRA

def train(config_path='configs/config.yaml'):
    print("--- 🚀 PAG-FSL Training Start ---")

    # ---------------------------
    # 1. 配置加载
    # ---------------------------
    if os.path.exists(config_path):
        config = yaml.safe_load(open(config_path))
        print(f"Loaded config: {config_path}")
    else:
        print("⚠ Config not found, using defaults.")
        config = {
            'system': {'common_dim': 512,'device':'cuda'},
            'train': {'lr': 1e-3, 'epochs': 100},  # lr 调低
            'few_shot': {'n_way': 5, 'k_shot': 5, 'q_query': 15, 'batch_size': 64}
        }
    device_str = config['system']['device']
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------
    # 2. 数据集
    # ---------------------------
    dataset = CASIASiluDataset('/datasets/CASIA-B/silu')

    # ---------------------------
    # 3. 初始化模型
    # ---------------------------
    model = PAGFSLModel(common_dim=config['system']['common_dim']).to(device)
    model.train()

    # ---------------------------
    # 4. 精细化冻结策略 (Frozen Strategy)
    # ---------------------------
    # A. 先把整个模型全冻上 (包括 struct_backbone, fam 等不用的部分)
    for p in model.parameters():
        p.requires_grad = False
    # B. 只解冻 视觉流相关 的部分
    #    1. LoRA 和 Norm 层
    for name, p in model.vis_backbone.named_parameters():
        if "lora" in name.lower() or "ln" in name.lower() or "norm" in name.lower():
            p.requires_grad = True
    #    2. 投影头 (Projection Head) -> 必须训练，把 192 映射到 512
    for p in model.proj_vis.parameters():
        p.requires_grad = True
    # 确认参数状态
    print("--- Parameter Status ---")
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    # print("Trainable parameters:", trainable)
    print(f"Total trainable tensors: {len(trainable)}")
    # 检查是不是混入了 struct 或者 fam (应该没有)
    for n in trainable:
        if "struct" in n or "fam" in n:
            print(f"⚠️ Warning: Unexpected trainable param: {n}")
    assert len(trainable) > 0, "❌ LoRA 层没有可训练参数！"

    # ---------------------------
    # 5. 优化器
    # ---------------------------
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['train']['lr']
    )

    criterion_proto = PrototypicalLoss().to(device)
    scaler = torch.amp.GradScaler('cuda')
    # ---------------------------
    # 6. 训练循环
    # ---------------------------
    epochs = config['train']['epochs']
    n_way = config['few_shot']['n_way']
    k_shot = config['few_shot']['k_shot']
    q_query = config['few_shot']['q_query']
    batch_size_vit = config['few_shot'].get('batch_size', 64)

    ema_alpha = 0.9
    ema_proto_loss = None
    sampler = FewShotSampler(dataset, n_way, k_shot, q_query)
    print(f"Start training: {n_way}-way {k_shot}-shot, {q_query} queries.")
    for epoch in range(1, epochs + 1):
        # sampler 返回数据结构: [Class0 (K+Q), Class1 (K+Q), ..., ClassN (K+Q)]
        X_vis, _, _= sampler.get_episode(mode='train')# labels 这里不需要了，我们会重新生成?
        # 🔥🔥🔥 DEBUG START 🔥🔥🔥
        if epoch == 1:
            print("\n🔍 --- DATA SANITY CHECK ---")
            print(f"X_vis shape: {X_vis.shape}")
            print(f"X_vis min: {X_vis.min().item()}, max: {X_vis.max().item()}, mean: {X_vis.mean().item()}")

            # 检查是否有非零元素
            if X_vis.sum().item() == 0:
                print("❌❌❌ CRITICAL ALARM: Input is ALL ZEROS! The dataloader is failing to read images.")
                print("Please check your dataset path and file extensions.")
                exit()  # 直接停止，不要浪费电了
            else:
                print("✅ Data looks valid (non-zero). Continuing...")
            print("🔍 -------------------------\n")
        # 🔥🔥🔥 DEBUG END 🔥🔥🔥
        # ========== reshape ==========
        B, T, C, H, W = X_vis.shape
        total_BT = B * T
        X_flat = X_vis.reshape(total_BT, C, H, W).to(device)

        # ============================================================
        # 🔥 分chunk + autocast + 完全冻结的 backbone 提特征
        # ============================================================
        feats = []
        # backbone 冻结
        for i in range(0, total_BT, batch_size_vit):
            chunk = X_flat[i:i + batch_size_vit]
            with torch.amp.autocast( 'cuda', dtype=torch.float16):
                f = model.vis_backbone(chunk)  # LoRA 前向
                f_proj = model.proj_vis(f)
            feats.append(f_proj)

        F_flat = torch.cat(feats, dim=0).float()  # (B*T, D) fp32

        # ========== 时间维平均 ==========
        F_vis = F_flat.reshape(B, T, -1)
        f_final = F_vis.mean(dim=1)#B,D

        # ========== 划分 support/query ==========
        # f_final 当前形状: (N_way * (K+Q), D)
        # 1. 恢复维度为 (N_way, K+Q, D)
        f_final_reshaped = f_final.view(n_way, k_shot + q_query, -1)

        # 2. 切分 Support (前 K 个)
        # shape: (N_way, K_shot, D) -> (N_way * K_shot, D)
        f_supp = f_final_reshaped[:, :k_shot, :].contiguous().view(n_way * k_shot, -1)

        # 3. 切分 Query (后 Q 个)
        # shape: (N_way, Q_query, D) -> (N_way * Q_query, D)
        f_query = f_final_reshaped[:, k_shot:, :].contiguous().view(n_way * q_query, -1)

        # 4. 生成正确的 Query 标签
        # Query 也是按类别顺序排列的: 00..0 (Q个), 11..1 (Q个) ...
        labels_query = torch.arange(n_way).repeat_interleave(q_query).to(device)
        # 调试打印 (第一次迭代时检查)
        if epoch == 1:
            print(f"DEBUG: Reshaped f_final: {f_final_reshaped.shape}")
            print(f"DEBUG: f_supp shape: {f_supp.shape}")  # 应为 (25, D)
            print(f"DEBUG: f_query shape: {f_query.shape}")  # 应为 (75, D)
            print(f"DEBUG: labels_query: {labels_query}")  # 应为 [0...0, 1...1, ..., 4...4]
        # ============================================================
        # 🔥 loss 计算也用 autocast（关键）
        # ============================================================
        loss_proto, acc_proto = criterion_proto(
            f_supp, f_query, labels_query, n_way, k_shot)


        # EMA 更新
        if ema_proto_loss is None:
            ema_proto_loss = loss_proto.item()
        else:
            ema_proto_loss = ema_alpha * ema_proto_loss + (1 - ema_alpha) * loss_proto.item()

        # ========== 优化器 ==========
        optimizer.zero_grad()
        scaler.scale(loss_proto).backward()
        scaler.step(optimizer)
        scaler.update()

        if epoch % 5 == 0:
            print(f"[Epoch {epoch}/{epochs}] Proto: {loss_proto.item():.4f} "
                  f"(Acc: {acc_proto:.4f}, EMA: {ema_proto_loss:.4f})")

    # ---------------------------
    # 6. 保存模型
    # ---------------------------
    save_dir = 'logs/checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'pag_fsl_vis.pth')
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training finished. Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/lwy/projects/fsan/configs/config.yaml')
    args = parser.parse_args()
    train(args.config)
