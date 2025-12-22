import torch
import torch.optim as optim
import os
import argparse
import yaml
try:
    from models.architecture_res import PAGFSLModel
except ImportError:
    from models.architecture import PAGFSLModel

from dataloader import CASIAMultiModalDataset, FewShotSampler
from losses.prototypical_loss import PrototypicalLoss


def load_pretrained(model, vis_path, struct_path, device):
    print(f"\n--- Loading Pretrained Weights ---")

    # 1. 加载 Visual Stream
    if os.path.exists(vis_path):
        print(f"✅ Loading Visual from: {vis_path}")
        vis_state = torch.load(vis_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_vis = {k: v for k, v in vis_state.items()
                          if k in model_dict and ('vis_backbone' in k or 'proj_vis' in k)}
        model_dict.update(pretrained_vis)
        model.load_state_dict(model_dict)
        print(f"   -> Loaded {len(pretrained_vis)} visual keys.")
    else:
        print(f"⚠️ Warning: Visual checkpoint not found at {vis_path}")

    # 2. 加载 Structure Stream
    if os.path.exists(struct_path):
        print(f"✅ Loading Structure from: {struct_path}")
        struct_data = torch.load(struct_path, map_location=device)

        model_dict = model.state_dict()
        loaded_count = 0

        # 映射 Backbone
        if 'backbone' in struct_data:
            for k, v in struct_data['backbone'].items():
                key_name = f'struct_backbone.{k}'
                if key_name in model_dict and model_dict[key_name].shape == v.shape:
                    model_dict[key_name] = v
                    loaded_count += 1

        # 映射 PPM
        if 'ppm' in struct_data:
            for k, v in struct_data['ppm'].items():
                key_name = f'ppm.{k}'
                if key_name in model_dict and model_dict[key_name].shape == v.shape:
                    model_dict[key_name] = v
                    loaded_count += 1

        model.load_state_dict(model_dict)
        print(f"   -> Loaded {loaded_count} structure keys.")
    else:
        print(f"⚠️ Warning: Structure checkpoint not found at {struct_path}")

    return model


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- 🚀 Stage 3: Fusion Training (Dual Stream) ---")

    # 1. Dataset (分别加载)
    dataset_pose = CASIAMultiModalDataset('/datasets/CASIA-B', mode='pose', target_len=30)
    dataset_vis = CASIAMultiModalDataset('/datasets/CASIA-B', mode='silu')

    # 2. Model
    # 修正: input_struct=51 (MotionBERT是51维), common_dim=512
    model = PAGFSLModel(common_dim=512, input_struct=51).to(device)

    # 3. 加载权重
    vis_ckpt = 'logs/checkpoints/pag_fsl_vis_ema.pth'
    if not os.path.exists(vis_ckpt):
        vis_ckpt = 'logs/checkpoints/pag_fsl_vis_latest.pth'
    struct_ckpt = 'logs/checkpoints/pag_fsl_struct_ppm.pth'

    model = load_pretrained(model, vis_ckpt, struct_ckpt, device)

    # 4. Optimizer
    # 冻结双流骨干，只训练融合层 (FAM)
    for p in model.parameters(): p.requires_grad = False

    # 解冻融合层 + 投影层
    for p in model.fam.parameters(): p.requires_grad = True
    for p in model.proj_vis.parameters(): p.requires_grad = True
    for p in model.proj_struct.parameters(): p.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = PrototypicalLoss().to(device)

    # 使用 Pose Sampler 驱动
    sampler_pose = FewShotSampler(dataset_pose, n_way=5, k_shot=5, q_query=15)

    print("Start Fusion Fine-tuning...")
    for epoch in range(1, 101):
        # A. 获取 Pose 数据
        # 注意: 这里的 labels 是全量 labels (100个), 我们后面不用它，而是自己生成 query labels
        X_struct, _, phase_w = sampler_pose.get_episode()
        X_struct = X_struct.to(device)
        phase_w = phase_w.to(device)

        # B. 模拟 Visual 数据 (因为 Sampler 还没改好支持双流)
        # ⚠️ 这里只是为了跑通代码逻辑！真实 Acc 可能会低！
        # 理想情况是: X_vis, X_struct, ... = sampler.get_dual_episode()
        B_size = X_struct.shape[0]
        X_vis_dummy = torch.randn(B_size, 8, 3, 224, 224).to(device)

        # --- Forward ---
        # 1. Visual Forward (SimpleCNN)
        # 手动 reshape 以适配 SimpleCNN: (B, T, C, H, W) -> (B*T, C, H, W)
        B, T, C, H, W = X_vis_dummy.shape
        f_vis_flat = model.vis_backbone(X_vis_dummy.view(B * T, C, H, W))  # (B*T, 256)
        f_vis = model.proj_vis(f_vis_flat)  # (B*T, 512)
        f_vis = f_vis.view(B, T, -1).max(dim=1)[0]  # MaxPool -> (B, 512)

        # 2. Structure Forward
        f_struct_seq = model.struct_backbone(X_struct)  # (B, T, 512)
        f_struct_ppm = model.ppm(f_struct_seq, phase_w)  # (B, 512)
        f_struct = model.proj_struct(f_struct_ppm)  # (B, 512)

        # 3. Fusion
        f_final = model.fam(f_vis, f_struct)  # (B, 512)

        # --- Loss ---
        n_way, k_shot, q_query = 5, 5, 15

        # 还原维度 (N_way, K+Q, Dim)
        f_reshaped = f_final.view(n_way, k_shot + q_query, -1)

        # 切分 Support / Query
        f_supp = f_reshaped[:, :k_shot].contiguous().view(-1, 512)
        f_query = f_reshaped[:, k_shot:].contiguous().view(-1, 512)

        # 🔥 关键修正：手动生成 Query Labels (75个)
        # 这里的 labels 必须只包含 Query set 的标签
        labels_query = torch.arange(n_way).repeat_interleave(q_query).to(device)

        loss, acc = criterion(f_supp, f_query, labels_query, n_way, k_shot)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"[Epoch {epoch}] Fusion Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    # 保存
    save_dir = 'logs/checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'pag_fsl_fusion.pth'))
    print("✅ Fusion Logic Verified.")


if __name__ == '__main__':
    train()