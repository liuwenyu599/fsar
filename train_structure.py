import os, sys, datetime, random, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

# 导入项目组件
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.backbones.structural import StructuralBackbone
from models.backbones.lora_handler import inject_view_lora
from models.heads.view_discriminator import ViewDiscriminator
from losses.prototypical_loss import PrototypicalLoss
from dataloader.multloader import CASIABMultiDataset, FewShotSampler


# ---------------------------------------------------------
# 1. 物理层：带时序采样的 Torso Unit Normalization
# ---------------------------------------------------------
def torso_unit_norm(pose, target_len=60):
    """
    pose: [T, 17, 3] or [T*51]
    """
    # 确保维度为 [T, 17, 3]
    if pose.dim() == 1:
        pose = pose.reshape(-1, 17, 3)
    elif pose.dim() == 2:
        pose = pose.reshape(-1, 17, 3)

    curr_len = pose.shape[0]

    # 🌟 时序对齐逻辑：确保输出固定为 target_len 帧
    #
    if curr_len > target_len:
        # 随机裁剪
        start = random.randint(0, curr_len - target_len)
        pose = pose[start:start + target_len]
    elif curr_len < target_len:
        # 循环补齐
        indices = np.arange(curr_len)
        pad_indices = np.random.choice(indices, target_len - curr_len)
        indices = np.sort(np.concatenate([indices, pad_indices]))
        pose = pose[indices]

    p = pose.clone().float()

    # 物理归一化 (以腰部为原点)
    hip_center = (p[:, 11:12, :2] + p[:, 12:13, :2]) / 2.0
    p[:, :, :2] -= hip_center

    # 尺度归一化 (躯干高度)
    neck_center = (p[:, 5:6, :2] + p[:, 6:7, :2]) / 2.0
    torso_h = torch.norm(neck_center - hip_center, dim=-1, keepdim=True).mean()
    p[:, :, :2] /= (torso_h + 1e-6)

    return p.reshape(target_len, 51)  # 最终输出固定 [60, 51]


# ---------------------------------------------------------
# 2. 核心加载逻辑：解决 Pickle 和 维度报错
# ---------------------------------------------------------
def safe_load_pose(path):
    try:
        if path.endswith('.pth'):
            data = torch.load(path, map_location='cpu')
        else:
            data = np.load(path, allow_pickle=True)
            if data.dtype == 'O' and data.shape == ():
                data = data.item()

        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)

        # 🌟 修复关键：现在支持任意长度的输入，自动采样为 60 帧
        return torso_unit_norm(data, target_len=60)
    except Exception as e:
        # 打印详细错误以便排查
        # print(f"加载失败: {path} | 错误: {e}")
        return torch.zeros(60, 51)


# ---------------------------------------------------------
# 3. 结构定义 (PPM)
# ---------------------------------------------------------
class StructuralPPM(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x, phase_w):
        phase_w = F.softmax(phase_w, dim=1).unsqueeze(-1)
        return torch.sum(x * phase_w, dim=1)


# ---------------------------------------------------------
# 4. 训练主程序
# ---------------------------------------------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 60 + "\n🛡️ Stage 2 V3: Live-LoRA & Physical Alignment\n" + "=" * 60)

    dataset = CASIABMultiDataset('/datasets/CASIA-B', mode='pose', seq_len=60)
    dataset.load_pose = safe_load_pose
    sampler = FewShotSampler(dataset, n_way=5, k_shot=5, q_query=5)

    model = StructuralBackbone().to(device)
    model = inject_view_lora(model, rank=16)
    ppm = StructuralPPM(feature_dim=512).to(device)
    discriminator = ViewDiscriminator(feature_dim=512, num_views=11).to(device)

    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 1e-4},
        {'params': ppm.parameters(), 'lr': 1e-4},
        {'params': discriminator.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000)
    scaler = GradScaler()
    criterion_id = PrototypicalLoss().to(device)
    criterion_view = nn.CrossEntropyLoss().to(device)

    writer = SummaryWriter(
        os.path.join("logs", "tensorboard", "V3_S2_" + datetime.datetime.now().strftime("%m%d-%H%M")))

    best_acc = 0.0
    temp = 16.0
    warmup = 500

    for step in range(1, 3001):
        model.train();
        ppm.train();
        discriminator.train()

        # 🌟 修复变量解包报错：使用 *args 接收所有返回值，只取前 4 个
        batch_data = sampler.get_episode(mode='train')
        X, Y, W, v_idx = batch_data[0], batch_data[1], batch_data[2], batch_data[3]

        X, Y, W, v_idx = X.to(device), Y.to(device), W.to(device), v_idx.to(device)

        alpha = min(1.0, (step - warmup) / 1000.0) if step > warmup else 0.0

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            feat_seq = F.normalize(model(X, view_idx=v_idx), p=2, dim=-1) * temp
            f_final = ppm(feat_seq, W)

            n, k, q = 5, 5, 5
            f_r = f_final.view(n, k + q, -1)
            q_labels = torch.arange(n).repeat_interleave(q).to(device)
            loss_id, train_acc = criterion_id(f_r[:, :k].reshape(-1, 512),
                                              f_r[:, k:].reshape(-1, 512), q_labels, n, k)

            view_pred = discriminator(f_final, alpha=alpha)
            loss_view = criterion_view(view_pred, v_idx)

            total_loss = loss_id + alpha * loss_view

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % 20 == 0:
            print(
                f"[Step {step}] L_ID: {loss_id.item():.3f} | L_View: {loss_view.item():.3f} | Acc: {train_acc.item():.4f}")
            writer.add_scalar('Train/Acc', train_acc.item(), step)

        if step % 200 == 0:
            # 同样应用解包修复逻辑 (在 validate 内部也需要注意)
            val_acc = validate(model, ppm, sampler, device, temp)
            print(f"🌈 Step {step} Validation Acc: {val_acc:.2f}%")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({'struct_backbone': model.state_dict(), 'ppm_state_dict': ppm.state_dict()},
                           'logs/checkpoints/ppgait_v3_best.pth')

    writer.close()


def validate(model, ppm, sampler, device, temp):
    model.eval();
    ppm.eval()
    accs = []
    for _ in range(50):
        # 🌟 同样的解包修复
        batch_data = sampler.get_episode(mode='test')
        X, Y, W, v_idx = batch_data[0], batch_data[1], batch_data[2], batch_data[3]

        X, Y, W, v_idx = X.to(device), Y.to(device), W.to(device), v_idx.to(device)
        with torch.no_grad():
            feat_seq = F.normalize(model(X, view_idx=v_idx), p=2, dim=-1) * temp
            f_final = ppm(feat_seq, W)
            n, k, q = 5, 5, 5
            f_r = f_final.view(n, k + q, -1)
            prototypes = f_r[:, :k].mean(dim=1)
            queries = f_r[:, k:].reshape(n * q, -1)
            dists = torch.cdist(queries, prototypes)
            pred = dists.argmin(dim=1)
            target = torch.arange(n).repeat_interleave(q).to(device)
            accs.append((pred == target).float().mean().item())
    return np.mean(accs) * 100


if __name__ == '__main__':
    train()