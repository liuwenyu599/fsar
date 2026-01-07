import sys, os, torch, random, numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# ---------------------------------------------------------
# 1. 环境与组件
# ---------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones.structural import StructuralBackbone
from models.backbones.lora_handler import inject_lora_to_motionbert
from dataloader.multloader import CASIABMultiDataset as S_DS
from dataloader.silu_resnet_loader import CASIASiluDataset as V_DS


def unified_align_pose(pose_data):
    if not isinstance(pose_data, torch.Tensor):
        pose_data = torch.from_numpy(pose_data)
    flat = pose_data.reshape(-1)
    new_data = torch.zeros(3060)
    new_data[:min(flat.shape[0], 3060)] = flat[:min(flat.shape[0], 3060)]
    return new_data.view(60, 51).float()


class VisualFeatureExtractor(nn.Module):
    def __init__(self, dim=512, num_classes=125):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.proj = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, dim))
        self.classifier = nn.Linear(dim, num_classes)  # 用于视觉身份预热

    def forward(self, x, return_logits=False):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.backbone(x)
        feat = self.proj(feat).view(B, T, -1).mean(dim=1)
        if return_logits:
            return feat, self.classifier(feat)
        return feat


class BCAFusionLayer(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.v2s_cross = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.s2v_cross = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm_v, self.norm_s = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, 128), nn.ReLU(), nn.Linear(128, 2))
        # 🌟 初始化：初始让 Alpha_S 偏小，避开 1.0 陷阱
        nn.init.constant_(self.gate[2].bias, -1.0)

    def forward(self, v_f, s_f):
        v, s = v_f.view(-1, 512), s_f.view(-1, 512)
        v_enh = self.norm_v(v + self.v2s_cross(v.unsqueeze(1), s.unsqueeze(1), s.unsqueeze(1))[0].squeeze(1))
        s_enh = self.norm_s(s + self.s2v_cross(s.unsqueeze(1), v.unsqueeze(1), v.unsqueeze(1))[0].squeeze(1))
        w = F.softmax(self.gate(torch.cat([v_enh, s_enh], dim=-1)), dim=-1)
        return w[:, 0:1] * v_enh + w[:, 1:2] * s_enh, w[:, 1:2]


# ---------------------------------------------------------
# 2. 核心训练主程序
# ---------------------------------------------------------
def train_fusion():
    device = "cuda"
    print("\n🔥 [Stage 3.0] 强迫参与训练模式：注入 Modal Dropout ...")

    s_model = inject_lora_to_motionbert(StructuralBackbone(), rank=8).to(device)
    v_model = VisualFeatureExtractor(512, num_classes=125).to(device)
    fusion = BCAFusionLayer(512).to(device)

    # 尝试加载最新权重继续
    ckpt_path = 'logs/checkpoints/ppgait_fusion_triplet_final.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        s_model.load_state_dict(ckpt['struct_lora_state_dict'], strict=False)
        fusion.load_state_dict(ckpt['fusion_layer_state_dict'], strict=False)
        if 'visual_state_dict' in ckpt:
            v_model.load_state_dict(ckpt['visual_state_dict'], strict=False)

    # 梯度解冻：LoRA + Proj + Fusion
    for name, p in s_model.named_parameters(): p.requires_grad = True if 'lora_' in name else False
    for p in v_model.backbone.parameters(): p.requires_grad = False
    for p in v_model.proj.parameters(): p.requires_grad = True
    for p in fusion.parameters(): p.requires_grad = True

    optimizer = optim.AdamW([
        {'params': fusion.parameters(), 'lr': 2e-4},
        {'params': v_model.proj.parameters(), 'lr': 2e-4},
        {'params': [p for p in s_model.parameters() if p.requires_grad], 'lr': 2e-5}  # 略微调高 LoRA 学习率
    ])

    criterion_tri = nn.TripletMarginLoss(margin=0.4)
    criterion_cls = nn.CrossEntropyLoss()
    scaler = GradScaler()

    s_ds, v_ds = S_DS("/datasets/CASIA-B"), V_DS("/datasets/CASIA-B/silu", target_len=4)
    train_ids = [sid for sid in s_ds.all_subject_ids if int(sid) <= 74]

    pbar = tqdm(range(1, 1001), desc="Forced Alignment")
    for step in pbar:
        # 🌟 健壮性采样
        Xs, Xv, labels, raw_ids = [], [], [], []
        while len(labels) < 8:
            sid = random.choice(train_ids)
            sid_str = str(sid).zfill(3)
            try:
                # 必须同时拥有 090 和 180
                p_090 = [p for p in s_ds.all_sequences[sid_str] if 'nm-01' in p and '090' in p][0]
                p_180 = [p for p in s_ds.all_sequences[sid_str] if 'nm-01' in p and '180' in p][0]
                v_090 = v_ds.all_sequences[sid_str]['nm-01']['090']
                v_180 = v_ds.all_sequences[sid_str]['nm-01']['180']

                curr_label = len(raw_ids) // 2
                for p_p, v_v in [(p_090, v_090), (p_180, v_180)]:
                    Xs.append(unified_align_pose(s_ds.load_pose(p_p)))
                    Xv.append(v_ds.load_sequence(v_v))
                    labels.append(curr_label)
                    raw_ids.append(int(sid) - 1)
            except:
                continue

        Xs_t, Xv_t = torch.stack(Xs).to(device), torch.stack(Xv).to(device)
        y_t, rid_t = torch.tensor(labels).to(device), torch.tensor(raw_ids).to(device)

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            # 1. 视觉分支身份预热
            feat_v, logits_v = v_model(Xv_t, return_logits=True)
            l_cls = criterion_cls(logits_v, rid_t)

            # 2. 骨骼分支
            sf = s_model(Xs_t).mean(dim=1)

            # 🌟 [关键点] 模态丢弃：30% 概率关闭骨骼信号，强迫模型练视觉
            if random.random() < 0.3:
                sf = torch.zeros_like(sf)

            # 3. 融合与 Triplet
            ff, alpha = fusion(feat_v, sf)
            ff = F.normalize(ff, dim=-1)

            l_tri = 0
            for k in range(0, len(ff), 2):
                a, p = ff[k:k + 1], ff[k + 1:k + 2]
                neg_indices = [idx for idx in range(len(ff)) if labels[idx] != labels[k]]
                n = ff[random.choice(neg_indices):random.choice(neg_indices) + 1]
                l_tri += criterion_tri(a, p, n)

            loss = 1.0 * l_cls + 2.0 * (l_tri / (len(ff) / 2))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 5 == 0:
            pbar.set_postfix(
                {"L_Cls": f"{l_cls.item():.2f}", "L_Tri": f"{l_tri.item():.2f}", "Alpha": f"{alpha.mean().item():.3f}"})

    # 保存
    torch.save({
        'fusion_layer_state_dict': fusion.state_dict(),
        'struct_lora_state_dict': {k: v for k, v in s_model.state_dict().items() if 'lora_' in k},
        'visual_state_dict': v_model.state_dict()
    }, 'logs/checkpoints/ppgait_fusion_triplet_final.pth')


if __name__ == "__main__":
    train_fusion()