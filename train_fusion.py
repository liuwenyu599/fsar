import sys, os, torch, random, numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------
# 1. 环境配置与路径
# ---------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones.structural import StructuralBackbone
from models.backbones.lora_handler import inject_view_lora
from dataloader.multloader import CASIABMultiDataset as S_DS
from dataloader.silu_resnet_loader import CASIASiluDataset as V_DS

TARGET_VIEWS = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
VIEW_TO_IDX = {v: i for i, v in enumerate(TARGET_VIEWS)}


# ---------------------------------------------------------
# 2. 模型架构组件
# ---------------------------------------------------------
class VisualFeatureExtractor(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.proj = nn.Sequential(nn.Linear(2048, dim))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        with torch.no_grad():
            feat = self.backbone(x)
        return self.proj(feat).view(B, T, -1).mean(dim=1)


class CredibilityGate(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.view_emb = nn.Embedding(11, 128)
        self.mapper = nn.Sequential(
            nn.Linear(128 + dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, v_f, s_f, v_idx):
        v, s = F.normalize(v_f, p=2, dim=-1), F.normalize(s_f, p=2, dim=-1)
        raw_out = self.mapper(torch.cat([v, s, self.view_emb(v_idx)], dim=-1))
        weights = torch.softmax(raw_out, dim=-1)
        f_fused = weights[:, 0:1] * v + weights[:, 1:2] * s
        return F.normalize(f_fused, p=2, dim=-1), weights


# ---------------------------------------------------------
# 3. 手术级加载与对齐工具
# ---------------------------------------------------------
def surgery_load(model, ckpt_path, block_name):
    if not os.path.exists(ckpt_path):
        print(f"⚠️ 警告: 未找到 {ckpt_path}");
        return
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt.get(block_name, ckpt)
    model.load_state_dict({k.replace('module.', '').replace('backbone.', ''): v
                           for k, v in sd.items() if
                           k.replace('module.', '').replace('backbone.', '') in model.state_dict()}, strict=False)


def build_aligned_cache(s_ds, v_ds, ids):
    cache = []
    for sid in ids:
        sid_str = str(sid).zfill(3)
        if sid_str not in s_ds.all_sequences or sid_str not in v_ds.all_sequences: continue
        s_v_map = {}
        for p in s_ds.all_sequences[sid_str]:
            parts = p.split(os.sep)
            try:
                v_name, seq_name = parts[-2], parts[-3]
            except:
                continue
            if seq_name not in s_v_map: s_v_map[seq_name] = {}
            s_v_map[seq_name][v_name] = p
        v_seq_dict = v_ds.all_sequences[sid_str]
        for seq_name, v_dict in s_v_map.items():
            if seq_name in v_seq_dict:
                v_data = v_seq_dict[seq_name]
                common = [v for v in TARGET_VIEWS if v in v_dict and v in v_data]
                if len(common) >= 2:
                    cache.append({'sid': sid_str, 'views': common, 's_map': v_dict, 'v_map': v_data})
    return cache


def get_safe_batch(cache, s_ds, v_ds, batch_size=14):
    Xs, Xv, vi, labels = [], [], [], []
    selected = random.sample(cache, batch_size)
    for i, item in enumerate(selected):
        v1, v2 = random.sample(item['views'], 2)
        for v in [v1, v2]:
            pose = s_ds.load_pose(item['s_map'][v])
            if not isinstance(pose, torch.Tensor): pose = torch.from_numpy(pose)
            Xs.append(pose.view(60, 51).float())
            Xv.append(v_ds.load_sequence(item['v_map'][v]))
            vi.append(VIEW_TO_IDX[v]);
            labels.append(i)
    return torch.stack(Xs), torch.stack(Xv), torch.tensor(vi), torch.tensor(labels)


# ---------------------------------------------------------
# 4. 协同精炼训练逻辑 (Stage 3.4)
# ---------------------------------------------------------
def train_synergy():
    device = "cuda"
    print("\n🎯 [Stage 3.4] PPGait 协同精炼：固定天平，精炼骨架")

    # A. 结构流：激活 LoRA，加载 Stage 2 底座
    s_model = StructuralBackbone().to(device)
    s_model = inject_view_lora(s_model, rank=16)
    surgery_load(s_model, 'logs/checkpoints/ppgait_struct_adversarial.pth', 'struct_backbone')

    # 🌟 关键：只允许 LoRA 参数更新
    for n, p in s_model.named_parameters():
        p.requires_grad = True if 'lora_' in n else False
    s_model.train()

    # B. 视觉流：绝对冻结作为特征锚点
    v_model = VisualFeatureExtractor(512).to(device)
    v_model.eval()
    for p in v_model.parameters(): p.requires_grad = False

    # C. Gate：固化 Stage 3.3 的权重天平
    gate = CredibilityGate(512).to(device)
    surgery_load(gate, 'logs/checkpoints/ppgait_minimalist_gate.pth', 'gate')
    gate.eval()
    for p in gate.parameters(): p.requires_grad = False

    # D. 优化器与损失函数
    optimizer = optim.AdamW([p for p in s_model.parameters() if p.requires_grad], lr=8e-5)
    criterion = nn.TripletMarginLoss(margin=0.3)

    # E. 数据加载
    s_ds, v_ds = S_DS("/datasets/CASIA-B"), V_DS("/datasets/CASIA-B/silu", target_len=4)
    train_ids = [sid for sid in s_ds.all_subject_ids if int(sid) <= 74]
    cache = build_aligned_cache(s_ds, v_ds, train_ids)
    print(f"✅ 数据扫描完成：共 {len(cache)} 组对齐样本。")

    print("\n" + "-" * 85)
    print(f"{'Step':<8} | {'Loss':<8} | {'PoseW':<8} | {'TripAcc':<8} | {'LR':<8}")
    print("-" * 85)
    # F. 训练循环
    for step in range(1, 1501):
        Xs, Xv, vi, labels = get_safe_batch(cache, s_ds, v_ds, batch_size=14)
        Xs, Xv, vi, labels = Xs.to(device), Xv.to(device), vi.to(device), labels.to(device)

        optimizer.zero_grad()

        # 1. 提取骨骼特征 (必须在 grad 模式下，因为它连接着我们要练的 LoRA)
        f_s = s_model(Xs, view_idx=vi).mean(dim=1)

        # 2. 提取视觉特征 (可以在 no_grad 下，因为它完全冻结且不回传梯度)
        with torch.no_grad():
            f_v = v_model(Xv)

        # 3. 🌟 关键修正：gate 调用必须在 no_grad 之外！
        # 虽然 gate.parameters 已经 requires_grad=False，但它需要转发 f_s 的梯度
        f_fused, weights = gate(f_v, f_s, vi)

        # 4. 计算 Loss
        loss = 0
        hits = 0
        count = 0
        for i in range(0, len(f_fused), 2):
            anc, pos = f_fused[i:i + 1], f_fused[i + 1:i + 2]
            neg_idx = random.choice([idx for idx in range(len(f_fused)) if labels[idx] != labels[i]])
            neg = f_fused[neg_idx:neg_idx + 1]

            l_tri = criterion(anc, pos, neg)
            loss += l_tri

            # 统计 Batch 内的 Triplet Accuracy
            if F.pairwise_distance(anc, pos) < F.pairwise_distance(anc, neg):
                hits += 1
            count += 1

        loss = loss / count

        # 5. 回传梯度 (现在路径通了：Loss -> f_fused -> gate -> f_s -> LoRA)
        loss.backward()
        optimizer.step()

        # 每 100 次迭代输出详细报告
        if step % 100 == 0 or step == 1:
            pw = weights[:, 1].mean().item()
            t_acc = (hits / count) * 100
            print(
                f"Step {step:04d} | {loss.item():.4f} | {pw:.4f}   | {t_acc:6.1f}% | {optimizer.param_groups[0]['lr']:.2e}")
    # G. 保存合体后的最终权重
    torch.save({
        'refined_lora': {k: v for k, v in s_model.state_dict().items() if 'lora_' in k},
        'gate_state_dict': gate.state_dict(),
        'visual_proj': v_model.proj.state_dict()
    }, 'logs/checkpoints/ppgait_synergy_final.pth')
    print("-" * 85)
    print("✅ Stage 3.4 协同精炼完成。系统已达到最终稳态。")


if __name__ == "__main__":
    train_synergy()