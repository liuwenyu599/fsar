import sys, os, torch, random, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# 1. 环境与路径
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones.structural import StructuralBackbone
from models.backbones.lora_handler import inject_view_lora
from dataloader.multloader import CASIABMultiDataset as S_DS
from dataloader.silu_resnet_loader import CASIASiluDataset as V_DS

TARGET_VIEWS = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
VIEW_TO_IDX = {v: i for i, v in enumerate(TARGET_VIEWS)}


# --- 模型定义 ---
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
        with torch.no_grad(): feat = self.backbone(x)
        return self.proj(feat).view(B, T, -1).mean(dim=1)


class CredibilityGate(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.view_emb = nn.Embedding(11, 128)
        self.mapper = nn.Sequential(nn.Linear(128 + dim * 2, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, v_f, s_f, v_idx):
        v, s = F.normalize(v_f, p=2, dim=-1), F.normalize(s_f, p=2, dim=-1)
        raw_out = self.mapper(torch.cat([v, s, self.view_emb(v_idx)], dim=-1))
        weights = torch.softmax(raw_out, dim=-1)
        return weights[:, 0:1] * v + weights[:, 1:2] * s, weights


def surgery_load(model, sd, name):
    model_dict = model.state_dict()
    new_sd = {k.replace('module.', '').replace('backbone.', ''): v
              for k, v in sd.items() if k.replace('module.', '').replace('backbone.', '') in model_dict
              and v.shape == model_dict[k.replace('module.', '').replace('backbone.', '')].shape}
    model.load_state_dict(new_sd, strict=False)
    print(f"📦 [{name}] 成功缝合 {len(new_sd)} Keys")


def build_aligned_cache(s_ds, v_ds, ids):
    print("🔍 正在扫描测试集跨视角对齐数据...")
    cache = []
    for sid in ids:
        sid_str = str(sid).zfill(3)
        if sid_str not in s_ds.all_sequences or sid_str not in v_ds.all_sequences: continue
        s_v_map = {}
        for p in s_ds.all_sequences[sid_str]:
            parts = p.split(os.sep)
            try:
                v_name, seq_name = parts[-2], parts[-3]
                if seq_name not in s_v_map: s_v_map[seq_name] = {}
                s_v_map[seq_name][v_name] = p
            except:
                continue
        v_seq_dict = v_ds.all_sequences[sid_str]
        # 对该 Subject 寻找跨序列的交集
        for v in TARGET_VIEWS:
            valid_seqs = [seq for seq in s_v_map if seq in v_seq_dict and v in s_v_map[seq] and v in v_seq_dict[seq]]
            if len(valid_seqs) >= 2:  # 必须同一视角有两个不同序列
                cache.append({'sid': sid_str, 'view': v, 'seqs': valid_seqs, 's_data': s_v_map, 'v_data': v_seq_dict})
    return cache


# ---------------------------------------------------------
# 执行测试
# ---------------------------------------------------------
def test_synergy():
    device = "cuda"
    s_model = StructuralBackbone().to(device)
    s_model = inject_view_lora(s_model, rank=16)
    v_model = VisualFeatureExtractor(512).to(device)
    gate = CredibilityGate(512).to(device)

    # A. 加载最终权重
    try:
        final_ckpt = torch.load('logs/checkpoints/ppgait_synergy_final.pth', map_location=device)
        base_ckpt = torch.load('logs/checkpoints/ppgait_struct_adversarial.pth', map_location=device)

        surgery_load(s_model, base_ckpt['struct_backbone'], "Base_Backbone")
        surgery_load(s_model, final_ckpt['refined_lora'], "Refined_LoRA")
        surgery_load(gate, final_ckpt['gate_state_dict'], "Gate")
        surgery_load(v_model.proj, final_ckpt['visual_proj'], "Visual_Proj")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return

    s_model.eval();
    v_model.eval();
    gate.eval()

    # B. 数据准备
    s_ds, v_ds = S_DS("/datasets/CASIA-B"), V_DS("/datasets/CASIA-B/silu", target_len=4)
    test_ids = [sid for sid in s_ds.all_subject_ids if int(sid) > 74]
    cache = build_aligned_cache(s_ds, v_ds, test_ids)
    print(f"✅ 对齐完成，共找到 {len(cache)} 组有效测试单元。")

    results = {v: {'f': [], 'p': []} for v in TARGET_VIEWS}

    for _ in tqdm(range(600), desc="Evaluating Synergy"):
        v_name = random.choice(TARGET_VIEWS)
        v_idx = VIEW_TO_IDX[v_name]
        # 🌟 修正：匹配正确的键名 'view'
        items = [i for i in cache if i['view'] == v_name]
        if len(items) < 5: continue

        sampled = random.sample(items, 5)
        sup_f, que_f, sup_p, que_p = [], [], [], []

        with torch.no_grad():
            for item in sampled:
                # 采样该视角下的两个不同序列（如 nm-01, nm-02）
                seqs = random.sample(item['seqs'], 2)
                for i, seq in enumerate(seqs):
                    # Pose Feature
                    pose_path = item['s_data'][seq][v_name]
                    fs = s_model(s_ds.load_pose(pose_path).to(device).view(1, 60, 51).float(),
                                 view_idx=torch.tensor([v_idx]).to(device)).mean(dim=1)
                    # Visual Feature
                    vis_path = item['v_data'][seq][v_name]
                    fv = v_model(v_ds.load_sequence(vis_path).unsqueeze(0).to(device))

                    # Fusion
                    ff, _ = gate(fv, fs, torch.tensor([v_idx]).to(device))

                    if i == 0:  # 第一个序列进 Support
                        sup_f.append(ff)
                        sup_p.append(F.normalize(fs, p=2, dim=-1))
                    else:  # 第二个序列进 Query
                        que_f.append(ff)
                        que_p.append(F.normalize(fs, p=2, dim=-1))

        def calc_acc(q, s):
            if len(q) < 5 or len(s) < 5: return 0.0
            sim = torch.mm(torch.cat(q), torch.cat(s).t())
            return (sim.argmax(dim=1) == torch.arange(5).to(device)).float().mean().item()

        results[v_name]['f'].append(calc_acc(que_f, sup_f))
        results[v_name]['p'].append(calc_acc(que_p, sup_p))

    # C. 打印对比表
    print("\n" + "=" * 50)
    print(f"{'View':<6} | {'Pose Acc (%)':<15} | {'Synergy Acc (%)':<15}")
    print("-" * 50)
    p_scores, f_scores = [], []
    for v in TARGET_VIEWS:
        if results[v]['f']:
            pa, fa = np.mean(results[v]['p']) * 100, np.mean(results[v]['f']) * 100
            p_scores.append(pa);
            f_scores.append(fa)
            print(f"{v:<6} | {pa:<15.2f} | {fa:<15.2f}")
    print("-" * 50)
    print(f"MEAN   | {np.mean(p_scores):<15.2f} | {np.mean(f_scores):<15.2f}")
    print("=" * 50)


if __name__ == "__main__":
    test_synergy()