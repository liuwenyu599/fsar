import sys, os, torch, random, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# 1. 环境配置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.backbones.structural import StructuralBackbone
from models.backbones.lora_handler import inject_view_lora
from dataloader.multloader import CASIABMultiDataset as S_DS


# ---------------------------------------------------------
# 🌟 物理归一化方案库
# ---------------------------------------------------------

def norm_max_abs(p):
    """方案 A: 全局最大值缩放 (Pelvis 为中心)"""
    p = p.clone()
    center = (p[:, 11:12, :2] + p[:, 12:13, :2]) / 2.0
    p[:, :, :2] -= center
    scale = torch.max(torch.abs(p[:, :, :2])) + 1e-6
    p[:, :, :2] /= scale
    return p


def norm_torso_unit(p):
    """方案 B: 躯干高度标准化 (步态识别 SOTA 常用)"""
    p = p.clone()
    hip = (p[:, 11:12, :2] + p[:, 12:13, :2]) / 2.0
    p[:, :, :2] -= hip
    neck = (p[:, 5:6, :2] + p[:, 6:7, :2]) / 2.0
    torso_h = torch.norm(neck - hip, dim=-1, keepdim=True).mean()
    p[:, :, :2] /= (torso_h + 1e-6)
    return p


def norm_pixel_div(p):
    """方案 C: 图像像素比例缩放 (假设输入是像素坐标)"""
    p = p.clone()
    # 假设 CASIA-B 裁剪后的对齐尺寸或原始尺寸
    p[:, :, 0] /= 640.0
    p[:, :, 1] /= 480.0
    p[:, :, :2] -= 0.5
    return p


# ---------------------------------------------------------
# 2. 核心扫描逻辑
# ---------------------------------------------------------

def run_deep_scan():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 70 + "\n🔎 PPGait 底座物理协议深挖 (针对 Stage 2 权重)\n" + "=" * 70)

    # A. 模型初始化并注入 LoRA
    model = StructuralBackbone().to(device)
    model = inject_view_lora(model, rank=16)

    # B. 加载 Stage 2 权重
    ckpt_path = 'logs/checkpoints/ppgait_struct_adversarial.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt.get('struct_backbone', ckpt)
        model.load_state_dict({k.replace('module.', '').replace('backbone.', ''): v
                               for k, v in sd.items() if
                               k.replace('module.', '').replace('backbone.', '') in model.state_dict()}, strict=False)
        print(f"📦 已加载 Stage 2 权重: {ckpt_path}")
    else:
        print(f"❌ 找不到权重文件: {ckpt_path}");
        return

    model.eval()
    ds = S_DS("/datasets/CASIA-B")
    test_ids = [sid for sid in ds.all_subject_ids if int(sid) > 74]

    # C. 扫描矩阵：归一化方式 x 置信度处理
    # 置信度处理：'raw' (原值), 'one' (强行设为 1), 'zero' (抹除)
    scan_configs = [
        ("Max_Abs_Norm", norm_max_abs, "raw"),
        ("Max_Abs_Norm", norm_max_abs, "one"),
        ("Torso_Unit_Norm", norm_torso_unit, "raw"),
        ("Torso_Unit_Norm", norm_torso_unit, "one"),
        ("Pixel_Scale_Norm", norm_pixel_div, "raw"),
        ("Pixel_Scale_Norm", norm_pixel_div, "one")
    ]

    print(f"\n{'Normalization Strategy':<25} | {'Conf Mode':<10} | {'Acc (5-way)':<12}")
    print("-" * 70)

    for name, norm_func, conf_mode in scan_configs:
        acc = evaluate_strategy(model, ds, test_ids, norm_func, conf_mode, device)
        print(f"{name:<25} | {conf_mode:<10} | {acc:10.2f}%")


def evaluate_strategy(model, ds, ids, norm_func, conf_mode, device):
    accs = []
    # 建立测试缓存 (nm-01, nm-02)
    cache = []
    for sid in ids:
        seqs = [p for p in ds.all_sequences[sid] if 'nm-01' in p or 'nm-02' in p]
        if len(seqs) >= 2: cache.append(seqs)

    if len(cache) < 10: return 0.0

    for _ in range(80):
        sampled = random.sample(cache, 5)
        sup, que = [], []
        with torch.no_grad():
            for person in sampled:
                paths = random.sample(person, 2)
                for i, path in enumerate(paths):
                    raw = ds.load_pose(path)
                    if not isinstance(raw, torch.Tensor): raw = torch.from_numpy(raw)

                    # 1. 归一化
                    p = norm_func(raw)

                    # 2. 置信度修正
                    if conf_mode == 'one':
                        p[:, :, 2] = 1.0
                    elif conf_mode == 'zero':
                        p[:, :, 2] = 0.0

                    # 3. 推理 (固定视角 90 度测试底座感知力)
                    p_in = p.to(device).reshape(1, 60, 51).float()
                    f = model(p_in, view_idx=torch.tensor([5]).to(device)).mean(dim=1)
                    f = F.normalize(f, p=2, dim=-1)
                    (sup if i == 0 else que).append(f)

        if len(que) == 5:
            sim = torch.mm(torch.cat(que), torch.cat(sup).t())
            accs.append((sim.argmax(dim=1) == torch.arange(5).to(device)).float().mean().item())

    return np.mean(accs) * 100


if __name__ == "__main__":
    run_deep_scan()