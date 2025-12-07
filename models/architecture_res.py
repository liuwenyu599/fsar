# models/architecture_res.py
import torch
import torch.nn as nn
from models.backbones.visual_resnet import VisualBackbone
from models.backbones.structural import StructuralBackbone
from models.heads.projections import ProjectionHead
from models.heads.ppm import PPMStructuredAggregator
from models.decoders.fam_fusion import FAMFusion


class PAGFSLModel(nn.Module):
    """
    PAG-FSL 主模型
    视觉流: ViT + LoRA (显存友好分批)
    结构流: MotionBERT / 简化版
    特征融合: PPM + FAM
    """
    def __init__(self, common_dim=512, input_struct=51, lora_rank=16):
        super().__init__()

        # --- 1. 视觉流骨干 ---
        self.vis_backbone = VisualBackbone(lora_rank=lora_rank,pretrained=False)
        vis_out_dim = self.vis_backbone.out_dim

        # --- 2. 结构流骨干 ---
        self.struct_backbone = StructuralBackbone(input_dim=input_struct, embed_dim=common_dim)
        struct_out_dim = self.struct_backbone.out_dim

        # --- 3. PPM 强化结构流 ---
        self.ppm = PPMStructuredAggregator(feature_dim=struct_out_dim)

        # --- 4. 投影头 (对齐到 common_dim) ---
        self.proj_vis = ProjectionHead(input_dim=vis_out_dim, common_dim=common_dim)
        self.proj_struct = ProjectionHead(input_dim=struct_out_dim, common_dim=common_dim)

        # --- 5. FAM 融合模块 ---
        self.fam = FAMFusion(common_dim)

    def forward_feature(self, x_vis, x_struct, phase_weights, batch_size=16):
        """
        前向传播 (显存友好)
        Args:
            x_vis: (B, T, C, H, W)
            x_struct: (B, T, D_struct)
            phase_weights: (B, T)
            batch_size: ViT 分批大小
        Returns:
            f_final: (B, common_dim)
            f_vis: (B, common_dim)
            f_struct: (B, common_dim)
        """
        B, T, C, H, W = x_vis.shape

        # --- A. 视觉流: 分批送 ViT ---
        x_vis_flat = x_vis.view(B*T, C, H, W)
        features_vis = []

        for i in range(0, B*T, batch_size):
            x_batch = x_vis_flat[i:i+batch_size]
            f_batch = self.vis_backbone(x_batch)  # (batch, vis_out_dim)
            features_vis.append(f_batch)

        f_vis_flat = torch.cat(features_vis, dim=0)  # (B*T, vis_out_dim)
        f_vis_seq = f_vis_flat.view(B, T, -1)        # (B, T, vis_out_dim)
        f_vis_pool = f_vis_seq.mean(dim=1)           # (B, vis_out_dim)
        f_vis = self.proj_vis(f_vis_pool)            # (B, common_dim)

        # --- B. 结构流 ---
        # 注意：struct_backbone 输出 shape 已是 (B, T, common_dim)
        f_struct_seq = self.struct_backbone(x_struct)        # (B, T, common_dim)
        f_struct_ppm = self.ppm(f_struct_seq, phase_weights) # (B, common_dim)
        f_struct = self.proj_struct(f_struct_ppm)           # (B, common_dim)

        # --- C. 融合 ---
        f_final = self.fam(f_vis, f_struct)                 # (B, common_dim)

        return f_final, f_vis, f_struct
